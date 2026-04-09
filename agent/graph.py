"""LangGraph routing for direct translation and institutional review.
The graph shape is fixed: agent -> tools -> agent with conditional routing to END.
"""

import json
import operator
import os
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from openai import OpenAI

from agent.tools import TOOLS, load_local_env


SYSTEM_PROMPT = (
    "You are an English-to-Spanish translation router. "
    "Use translate_with_custom_model for general translation requests or 'how do you say' questions. "
    "Use rag_translate for parliamentary, committee, motion, session, council, or other institutional translation requests. "
    "If a tool result is already present, answer directly without calling another tool."
)


class AgentState(TypedDict):
    """State carried through the LangGraph agent loop.
    Messages are appended rather than replaced on each node return.
    """

    messages: Annotated[list[AnyMessage], operator.add]


def _build_openai_client() -> Optional[OpenAI]:
    """Create an OpenAI client for tool routing when a key is available.
    Returns `None` so the graph can run in offline heuristic mode.
    """
    load_local_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key, timeout=30.0, max_retries=2)


def _tool_schemas() -> list[dict]:
    """Return OpenAI tool schemas for the supported agent actions.
    The schema is intentionally narrow so GPT-4o-mini picks one tool cleanly.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "translate_with_custom_model",
                "description": "Translate English text into Spanish using the custom FastAPI model.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The English text to translate into Spanish.",
                        }
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rag_translate",
                "description": "Translate English text into Spanish using retrieved translation-memory context for institutional terminology.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The English text to translate with translation-memory context.",
                        }
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _stringify_message_content(content: object) -> str:
    """Convert LangChain message content into plain text for OpenAI requests.
    Non-string content is JSON-encoded to avoid silently dropping information.
    """
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _langchain_to_openai_messages(messages: list[AnyMessage]) -> list[dict]:
    """Convert LangChain message objects into Chat Completions messages.
    Tool messages preserve their `tool_call_id` so OpenAI can continue the exchange.
    """
    openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for message in messages:
        if isinstance(message, HumanMessage):
            openai_messages.append(
                {"role": "user", "content": _stringify_message_content(message.content)}
            )
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                tool_calls = []
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["args"], ensure_ascii=False),
                            },
                        }
                    )
                openai_messages.append(
                    {
                        "role": "assistant",
                        "content": _stringify_message_content(message.content),
                        "tool_calls": tool_calls,
                    }
                )
            else:
                openai_messages.append(
                    {
                        "role": "assistant",
                        "content": _stringify_message_content(message.content),
                    }
                )
        elif isinstance(message, ToolMessage):
            openai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": _stringify_message_content(message.content),
                }
            )

    return openai_messages


def _extract_latest_user_text(messages: list[AnyMessage]) -> str:
    """Return the most recent human message content for routing heuristics.
    The fallback router only needs the latest user turn to choose a tool.
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _stringify_message_content(message.content)
    return ""


def _heuristic_tool_call(user_text: str) -> Optional[dict]:
    """Select a tool deterministically when OpenAI is unavailable.
    The fallback is intentionally narrow and targets the focused translation path.
    """
    lowered = user_text.lower().strip()
    if not lowered:
        return None

    if "translate" in lowered or "how do you say" in lowered or "to spanish" in lowered:
        text_match = None
        if "'" in user_text:
            parts = user_text.split("'")
            if len(parts) >= 3:
                text_match = parts[1].strip()
        if not text_match:
            text_match = user_text
        rag_keywords = [
            "parliament",
            "parliamentary",
            "session",
            "adjourned",
            "motion",
            "committee",
            "commission",
            "council",
            "amendment",
            "rapporteur",
            "plenary",
        ]
        tool_name = (
            "rag_translate"
            if any(keyword in lowered for keyword in rag_keywords)
            else "translate_with_custom_model"
        )
        return {
            "name": tool_name,
            "args": {"text": text_match},
            "id": "offline_translate",
            "type": "tool_call",
        }
    return None


def _route_with_openai(messages: list[AnyMessage]) -> Optional[AIMessage]:
    """Ask GPT-4o-mini to choose a tool call when an API key is available.
    Returns `None` so the caller can fall back to deterministic routing.
    """
    client = _build_openai_client()
    if client is None:
        return None

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=_langchain_to_openai_messages(messages),
        tools=_tool_schemas(),
        tool_choice="auto",
    )
    assistant_message = response.choices[0].message

    if assistant_message.tool_calls:
        tool_calls = []
        for tool_call in assistant_message.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"text": tool_call.function.arguments or ""}
            tool_calls.append(
                {
                    "name": tool_call.function.name,
                    "args": args,
                    "id": tool_call.id,
                    "type": "tool_call",
                }
            )
        return AIMessage(
            content=assistant_message.content or "",
            tool_calls=tool_calls,
        )

    return AIMessage(content=assistant_message.content or "")


def agent_node(state: AgentState) -> dict[str, list[AnyMessage]]:
    """Route the latest user request or finalize after a tool call.
    Tool outputs are returned directly on the second pass through the agent node.
    """
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, ToolMessage):
        return {
            "messages": [
                AIMessage(content=_stringify_message_content(last_message.content))
            ]
        }

    routed_message = _route_with_openai(messages)
    if routed_message is not None:
        return {"messages": [routed_message]}

    user_text = _extract_latest_user_text(messages)
    tool_call = _heuristic_tool_call(user_text)
    if tool_call is not None:
        return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}

    return {
        "messages": [
            AIMessage(
                content=(
                    "I can translate English to Spanish and route institutional sentences "
                    "through the translation review path."
                )
            )
        ]
    }


def should_continue(state: AgentState) -> str:
    """Route to ToolNode when the agent emitted tool calls.
    Returning `END` stops the graph after a direct response.
    """
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_graph():
    """Build and compile the fixed LangGraph routing graph.
    The graph shape remains `agent -> tools -> agent` until no tool calls remain.
    """
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()
