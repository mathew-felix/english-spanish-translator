"""Lazy-loaded retrieval over a persistent ChromaDB translation memory.
The collection stores English Europarl sentences with Spanish translations in metadata.
"""

import os
import threading

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb
import torch
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "translation_memory"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3

_CLIENT = None
_COLLECTION = None
_EMBEDDING_MODEL = None
_LOCK = threading.RLock()


def _repo_root() -> str:
    """Return the repository root for repo-relative RAG paths.
    The retriever must work no matter which working directory invokes it.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _db_path() -> str:
    """Return the persistent Chroma directory used by this project.
    The path is fixed to `rag/chroma_db` and is gitignored.
    """
    return os.path.join(_repo_root(), "rag", "chroma_db")


def _embedding_device() -> str:
    """Select the local embedding device for sentence-transformers.
    CUDA is used when available; otherwise CPU is used.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_client():
    """Create or return the persistent Chroma client.
    The client is cached process-wide to avoid reopening the DB repeatedly.
    """
    global _CLIENT

    if _CLIENT is None:
        with _LOCK:
            if _CLIENT is None:
                os.makedirs(_db_path(), exist_ok=True)
                _CLIENT = chromadb.PersistentClient(
                    path=_db_path(),
                    settings=Settings(anonymized_telemetry=False),
                )
    return _CLIENT


def _get_embedding_model():
    """Create or return the sentence-transformer used for retrieval.
    The embedding model is loaded lazily because startup is relatively expensive.
    """
    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is None:
        with _LOCK:
            if _EMBEDDING_MODEL is None:
                _EMBEDDING_MODEL = SentenceTransformer(
                    EMBEDDING_MODEL_NAME,
                    device=_embedding_device(),
                )
    return _EMBEDDING_MODEL


def _get_collection():
    """Return the persisted translation-memory collection.
    Raises a clear error when the index has not been built yet.
    """
    global _COLLECTION

    if _COLLECTION is None:
        with _LOCK:
            if _COLLECTION is None:
                client = _get_client()
                try:
                    _COLLECTION = client.get_collection(name=COLLECTION_NAME)
                except NotFoundError as exc:
                    raise RuntimeError(
                        "RAG translation memory is not built. "
                        "Run `venv/bin/python rag/build_index.py` first."
                    ) from exc
    return _COLLECTION


def retrieve_similar_translations(query: str, k: int = DEFAULT_TOP_K) -> list[dict]:
    """Retrieve the top-k similar English-to-Spanish memory pairs.
    The query must be non-empty. When the translation-memory index has not been
    built yet, returns an empty list so API review can still run without RAG context.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query must not be empty.")
    if k < 1:
        raise ValueError("k must be at least 1.")

    try:
        collection = _get_collection()
    except RuntimeError:
        return []

    model = _get_embedding_model()
    query_embedding = model.encode(
        [cleaned_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].tolist()

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    retrieved_rows = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        retrieved_rows.append(
            {
                "english": document,
                "spanish": metadata.get("spanish", ""),
                "corpus": metadata.get("corpus", ""),
                "source_index": metadata.get("source_index"),
                "distance": round(float(distance), 6),
            }
        )
    return retrieved_rows
