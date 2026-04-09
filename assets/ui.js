function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function bindExampleChips() {
  const textarea = document.getElementById("review-text");
  document.querySelectorAll(".example-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      textarea.value = chip.dataset.value || "";
      textarea.focus();
    });
  });
}

function resetSteps() {
  document.querySelectorAll(".review-step").forEach((step) => {
    step.classList.remove("visible", "active");
  });
  document.getElementById("draft-output").textContent =
    "The first-pass Spanish translation will appear here.";
  document.getElementById("context-output").textContent =
    "Similar bilingual examples will appear here.";
  document.getElementById("decision-output").textContent =
    "The review decision will appear here.";
  document.getElementById("final-output").textContent =
    "The final reviewed Spanish translation will appear here.";
  document.getElementById("status-badge").textContent = "Waiting for input";
  document.getElementById("latency-badge").textContent = "Waiting";
}

function setActiveStep(stepId, statusText) {
  document.querySelectorAll(".review-step").forEach((step) => {
    step.classList.remove("active");
  });
  const step = document.getElementById(stepId);
  if (step) {
    step.classList.add("visible", "active");
  }
  document.getElementById("status-badge").textContent = statusText;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderExamples(examples) {
  return examples
    .map(
      (example, index) => `
        <article class="context-item">
          <strong>Example ${index + 1}</strong>
          <div><b>English:</b> ${escapeHtml(example.english)}</div>
          <div><b>Spanish:</b> ${escapeHtml(example.spanish)}</div>
          <div class="distance-line">Similarity distance: ${example.distance}</div>
        </article>
      `
    )
    .join("");
}

async function submitReview(text) {
  const response = await fetch("/institutional-review", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const payload = await response.json();
  if (!response.ok) {
    const detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
    throw new Error(detail || "Review request failed.");
  }
  return payload;
}

function bindReviewForm() {
  const form = document.getElementById("review-form");
  const textarea = document.getElementById("review-text");
  const button = form.querySelector("button[type='submit']");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    resetSteps();

    button.disabled = true;
    button.textContent = "Running review...";
    document.getElementById("status-badge").textContent = "Generating custom model draft";

    try {
      const payload = await submitReview(textarea.value);

      setActiveStep("step-draft", "Step 1 of 4: draft ready");
      document.getElementById("draft-output").textContent = payload.draft_translation;
      await wait(1200);

      setActiveStep("step-context", "Step 2 of 4: similar examples loaded");
      document.getElementById("context-output").innerHTML = renderExamples(payload.retrieved_examples);
      await wait(1400);

      setActiveStep("step-decision", "Step 3 of 4: review decision");
      document.getElementById("decision-output").textContent = payload.decision;
      await wait(1400);

      setActiveStep("step-final", "Step 4 of 4: final translation ready");
      document.getElementById("final-output").textContent = payload.final_translation;
      document.getElementById("latency-badge").textContent = `${payload.latency_ms} ms`;
    } catch (error) {
      setActiveStep("step-final", "Review failed");
      document.getElementById("final-output").textContent = error.message;
      document.getElementById("latency-badge").textContent = "Error";
    } finally {
      button.disabled = false;
      button.textContent = "Run review";
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  bindExampleChips();
  bindReviewForm();
  resetSteps();
});
