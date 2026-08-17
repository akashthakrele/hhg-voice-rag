/**
 * Voice-Enabled RAG Pipeline — Frontend Application Logic
 * Pure ES6+ JavaScript, zero build step.
 */

(() => {
  "use strict";

  // ── DOM Element References ──────────────────────────────────
  const healthIndicator = document.getElementById("health-indicator");
  const healthText = document.getElementById("health-text");

  const languageSelect = document.getElementById("language-select");
  const recordBtn = document.getElementById("record-btn");
  const recordLabel = document.getElementById("record-label");
  const recordingTimer = document.getElementById("recording-timer");
  const audioPreviewContainer = document.getElementById("audio-preview-container");
  const audioPreview = document.getElementById("audio-preview");
  const reRecordBtn = document.getElementById("re-record-btn");

  const textForm = document.getElementById("text-query-form");
  const textInput = document.getElementById("text-query-input");
  const submitTextBtn = document.getElementById("submit-text-btn");
  const promptChips = document.querySelectorAll(".prompt-chip");

  const errorBanner = document.getElementById("error-banner");
  const errorTitle = document.getElementById("error-title");
  const errorMessage = document.getElementById("error-message");
  const errorCloseBtn = document.getElementById("error-close-btn");
  const errorDetailsWrapper = document.getElementById("error-details-wrapper");
  const toggleErrorDetailsBtn = document.getElementById("toggle-error-details");
  const errorDetails = document.getElementById("error-details");

  const placeholderState = document.getElementById("placeholder-state");
  const loadingState = document.getElementById("loading-state");
  const loadingStageText = document.getElementById("loading-stage-text");
  const responseContent = document.getElementById("response-content");

  const sourceBadge = document.getElementById("source-badge");
  const targetLatencyBadge = document.getElementById("target-latency-badge");

  const timingTotal = document.getElementById("timing-total");
  const timingStatus = document.getElementById("timing-status");
  const timingStt = document.getElementById("timing-stt");
  const timingRetrieval = document.getElementById("timing-retrieval");
  const timingGeneration = document.getElementById("timing-generation");
  const timingGuardrail = document.getElementById("timing-guardrail");

  const groundingCard = document.getElementById("grounding-card");
  const groundingIcon = document.getElementById("grounding-icon");
  const groundingText = document.getElementById("grounding-text");

  const guardrailCard = document.getElementById("guardrail-card");
  const guardrailIcon = document.getElementById("guardrail-icon");
  const guardrailText = document.getElementById("guardrail-text");
  const guardrailAlert = document.getElementById("guardrail-alert");
  const guardrailReasonText = document.getElementById("guardrail-reason-text");

  const transcribedBox = document.getElementById("transcribed-box");
  const transcribedText = document.getElementById("transcribed-text");

  const answerText = document.getElementById("answer-text");
  const copyAnswerBtn = document.getElementById("copy-answer-btn");

  const toggleChunksBtn = document.getElementById("toggle-chunks-btn");
  const chunksCount = document.getElementById("chunks-count");
  const chunksBody = document.getElementById("chunks-body");
  const chunksList = document.getElementById("chunks-list");
  const accordionArrow = document.getElementById("accordion-arrow");

  // ── State ──────────────────────────────────────────────────
  let mediaRecorder = null;
  let audioChunks = [];
  let recordingStartTime = null;
  let timerInterval = null;
  let currentAudioBlob = null;
  let isRecording = false;
  let isProcessing = false;

  // ── Health Check ───────────────────────────────────────────
  async function checkHealth() {
    try {
      const res = await fetch("/api/v1/health");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      healthIndicator.className = "status-badge";
      if (data.status === "healthy") {
        healthIndicator.classList.add("status-online");
        healthText.textContent = "Pipeline Online";
      } else if (data.status === "degraded") {
        healthIndicator.classList.add("status-degraded");
        healthText.textContent = "Degraded (Qdrant)";
      } else {
        healthIndicator.classList.add("status-offline");
        healthText.textContent = "Offline";
      }
    } catch (err) {
      healthIndicator.className = "status-badge status-offline";
      healthText.textContent = "API Offline";
    }
  }

  // ── Audio Recording ────────────────────────────────────────
  async function startRecording() {
    hideError();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError(
        "Microphone Unsupported",
        "Your browser does not support audio recording or you are accessing this page over an insecure connection (HTTPS or localhost is required)."
      );
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Determine best supported MIME type
      let mimeType = "audio/webm";
      if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
        mimeType = "audio/webm;codecs=opus";
      } else if (MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) {
        mimeType = "audio/ogg;codecs=opus";
      } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
        mimeType = "audio/mp4";
      }

      mediaRecorder = new MediaRecorder(stream, { mimeType });
      audioChunks = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // Stop all tracks to release the mic
        stream.getTracks().forEach((track) => track.stop());
        clearInterval(timerInterval);

        const durationMs = Date.now() - recordingStartTime;
        if (durationMs < 400 || audioChunks.length === 0) {
          resetRecordingUI();
          showError("Recording Too Short", "Please hold and speak for at least 1 second.");
          return;
        }

        currentAudioBlob = new Blob(audioChunks, { type: mimeType });
        const audioUrl = URL.createObjectURL(currentAudioBlob);
        audioPreview.src = audioUrl;
        audioPreviewContainer.classList.remove("hidden");

        // Automatically dispatch voice query
        await submitVoiceQuery(currentAudioBlob, mimeType);
      };

      mediaRecorder.start(250); // collect in 250ms chunks
      isRecording = true;
      recordingStartTime = Date.now();
      updateTimerDisplay(0);
      recordingTimer.classList.remove("hidden");
      timerInterval = setInterval(() => {
        const elapsedSec = Math.floor((Date.now() - recordingStartTime) / 1000);
        updateTimerDisplay(elapsedSec);
      }, 1000);

      recordBtn.className = "record-btn state-recording";
      recordLabel.textContent = "Stop & Send";
    } catch (err) {
      console.error("Mic access error:", err);
      let msg = "Could not access microphone.";
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        msg = "Microphone permission was denied. Please allow microphone access in your browser settings.";
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        msg = "No microphone hardware detected on this device.";
      }
      showError("Microphone Access Error", msg, err.stack || err.toString());
      resetRecordingUI();
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
      isRecording = false;
      recordBtn.className = "record-btn state-processing";
      recordLabel.textContent = "Processing...";
    }
  }

  function resetRecordingUI() {
    isRecording = false;
    clearInterval(timerInterval);
    recordingTimer.classList.add("hidden");
    recordBtn.className = "record-btn state-idle";
    recordLabel.textContent = "Click to Speak";
  }

  function updateTimerDisplay(seconds) {
    const mins = String(Math.floor(seconds / 60)).padStart(2, "0");
    const secs = String(seconds % 60).padStart(2, "0");
    recordingTimer.textContent = `${mins}:${secs}`;
  }

  // ── Voice Query Submission ─────────────────────────────────
  async function submitVoiceQuery(blob, mimeType) {
    setProcessing(true, "Transcribing voice audio & running RAG pipeline...");
    hideError();

    try {
      const language = languageSelect.value || "hi-IN";
      const formData = new FormData();

      const ext = mimeType.includes("ogg") ? "ogg" : mimeType.includes("mp4") ? "mp4" : "webm";
      const audioFile = new File([blob], `query.${ext}`, { type: mimeType });
      formData.append("file", audioFile);
      formData.append("language", language);

      const response = await fetch(`/api/v1/query/voice?language=${encodeURIComponent(language)}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await parseResponseError(response);
        throw new Error(errorData);
      }

      const data = await response.json();
      renderResponse(data, "voice");
    } catch (err) {
      console.error("Voice Query Error:", err);
      showError("Voice Query Failed", err.message || "Failed to process audio query.", err.stack);
    } finally {
      setProcessing(false);
      resetRecordingUI();
    }
  }

  // ── Text Query Submission ──────────────────────────────────
  async function submitTextQuery(queryText) {
    const text = (queryText || textInput.value || "").trim();
    if (!text) {
      showError("Empty Query", "Please enter a question before submitting.");
      return;
    }

    setProcessing(true, "Retrieving context & generating answer...");
    hideError();

    try {
      const language = languageSelect.value || "en";
      const response = await fetch("/api/v1/query/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text, language }),
      });

      if (!response.ok) {
        const errorData = await parseResponseError(response);
        throw new Error(errorData);
      }

      const data = await response.json();
      renderResponse(data, "text");
    } catch (err) {
      console.error("Text Query Error:", err);
      showError("Text Query Failed", err.message || "Failed to process text query.", err.stack);
    } finally {
      setProcessing(false);
    }
  }

  // ── Helper: Parse Server Errors ────────────────────────────
  async function parseResponseError(response) {
    try {
      const json = await response.json();
      if (json && json.detail) {
        if (typeof json.detail === "string") return json.detail;
        if (Array.isArray(json.detail)) {
          return json.detail.map((d) => d.msg || JSON.stringify(d)).join(", ");
        }
        return JSON.stringify(json.detail);
      }
    } catch (e) {
      // Not JSON
    }
    return `Server returned HTTP ${response.status} (${response.statusText || "Error"})`;
  }

  // ── UI State Management ────────────────────────────────────
  function setProcessing(loading, stageText = "Processing query...") {
    isProcessing = loading;
    if (loading) {
      placeholderState.classList.add("hidden");
      responseContent.classList.add("hidden");
      loadingState.classList.remove("hidden");
      loadingStageText.textContent = stageText;

      submitTextBtn.disabled = true;
      submitTextBtn.querySelector(".btn-text").textContent = "Sending...";
      submitTextBtn.querySelector(".btn-spinner").classList.remove("hidden");
    } else {
      loadingState.classList.add("hidden");
      submitTextBtn.disabled = false;
      submitTextBtn.querySelector(".btn-text").textContent = "Send Query";
      submitTextBtn.querySelector(".btn-spinner").classList.add("hidden");
    }
  }

  // ── Response Rendering ─────────────────────────────────────
  function renderResponse(data, source = "text") {
    placeholderState.classList.add("hidden");
    loadingState.classList.add("hidden");
    responseContent.classList.remove("hidden");

    // Source Badge
    sourceBadge.classList.remove("hidden");
    sourceBadge.textContent = source === "voice" ? "🎤 Voice Input" : "⌨️ Text Input";

    // Target Latency Badge & Timings
    const timings = data.timings || {};
    const totalMs = timings.total_ms != null ? timings.total_ms : 0;
    timingTotal.textContent = totalMs.toFixed(1);

    targetLatencyBadge.classList.remove("hidden");
    if (totalMs <= 200) {
      targetLatencyBadge.className = "badge badge-success";
      targetLatencyBadge.textContent = "⚡ Target Met (<200ms)";
      timingStatus.className = "metric-status status-met";
      timingStatus.textContent = "Target Met (<200ms)";
    } else {
      targetLatencyBadge.className = "badge badge-warning";
      targetLatencyBadge.textContent = `Target Exceeded (${totalMs.toFixed(0)}ms)`;
      timingStatus.className = "metric-status status-exceeded";
      timingStatus.textContent = `Exceeds 200ms target`;
    }

    // Breakdown Timings
    timingStt.textContent = timings.stt_ms != null ? timings.stt_ms.toFixed(1) : (source === "voice" ? "--" : "N/A");
    timingRetrieval.textContent = timings.retrieval_ms != null ? timings.retrieval_ms.toFixed(1) : "--";
    timingGeneration.textContent = timings.generation_ms != null ? timings.generation_ms.toFixed(1) : "--";
    timingGuardrail.textContent = timings.guardrail_ms != null ? timings.guardrail_ms.toFixed(1) : "--";

    // Grounding Status
    if (data.grounded === true) {
      groundingCard.className = "status-card status-card-success";
      groundingIcon.textContent = "✓";
      groundingText.textContent = "Grounded in Context";
    } else {
      groundingCard.className = "status-card status-card-warning";
      groundingIcon.textContent = "⚠️";
      groundingText.textContent = "Ungrounded / Low Support";
    }

    // Guardrail Status
    if (data.guardrail_triggered) {
      guardrailCard.className = "status-card status-card-danger";
      guardrailIcon.textContent = "🛡️";
      guardrailText.textContent = "Triggered (Refusal)";

      guardrailAlert.classList.remove("hidden");
      guardrailReasonText.textContent = data.guardrail_reason || "Query flagged as off-topic or context insufficient.";
    } else {
      guardrailCard.className = "status-card status-card-success";
      guardrailIcon.textContent = "🛡️";
      guardrailText.textContent = "Passed (On-Topic)";
      guardrailAlert.classList.add("hidden");
    }

    // Transcribed Voice Text
    if (source === "voice" && data.query) {
      transcribedBox.classList.remove("hidden");
      transcribedText.textContent = data.query;
    } else {
      transcribedBox.classList.add("hidden");
    }

    // Generated Answer Text
    answerText.textContent = data.answer || "(No response generated)";

    // Retrieved Chunks Accordion
    const chunks = data.retrieved_chunks || [];
    chunksCount.textContent = chunks.length;
    renderChunks(chunks);
  }

  function renderChunks(chunks) {
    chunksList.innerHTML = "";
    if (chunks.length === 0) {
      chunksList.innerHTML = `<p class="input-hint">No context chunks retrieved from vector database.</p>`;
      return;
    }

    chunks.forEach((chunk, index) => {
      const card = document.createElement("div");
      card.className = "chunk-card";

      const score = chunk.score != null ? chunk.score.toFixed(3) : "N/A";
      const meta = chunk.metadata || {};
      const strategy = meta.strategy || "fixed_size";
      const passageId = meta.passage_id ? `Passage: ${meta.passage_id}` : `Chunk #${index + 1}`;

      card.innerHTML = `
        <div class="chunk-meta">
          <span><strong>${passageId}</strong> &bull; <span class="chunk-strategy">${strategy}</span></span>
          <span class="chunk-score">Score: ${score}</span>
        </div>
        <p class="chunk-text">${escapeHtml(chunk.text || "")}</p>
      `;
      chunksList.appendChild(card);
    });
  }

  // ── Error Management ───────────────────────────────────────
  function showError(title, message, details = null) {
    errorTitle.textContent = title;
    errorMessage.textContent = message;
    errorBanner.classList.remove("hidden");

    if (details) {
      errorDetailsWrapper.classList.remove("hidden");
      errorDetails.textContent = details;
    } else {
      errorDetailsWrapper.classList.add("hidden");
      errorDetails.textContent = "";
    }
  }

  function hideError() {
    errorBanner.classList.add("hidden");
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // ── Event Listeners ────────────────────────────────────────

  // Record Button Click
  recordBtn.addEventListener("click", () => {
    if (isProcessing) return;
    if (!isRecording) {
      startRecording();
    } else {
      stopRecording();
    }
  });

  // Re-record Button
  reRecordBtn.addEventListener("click", () => {
    audioPreviewContainer.classList.add("hidden");
    audioPreview.src = "";
    currentAudioBlob = null;
    resetRecordingUI();
  });

  // Text Form Submit
  textForm.addEventListener("submit", (e) => {
    e.preventDefault();
    submitTextQuery(textInput.value);
  });

  // Text Form Ctrl+Enter shortcut
  textInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      submitTextQuery(textInput.value);
    }
  });

  // Sample Prompt Chips
  promptChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.dataset.query;
      const lang = chip.dataset.lang;
      if (query) {
        textInput.value = query;
      }
      if (lang) {
        languageSelect.value = lang;
      }
      submitTextQuery(query);
    });
  });

  // Copy Answer Button
  copyAnswerBtn.addEventListener("click", async () => {
    const text = answerText.textContent;
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      const orig = copyAnswerBtn.textContent;
      copyAnswerBtn.textContent = "Copied!";
      setTimeout(() => {
        copyAnswerBtn.textContent = orig;
      }, 1800);
    } catch (e) {
      console.warn("Clipboard write failed:", e);
    }
  });

  // Accordion Toggle
  toggleChunksBtn.addEventListener("click", () => {
    const isHidden = chunksBody.classList.contains("hidden");
    if (isHidden) {
      chunksBody.classList.remove("hidden");
      accordionArrow.style.transform = "rotate(180deg)";
    } else {
      chunksBody.classList.add("hidden");
      accordionArrow.style.transform = "rotate(0deg)";
    }
  });

  // Close Error Banner
  errorCloseBtn.addEventListener("click", hideError);

  // Toggle Technical Error Details
  toggleErrorDetailsBtn.addEventListener("click", () => {
    const isHidden = errorDetails.classList.contains("hidden");
    if (isHidden) {
      errorDetails.classList.remove("hidden");
      toggleErrorDetailsBtn.textContent = "Hide technical details";
    } else {
      errorDetails.classList.add("hidden");
      toggleErrorDetailsBtn.textContent = "Show technical details";
    }
  });

  // ── Initialization ─────────────────────────────────────────
  checkHealth();
  // Periodic health check every 30s
  setInterval(checkHealth, 30000);
})();
