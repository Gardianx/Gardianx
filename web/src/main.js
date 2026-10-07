import "./style.css";
import "./controls.css";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const STORAGE_KEY = "gardianx-app-password";
const app = document.querySelector("#app");

const icons = {
  spark: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 3 1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7L12 3Z"/><path d="m19 14 .9 2.1L22 17l-2.1.9L19 20l-.9-2.1L16 17l2.1-.9L19 14Z"/></svg>',
  send: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m21 3-7.2 18-3.9-7.9L2 9.2 21 3Z"/><path d="M10 13 15 8"/></svg>',
  mic: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="3" width="6" height="12" rx="3"/><path d="M5 11v1a7 7 0 0 0 14 0v-1M12 19v3m-4 0h8"/></svg>',
  upload: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V4m-5 5 5-5 5 5"/><path d="M5 14v5a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-5"/></svg>',
  chart: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 19V5m0 14h17M8 15l4-5 3 2 5-7"/></svg>',
  close: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m18 6-12 12M6 6l12 12"/></svg>',
};

const state = {
  password: sessionStorage.getItem(STORAGE_KEY) || "",
  provider: "groq",
  model: "",
  recording: false,
  mediaRecorder: null,
  audioChunks: [],
  conversation: [],
  performance: {
    requests: 0,
    failures: 0,
    avgLatency: 0,
    transcriptionCount: 0,
    audioErrors: 0,
    transcriptionMs: 0,
    lastLatency: 0,
    lastTranscription: 0,
    estimatedTokens: 0,
    latencies: [],
    inputMode: "text",
  },
};

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
}

function renderLogin(message = "") {
  app.innerHTML = `
    <main class="login-shell">
      <section class="login-card">
        <div class="brand-mark">${icons.spark}</div>
        <p class="eyebrow">GARDIANX · PRIVATE WORKSPACE</p>
        <h1>Welcome back.</h1>
        <p class="muted">Enter the shared app password configured for this service.</p>
        <form id="login-form">
          <label for="app-password">App password</label>
          <input id="app-password" type="password" autocomplete="current-password" required />
          <p class="error-text">${escapeHtml(message)}</p>
          <button class="primary-button full-button" type="submit">Open assistant <span>→</span></button>
        </form>
      </section>
    </main>`;
  app.querySelector("#login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    state.password = app.querySelector("#app-password").value;
    sessionStorage.setItem(STORAGE_KEY, state.password);
    try {
      await api("/api/performance");
      renderApp();
      await refreshPerformance();
    } catch (error) {
      state.password = "";
      sessionStorage.removeItem(STORAGE_KEY);
      renderLogin(error.message);
    }
  });
}

function renderApp() {
  app.innerHTML = `
    <div class="app-shell">
      <aside class="sidebar">
        <a class="brand" href="#" aria-label="Gardianx home"><span class="brand-mark">${icons.spark}</span><span>gardianx<span class="brand-dot">.</span></span></a>
        <div class="workspace-label">WORKSPACE</div>
        <button class="side-link active" id="assistant-link"><span class="side-icon">✳</span> Assistant</button>
        <div class="side-spacer"></div>
        <div class="privacy-card"><span class="privacy-dot"></span><div><strong>Protected session</strong><small>Provider keys stay server-side</small></div></div>
        <button class="side-link signout" id="signout">Sign out</button>
        <div class="user-chip"><span class="avatar">G</span><span><strong>Gardianx</strong><small>Personal workspace</small></span><span class="user-menu">···</span></div>
      </aside>
      <main class="main-area">
        <header class="topbar"><div><span class="crumb">Workspace</span><span class="crumb-separator">/</span><strong>Assistant</strong></div><div class="topbar-right"><label class="provider-select-label" for="provider-select">AI</label><select id="provider-select" class="provider-select"><option value="groq">Groq Cloud</option><option value="google">Google AI Studio</option></select><span class="secure-label"><span class="status-dot"></span> Secure session</span><button class="icon-button" id="refresh-stats" title="Refresh performance" aria-label="Refresh performance">${icons.chart}</button></div></header>
        <div class="content-grid">
          <section class="chat-column">
            <div class="welcome-block"><div class="welcome-icon">${icons.spark}</div><p class="eyebrow">YOUR AI COMPANION</p><h1>What can I help<br/>you with today?</h1><p class="welcome-copy">Ask questions, talk through an issue, or just have a conversation.</p></div>
            <div class="suggestion-row"><button class="suggestion" data-prompt="Can you help me understand an error message?">Explain an error <span>↗</span></button><button class="suggestion" data-prompt="What can you help me with?">What can you do? <span>↗</span></button></div>
            <div class="conversation" id="conversation" aria-live="polite"></div>
            <div class="composer-wrap">
              <div class="input-tabs" role="tablist" aria-label="Choose input mode"><button class="input-tab selected" data-mode="text">Message</button><button class="input-tab" data-mode="voice">${icons.mic} Voice</button><button class="input-tab" data-mode="audio">${icons.upload} Audio file</button></div>
              <div class="composer" id="composer">
                <textarea id="message-input" rows="1" maxlength="4000" placeholder="Message your assistant…"></textarea>
                <div class="composer-actions"><span class="composer-hint">Enter to send · Shift + Enter for a new line</span><button class="send-button" id="send-button" aria-label="Send message">${icons.send}</button></div>
              </div>
              <div class="voice-panel hidden" id="voice-panel"><div class="voice-orb" id="voice-orb">${icons.mic}</div><div class="voice-copy"><strong id="voice-title">Ready when you are</strong><span id="voice-caption">Your voice is transcribed securely for this conversation.</span></div><button class="primary-button" id="record-button">${icons.mic} Start recording</button></div>
              <div class="audio-panel hidden" id="audio-panel"><div class="upload-icon">${icons.upload}</div><div class="upload-copy"><strong>Upload an audio recording</strong><span>WAV, MP3, M4A, AAC, FLAC, OGG or WebM · max 15 MB</span></div><label class="secondary-button upload-button">Choose audio<input id="audio-input" type="file" accept="audio/*,.wav,.mp3,.m4a,.ogg,.webm,.flac" hidden /></label></div>
              <p class="composer-footnote"><span>✦</span> AI can make mistakes. Trade requests are never executed.</p>
            </div>
          </section>
          <aside class="insights-column">
            <div class="panel-heading"><div><p class="eyebrow">LIVE OVERVIEW</p><h2>AI performance</h2></div><span class="live-indicator"><span class="status-dot"></span> LIVE</span></div>
            <div class="model-card"><div class="model-logo">${icons.spark}</div><div class="model-meta"><span>ACTIVE MODEL</span><strong id="model-name">Connecting…</strong><small id="provider-name">Hosted AI</small></div><span class="model-status"></span></div>
            <div class="stats-grid"><div class="stat-card"><span>Messages</span><strong id="stat-requests">0</strong><small>this session</small></div><div class="stat-card"><span>Avg. response</span><strong id="stat-latency">—</strong><small>milliseconds</small></div><div class="stat-card"><span>Voice clips</span><strong id="stat-audio">0</strong><small>transcribed</small></div><div class="stat-card"><span>Last response</span><strong id="stat-last">—</strong><small>milliseconds</small></div></div>
            <div class="activity-card"><div class="activity-head"><h3>Session activity</h3><span id="activity-count">0 events</span></div><div class="activity-chart"><div class="chart-y"><span>2s</span><span>1s</span><span>0</span></div><div class="chart-area"><div class="chart-grid"><i></i><i></i><i></i></div><div class="chart-empty" id="chart-empty"><span>◌</span><small>Performance appears here<br/>as you chat</small></div><svg class="chart-line hidden" id="chart-line" viewBox="0 0 300 112" preserveAspectRatio="none"><path class="chart-fill" d=""/><path class="chart-stroke" d=""/></svg><div class="chart-x"><span>Earlier</span><span>Recent</span></div></div></div></div>
            <div class="health-card"><div class="health-icon">✓</div><div><strong>Ready to assist</strong><span id="health-message">Protected by the shared app password.</span></div></div>
            <div class="privacy-note"><span>◈</span><p>Your messages are processed by the selected AI provider. Audio is transcribed by this app and only the transcript is sent for chat.</p></div>
          </aside>
        </div>
      </main>
      <div class="toast hidden" id="toast" role="status"></div>
    </div>`;
  bindEvents();
  loadHealth();
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-App-Password", state.password);
  if (options.body && !(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  } catch {
    throw new Error("Could not reach the assistant service. Check the deployment/API URL.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) {
      state.password = "";
      sessionStorage.removeItem(STORAGE_KEY);
      renderLogin(payload.detail || "Session expired.");
    }
    throw new Error(payload.detail || `Request failed (${response.status}).`);
  }
  return payload;
}

function bindEvents() {
  app.querySelector(".stats-grid").insertAdjacentHTML(
    "beforeend",
    '<div class="stat-card"><span>Errors</span><strong id="stat-errors">0</strong><small>chat + audio</small></div><div class="stat-card"><span>Est. tokens</span><strong id="stat-tokens">0</strong><small>approximate</small></div>',
  );
  app.querySelector("#signout").addEventListener("click", () => {
    state.password = "";
    state.conversation = [];
    sessionStorage.removeItem(STORAGE_KEY);
    renderLogin();
  });
  app.querySelector("#refresh-stats").addEventListener("click", refreshPerformance);
  app.querySelector("#provider-select").addEventListener("change", (event) => {
    state.provider = event.target.value;
    const provider = state.provider === "google" ? "Google AI Studio" : "Groq Cloud";
    app.querySelector("#provider-name").textContent = provider;
    app.querySelector("#model-name").textContent = state.models?.[state.provider]?.model || "Model unavailable";
    const isReady = state.models?.[state.provider]?.configured;
    app.querySelector("#health-message").textContent = isReady
      ? `${provider} connected · requests protected by app password.`
      : `Add the ${state.provider === "google" ? "GOOGLE_API_KEY" : "GROQ_API_KEY"} secret in Hugging Face Space settings to enable chat.`;
  });
  app.querySelectorAll(".input-tab").forEach((button) => {
    button.addEventListener("click", () => selectMode(button.dataset.mode));
  });
  app.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => {
      app.querySelector("#message-input").value = button.dataset.prompt;
      app.querySelector("#message-input").focus();
    });
  });
  app.querySelector("#send-button").addEventListener("click", sendMessage);
  app.querySelector("#message-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });
  app.querySelector("#record-button").addEventListener("click", toggleRecording);
  app.querySelector("#audio-input").addEventListener("change", handleAudioFile);
}

function selectMode(mode) {
  state.performance.inputMode = mode;
  app.querySelectorAll(".input-tab").forEach((button) => button.classList.toggle("selected", button.dataset.mode === mode));
  app.querySelector("#composer").classList.toggle("hidden", mode !== "text");
  app.querySelector("#voice-panel").classList.toggle("hidden", mode !== "voice");
  app.querySelector("#audio-panel").classList.toggle("hidden", mode !== "audio");
}

function setBusy(isBusy) {
  app.querySelector("#send-button").disabled = isBusy;
  app.querySelector("#record-button").disabled = isBusy;
  app.querySelector("#audio-input").disabled = isBusy;
  app.querySelector("#composer").classList.toggle("busy", isBusy);
}

function addMessage(role, content, extra = "") {
  const conversation = app.querySelector("#conversation");
  const row = document.createElement("div");
  row.className = `message-row ${role}`;
  const label = role === "assistant" ? "G" : "You";
  row.innerHTML = `<span class="message-avatar ${role}">${label}</span><div class="message-body"><span class="message-author">${role === "assistant" ? "Gardianx assistant" : "You"}</span><div class="message-content">${escapeHtml(content).replace(/\n/g, "<br/>")}</div>${extra}</div>`;
  conversation.append(row);
  conversation.scrollTop = conversation.scrollHeight;
}

async function sendMessage(transcribedText = null) {
  const input = app.querySelector("#message-input");
  const message = (transcribedText ?? input.value).trim();
  if (!message) return;
  setBusy(true);
  if (transcribedText === null) input.value = "";
  state.conversation.push({ role: "user", content: message });
  addMessage("user", message, state.performance.inputMode !== "text" ? `<span class="message-tag">${escapeHtml(state.performance.inputMode)} input</span>` : "");
  const started = performance.now();
  try {
    const result = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        message,
        provider: state.provider,
        conversation: state.conversation.slice(0, -1).slice(-20),
      }),
    });
    const elapsed = Math.max(0, Math.round(performance.now() - started));
    state.performance.requests += 1;
    state.performance.lastLatency = result.latency_ms ?? elapsed;
    state.performance.avgLatency += (state.performance.lastLatency - state.performance.avgLatency) / state.performance.requests;
    state.performance.latencies.push(state.performance.lastLatency);
    state.performance.latencies = state.performance.latencies.slice(-8);
    state.performance.estimatedTokens +=
      (result.estimated_tokens?.input || 0) + (result.estimated_tokens?.output || 0);
    state.conversation.push({ role: "assistant", content: result.reply });
    addMessage("assistant", result.reply, `<span class="message-tag">${escapeHtml(result.provider)} · ${Math.round(result.latency_ms)} ms</span>`);
    updatePerformance();
  } catch (error) {
    state.performance.failures += 1;
    addMessage("assistant", `I couldn't complete that request: ${error.message}`);
    updatePerformance();
    showToast(error.message);
  } finally {
    setBusy(false);
    input.focus();
  }
}

async function toggleRecording() {
  if (state.recording) {
    state.mediaRecorder.stop();
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    showToast("Microphone recording is not available in this browser or context.");
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const supportedType = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg"].find((type) => MediaRecorder.isTypeSupported(type));
    state.mediaRecorder = new MediaRecorder(stream, supportedType ? { mimeType: supportedType } : undefined);
    state.audioChunks = [];
    state.mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) state.audioChunks.push(event.data);
    });
    state.mediaRecorder.addEventListener("stop", async () => {
      stream.getTracks().forEach((track) => track.stop());
      state.recording = false;
      app.querySelector("#record-button").innerHTML = `${icons.mic} Start recording`;
      app.querySelector("#voice-title").textContent = "Transcribing your voice…";
      app.querySelector("#voice-caption").textContent = "Audio is securely sent to the app for transcription.";
      const blob = new Blob(state.audioChunks, { type: state.mediaRecorder.mimeType || "audio/webm" });
      await transcribeBlob(blob, "voice-note.webm");
      app.querySelector("#voice-title").textContent = "Ready when you are";
      app.querySelector("#voice-caption").textContent = "Your voice is transcribed securely for this conversation.";
    });
    state.mediaRecorder.start();
    state.recording = true;
    app.querySelector("#record-button").innerHTML = `${icons.close} Stop recording`;
    app.querySelector("#voice-title").textContent = "Listening…";
    app.querySelector("#voice-caption").textContent = "Select stop when you have finished speaking.";
    app.querySelector("#voice-orb").classList.add("recording");
  } catch (error) {
    stream?.getTracks().forEach((track) => track.stop());
    showToast(error.name === "NotAllowedError" ? "Microphone permission was denied." : `Could not start microphone: ${error.message}`);
  }
}

async function handleAudioFile(event) {
  const file = event.target.files?.[0];
  event.target.value = "";
  if (!file) return;
  if (file.size > 15 * 1024 * 1024) {
    showToast("Audio upload exceeds the 15 MB limit.");
    return;
  }
  await transcribeBlob(file, file.name);
}

async function transcribeBlob(blob, filename) {
  setBusy(true);
  const started = performance.now();
  const form = new FormData();
  form.append("file", blob, filename);
  try {
    const result = await api("/api/transcribe", { method: "POST", body: form });
    const elapsed = result.latency_ms ?? Math.round(performance.now() - started);
    state.performance.transcriptionCount += 1;
    state.performance.lastTranscription = elapsed;
    state.performance.transcriptionMs += elapsed;
    state.performance.latencies.push(elapsed);
    state.performance.latencies = state.performance.latencies.slice(-8);
    updatePerformance();
    selectMode("text");
    app.querySelector("#message-input").value = result.transcript;
    app.querySelector("#message-input").focus();
    showToast("Transcript ready — review it, then send.");
  } catch (error) {
    state.performance.audioErrors += 1;
    updatePerformance();
    showToast(error.message);
  } finally {
    app.querySelector("#voice-orb")?.classList.remove("recording");
    setBusy(false);
  }
}

async function loadHealth() {
  try {
    const health = await api("/api/health");
    state.provider = health.provider;
    state.model = health.model;
    state.models = health.providers;
    app.querySelector("#provider-select").value = health.provider;
    app.querySelector("#model-name").textContent = state.models[health.provider]?.model || health.model;
    app.querySelector("#provider-name").textContent = health.provider === "google" ? "Google AI Studio" : "Groq Cloud";
    if (!health.password_required) {
      app.querySelector("#health-message").textContent = "Service configuration required before chat.";
    } else if (!health.provider_configured) {
      app.querySelector("#health-message").textContent = `Add a ${health.provider} API key in Hugging Face Space settings to enable chat.`;
    } else {
      app.querySelector("#health-message").textContent = "Provider key configured · requests protected by app password.";
    }
    await refreshPerformance();
  } catch (error) {
    app.querySelector("#model-name").textContent = "API unavailable";
    app.querySelector("#health-message").textContent = error.message;
    showToast(error.message);
  }
}

async function refreshPerformance() {
  try {
    const result = await api("/api/performance");
    if (result.chat_requests) {
      state.performance.requests = result.chat_requests - result.chat_errors;
      state.performance.avgLatency = result.average_chat_latency_ms;
      state.performance.lastLatency = result.last_chat_latency_ms;
      state.performance.failures = result.chat_errors;
    }
    if (result.audio_transcriptions) {
      state.performance.transcriptionCount = result.audio_transcriptions;
      state.performance.transcriptionMs = result.average_transcription_latency_ms * result.audio_transcriptions;
      state.performance.lastTranscription = result.last_transcription_latency_ms;
    }
    state.performance.audioErrors = result.audio_errors || 0;
    updatePerformance();
  } catch {
    updatePerformance();
  }
}

function updatePerformance() {
  const perf = state.performance;
  app.querySelector("#stat-requests").textContent = perf.requests;
  app.querySelector("#stat-latency").textContent = perf.requests ? Math.round(perf.avgLatency).toLocaleString() : "—";
  app.querySelector("#stat-audio").textContent = perf.transcriptionCount;
  app.querySelector("#stat-last").textContent = perf.lastLatency ? Math.round(perf.lastLatency).toLocaleString() : "—";
  app.querySelector("#stat-errors").textContent = perf.failures + perf.audioErrors;
  app.querySelector("#stat-tokens").textContent = perf.estimatedTokens.toLocaleString();
  const count = perf.requests + perf.transcriptionCount + perf.failures + perf.audioErrors;
  app.querySelector("#activity-count").textContent = `${count} ${count === 1 ? "event" : "events"}`;
  if (!count) return;
  app.querySelector("#chart-empty").classList.add("hidden");
  const line = app.querySelector("#chart-line");
  line.classList.remove("hidden");
  const latencies = perf.latencies.slice(-8);
  const points = latencies.map((latency, index) => {
    const x = latencies.length < 2 ? 10 : 10 + (index * 280) / (latencies.length - 1);
    const y = Math.max(8, 100 - Math.min(90, latency / 25));
    return [x, y];
  });
  if (points.length === 1) points.unshift([0, points[0][1]]);
  const path = points.map(([x, y], index) => `${index ? "L" : "M"}${x},${y}`).join(" ");
  line.querySelector(".chart-stroke").setAttribute("d", path);
  line.querySelector(".chart-fill").setAttribute("d", `${path} L${points.at(-1)[0]},105 L${points[0][0]},105 Z`);
}

let toastTimer;
function showToast(message) {
  const toast = app.querySelector("#toast");
  if (!toast) return;
  toast.textContent = message;
  toast.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add("hidden"), 4500);
}

if (state.password) renderApp();
else renderLogin();
