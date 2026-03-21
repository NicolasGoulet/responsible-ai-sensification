"use strict";

// ── State ──────────────────────────────────────────────────────────────────
let ws = null;
let isRunning = false;
let tokenCount = 0;

// ── DOM refs ───────────────────────────────────────────────────────────────
const prompt     = document.getElementById("prompt");
const btnStart   = document.getElementById("btn-start");
const btnStop    = document.getElementById("btn-stop");
const statusEl   = document.getElementById("status");
const canvas     = document.getElementById("waveform");
const ctx        = canvas.getContext("2d");

const layerInput  = document.getElementById("layer");
const layerVal    = document.getElementById("layer-val");
const widthSel    = document.getElementById("width");
const clustersIn  = document.getElementById("clusters");
const clustersVal = document.getElementById("clusters-val");
const maxTokensIn = document.getElementById("max-tokens");
const maxTokensVal= document.getElementById("max-tokens-val");
const loopCb      = document.getElementById("loop");

// ── Populate defaults ──────────────────────────────────────────────────────
async function loadDefaults() {
  try {
    const res = await fetch("/api/config/defaults");
    const d = await res.json();
    applyParams(d);
  } catch (e) {
    console.warn("Could not load defaults", e);
  }
}

function applyParams(p) {
  if (p.prompt !== undefined)    prompt.value = p.prompt;
  if (p.layer  !== undefined)  { layerInput.value = p.layer; layerVal.textContent = p.layer; }
  if (p.width  !== undefined)    widthSel.value = p.width;
  if (p.clusters !== undefined){ clustersIn.value = p.clusters; clustersVal.textContent = p.clusters; }
  if (p.max_tokens !== undefined){ maxTokensIn.value = p.max_tokens; maxTokensVal.textContent = p.max_tokens; }
  if (p.loop !== undefined)      loopCb.checked = p.loop;

  if (p.strategy !== undefined) {
    document.querySelectorAll("[data-strategy]").forEach(b => {
      b.classList.toggle("active", b.dataset.strategy === p.strategy);
    });
  }
  if (p.mode !== undefined) {
    document.querySelectorAll("[data-mode]").forEach(b => {
      b.classList.toggle("active", b.dataset.mode === p.mode);
    });
  }
}

// ── Collect current params ─────────────────────────────────────────────────
function collectParams() {
  const strategy = document.querySelector("[data-strategy].active")?.dataset.strategy ?? "identity";
  const mode     = document.querySelector("[data-mode].active")?.dataset.mode ?? "timed";
  return {
    prompt:     prompt.value,
    strategy,
    layer:      parseInt(layerInput.value),
    width:      widthSel.value,
    clusters:   parseInt(clustersIn.value),
    max_tokens: parseInt(maxTokensIn.value),
    loop:       loopCb.checked,
    mode,
  };
}

// ── WebSocket ──────────────────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws/stream`);

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    handleMessage(msg);
  };

  ws.onerror = () => setStatus("WebSocket error");
  ws.onclose = () => { ws = null; setIdle(); };
}

function handleMessage(msg) {
  switch (msg.type) {
    case "ready":
      applyParams(msg.params);
      setStatus("Connected — ready");
      break;

    case "loading":
      setStatus(msg.stage);
      break;

    case "token":
      tokenCount++;
      setStatus(`Tokens: ${tokenCount}`);
      drawNotes(msg.notes ?? []);
      break;

    case "stopped":
      setIdle();
      break;

    case "error":
      setStatus(`Error: ${msg.message}`);
      setIdle();
      break;
  }
}

// ── Canvas rendering ───────────────────────────────────────────────────────
const FREQ_MIN = 20;
const FREQ_MAX = 20000;

// Distinct colours for clusters/instruments
const CLUSTER_COLORS = [
  "#00d4aa", "#ff6b6b", "#ffd93d", "#6bcb77",
  "#4d96ff", "#ff922b", "#cc5de8", "#f06595",
];

function drawNotes(notes) {
  const W = canvas.offsetWidth;
  const H = canvas.offsetHeight;
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W;
    canvas.height = H;
  }

  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, W, H);

  if (!notes.length) return;

  // Find max amplitude for normalisation
  const maxAmp = Math.max(...notes.map(n => n.amplitude ?? 0), 1);

  for (const note of notes) {
    const freq = note.freq ?? 440;
    const amp  = (note.amplitude ?? 0) / maxAmp;
    const cluster = note.cluster ?? null;

    // Log-scale frequency → X position
    const logMin = Math.log10(FREQ_MIN);
    const logMax = Math.log10(FREQ_MAX);
    const x = Math.round(((Math.log10(Math.max(freq, FREQ_MIN)) - logMin) / (logMax - logMin)) * W);

    const barH = Math.max(2, Math.round(amp * H));
    const color = cluster !== null ? CLUSTER_COLORS[cluster % CLUSTER_COLORS.length] : "#00d4aa";

    ctx.fillStyle = color;
    ctx.fillRect(x, H - barH, 2, barH);
  }
}

// ── UI helpers ─────────────────────────────────────────────────────────────
function setStatus(msg) {
  statusEl.textContent = msg;
}

function setIdle() {
  isRunning = false;
  btnStart.disabled = false;
  btnStop.disabled  = true;
}

function setRunning() {
  isRunning = true;
  tokenCount = 0;
  btnStart.disabled = true;
  btnStop.disabled  = false;
}

// ── Control wiring ─────────────────────────────────────────────────────────
btnStart.addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  setRunning();
  ws.send(JSON.stringify({ action: "start", params: collectParams() }));
});

btnStop.addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ action: "stop" }));
});

// Live param updates while running
function sendParamUpdate(partial) {
  if (!isRunning || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ action: "update_params", params: partial }));
}

layerInput.addEventListener("input", () => {
  layerVal.textContent = layerInput.value;
  sendParamUpdate({ layer: parseInt(layerInput.value) });
});

clustersIn.addEventListener("input", () => {
  clustersVal.textContent = clustersIn.value;
  sendParamUpdate({ clusters: parseInt(clustersIn.value) });
});

maxTokensIn.addEventListener("input", () => {
  maxTokensVal.textContent = maxTokensIn.value;
  sendParamUpdate({ max_tokens: parseInt(maxTokensIn.value) });
});

widthSel.addEventListener("change", () => sendParamUpdate({ width: widthSel.value }));
loopCb.addEventListener("change",   () => sendParamUpdate({ loop: loopCb.checked }));

document.querySelectorAll("[data-strategy]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("[data-strategy]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    sendParamUpdate({ strategy: btn.dataset.strategy });
  });
});

document.querySelectorAll("[data-mode]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("[data-mode]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    sendParamUpdate({ mode: btn.dataset.mode });
  });
});

// ── Init ───────────────────────────────────────────────────────────────────
loadDefaults();
connectWS();
