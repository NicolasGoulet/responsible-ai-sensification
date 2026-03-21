"use strict";

// ── State ──────────────────────────────────────────────────────────────────
let ws = null;
let isRunning = false;
let tokenCount = 0;
let catalogue = {};        // modelId -> { layers: [...], widths: [...] }
let strategyDescs = {};    // value -> description
let modeDescs = {};        // value -> description

// ── Local storage ──────────────────────────────────────────────────────────
const STORAGE_KEY = "sae_ui_params";

function saveParams() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(collectParams())); } catch {}
}

function loadSavedParams() {
  try { const raw = localStorage.getItem(STORAGE_KEY); return raw ? JSON.parse(raw) : null; } catch { return null; }
}

// ── DOM refs ───────────────────────────────────────────────────────────────
const prompt          = document.getElementById("prompt");
const btnSend         = document.getElementById("btn-send");
const btnStart        = document.getElementById("btn-start");
const btnStop         = document.getElementById("btn-stop");
const statusEl        = document.getElementById("status");
const statusText      = document.getElementById("status-text");
const loopCountEl     = document.getElementById("loop-count-display");
const canvas          = document.getElementById("waveform");
const ctx             = canvas.getContext("2d");

const modelSel        = document.getElementById("model");
const layerSel        = document.getElementById("layer");
const widthSel        = document.getElementById("width");
const strategySel     = document.getElementById("strategy");
const strategyHelp    = document.getElementById("strategy-help");
const clustersGroup   = document.getElementById("clusters-group");
const clustersIn      = document.getElementById("clusters");
const maxTokensIn     = document.getElementById("max-tokens");
const modeSel         = document.getElementById("mode");
const modeHelp        = document.getElementById("mode-help");
const bpmGroup        = document.getElementById("bpm-group");
const bpmIn           = document.getElementById("bpm");
const loopCb          = document.getElementById("loop");

// ── Load options + defaults ─────────────────────────────────────────────────
async function loadOptions() {
  try {
    const res = await fetch("/api/config/model-options");
    const data = await res.json();

    catalogue = data.model_catalogue ?? {};

    // Populate model dropdown
    data.models.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      modelSel.appendChild(opt);
    });

    // Populate layer/width based on first model
    if (modelSel.value) populateLayerWidth(modelSel.value);

    // Populate strategy dropdown
    data.strategies.forEach(s => {
      const opt = document.createElement("option");
      opt.value = s.value;
      opt.textContent = s.label;
      strategySel.appendChild(opt);
      strategyDescs[s.value] = s.description;
    });
    updateStrategyHelp();

    // Populate mode dropdown
    data.modes.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.value;
      opt.textContent = m.label;
      modeSel.appendChild(opt);
      modeDescs[m.value] = m.description;
    });
    updateModeHelp();

  } catch (e) {
    console.warn("Could not load model options", e);
  }

  try {
    const res = await fetch("/api/config/defaults");
    const d = await res.json();
    applyParams(d);
  } catch (e) {
    console.warn("Could not load defaults", e);
  }

  const saved = loadSavedParams();
  if (saved) applyParams(saved);
}

function populateLayerWidth(modelId) {
  const info = catalogue[modelId] ?? { layers: [], widths: [] };

  layerSel.innerHTML = "";
  info.layers.forEach(l => {
    const opt = document.createElement("option");
    opt.value = l;
    opt.textContent = l;
    layerSel.appendChild(opt);
  });

  widthSel.innerHTML = "";
  info.widths.forEach(w => {
    const opt = document.createElement("option");
    opt.value = w;
    opt.textContent = w;
    widthSel.appendChild(opt);
  });
}

function updateStrategyHelp() {
  strategyHelp.dataset.tooltip = strategyDescs[strategySel.value] ?? "";
}

function updateModeHelp() {
  modeHelp.dataset.tooltip = modeDescs[modeSel.value] ?? "";
}

function applyParams(p) {
  if (p.prompt     !== undefined) prompt.value       = p.prompt;
  if (p.model      !== undefined) modelSel.value     = p.model;
  if (p.layer      !== undefined) layerSel.value     = p.layer;
  if (p.width      !== undefined) widthSel.value     = p.width;
  if (p.strategy   !== undefined) { strategySel.value = p.strategy; updateStrategyHelp(); }
  if (p.clusters   !== undefined) clustersIn.value   = p.clusters;
  if (p.max_tokens !== undefined) maxTokensIn.value  = p.max_tokens;
  if (p.mode       !== undefined) { modeSel.value = p.mode; updateModeHelp(); }
  if (p.bpm        !== undefined) bpmIn.value        = p.bpm;
  if (p.loop       !== undefined) loopCb.checked     = p.loop;

  // Sync conditional visibility
  if (p.strategy !== undefined) syncClustersVisibility();
  if (p.mode     !== undefined) syncBpmVisibility();
}

// ── Collect current params ─────────────────────────────────────────────────
function collectParams() {
  return {
    prompt:     prompt.value,
    model:      modelSel.value,
    layer:      parseInt(layerSel.value),
    width:      widthSel.value,
    strategy:   strategySel.value,
    clusters:   parseInt(clustersIn.value),
    max_tokens: parseInt(maxTokensIn.value),
    mode:       modeSel.value,
    bpm:        parseInt(bpmIn.value),
    loop:       loopCb.checked,
  };
}

// ── Conditional visibility ─────────────────────────────────────────────────
function syncClustersVisibility() {
  clustersGroup.classList.toggle("hidden", strategySel.value !== "cluster");
}

function syncBpmVisibility() {
  bpmGroup.classList.toggle("hidden", modeSel.value !== "timed");
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
      if (msg.loop_count !== undefined) {
        loopCountEl.textContent = `Loop: ${msg.loop_count}`;
        loopCountEl.classList.remove("hidden");
      }
      drawNotes(msg.notes ?? []);
      break;

    case "done":
      setStatus(`Done (${tokenCount} tokens) — loop or send a new prompt`);
      break;

    case "silent":
      setStatus("Silent");
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

  const maxAmp = Math.max(...notes.map(n => n.amplitude ?? 0), 1);

  for (const note of notes) {
    const freq = note.freq ?? 440;
    const amp  = (note.amplitude ?? 0) / maxAmp;
    const cluster = note.cluster ?? null;

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
  statusText.textContent = msg;
}

function setIdle() {
  isRunning = false;
  btnStart.disabled = false;
  btnStop.disabled  = true;
  btnSend.disabled  = true;
}

function setRunning() {
  isRunning = true;
  tokenCount = 0;
  loopCountEl.classList.add("hidden");
  loopCountEl.textContent = "Loop: 0";
  btnStart.disabled = true;
  btnStop.disabled  = false;
  btnSend.disabled  = false;
}

// ── Control wiring ─────────────────────────────────────────────────────────
function startPipeline() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  setRunning();
  ws.send(JSON.stringify({ action: "start", params: collectParams() }));
}

btnStart.addEventListener("click", startPipeline);
btnSend.addEventListener("click", startPipeline);

btnStop.addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ action: "stop" }));
});

function sendParamUpdate(partial) {
  if (!isRunning || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ action: "update_params", params: partial }));
}

prompt.addEventListener("input", () => saveParams());

modelSel.addEventListener("change", () => {
  populateLayerWidth(modelSel.value);
  sendParamUpdate({ model: modelSel.value });
  saveParams();
});

layerSel.addEventListener("change", () => { sendParamUpdate({ layer: parseInt(layerSel.value) }); saveParams(); });
widthSel.addEventListener("change", () => { sendParamUpdate({ width: widthSel.value }); saveParams(); });

strategySel.addEventListener("change", () => {
  updateStrategyHelp();
  syncClustersVisibility();
  sendParamUpdate({ strategy: strategySel.value });
  saveParams();
});

clustersIn.addEventListener("input", () => { sendParamUpdate({ clusters: parseInt(clustersIn.value) }); saveParams(); });
maxTokensIn.addEventListener("input", () => { sendParamUpdate({ max_tokens: parseInt(maxTokensIn.value) }); saveParams(); });

modeSel.addEventListener("change", () => {
  updateModeHelp();
  syncBpmVisibility();
  sendParamUpdate({ mode: modeSel.value });
  saveParams();
});

bpmIn.addEventListener("input", () => { sendParamUpdate({ bpm: parseInt(bpmIn.value) }); saveParams(); });
loopCb.addEventListener("change", () => { sendParamUpdate({ loop: loopCb.checked }); saveParams(); });

// ── Init ───────────────────────────────────────────────────────────────────
loadOptions();
connectWS();
