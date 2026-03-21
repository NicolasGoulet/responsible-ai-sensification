# Feature: Streaming Live Audio Pipeline

## Feature Description
Transform the current batch pipeline (extract → JSON → synthesize → WAV) into a real-time streaming pipeline where SAE feature activations are emitted token-by-token, optionally transformed into structured musical parameters (with instrument assignment via semantic clustering), and played live through the soundcard. A loop mode replays the recorded generation indefinitely. Two synthesis timing modes are supported: timed (one note per 0.5 s, fixed) and sustain (each note plays until the next token arrives, tracking inference speed).

## User Story
As a researcher / artist collaborator
I want to run `uv run python extract.py "my prompt" --stream | uv run python transform.py --strategy cluster | uv run python synthesize.py --live --mode sustain`
So that I can hear the model's internal representations as live music in real-time, with instrument timbres reflecting semantic clusters of active features, and loop the recording for continuous listening

## Problem Statement
The current pipeline is batch-only: extract runs fully, writes a JSON file, then synthesize reads it and writes a WAV. There is no way to hear audio while the model is generating, no semantic structure in instrument assignment (all features map to the same sine-wave timbre), no loop playback, and no ability to sustain a note for the natural duration of token generation.

## Solution Statement
- `extract.py` gains `--stream` (NDJSON to stdout, one line per token) and `--loop` (replay the recorded tokens indefinitely after generation ends, Ctrl+C to stop). Each NDJSON line is either a `meta` header (emitted first) or a `token` event (emitted per generated token, including `elapsed_ms` for accurate replay timing).
- `transform.py` (new) reads NDJSON from stdin, applies a transformation strategy, and emits NDJSON `MusicalEvent` lines. Two strategies:
  - `identity`: direct feature→frequency mapping (same formula as current `synthesize.py`)
  - `cluster`: embeds feature descriptions with a sentence transformer, k-means clusters them, assigns each cluster an instrument name; the `meta` line triggers cluster pre-computation before any tokens arrive
- `synthesize.py` gains `--live` flag and `--mode timed|sustain` for real-time playback via `sounddevice`. Timed plays exactly TOKEN_DURATION (0.5 s) of audio per token. Sustain hard-cuts to the new sound the moment the next token event arrives.

## Relevant Files

- `extract.py` — add `--stream` and `--loop` flags; refactor the generation loop to optionally yield per-token NDJSON to stdout with `flush=True`; emit a `meta` header line first; record `elapsed_ms` per token; replay recorded tokens in loop mode with original timing
- `synthesize.py` — add `--live` / `--mode timed|sustain`; implement timed playback using `sounddevice.play` per token; implement sustain playback using a `sounddevice.OutputStream` callback that hot-swaps the current audio buffer on new token arrival
- `export.py` — unchanged

### New Files
- `transform.py` — NDJSON→NDJSON transformer; reads `meta` then `token` events; identity strategy replicates current feature→freq mapping; cluster strategy embeds descriptions with `sentence-transformers`, runs k-means via `scikit-learn`, caches cluster assignments to `neuronpedia_cache/{model_id}_{layer}_{width}_clusters_{n}.json` for fast reuse; emits `MusicalEvent` lines with `notes: [{freq, amplitude, cluster}]`
- `specs/streaming-live-audio-pipeline.md` — this plan

## NDJSON Wire Formats

### extract.py stdout (TokenStream)

First line — metadata header:
```json
{"type": "meta", "model_id": "google/gemma-3-1b-pt", "layer": 22, "sae_width": "65k"}
```

Subsequent lines — one per generated token:
```json
{"type": "token", "token_id": 1234, "token": "the", "l0": 42, "active_features": [{"index": 5, "activation": 1.3, "description": "..."}], "elapsed_ms": 312}
```

### transform.py stdout (MusicalEvent stream)

First line — pass-through metadata:
```json
{"type": "meta", "model_id": "google/gemma-3-1b-pt", "layer": 22, "sae_width": "65k", "strategy": "cluster", "n_clusters": 8}
```

Subsequent lines — one per token:
```json
{"type": "token", "token": "the", "token_id": 1234, "elapsed_ms": 312, "notes": [{"freq": 440.0, "amplitude": 1.3, "cluster": 3}]}
```

For `identity` strategy, `cluster` is `null` in every note.

## Implementation Plan

### Phase 1: Foundation — streaming extract
Refactor `inspect_live()` in `extract.py` to yield `TokenAnalysis` objects one at a time (generator) instead of collecting all into a list. This unblocks the stream without changing the batch path (batch mode just exhausts the generator and collects).

### Phase 2: Core Implementation — transform.py + live synth
Implement `transform.py` with both strategies. Implement live audio in `synthesize.py` using `sounddevice`. The cluster strategy pre-computes embeddings and k-means during the window between reading the `meta` line and the first `token` line — extract.py takes at least several seconds to load the model and generate the first token, giving transform.py plenty of time.

### Phase 3: Integration — loop mode + replay timing
Add `--loop` to `extract.py`: after generation ends (and optional `--output` save), re-emit the recorded token lines with the original `elapsed_ms` delays, looping forever until SIGINT (Ctrl+C).

## Step by Step Tasks

### Step 1: Add `sounddevice` and `sentence-transformers` and `scikit-learn` dependencies
- `uv add sounddevice sentence-transformers scikit-learn`

### Step 2: Refactor `inspect_live()` in `extract.py` to a generator
- Change return type from `GenerationAnalysis` to `Generator[TokenAnalysis, None, GenerationAnalysis]`
- Replace the list-collect pattern with `yield token_analysis` per token inside the loop
- The caller collects all yielded items to build `GenerationAnalysis` for batch mode
- For stream mode, the caller processes each item immediately as it's yielded

### Step 3: Add `--stream` and `--loop` flags to `extract.py`
- `--stream`: emit one NDJSON line per token to stdout (with `flush=True`); first line is the `meta` header
- `--loop`: after generation completes, replay the list of recorded token NDJSON lines in sequence, sleeping `elapsed_ms` ms between each; loop forever, catch `KeyboardInterrupt` to exit cleanly
- Both flags can be combined: `--stream --loop`
- Record `elapsed_ms` per token: measure wall-clock time from immediately before the forward pass to immediately after `yield`; this reflects actual inference latency including SAE encoding
- `--output` still saves the full `GenerationAnalysis` JSON at the end of the first (live) generation pass; in loop mode, subsequent passes are replay-only (no re-inference, no re-save)

### Step 4: Create `transform.py`
- CLI: `uv run python transform.py [input] [--strategy identity|cluster] [--clusters N] [--embed-model MODEL]`
  - `input` (optional positional): path to a batch JSON file produced by `extract.py` (i.e. a `GenerationAnalysis` JSON). When omitted, reads NDJSON from stdin (pipe mode).
  - `--strategy`: default `identity`
  - `--clusters`: default `8`
  - `--embed-model`: default `all-MiniLM-L6-v2`

**Standalone mode (batch JSON input):**
- Read the JSON file, reconstruct a synthetic `meta` event from its `model_id`, `layer`, `sae_width` fields, and a sequence of `token` events from `generated_tokens`
- `elapsed_ms` is set to `0` for all tokens in standalone mode (no timing info in the batch file)
- Feed these synthetic events through the same strategy pipeline as pipe mode
- Emit NDJSON to stdout (can be piped directly into `synthesize.py --live`)

**Pipe mode (stdin, default when no positional arg):**
- Read NDJSON from stdin line by line
- On `meta` line:
  - Parse `model_id`, `layer`, `sae_width`
  - If strategy is `cluster`, load or build the cluster map (see below), then emit the enriched `meta` line to stdout
  - If strategy is `identity`, emit the `meta` line with `strategy: "identity"` and pass through
- On `token` line:
  - Apply the active strategy to `active_features` → list of `{freq, amplitude, cluster}`
  - Emit a `MusicalEvent` NDJSON line to stdout with `flush=True`

**Identity strategy:**
- Same `feature_to_frequency(index)` formula currently in `synthesize.py`
- `cluster` field is `null`

**Cluster strategy — build_cluster_map:**
- Cache path: `neuronpedia_cache/{model_id}_{layer}_{sae_width}_clusters_{n}.json`
- If cache exists: load and return `{feature_index: {cluster_id, instrument}}`
- If not:
  1. Load neuronpedia cache from `neuronpedia_cache/{model_id}_{layer}_{sae_width}.jsonl` (already downloaded by extract.py)
  2. Build list of `(index, description)` pairs — skip entries with null description
  3. Detect device: `embed_device = "cuda" if torch.cuda.is_available() else "cpu"`; log the chosen device to stderr
  4. Embed descriptions using `SentenceTransformer(embed_model, device=embed_device).encode(descriptions, batch_size=512, show_progress_bar=True)` — `batch_size=512` is safe on a 24 GB GPU; on CPU use `batch_size=64`
  5. Run `sklearn.cluster.MiniBatchKMeans(n_clusters=n, random_state=42).fit(embeddings)`
  6. Build cluster→instrument map: cycle through an instrument list `["piano", "guitar", "bass", "strings", "pad", "bell", "flute", "brass"]` (wrap if `n_clusters > 8`)
  7. Build `{feature_index: {cluster_id, instrument}}` dict and save to cache JSON
- Feature→frequency inside cluster strategy: same log-scale formula as identity (frequency encodes position within cluster's feature range), amplitude from activation, cluster from the cluster map

### Step 5: Add `--live`, `--mode` to `synthesize.py`

**Add shared `feature_to_frequency` import path:**
- Extract `feature_to_frequency()` into a small shared module `audio_utils.py` so both `transform.py` (identity strategy) and `synthesize.py` (legacy batch path) import from the same place

**Timed mode (`--mode timed`):**
- For each incoming `MusicalEvent` NDJSON line, call `generate_token_audio(notes)` to build `SAMPLES_PER_TOKEN` samples
- Play with `sounddevice.play(segment, samplerate=SAMPLE_RATE, blocking=True)` — this blocks for exactly TOKEN_DURATION (0.5 s) before processing the next token
- If a token's audio is still playing when the next arrives in the stream buffer, it finishes its full 0.5 s (timed is always fixed-duration)

**Sustain mode (`--mode sustain`):**
- Open a `sounddevice.OutputStream` with a callback
- The callback reads from a shared `current_buffer: np.ndarray` (protected by a `threading.Lock`) in a loop, tracking position within the buffer; when the end of the buffer is reached, wrap around (sustain = loop the current note)
- A reader thread reads NDJSON lines from stdin; on each new `token` event, synthesize the audio buffer for that token's notes and atomically swap `current_buffer` (hard cut, no crossfade)
- Main thread blocks until stdin closes or `KeyboardInterrupt`

**Instrument timbre per cluster:**
- When `cluster` field is non-null, modify the synthesis to approximate the instrument:
  - `piano`: sine + 0.5× second harmonic
  - `guitar`: sine + 0.3× second + 0.2× third harmonic
  - `bass`: sine + 0.6× octave below (half-frequency)
  - `strings`: sine + small random detune (±2 Hz) on a copy of itself summed
  - `pad`: sine with slow amplitude envelope (fade in/out over TOKEN_DURATION)
  - `bell`: sine + 0.4× second + 0.2× fifth harmonic, amplitude decays exponentially
  - `flute`: pure sine (same as current, no harmonics)
  - `brass`: sine + 0.7× second + 0.5× third + 0.3× fourth harmonic
  - Default (null cluster / identity): pure sine (current behavior)

### Step 6: Create `audio_utils.py`
- Move `feature_to_frequency()`, `FREQ_MIN`, `FREQ_MAX`, `NUM_INSTRUMENTS`, `FEATURES_PER_INSTRUMENT` from `synthesize.py` into `audio_utils.py`
- Move `generate_token_audio()` into `audio_utils.py`, generalized to accept a list of `{freq, amplitude}` dicts (instrument timbre logic lives here)
- Update `synthesize.py` and `transform.py` to import from `audio_utils`

### Step 7: Update `README.md`
- Add streaming pipeline section showing the three-stage pipe command
- Document all new flags: `--stream`, `--loop`, `--live`, `--mode`, `--strategy`, `--clusters`
- Show example: build clusters first, then stream

### Step 8: Validate
- Run the validation commands below

## Testing Strategy

### Unit Tests
No formal test suite. Manual smoke tests.

### Integration Tests
- Batch path regression: `extract.py ... && synthesize.py ...` still produces a WAV (no regression)
- Stream path: `extract.py --stream` emits valid NDJSON with `meta` first, then `token` lines
- Transform identity: output notes match expected frequency for known feature indices
- Transform cluster: cluster map is built and cached; second run loads from cache (fast)
- Live timed: audio plays without errors for at least 3 tokens
- Live sustain: audio callback runs continuously; buffer swaps without glitches

### Edge Cases
- No active features on a token → emit an empty `notes: []` event; synth plays silence for that token
- Cluster strategy, feature with null description → omit from embedding; assign to cluster 0 as fallback
- `--loop` with `--max-tokens 1` → loops a single token indefinitely (should work)
- `KeyboardInterrupt` mid-generation in `--stream --loop` → exits cleanly, no partial JSON written to `--output`
- `n_clusters` larger than number of features with descriptions → `MiniBatchKMeans` will error; clamp `n_clusters` to `min(n_clusters, len(embeddings))`

## Acceptance Criteria
- `uv run python extract.py "test" --stream --max-tokens 5` emits a `meta` line then 5 `token` NDJSON lines to stdout
- `uv run python extract.py "test" --stream --max-tokens 5 | uv run python transform.py --strategy identity` emits a `meta` line then 5 `MusicalEvent` lines with non-null `freq` values
- `uv run python extract.py "test" --stream --max-tokens 5 | uv run python transform.py --strategy cluster --clusters 4` builds a cluster map (or loads from cache) and emits `MusicalEvent` lines with `cluster` in `0..3`
- `--loop` causes the stream to replay after generation ends, indefinitely, until Ctrl+C
- `--live --mode timed` plays audio to the soundcard for each token, approximately 0.5 s per token
- `--live --mode sustain` plays audio continuously, hard-cutting to the new sound on each token arrival
- Batch mode (`extract.py` without `--stream`, `synthesize.py` without `--live`) continues to work unchanged
- Cluster map is cached to `neuronpedia_cache/` and reused on subsequent runs

## Validation Commands

```bash
# 1. Verify new dependencies installed
uv run python -c "import sounddevice, sentence_transformers, sklearn; print('deps OK')"

# 2. Smoke-test extract stream mode (no model needed — will fail on model load, but NDJSON structure is verifiable with a mock)
uv run python extract.py --help | grep -E "stream|loop"

# 3. Smoke-test transform CLI
uv run python transform.py --help | grep -E "strategy|clusters"

# 4. Smoke-test synthesize live flags
uv run python synthesize.py --help | grep -E "live|mode"

# 5. Batch regression: extract + synthesize still works
uv run python extract.py "The law of conservation of energy" --layer 22 --width 65k --output runs/stream_test.json --verbose
uv run python synthesize.py runs/stream_test.json --method additive --output-dir audio

# 6. Full stream pipeline (identity, timed — requires model loaded)
uv run python extract.py "hello world" --stream --max-tokens 10 \
  | uv run python transform.py --strategy identity \
  | uv run python synthesize.py --live --mode timed

# 7. Full stream pipeline (cluster, sustain)
uv run python extract.py "hello world" --stream --max-tokens 10 \
  | uv run python transform.py --strategy cluster --clusters 4 \
  | uv run python synthesize.py --live --mode sustain

# 8. Verify cluster cache written
ls neuronpedia_cache/*_clusters_4.json

# 9. Loop mode (must Ctrl+C to stop — run for 5 s in a subshell to validate it loops)
timeout 5 uv run python extract.py "hello" --stream --loop --max-tokens 3 \
  | uv run python transform.py --strategy identity \
  | uv run python synthesize.py --live --mode timed || true

# 10. Standalone transform from batch JSON (identity)
uv run python transform.py runs/stream_test.json --strategy identity | head -3

# 11. Standalone transform from batch JSON piped into live synth (cluster)
uv run python transform.py runs/stream_test.json --strategy cluster --clusters 4 \
  | uv run python synthesize.py --live --mode timed
```

## Notes

**On terminal pipe buffering:** Python's stdout switches to block-buffering when piped (4–8 KB buffer). Every `print()` in extract.py and transform.py must use `flush=True`. Alternatively, launch with `python -u` (unbuffered), but `flush=True` is more explicit and portable.

**Cluster pre-computation window:** The window between `meta` line emission and the first `token` line from extract.py is large (model load + first forward pass = tens of seconds). This gives transform.py ample time to build the cluster map from the local neuronpedia cache. No coordination primitives needed — the pipe naturally back-pressures if transform.py is not yet ready to emit.

**Embedding scale:** The neuronpedia cache for one scope can have 65k feature descriptions. Embedding all of them with `all-MiniLM-L6-v2` (384-dim, fast) takes ~30–90 s on CPU, under 10 s on an RTX 4090 (confirmed on this machine). `transform.py` auto-detects CUDA via `torch.cuda.is_available()` and uses `batch_size=512` on GPU vs `batch_size=64` on CPU. This only happens on the first run; subsequent runs load from the cache JSON.

**Loop replay timing:** During replay, sleep `elapsed_ms` ms between emitting each token line to reproduce the original generation rhythm in sustain mode. In timed mode, the synth controls timing (0.5 s fixed), so replay emissions can be as fast as the pipe allows.

**Future: OSC output from transform.py:** Adding `--output-format osc` to transform.py would emit UDP OSC messages instead of NDJSON stdout, enabling SuperCollider / Max/MSP / Pure Data as the synthesis backend — the natural next step for multi-instrument richness. The cluster map maps directly to OSC address namespacing (e.g. `/sae/cluster/3`).

**New libraries:**
- `sounddevice` — callback-based real-time audio I/O
- `sentence-transformers` — lightweight sentence embedding for feature description clustering
- `scikit-learn` — `MiniBatchKMeans` for fast large-scale clustering
```

Install: `uv add sounddevice sentence-transformers scikit-learn`
