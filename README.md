# responsible-ai-sensification

Sensification of the output of Gemma 3 — turns SAE feature activations into live generative audio via a browser UI.

> **Requires a HuggingFace login** and acceptance of the [Gemma 3 license](https://huggingface.co/google/gemma-3-1b-pt) before first use.

## Quick Start (Web UI)

```bash
# Install system dependency for audio (only needed for CLI --live flag)
sudo apt install libportaudio2

# Start the server
./scripts/start.sh

# Open in browser
http://localhost:8080
```

```bash
# Stop the server
./scripts/stop.sh
```

The browser UI is the primary interface. Set a prompt, choose model/strategy/layer/width/clusters/mode/BPM, then click **Start**. Parameters can be tweaked mid-generation without restarting.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| **Prompt** | `Hello world` | Text prompt fed to the model |
| **Model** | `google/gemma-3-1b-pt` | HuggingFace model (`gemma-3-1b-pt` or `gemma-3-4b-pt`) |
| **Layer** | `22` | Transformer layer index to hook SAE onto |
| **Width** | `65k` | SAE width |
| **Strategy** | `identity` | `identity`: maps each feature directly to a frequency; `cluster`: groups features by semantic similarity (k-means on Neuronpedia embeddings) |
| **Clusters** | `8` | Number of k-means clusters (cluster strategy only) |
| **Mode** | `timed` | `timed`: notes play for a fixed BPM-derived duration; `sustain`: notes hold until the next token arrives |
| **BPM** | `120` | Tempo for timed mode |
| **Loop** | off | Replay generated tokens indefinitely after generation ends |

### Verbose logging

```bash
./scripts/start.sh --verbose
# or
VERBOSE=1 ./scripts/start.sh
```

---

## Prerequisites

```bash
sudo apt install libportaudio2   # required for CLI --live playback only
```

## Directory Layout

```
app/
  client/          # Vanilla-JS browser frontend
    index.html
    style.css
    main.js
  server/
    main.py        # FastAPI app factory
    session.py     # PipelineParams + PipelineSession
    pipeline/
      extract.py
      transform.py
      synthesize.py
      audio_utils.py
      export.py
    routers/
      config.py    # GET /api/config/defaults, GET /api/config/model-options
      stream.py    # WS /ws/stream
scripts/
  start.sh         # Start uvicorn on port 8080
  stop.sh          # Kill uvicorn on port 8080
specs/             # Feature specs + TODO backlog
```

---

## CLI Tools

The pipeline can also be used directly from the command line.

### extract.py

Loads a model + SAE + Neuronpedia explanations, generates tokens autoregressively, and writes a JSON file with per-token SAE feature activations.

```
uv run python app/server/pipeline/extract.py PROMPT [--model MODEL] [--layer LAYER] [--width WIDTH]
                                [--l0 L0] [--max-tokens N] [--output PATH] [--verbose]
                                [--stream] [--loop]
```

| Flag | Default | Description |
|---|---|---|
| `prompt` | *(required)* | Prompt string |
| `--model` | `google/gemma-3-1b-pt` | HuggingFace model ID |
| `--layer` | `22` | Transformer layer index |
| `--width` | `65k` | SAE width |
| `--l0` | `medium` | SAE L0 target |
| `--max-tokens` | `200` | Maximum new tokens |
| `--output` | `runs/analysis.json` | Output JSON path |
| `--verbose` | off | Print progress to stderr |
| `--stream` | off | Emit one NDJSON line per token to stdout (meta header first) |
| `--loop` | off | Replay recorded tokens indefinitely after generation (Ctrl+C to stop) |

### synthesize.py

Reads a `GenerationAnalysis` JSON and renders a WAV file, or plays live audio from a `MusicalEvent` NDJSON stream.

```
uv run python app/server/pipeline/synthesize.py [INPUT] [--method METHOD] [--output-dir DIR]
                             [--live] [--mode timed|sustain]
```

| Flag | Default | Description |
|---|---|---|
| `input` | *(required in batch mode)* | Path to JSON produced by `extract.py` |
| `--method` | `additive` | Synthesis method (`additive`) |
| `--output-dir` | `audio` | Output directory |
| `--live` | off | Play audio live from NDJSON stdin |
| `--mode` | `timed` | Live mode: `timed` (0.5 s/token) or `sustain` (hold until next token) |

### transform.py

Transforms a `TokenStream` NDJSON (from `extract.py --stream`) into a `MusicalEvent` NDJSON stream with frequency/amplitude/instrument assignments.

```
uv run python app/server/pipeline/transform.py [INPUT] [--strategy identity|cluster] [--clusters N] [--embed-model MODEL]
```

| Flag | Default | Description |
|---|---|---|
| `input` | *(stdin if omitted)* | Batch JSON file or omit to read NDJSON from stdin |
| `--strategy` | `identity` | `identity`: direct feature→freq; `cluster`: semantic clustering |
| `--clusters` | `8` | Number of k-means clusters (cluster strategy) |
| `--embed-model` | `all-MiniLM-L6-v2` | Sentence transformer model for embeddings |

### Basic pipeline

```bash
uv run python app/server/pipeline/extract.py "The law of conservation of energy" --layer 22 --width 65k --verbose
uv run python app/server/pipeline/synthesize.py runs/analysis.json --method additive
```

### Streaming live pipeline

```bash
# Identity strategy, timed mode (0.5 s per token):
uv run python app/server/pipeline/extract.py "hello world" --stream --max-tokens 20 \
  | uv run python app/server/pipeline/transform.py --strategy identity \
  | uv run python app/server/pipeline/synthesize.py --live --mode timed

# Cluster strategy, sustain mode (each note holds until next token arrives):
uv run python app/server/pipeline/extract.py "hello world" --stream --max-tokens 20 \
  | uv run python app/server/pipeline/transform.py --strategy cluster --clusters 8 \
  | uv run python app/server/pipeline/synthesize.py --live --mode sustain

# Loop mode — replay the generation indefinitely (Ctrl+C to stop):
uv run python app/server/pipeline/extract.py "hello world" --stream --loop --max-tokens 20 \
  | uv run python app/server/pipeline/transform.py --strategy identity \
  | uv run python app/server/pipeline/synthesize.py --live --mode timed

# Build cluster map first from a batch JSON, then stream it live:
uv run python app/server/pipeline/transform.py runs/analysis.json --strategy cluster --clusters 8 \
  | uv run python app/server/pipeline/synthesize.py --live --mode timed
```

---

Source: https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r#scrollTo=nOBcV4om7mrT
