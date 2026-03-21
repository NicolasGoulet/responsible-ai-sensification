# responsible-ai-sensification
sensification of the output of gemma 3

don't forget to login with huggingface to be able to download the model as well as accepting the gemma-3 license on huggingface

## Prerequisites

```bash
# System library required for live audio playback (--live flag)
sudo apt install libportaudio2
```

source code : https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r#scrollTo=nOBcV4om7mrT

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
    pipeline/      # Moved pipeline scripts
      extract.py
      transform.py
      synthesize.py
      audio_utils.py
      export.py
    routers/
      config.py    # GET /api/config/defaults
      stream.py    # WS /ws/stream
scripts/
  start.sh         # Start uvicorn server
  stop.sh          # Kill uvicorn on port 8000
specs/             # Feature specs + TODO backlog
```

## Web Server

```bash
# Start the server and open http://localhost:8000
./scripts/start.sh

# Stop the server
./scripts/stop.sh
```

The browser UI lets you set a prompt, choose strategy/layer/width/clusters, and click **Start** — all without touching the CLI. Parameters can be tweaked mid-generation.

## CLI Tools

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
