# responsible-ai-sensification
sensification of the output of gemma 3

don't forget to login with huggingface to be able to download the model as well as accepting the gemma-3 license on huggingface

source code : https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r#scrollTo=nOBcV4om7mrT

## CLI Tools

### extract.py

Loads a model + SAE + Neuronpedia explanations, generates tokens autoregressively, and writes a JSON file with per-token SAE feature activations.

```
uv run python extract.py PROMPT [--model MODEL] [--layer LAYER] [--width WIDTH]
                                [--l0 L0] [--max-tokens N] [--output PATH] [--verbose]
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
| `--verbose` | off | Print progress |

### synthesize.py

Reads a `GenerationAnalysis` JSON and renders a WAV file by mapping SAE feature activations to audio.

```
uv run python synthesize.py INPUT [--method METHOD] [--output-dir DIR]
```

| Flag | Default | Description |
|---|---|---|
| `input` | *(required)* | Path to JSON produced by `extract.py` |
| `--method` | `additive` | Synthesis method (`additive`) |
| `--output-dir` | `audio` | Output directory |

### Basic pipeline

```bash
uv run python extract.py "The law of conservation of energy" --layer 22 --width 65k --verbose
uv run python synthesize.py runs/analysis.json --method additive
```
