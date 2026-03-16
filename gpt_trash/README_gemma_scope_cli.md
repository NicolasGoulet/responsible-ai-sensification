# Gemma Scope prompt CLI

This CLI analyzes a **prompt** with **Gemma 3 1B PT + Gemma Scope 2 SAE** and exports:

- a PNG report with:
  - top SAE features by prevalence
  - token × feature heatmap
  - L0 per token
- a JSON sidecar with the raw top-feature/token analysis

## Files

- `gemma_scope_prompt_cli.py`
- `requirements_gemma_scope_cli.txt`
- `setup_gemma_scope_cli.sh`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements_gemma_scope_cli.txt
```

Then:

1. Accept the Gemma license on Hugging Face for `google/gemma-3-1b-pt`
2. Export your HF token:

```bash
export HF_TOKEN=your_token_here
```

## Run

```bash
python gemma_scope_prompt_cli.py \
  --prompt "The law of conservation of energy states that energy cannot be created or destroyed." \
  --layer 22 \
  --width 65k \
  --l0 medium \
  --category resid_post \
  --aggregate sum_activation \
  --top-k 12 \
  --outdir runs/gemma_scope_prompt_report
```

## Useful switches

- `--aggregate count` ranks features by number of prompt tokens where they activate
- `--aggregate sum_activation` ranks features by total activation mass across the prompt
- `--top-k-heatmap 10` controls how many top features appear in the heatmap
- `--token-batch-size 64` reduces memory pressure during SAE encoding

## Output

You will get files like:

- `runs/gemma_scope_prompt_report/<prompt>_layer22_65k.png`
- `runs/gemma_scope_prompt_report/<prompt>_layer22_65k.json`
