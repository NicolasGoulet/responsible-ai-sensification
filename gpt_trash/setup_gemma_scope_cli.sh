#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements_gemma_scope_cli.txt

echo "Setup complete."
echo "Before running the CLI:"
echo "1) Accept the Gemma license on Hugging Face for google/gemma-3-1b-pt"
echo "2) Export your token: export HF_TOKEN=..."
