import json
from pathlib import Path

from main import GenerationAnalysis


def export_to_json(analysis: GenerationAnalysis, output_path: Path) -> None:
    descriptions: dict[str, str] = {}
    for tok in analysis.generated_tokens:
        for feat in tok.active_features:
            if feat.description is not None:
                descriptions[str(feat.index)] = feat.description

    data = {
        "prompt": analysis.prompt,
        "model_id": analysis.model_id,
        "layer": analysis.layer,
        "sae_width": analysis.sae_width,
        "descriptions": descriptions,
        "generated_tokens": [
            {
                "token_id": tok.token_id,
                "token": tok.token,
                "l0": tok.l0,
                "active_features": [
                    {
                        "index": feat.index,
                        "activation": feat.activation,
                        "description": feat.description,
                    }
                    for feat in tok.active_features
                ],
            }
            for tok in analysis.generated_tokens
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
