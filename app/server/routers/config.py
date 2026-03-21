"""config.py: Configuration and defaults API routes."""
import dataclasses

from fastapi import APIRouter

from app.server.session import PipelineParams

router = APIRouter(prefix="/api/config")

_DEFAULTS = PipelineParams()

MODEL_OPTIONS = [
    "google/gemma-3-1b-pt",
    "google/gemma-3-4b-pt",
]

LAYER_OPTIONS = list(range(29))       # 0–28 for Gemma-3 1B
WIDTH_OPTIONS = ["16k", "65k", "262k", "1m"]
STRATEGY_OPTIONS = ["identity", "cluster"]


@router.get("/defaults")
def get_defaults() -> dict:
    return dataclasses.asdict(_DEFAULTS)


@router.get("/model-options")
def get_model_options() -> dict:
    return {
        "models": MODEL_OPTIONS,
        "layers": LAYER_OPTIONS,
        "widths": WIDTH_OPTIONS,
        "strategies": STRATEGY_OPTIONS,
    }
