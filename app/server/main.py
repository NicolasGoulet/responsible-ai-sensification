"""main.py: FastAPI application factory."""
import logging
import os
from pathlib import Path

from fastapi import FastAPI

_log_level = logging.DEBUG if os.environ.get("VERBOSE") == "1" else logging.INFO
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.server.routers import config, stream

CLIENT_DIR = Path(__file__).parent.parent / "client"

app = FastAPI(title="Responsible AI Sensification")

app.include_router(config.router)
app.include_router(stream.router)

# Serve static files at /static so the root GET / below takes priority
app.mount("/static", StaticFiles(directory=CLIENT_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(CLIENT_DIR / "index.html")


@app.get("/{path:path}")
async def static_fallback(path: str) -> FileResponse:
    target = CLIENT_DIR / path
    if target.exists() and target.is_file():
        return FileResponse(target)
    return FileResponse(CLIENT_DIR / "index.html")
