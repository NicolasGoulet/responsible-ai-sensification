#!/usr/bin/env bash
PYTHONPATH=. uv run uvicorn app.server.main:app --host 0.0.0.0 --port 8080 --reload
