#!/usr/bin/env bash
LOG_LEVEL="info"
if [ "$1" = "--verbose" ] || [ "${VERBOSE:-0}" = "1" ]; then
    LOG_LEVEL="debug"
    export VERBOSE=1
fi
PYTHONPATH=. uv run uvicorn app.server.main:app \
    --host 0.0.0.0 --port 8080 --log-level "$LOG_LEVEL"
