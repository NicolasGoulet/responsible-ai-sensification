#!/usr/bin/env bash
fuser -k 8080/tcp 2>/dev/null && echo "Stopped" || echo "Not running"
