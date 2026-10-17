#!/bin/bash
# CPU-only path: unit tests + surrogate-policy smoke run (no Ollama, seconds).
set -euo pipefail
uv run pytest -q
uv run python -m dreamcache.experiments.runner configs/smoke_cpu.yaml --workers 2
