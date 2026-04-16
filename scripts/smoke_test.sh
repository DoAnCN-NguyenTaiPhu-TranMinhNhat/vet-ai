#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q ai_service/tests/test_smoke.py
