#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r modeling/api/requirements.txt
fi
if [ ! -d node_modules ]; then
  npm install
fi
.venv/bin/uvicorn modeling.api.app:app --port 8000 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT
npm run dev
