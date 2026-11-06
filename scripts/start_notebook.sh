#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
exec "$LOCAL_AI_ROOT/.venv/bin/python" -m notebook \
  --ServerApp.ip=127.0.0.1 \
  --ServerApp.open_browser=False \
  --ServerApp.root_dir="$LOCAL_AI_ROOT" \
  --ServerApp.port=8888

