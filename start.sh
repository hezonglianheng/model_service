#!/usr/bin/env bash
# 前台启动模型服务；可在任意目录执行：bash /path/to/model_service/start.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="$SCRIPT_DIR/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "未找到虚拟环境：$PY" >&2
  echo "请先运行：bash $SCRIPT_DIR/setup.sh" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
exec "$PY" "$SCRIPT_DIR/server.py" "$@"
