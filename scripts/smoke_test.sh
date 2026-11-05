#!/usr/bin/env bash
# 冒烟测试：/health -> /v1/models -> 一次短 chat 请求
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"

BASE_URL="${BASE_URL:-$("$PY" - "$ROOT" <<'PY' 2>/dev/null || echo "http://127.0.0.1:8000"
import sys
sys.path.insert(0, sys.argv[1])
import config
print(f"http://{config.HOST}:{config.PORT}")
PY
)}"
MODEL="${MODEL:-$("$PY" - "$ROOT" <<'PY' 2>/dev/null || echo "Qwen/Qwen3.8-27B"
import sys
sys.path.insert(0, sys.argv[1])
import config
print(config.SERVED_MODEL_NAME)
PY
)}"
API_KEY="$("$PY" - "$ROOT" <<'PY' 2>/dev/null || true
import sys
sys.path.insert(0, sys.argv[1])
import config
print(config.API_KEY or "")
PY
)"

AUTH=()
if [[ -n "$API_KEY" ]]; then
  AUTH=(-H "Authorization: Bearer $API_KEY")
fi

log() { printf '[smoke] %s\n' "$*"; }

log "健康检查：$BASE_URL/health"
curl -fsS --max-time 10 "$BASE_URL/health" >/dev/null
log "OK"

log "模型列表：$BASE_URL/v1/models"
MODELS_JSON="$(curl -fsS --max-time 10 "${AUTH[@]}" "$BASE_URL/v1/models")"
printf '%s' "$MODELS_JSON" | "$PY" -c '
import json, sys
data = json.load(sys.stdin)
ids = [m.get("id") for m in data.get("data", [])]
print("  models:", ", ".join(str(x) for x in ids))
'
if [[ "$MODELS_JSON" != *"$MODEL"* ]]; then
  echo "[smoke] 警告：/v1/models 中未找到 $MODEL" >&2
fi

log "chat 请求（关闭 thinking，max_tokens=64）"
RESPONSE="$(curl -fsS --max-time 300 "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -X POST "$BASE_URL/v1/chat/completions" \
  -d "$("$PY" - "$MODEL" <<'PY'
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": "用一句话说明你是什么模型。"}],
    "max_tokens": 64,
    "temperature": 0.7,
    "chat_template_kwargs": {"enable_thinking": False},
}, ensure_ascii=False))
PY
)")"

printf '%s' "$RESPONSE" | "$PY" -c '
import json, sys
data = json.load(sys.stdin)
msg = data["choices"][0]["message"]
content = (msg.get("content") or "").strip()
if not content:
    print("[smoke] 错误：返回内容为空", file=sys.stderr)
    print(json.dumps(data, ensure_ascii=False, indent=2)[:2000], file=sys.stderr)
    sys.exit(1)
print("[smoke] 回复：", content[:300])
print("[smoke] 通过")
'
