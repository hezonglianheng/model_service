#!/usr/bin/env bash
# 后台服务管理：start [--wait] / stop / restart [--wait] / status / logs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_DIR="$SCRIPT_DIR/run"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$RUN_DIR/server.pid"
LOG_FILE="$LOG_DIR/server.log"
PY="$SCRIPT_DIR/.venv/bin/python"

mkdir -p "$RUN_DIR" "$LOG_DIR"

BASE_URL="$("$PY" - <<'PY' 2>/dev/null || echo "http://127.0.0.1:8000"
import config
host = "127.0.0.1" if config.HOST == "0.0.0.0" else config.HOST
print(f"http://{host}:{config.PORT}")
PY
)"

log() { printf '[service] %s\n' "$*"; }
die() { printf '[service] 错误：%s\n' "$*" >&2; exit 1; }

read_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  printf '%s' "$pid"
}

pid_alive() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null
}

is_our_process() {
  local pid="$1"
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' <"/proc/$pid/cmdline" | grep -q -e 'server.py' -e 'vllm'
}

health_ok() {
  curl --noproxy '*' -fsS --max-time 3 "$BASE_URL/health" >/dev/null 2>&1
}

show_replicas() {
  "$PY" - <<'PY' || true
import json
import urllib.request
import config
host = "127.0.0.1" if config.HOST == "0.0.0.0" else config.HOST
headers = {"Authorization": f"Bearer {config.API_KEY}"} if config.API_KEY else {}
req = urllib.request.Request(f"http://{host}:{config.PORT}/replicas", headers=headers)
try:
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=3) as response:
        data = json.load(response)
    for b in data["backends"]:
        print(f"  {b['name']}: {b['state']}, healthy={b['healthy']}, "
              f"inflight={b['inflight']}, completed={b['completed']}")
    for g in data["gpus"]:
        if not g["eligible"]:
            print(f"  GPU {g['gpu']} skipped: {g['reason']}")
except Exception as exc:
    print(f"  副本详情暂不可用：{exc}")
PY
}

wait_ready() {
  local pid="$1"
  local waited=0
  local timeout="${STARTUP_TIMEOUT:-14400}"
  local idle_timeout="${STARTUP_IDLE_TIMEOUT:-1200}"
  local last_size=0
  local last_progress=0
  local size
  log "等待服务就绪（最多 ${timeout}s；日志 ${idle_timeout}s 无变化则判定卡住）"
  while (( waited < timeout )); do
    if ! pid_alive "$pid"; then
      log "进程已退出，最近日志："
      tail -n 80 "$LOG_FILE" >&2 || true
      return 1
    fi
    if health_ok; then
      log "服务已就绪（至少一个副本健康）：$BASE_URL"
      show_replicas
      return 0
    fi
    # 首次启动可能要先下载 55GB 模型；只要日志还在增长就继续等
    size="$(stat -c %s "$LOG_FILE" 2>/dev/null || echo 0)"
    if (( size != last_size )); then
      last_size=$size
      last_progress=$waited
    fi
    if (( waited - last_progress > idle_timeout )); then
      log "日志已 ${idle_timeout}s 无变化，判定启动卡住。最近日志："
      tail -n 80 "$LOG_FILE" >&2 || true
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
    if (( waited % 60 == 0 )); then
      log "已等待 ${waited}s，最近日志："
      tail -n 5 "$LOG_FILE" | sed 's/^/    /' || true
    fi
  done
  log "等待超时（${timeout}s）。最近日志："
  tail -n 80 "$LOG_FILE" >&2 || true
  return 1
}

cmd_start() {
  local wait_flag=0
  if [[ "${1:-}" == "--wait" ]]; then
    wait_flag=1
    shift
  fi

  if pid="$(read_pid)" && pid_alive "$pid"; then
    if ! is_our_process "$pid"; then
      die "PID 文件 $PID_FILE 指向其他进程（PID $pid），请先人工确认后删除该文件"
    fi
    log "服务已在运行（PID $pid）"
    if (( wait_flag )); then wait_ready "$pid"; fi
    return 0
  fi
  rm -f "$PID_FILE"

  [[ -x "$PY" ]] || die "未找到虚拟环境，请先运行：bash $SCRIPT_DIR/setup.sh"
  if [[ -f "$LOG_FILE" ]]; then
    log "日志追加到：$LOG_FILE"
  fi

  log "启动服务，日志：$LOG_FILE"
  PYTHONUNBUFFERED=1 nohup "$PY" "$SCRIPT_DIR/server.py" "$@" \
    >>"$LOG_FILE" 2>&1 &

  local pid=$!
  echo "$pid" >"$PID_FILE"
  log "PID $pid"

  sleep 2
  if ! pid_alive "$pid"; then
    log "启动失败，日志末尾："
    tail -n 80 "$LOG_FILE" >&2 || true
    rm -f "$PID_FILE"
    return 1
  fi

  if (( wait_flag )); then
    wait_ready "$pid"
  else
    log "已在后台启动。查看状态：bash $SCRIPT_DIR/service.sh status"
  fi
}

cmd_stop() {
  local pid
  if ! pid="$(read_pid)"; then
    log "未找到 PID 文件，服务可能未运行"
    rm -f "$PID_FILE"
    return 0
  fi
  if ! pid_alive "$pid"; then
    log "进程不存在（PID $pid），清理 PID 文件"
    rm -f "$PID_FILE"
    return 0
  fi
  if ! is_our_process "$pid"; then
    die "PID $pid 不属于本服务，拒绝发送信号。请人工检查 /proc/$pid/cmdline 后删除 $PID_FILE"
  fi

  log "发送 SIGINT（supervisor PID $pid），等待网关和所有副本退出..."
  kill -INT "$pid" 2>/dev/null || true
  for _ in $(seq 1 45); do
    pid_alive "$pid" || { rm -f "$PID_FILE"; log "已停止"; return 0; }
    sleep 1
  done

  log "优雅退出超时，发送 SIGTERM..."
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    pid_alive "$pid" || { rm -f "$PID_FILE"; log "已停止"; return 0; }
    sleep 1
  done

  log "仍未退出，发送 SIGKILL 给 supervisor；副本守护进程将按 REPLICA_CHILD_GRACE_SECONDS 回收 GPU 进程"
  kill -KILL "$pid" 2>/dev/null || true
  sleep 1
  rm -f "$PID_FILE"
  log "已强制停止 supervisor（副本清理可能仍在进行，见 logs/gpu-*.log）"
}

cmd_restart() {
  cmd_stop
  cmd_start "$@"
}

cmd_status() {
  local pid
  if ! pid="$(read_pid)" || ! pid_alive "$pid"; then
    log "状态：未运行"
    return 3
  fi
  if ! is_our_process "$pid"; then
    log "警告：PID $pid 存活，但不像本服务的进程"
  fi
  if health_ok; then
    log "状态：运行中（PID $pid），$BASE_URL/health 正常"
    show_replicas
    return 0
  fi
  log "状态：进程存活（PID $pid），但健康检查未通过（可能仍在加载模型）"
  return 1
}

cmd_logs() {
  tail -n 200 -f "$LOG_FILE"
}

case "${1:-}" in
  start) shift; cmd_start "$@" ;;
  stop) cmd_stop ;;
  restart) shift; cmd_restart "$@" ;;
  status) cmd_status ;;
  logs) cmd_logs ;;
  *)
    cat <<EOF
用法：bash $SCRIPT_DIR/service.sh <command>

  start [--wait]    后台启动（--wait 轮询 /health 直到就绪）
  stop              SIGINT -> SIGTERM -> SIGKILL 依次尝试
  restart [--wait]  重启
  status            查看进程与健康状态
  logs              tail -f 日志

环境变量：
  STARTUP_TIMEOUT=14400        最长等待秒数（首次下载模型可能较久）
  STARTUP_IDLE_TIMEOUT=1200    日志无变化多少秒后判定卡住
EOF
    exit 2
    ;;
esac
