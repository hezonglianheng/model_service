#!/usr/bin/env bash
# 创建项目内 .venv，安装 cu129 版 torch + vLLM，并做 CUDA/bf16 冒烟测试。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="${VENV:-$SCRIPT_DIR/.venv}"
PY="$VENV/bin/python"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup] 警告：%s\033[0m\n' "$*" >&2; }
die() { printf '\033[1;31m[setup] 错误：%s\033[0m\n' "$*" >&2; exit 1; }

command -v nvidia-smi >/dev/null 2>&1 || die "未找到 nvidia-smi，请先确认 NVIDIA 驱动已安装"

log "GPU 概况"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap \
  --format=csv,noheader || true

DRIVER_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -n1 || true)"
DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 || true)"
log "驱动版本：${DRIVER_VER:-未知}，驱动支持的最高 CUDA：${DRIVER_CUDA:-未知}"
if [[ -n "$DRIVER_CUDA" ]]; then
  major="${DRIVER_CUDA%%.*}"
  minor="${DRIVER_CUDA#*.}"
  minor="${minor%%.*}"
  if (( major < 12 )); then
    die "驱动过旧（最高 CUDA ${DRIVER_CUDA}）。cu129 wheel 至少需要支持 CUDA 12.0 的驱动（>=525）。"
  fi
  if (( major == 12 && minor < 4 )); then
    warn "驱动仅支持 CUDA ${DRIVER_CUDA}，将依赖 CUDA minor-version compatibility；若启动失败请见 README 的 cuda-compat 方案。"
  fi
fi

if [[ ! -x "$PY" ]]; then
  log "创建虚拟环境：$VENV"
  if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV" --python 3.12
  else
    "$PYTHON_BIN" -m venv "$VENV"
  fi
else
  log "复用已有虚拟环境：$VENV"
fi
[[ -x "$PY" ]] || die "虚拟环境创建失败：$PY 不存在"

if command -v uv >/dev/null 2>&1; then
  PIP=(uv pip install --python "$PY")
else
  "$PY" -m pip install --upgrade pip
  PIP=("$PY" -m pip install)
fi

VLLM_VERSION="$("$PY" - <<'PY'
import config
print(config.VLLM_VERSION)
PY
)"
VLLM_CUDA_VARIANT="$("$PY" - <<'PY'
import config
print(config.VLLM_CUDA_VARIANT)
PY
)"
TORCH_VERSION="$("$PY" - <<'PY'
import config
print(config.TORCH_VERSION)
PY
)"
TORCHVISION_VERSION="$("$PY" - <<'PY'
import config
print(config.TORCHVISION_VERSION)
PY
)"
TORCHAUDIO_VERSION="$("$PY" - <<'PY'
import config
print(config.TORCHAUDIO_VERSION)
PY
)"

log "安装 torch ${TORCH_VERSION}+${VLLM_CUDA_VARIANT}（PyTorch 官方源）"
"${PIP[@]}" \
  --extra-index-url "https://download.pytorch.org/whl/${VLLM_CUDA_VARIANT}" \
  "torch==${TORCH_VERSION}+${VLLM_CUDA_VARIANT}" \
  "torchvision==${TORCHVISION_VERSION}+${VLLM_CUDA_VARIANT}" \
  "torchaudio==${TORCHAUDIO_VERSION}+${VLLM_CUDA_VARIANT}"

log "安装 vLLM ${VLLM_VERSION}+${VLLM_CUDA_VARIANT}（vLLM wheel 源）"
"${PIP[@]}" \
  --extra-index-url "https://wheels.vllm.ai/${VLLM_VERSION}/${VLLM_CUDA_VARIANT}" \
  "vllm==${VLLM_VERSION}+${VLLM_CUDA_VARIANT}"

log "检查 torchcodec 与 cu129 的兼容性"
if "$PY" -c "import torchcodec" >/dev/null 2>&1; then
  log "torchcodec 可正常导入"
else
  warn "torchcodec 无法在当前 cu129 环境导入（PyPI wheel 通常面向 CUDA 13）"
  warn "卸载 torchcodec；音频/视频解码会回退到 PyAV/OpenCV"
  if command -v uv >/dev/null 2>&1; then
    uv pip uninstall --python "$PY" torchcodec || true
  else
    "$PY" -m pip uninstall -y torchcodec || true
  fi
fi

log "安装客户端与网关依赖（openai / huggingface_hub / socksio / ninja / aiohttp）"
"${PIP[@]}" "openai>=1.0" "huggingface_hub>=0.30" "socksio>=1.0" "ninja>=1.11" "aiohttp>=3.10,<4"

log "预下载 vLLM 预编译 kernel（失败不影响启动）"
if [[ "${SKIP_KERNELS:-0}" == "1" ]]; then
  log "SKIP_KERNELS=1，跳过"
else
  "$VENV/bin/vllm" download-kernels || warn "vllm download-kernels 失败，跳过"
fi

log "版本检查"
"$PY" - <<'PY'
import torch
import vllm
print(f"torch={torch.__version__} (CUDA {torch.version.cuda})")
print(f"vllm={vllm.__version__}")
PY

log "CUDA 冒烟测试（可见 GPU，逐卡检查）"
set +e
"$PY" - <<'PY'
import os
import sys
import config
from gpu_planner import discover_gpus, visible_gpu_uuids
from server import build_env

os.environ.update(build_env())
gpus = discover_gpus()
visible = visible_gpu_uuids(gpus)
selected = [g for g in gpus if (config.GPU_IDS == "auto" or g.index in config.GPU_IDS)
            and not g.mig_enabled and (visible is None or g.uuid.lower() in visible)]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(g.uuid for g in selected)
import torch

if not torch.cuda.is_available():
    print("torch.cuda.is_available() = False", file=sys.stderr)
    sys.exit(2)

passed = 0
for index in range(torch.cuda.device_count()):
    try:
        with torch.cuda.device(index):
            cap = torch.cuda.get_device_capability(index)
            dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
            print(f"设备 {index}: {torch.cuda.get_device_name(index)}, SM{cap}, {dtype}")
            a = torch.randn(256, 256, device=f"cuda:{index}", dtype=dtype)
            b = a @ a
            torch.cuda.synchronize()
            print(f"matmul OK，结果均值 {b.float().abs().mean().item():.4f}")
            del a, b
            torch.cuda.empty_cache()
            passed += 1
    except Exception as exc:
        print(f"设备 {index} 冒烟失败：{exc}", file=sys.stderr)
if not passed:
    sys.exit(2)
PY
SMOKE_RC=$?
set -e
if [[ $SMOKE_RC -ne 0 ]]; then
  cat >&2 <<'EOF'

[setup] CUDA 冒烟测试失败。常见原因与处理：
  1) 驱动（550.x，最高 CUDA 12.4）与 cu129 wheel 的兼容问题：
     运行 bash scripts/fetch_cuda_compat.sh 解压用户态 cuda-compat-12-9，
     然后在 config.py 中把 USE_CUDA_COMPAT 改为 True，再重试。
  2) 显卡被其他进程占满：nvidia-smi 查看显存占用。
  3) 容器/权限问题：确认当前用户能访问 /dev/nvidia*。
EOF
  exit "$SMOKE_RC"
fi

log "安装完成。下一步："
printf '  1) 下载模型：%s download_model.py\n' "$PY"
printf '  2) 前台启动：bash %s/start.sh\n' "$SCRIPT_DIR"
printf '  3) 后台启动：bash %s/service.sh start --wait\n' "$SCRIPT_DIR"
