#!/usr/bin/env bash
# 无 sudo 获取 CUDA 12.9 用户态前向兼容库（cuda-compat-12-9）。
#
# 背景：cu129 wheel 在只支持 CUDA 12.4 的驱动（550.x）上通常可用
# （CUDA minor-version compatibility）；若启动报
# "CUDA driver version is insufficient for CUDA runtime version"，
# 用本脚本解压用户态兼容库，并把 config.USE_CUDA_COMPAT 改为 True。
#
# 安全说明：只下载 .deb 并用 dpkg-deb 解压，不执行其中任何文件。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET="$ROOT/.cuda-compat"

URL="${CUDA_COMPAT_URL:-https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-compat-12-9_575.57.08-0ubuntu1_amd64.deb}"
DEB_NAME="$(basename "$URL")"

command -v dpkg-deb >/dev/null 2>&1 || {
  echo "错误：需要 dpkg-deb（Ubuntu/Debian）。" >&2
  exit 1
}

mkdir -p "$ROOT/.downloads"
WORK="$(mktemp -d "$ROOT/.downloads/cuda-compat.XXXXXX")"
cleanup() {
  case "$WORK" in
    "$ROOT"/.downloads/cuda-compat.*) rm -rf "$WORK" ;;
    *) echo "跳过清理非预期路径：$WORK" >&2 ;;
  esac
}
trap cleanup EXIT

echo "[compat] 下载：$URL"
curl -fL --retry 3 --connect-timeout 20 --max-time 600 -o "$WORK/$DEB_NAME" "$URL"

echo "[compat] 解压到独立临时目录（不执行其中内容）"
mkdir "$WORK/extract"
dpkg-deb -x "$WORK/$DEB_NAME" "$WORK/extract"

COMPAT_LIB="$(find "$WORK/extract" -name 'libcuda.so.1' -print -quit)"
if [[ -z "$COMPAT_LIB" ]]; then
  echo "错误：解压后未找到 libcuda.so.1" >&2
  exit 1
fi
COMPAT_DIR="$(dirname "$COMPAT_LIB")"
if [[ ! -e "$COMPAT_DIR/libcuda.so.1" ]]; then
  echo "错误：libcuda.so.1 链接目标缺失：$COMPAT_DIR" >&2
  exit 1
fi

echo "[compat] 安装用户态库到：$TARGET"
mkdir -p "$TARGET"
cp -a "$COMPAT_DIR/." "$TARGET/"

echo "[compat] 完成。库文件："
ls -1 "$TARGET" | sed 's/^/  /'

cat <<EOF

下一步：
  1) 编辑 config.py，把 USE_CUDA_COMPAT 改为 True
  2) 重启服务：bash $ROOT/service.sh restart --wait

说明：这些是用户态 CUDA 兼容库，只影响本服务进程；无需 sudo。
EOF
