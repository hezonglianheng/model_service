#!/usr/bin/env python3
"""预下载并校验模型快照（可选，首次启动会自动执行同样的下载）。

用法：
    .venv/bin/python download_model.py              # 按 config 联网下载/复用缓存
    .venv/bin/python download_model.py --check-only # 只检查本地缓存，不联网
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def env_path(value: str) -> str:
    """配置里的路径转绝对路径；相对路径以项目根目录为基准（支持任意 CWD）。"""
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


if config.HF_ENDPOINT:
    os.environ["HF_ENDPOINT"] = config.HF_ENDPOINT
else:
    os.environ.pop("HF_ENDPOINT", None)
os.environ["HF_HOME"] = env_path(config.HF_HOME)
os.environ["HF_HUB_CACHE"] = env_path(config.MODEL_CACHE_DIR)
os.environ["HF_HUB_DISABLE_XET"] = "1"

import model_utils  # noqa: E402


def human_size(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"


def snapshot_size(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    for file in path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
            count += 1
    return total, count


def main() -> int:
    parser = argparse.ArgumentParser(description="下载并校验模型权重")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查本地缓存/LOCAL_MODEL_PATH，不联网",
    )
    args = parser.parse_args()

    if args.check_only:
        config.ONLINE_LOAD = False
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("[download] 检查离线缓存（不联网）", flush=True)
    else:
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ.pop(key, None)

    try:
        path = model_utils.resolve_model_path()
    except Exception as exc:
        print(f"[download] 失败：{exc}", file=sys.stderr)
        return 1

    total, count = snapshot_size(path)
    print(f"[download] 模型目录：{path}")
    print(f"[download] 文件数：{count}，总大小：{human_size(total)}")
    print("[download] 校验通过：config.json 与权重分片齐全")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
