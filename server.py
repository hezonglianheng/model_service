#!/usr/bin/env python3
"""配置驱动的 vLLM 启动器。

流程：读取 config.py -> 设置 HF/CUDA 环境 -> 解析/下载模型 -> 组装
`vllm serve` 参数 -> os.execvpe 交给 vLLM（PID、信号、日志直通）。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def env_path(value: Any) -> str:
    """配置里的路径转绝对路径；相对路径以项目根目录为基准（支持任意 CWD）。"""
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


# ---- 这些变量必须在 import huggingface_hub 之前生效 ----
if config.HF_ENDPOINT:
    os.environ["HF_ENDPOINT"] = config.HF_ENDPOINT
else:
    os.environ.pop("HF_ENDPOINT", None)
os.environ["HF_HOME"] = env_path(config.HF_HOME)
os.environ["HF_HUB_CACHE"] = env_path(config.MODEL_CACHE_DIR)
os.environ["HF_HUB_DISABLE_XET"] = "1"  # 避免 Xet CAS 绕过镜像
if not config.ONLINE_LOAD:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
else:
    # config 是唯一事实来源：避免外部 shell 里的离线变量阻止联网加载
    for _key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(_key, None)

import model_utils  # noqa: E402


def log(msg: str) -> None:
    print(f"[server] {msg}", file=sys.stderr, flush=True)


def find_vllm_bin() -> str:
    sibling = Path(sys.executable).with_name("vllm")
    if sibling.is_file():
        return str(sibling)
    found = shutil.which("vllm")
    if found:
        return found
    raise SystemExit(
        "未找到 vllm 可执行文件。请先运行：\n"
        f"  bash {ROOT / 'setup.sh'}\n"
        f"或使用启动脚本：bash {ROOT / 'start.sh'}"
    )


def validate_config() -> None:
    if not isinstance(config.GPU_IDS, (list, tuple)) or not config.GPU_IDS:
        raise SystemExit("config.GPU_IDS 必须是非空列表，例如 [1]")
    if not all(isinstance(x, int) and x >= 0 for x in config.GPU_IDS):
        raise SystemExit(f"config.GPU_IDS 必须是非负整数列表，当前为 {config.GPU_IDS!r}")
    if len(set(config.GPU_IDS)) != len(config.GPU_IDS):
        raise SystemExit(f"config.GPU_IDS 有重复项：{config.GPU_IDS!r}")

    tp = tensor_parallel_size()
    if tp < 1:
        raise SystemExit(f"tensor parallel size 非法：{tp}")

    if not (0.05 <= float(config.GPU_MEMORY_UTILIZATION) <= 0.98):
        raise SystemExit(
            "config.GPU_MEMORY_UTILIZATION 应在 0.05~0.98 之间，"
            f"当前为 {config.GPU_MEMORY_UTILIZATION!r}"
        )
    if not (0 < int(config.PORT) < 65536):
        raise SystemExit(f"config.PORT 非法：{config.PORT!r}")

    if isinstance(config.MAX_MODEL_LEN, str) and config.MAX_MODEL_LEN != "auto":
        raise SystemExit(f"config.MAX_MODEL_LEN 必须是正整数或 'auto'，当前为 {config.MAX_MODEL_LEN!r}")

    if str(config.KV_CACHE_DTYPE).startswith("fp8"):
        caps = query_compute_caps()
        selected = [caps.get(i) for i in config.GPU_IDS if i in caps]
        if selected and any(cap is not None and cap < 8.9 for cap in selected):
            log(
                "警告：A100（SM80）不支持 vLLM 的 fp8 KV cache，"
                "请把 config.KV_CACHE_DTYPE 改回 'auto'/'bfloat16'"
            )


def tensor_parallel_size() -> int:
    value = config.TENSOR_PARALLEL_SIZE
    if value in (None, "", "auto"):
        return len(config.GPU_IDS)
    tp = int(value)
    if tp != len(config.GPU_IDS):
        raise SystemExit(
            f"config.TENSOR_PARALLEL_SIZE={tp} 与 GPU_IDS 数量 "
            f"{len(config.GPU_IDS)} 不一致；多卡请把卡号都写进 GPU_IDS"
        )
    return tp


def _nvidia_smi(query: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return None
        return out.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def query_gpu_indices() -> set[int] | None:
    out = _nvidia_smi("index")
    if out is None:
        return None
    try:
        return {int(line.strip().split(",")[0]) for line in out.splitlines() if line.strip()}
    except ValueError:
        return None


def query_compute_caps() -> dict[int, float]:
    out = _nvidia_smi("index,compute_cap")
    if out is None:
        return {}
    caps: dict[int, float] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                caps[int(parts[0])] = float(parts[1])
            except ValueError:
                continue
    return caps


def validate_gpu_ids() -> None:
    available = query_gpu_indices()
    if available is None:
        log("警告：无法执行 nvidia-smi，跳过 GPU 编号校验")
        return
    missing = [i for i in config.GPU_IDS if i not in available]
    if missing:
        raise SystemExit(
            f"config.GPU_IDS={config.GPU_IDS} 中有本机不存在的卡：{missing}；"
            f"nvidia-smi 可见编号为 {sorted(available)}"
        )


def validate_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((config.HOST, int(config.PORT)))
        except OSError as exc:
            raise SystemExit(
                f"无法监听 {config.HOST}:{config.PORT}（{exc}）。"
                "端口可能被占用；可先 `bash service.sh stop` 或修改 config.PORT。"
            ) from exc


def find_compat_dir(root: Path) -> Path:
    if not root.is_dir():
        raise SystemExit(
            f"config.USE_CUDA_COMPAT=True，但目录不存在：{root}\n"
            f"请先运行：bash {ROOT / 'scripts' / 'fetch_cuda_compat.sh'}"
        )
    for candidate in root.rglob("libcuda.so.1"):
        return candidate.parent
    raise SystemExit(
        f"在 {root} 中未找到 libcuda.so.1。"
        f"请重新运行：bash {ROOT / 'scripts' / 'fetch_cuda_compat.sh'}"
    )


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in config.GPU_IDS)
    if config.HF_ENDPOINT:
        env["HF_ENDPOINT"] = config.HF_ENDPOINT
    else:
        env.pop("HF_ENDPOINT", None)
    env["HF_HOME"] = env_path(config.HF_HOME)
    env["HF_HUB_CACHE"] = env_path(config.MODEL_CACHE_DIR)
    env["HF_HUB_DISABLE_XET"] = "1"
    if config.HF_TOKEN:
        env["HF_TOKEN"] = str(config.HF_TOKEN)
        env["HUGGING_FACE_HUB_TOKEN"] = str(config.HF_TOKEN)
    env["VLLM_CACHE_ROOT"] = str(config.VLLM_CACHE_ROOT)
    env["VLLM_NO_USAGE_STATS"] = "1"
    env["DO_NOT_TRACK"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    # FlashInfer 运行时会调用 ninja（venv 里已安装）；不激活 venv 时也要能找到
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.setdefault("NO_PROXY", "127.0.0.1,localhost")

    if not config.ONLINE_LOAD:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
    else:
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            env.pop(key, None)

    if config.USE_RUST_FRONTEND:
        env["VLLM_USE_RUST_FRONTEND"] = "1"
    if config.ALLOW_EXCEED_NATIVE_CONTEXT:
        env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    if config.USE_CUDA_COMPAT:
        compat = find_compat_dir(Path(config.CUDA_COMPAT_DIR))
        env["VLLM_ENABLE_CUDA_COMPATIBILITY"] = "1"
        env["VLLM_CUDA_COMPATIBILITY_PATH"] = str(compat)
        env["LD_LIBRARY_PATH"] = f"{compat}:{env.get('LD_LIBRARY_PATH', '')}"
        log(f"CUDA 前向兼容库：{compat}")

    env.update({str(k): str(v) for k, v in config.EXTRA_ENV.items()})
    return env


def resolve_auto_flag(name: str, model_cfg: dict[str, Any], profile: dict[str, Any]) -> str | None:
    explicit = getattr(config, name)
    if explicit != "auto":
        return explicit or None
    if not config.AUTO_MODEL_PROFILE:
        return None
    return profile.get(name.lower()) or None


def build_argv(vllm_bin: str, model_path: Path, model_cfg: dict[str, Any]) -> list[str]:
    tp = tensor_parallel_size()
    max_len = model_utils.clamp_max_model_len(model_cfg)
    profile = model_utils.infer_model_profile(model_cfg)
    has_vision = model_utils.is_multimodal(model_cfg)

    tool_parser = resolve_auto_flag("TOOL_CALL_PARSER", model_cfg, profile)
    reasoning_parser = resolve_auto_flag("REASONING_PARSER", model_cfg, profile)
    mm_mode = resolve_auto_flag("MM_ENCODER_TP_MODE", model_cfg, profile)
    if not has_vision:
        mm_mode = None

    argv = [
        vllm_bin,
        "serve",
        str(model_path),
        "--host",
        config.HOST,
        "--port",
        str(config.PORT),
        "--served-model-name",
        config.SERVED_MODEL_NAME,
        "--tensor-parallel-size",
        str(tp),
        "--gpu-memory-utilization",
        str(config.GPU_MEMORY_UTILIZATION),
        "--dtype",
        config.DTYPE,
        "--uvicorn-log-level",
        config.UVICORN_LOG_LEVEL,
    ]

    if max_len:
        argv += ["--max-model-len", str(max_len)]
    if config.MAX_NUM_SEQS is not None:
        argv += ["--max-num-seqs", str(config.MAX_NUM_SEQS)]
    if config.MAX_NUM_BATCHED_TOKENS is not None:
        argv += ["--max-num-batched-tokens", str(config.MAX_NUM_BATCHED_TOKENS)]
    if config.KV_CACHE_DTYPE != "auto":
        argv += ["--kv-cache-dtype", config.KV_CACHE_DTYPE]
    if config.TRUST_REMOTE_CODE:
        argv += ["--trust-remote-code"]
    if config.QUANTIZATION:
        argv += ["--quantization", config.QUANTIZATION]
    if config.ENABLE_PREFIX_CACHING:
        argv += ["--enable-prefix-caching"]
    else:
        argv += ["--no-enable-prefix-caching"]
    if config.ENFORCE_EAGER:
        argv += ["--enforce-eager"]
    if config.API_KEY:
        argv += ["--api-key", config.API_KEY]

    if tool_parser:
        if config.ENABLE_AUTO_TOOL_CHOICE:
            argv += ["--enable-auto-tool-choice"]
        argv += ["--tool-call-parser", tool_parser]
    if reasoning_parser:
        argv += ["--reasoning-parser", reasoning_parser]

    if has_vision:
        if config.LANGUAGE_MODEL_ONLY:
            argv += ["--language-model-only"]
        if mm_mode:
            argv += ["--mm-encoder-tp-mode", mm_mode]
        if config.LIMIT_MM_PER_PROMPT:
            argv += ["--limit-mm-per-prompt", json.dumps(config.LIMIT_MM_PER_PROMPT)]

    if config.DEFAULT_CHAT_TEMPLATE_KWARGS:
        argv += ["--default-chat-template-kwargs", json.dumps(config.DEFAULT_CHAT_TEMPLATE_KWARGS)]
    if config.HF_OVERRIDES:
        argv += ["--hf-overrides", json.dumps(config.HF_OVERRIDES)]
    if config.SPECULATIVE_CONFIG:
        argv += ["--speculative-config", json.dumps(config.SPECULATIVE_CONFIG)]
    if config.OVERRIDE_GENERATION_CONFIG:
        argv += ["--override-generation-config", json.dumps(config.OVERRIDE_GENERATION_CONFIG)]
    if getattr(config, "MODEL_IMPL", "auto") != "auto":
        argv += ["--model-impl", config.MODEL_IMPL]

    argv += [str(x) for x in config.EXTRA_VLLM_ARGS]

    # 启动日志
    log(f"模型：{config.MODEL_ID}（实际加载：{model_path}）")
    log(f"加载地址：{model_utils.effective_model_url()}（ONLINE_LOAD={config.ONLINE_LOAD}）")
    log(f"GPU：CUDA_VISIBLE_DEVICES={','.join(map(str, config.GPU_IDS))}，TP={tp}")
    log(
        f"max_model_len={max_len}（模型原生：{model_utils.native_max_model_len(model_cfg)}），"
        f"gpu_memory_utilization={config.GPU_MEMORY_UTILIZATION}"
    )
    kv_bytes = model_utils.estimate_kv_bytes_per_token(model_cfg)
    if kv_bytes and max_len:
        kv_gib = kv_bytes * max_len / (1024 ** 3)
        log(f"估算 KV（BF16）≈ {kv_bytes / 1024:.0f} KiB/token，{max_len} tokens ≈ {kv_gib:.1f} GiB")
    if has_vision:
        log(
            f"多模态：已启用，language_model_only={config.LANGUAGE_MODEL_ONLY}，"
            f"mm_encoder_tp_mode={mm_mode}"
        )
    log(f"parser：tool={tool_parser}，reasoning={reasoning_parser}")

    return argv


def redact_env(env: dict[str, str]) -> dict[str, str]:
    masked = dict(env)
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "API_KEY"):
        if masked.get(key):
            masked[key] = "***"
    return masked


def redact_argv(argv: list[str]) -> list[str]:
    """打印命令时隐藏 --api-key 的值。"""
    masked = list(argv)
    for i, arg in enumerate(masked[:-1]):
        if arg == "--api-key":
            masked[i + 1] = "***"
    return masked


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 Qwen3.8-27B vLLM 服务")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印环境变量和最终命令，不启动服务",
    )
    args = parser.parse_args()

    vllm_bin = find_vllm_bin()
    validate_config()
    validate_gpu_ids()
    validate_port()

    log(f"服务地址：http://{config.HOST}:{config.PORT}")
    log(f"模型加载点：{model_utils.effective_model_url()}")
    model_path = model_utils.resolve_model_path()
    model_cfg = model_utils.read_model_config(model_path)
    argv = build_argv(vllm_bin, model_path, model_cfg)
    env = build_env()

    if args.dry_run:
        print("\n=== 环境变量（敏感项已打码）===")
        for key in sorted(redact_env(env)):
            if key in {
                "CUDA_VISIBLE_DEVICES",
                "HF_ENDPOINT",
                "HF_HOME",
                "HF_HUB_CACHE",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "VLLM_USE_RUST_FRONTEND",
                "VLLM_ENABLE_CUDA_COMPATIBILITY",
                "VLLM_CUDA_COMPATIBILITY_PATH",
                "LD_LIBRARY_PATH",
            }:
                print(f"{key}={redact_env(env)[key]}")
        print("\n=== vllm 命令 ===")
        print(shlex.join(redact_argv(argv)))
        return

    log("启动 vLLM：")
    print(f"  {shlex.join(redact_argv(argv))}", file=sys.stderr, flush=True)
    os.execvpe(vllm_bin, argv, env)


if __name__ == "__main__":
    main()
