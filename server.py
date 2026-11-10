#!/usr/bin/env python3
"""配置驱动的 vLLM 启动器。

流程：读取模型配置与权重 -> 探测所有 GPU -> 每卡独立 vLLM -> 统一异步网关。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shlex
import shutil
import signal
import socket
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
import gpu_planner  # noqa: E402


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


def _extra_arg_root(arg: Any) -> str:
    """按 vLLM FlexibleArgumentParser 的规则得到受保护参数根名。

    vLLM 会把 ``--max_num_seqs`` 规范成 ``--max-num-seqs``，并把
    ``--hf-overrides.text_config.x`` 合并进 ``--hf-overrides``。
    """
    flag = str(arg).split("=", 1)[0]
    if flag.startswith("--"):
        flag = "--" + flag[2:].split(".", 1)[0].replace("_", "-")
    return flag


def _is_reserved_extra(flag: str, reserved: set[str]) -> bool:
    """同时拦截 argparse 的长参数缩写和短参数连写。"""
    if flag in reserved or flag.startswith(("--data-parallel", "--pipeline-parallel")):
        return True
    if flag.startswith("--"):
        return any(name.startswith(flag) for name in reserved if name.startswith("--"))
    return any(name.startswith("-") and not name.startswith("--") and flag.startswith(name)
               for name in reserved)


def validate_config() -> None:
    if config.GPU_IDS != "auto":
        if not isinstance(config.GPU_IDS, (list, tuple)) or not config.GPU_IDS:
            raise SystemExit("config.GPU_IDS 必须是 'auto' 或非空物理 GPU 编号列表")
        if not all(type(x) is int and x >= 0 for x in config.GPU_IDS):
            raise SystemExit("config.GPU_IDS 必须是非负整数列表")
        if len(set(config.GPU_IDS)) != len(config.GPU_IDS):
            raise SystemExit("config.GPU_IDS 有重复项")
    tensor_parallel_size()

    if not (0.05 <= float(config.GPU_MEMORY_UTILIZATION) <= 0.98):
        raise SystemExit(
            "config.GPU_MEMORY_UTILIZATION 应在 0.05~0.98 之间，"
            f"当前为 {config.GPU_MEMORY_UTILIZATION!r}"
        )
    if not (0 < int(config.PORT) < 65536):
        raise SystemExit(f"config.PORT 非法：{config.PORT!r}")

    if isinstance(config.MAX_MODEL_LEN, str) and config.MAX_MODEL_LEN != "auto":
        raise SystemExit(f"config.MAX_MODEL_LEN 必须是正整数或 'auto'，当前为 {config.MAX_MODEL_LEN!r}")

    for name in ("MAX_NUM_SEQS", "MIN_NUM_SEQS", "MAX_NUM_BATCHED_TOKENS", "PROXY_MAX_BODY_MIB"):
        value = getattr(config, name, None)
        if value is None and name == "MAX_NUM_BATCHED_TOKENS":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SystemExit(f"config.{name} 必须是正整数或 None，当前为 {value!r}")

    if config.MIN_NUM_SEQS > config.MAX_NUM_SEQS:
        raise SystemExit("MIN_NUM_SEQS 不能超过 MAX_NUM_SEQS")
    if not 0 < config.BACKEND_PORT_START < 65536:
        raise SystemExit("BACKEND_PORT_START 非法")
    for name in ("GPU_FREE_MEMORY_RESERVE_GIB", "MIN_KV_CACHE_GIB", "RUNTIME_MEMORY_GIB",
                 "RUNTIME_MEMORY_PER_SEQ_MIB", "WEIGHT_MEMORY_MARGIN", "REPLICA_STARTUP_TIMEOUT",
                 "REPLICA_HEALTH_INTERVAL", "REPLICA_HEALTH_TIMEOUT", "REPLICA_SHUTDOWN_TIMEOUT",
                 "REPLICA_CHILD_GRACE_SECONDS", "PROXY_CONNECT_TIMEOUT", "PROXY_READ_TIMEOUT"):
        value = getattr(config, name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise SystemExit(f"config.{name} 必须是有限非负数")
        if ("TIMEOUT" in name or "INTERVAL" in name) and value == 0:
            raise SystemExit(f"config.{name} 必须大于 0")
    if config.WEIGHT_MEMORY_MARGIN < 1:
        raise SystemExit("WEIGHT_MEMORY_MARGIN 不能小于 1")
    for name in ("RESOURCE_OVERRIDES", "GPU_ROUTING_WEIGHTS", "MODEL_PROFILES",
                 "EXTRA_ENV", "LIMIT_MM_PER_PROMPT"):
        if not isinstance(getattr(config, name), dict):
            raise SystemExit(f"config.{name} 必须是 dict")

    reserved = {"--host", "--port", "--config", "--model", "--served-model-name", "--api-key",
                "--tensor-parallel-size", "-tp", "-pp", "-dp", "-dpl", "-dpr", "-dpn",
                "--max-model-len", "--max-num-seqs", "--max-num-batched-tokens",
                "--gpu-memory-utilization", "--kv-cache-memory-bytes", "--kv-cache-dtype",
                "--dtype", "--quantization", "-q", "--hf-overrides", "--speculative-config",
                "--num-gpu-blocks-override", "--cpu-offload-gb", "--distributed-executor-backend",
                "--data-parallel-size", "--pipeline-parallel-size"}
    for arg in config.EXTRA_VLLM_ARGS:
        flag = _extra_arg_root(arg)
        if _is_reserved_extra(flag, reserved):
            raise SystemExit(f"EXTRA_VLLM_ARGS 不得覆盖自动规划参数 {flag}；请使用 config.py 对应配置")
    for key in config.EXTRA_ENV:
        if key in {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"} or key.startswith("VLLM_DP_"):
            raise SystemExit(f"EXTRA_ENV 不得覆盖副本设备/拓扑：{key}")


def tensor_parallel_size() -> int:
    value = config.TENSOR_PARALLEL_SIZE
    if value not in (None, "", "auto", 1):
        raise SystemExit("独立副本模式要求 TENSOR_PARALLEL_SIZE=1；每卡加载完整模型")
    return 1


def validate_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if os.name == "posix":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        elif hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
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


def build_env(*, replica: gpu_planner.GPUDecision | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if replica is not None:
        env["CUDA_VISIBLE_DEVICES"] = replica.gpu.uuid
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
    if replica is not None:
        # 同机多副本并行首次启动时避免共享编译缓存产生竞争；代价是每卡独立缓存。
        env["VLLM_CACHE_ROOT"] = str(Path(config.VLLM_CACHE_ROOT) / f"gpu-{replica.gpu.index}")
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


def build_argv(vllm_bin: str, model_path: Path, model_cfg: dict[str, Any], *,
               replica: gpu_planner.GPUDecision | None = None, port: int | None = None,
               resolved_dtype: str | None = None) -> list[str]:
    model_cfg = gpu_planner.effective_model_config(model_cfg)
    tp = tensor_parallel_size()
    max_len = replica.max_model_len if replica else model_utils.clamp_max_model_len(model_cfg)
    utilization = replica.memory_utilization if replica else config.GPU_MEMORY_UTILIZATION
    seqs = replica.max_num_seqs if replica else config.MAX_NUM_SEQS
    dtype = resolved_dtype or config.DTYPE
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
        "127.0.0.1" if replica else config.HOST,
        "--port",
        str(port if port is not None else config.PORT),
        "--served-model-name",
        config.SERVED_MODEL_NAME,
        "--tensor-parallel-size",
        str(tp),
        "--gpu-memory-utilization",
        str(utilization),
        "--dtype",
        dtype,
        "--uvicorn-log-level",
        config.UVICORN_LOG_LEVEL,
    ]

    if max_len:
        argv += ["--max-model-len", str(max_len)]
    if seqs is not None:
        argv += ["--max-num-seqs", str(seqs)]
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
    log(f"GPU：{replica.gpu.uuid if replica else config.GPU_IDS}，TP={tp}，seqs={seqs}")
    log(
        f"max_model_len={max_len}（模型原生：{model_utils.native_max_model_len(model_cfg)}），"
        f"gpu_memory_utilization={utilization}，dtype={dtype}"
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

    validate_config()
    if not args.dry_run:
        validate_port()

    log(f"服务地址：http://{config.HOST}:{config.PORT}")
    log(f"模型加载点：{model_utils.effective_model_url()}")
    model_path = model_utils.resolve_model_path()
    model_cfg = model_utils.read_model_config(model_path)
    try:
        child_env = build_env()
        resources = gpu_planner.estimate_resources(model_path, model_cfg, env=child_env)
        decisions = gpu_planner.plan_replicas(gpu_planner.discover_gpus(), resources,
                                             visibility_env=child_env)
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"GPU/模型准入失败：{exc}") from exc
    log(f"模型估算：weights={resources.weights_bytes / 1024**3:.2f} GiB ({resources.weight_source})，"
        f"dtype={resources.dtype}，KV={resources.kv_bytes_per_token} bytes/token，"
        f"state={resources.state_bytes_per_seq / 1024**2:.1f} MiB/seq")
    eligible = [d for d in decisions if d.eligible]
    for d in decisions:
        log(f"GPU {d.gpu.index} {d.gpu.name} SM{d.gpu.compute_capability} "
            f"free/total={d.gpu.free_bytes / 1024**3:.1f}/{d.gpu.total_bytes / 1024**3:.1f} GiB："
            f"{'使用' if d.eligible else '跳过'}，seqs={d.max_num_seqs}，"
            f"utilization={d.memory_utilization}，{d.reason}")
    if not eligible:
        raise SystemExit("没有符合当前模型精度/上下文要求的 GPU；请查看逐卡原因")
    ports = range(config.BACKEND_PORT_START, config.BACKEND_PORT_START + len(eligible))
    if ports[-1] > 65535 or config.PORT in ports:
        raise SystemExit("副本端口超界或与 PORT 冲突，请调整 BACKEND_PORT_START")
    vllm_bin = "vllm" if args.dry_run else find_vllm_bin()
    if args.dry_run:
        for d, port in zip(eligible, ports):
            argv = build_argv(vllm_bin, model_path, model_cfg, replica=d, port=port,
                              resolved_dtype=resources.dtype)
            print(f"CUDA_VISIBLE_DEVICES={d.gpu.uuid} " + shlex.join(redact_argv(argv)))
        return
    raise SystemExit(asyncio.run(serve(decisions, vllm_bin, model_path, model_cfg, resources.dtype)))


async def serve(decisions, vllm_bin, model_path, model_cfg, resolved_dtype: str) -> int:
    from aiohttp import web
    from gateway import create_app
    from replica_manager import ReplicaManager

    manager = ReplicaManager(decisions,
        lambda d, port: build_argv(vllm_bin, model_path, model_cfg, replica=d, port=port,
                                   resolved_dtype=resolved_dtype),
        lambda d: build_env(replica=d), log_dir=ROOT / "logs", port_start=config.BACKEND_PORT_START,
        startup_timeout=config.REPLICA_STARTUP_TIMEOUT, health_interval=config.REPLICA_HEALTH_INTERVAL,
        health_timeout=config.REPLICA_HEALTH_TIMEOUT, shutdown_timeout=config.REPLICA_SHUTDOWN_TIMEOUT,
        api_key=config.API_KEY)
    app = create_app(manager.pool, api_key=config.API_KEY, status_provider=manager.status,
                     connect_timeout=config.PROXY_CONNECT_TIMEOUT, read_timeout=config.PROXY_READ_TIMEOUT,
                     max_body_mib=config.PROXY_MAX_BODY_MIB)
    runner = web.AppRunner(app, handler_cancellation=True, shutdown_timeout=config.REPLICA_SHUTDOWN_TIMEOUT)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))
    waiters = []
    try:
        await runner.setup()
        await web.TCPSite(runner, config.HOST, config.PORT).start()
        await manager.start()
        waiters = [asyncio.create_task(stop.wait()), asyncio.create_task(manager.all_failed.wait())]
        await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        return 1 if manager.all_failed.is_set() else 0
    finally:
        manager.pool.accepting = False
        for task in waiters:
            task.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
        try:
            await runner.cleanup()
        finally:
            await manager.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
