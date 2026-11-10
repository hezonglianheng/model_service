"""按实际模型与 GPU 状态规划独立副本；不导入 torch、不加载权重张量。"""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config
import model_utils

GIB = 1024 ** 3
MIB = 1024 ** 2

# 与 vLLM 0.32.0 config/model.py 的 _FLOAT16_NOT_SUPPORTED_MODELS 保持一致。
# 这些模型即使 checkpoint 是 FP32，也不能显式使用 --dtype float16。
_FLOAT16_UNSUPPORTED_MODEL_TYPES = frozenset({
    "gemma2", "gemma3", "gemma3_text", "glm4",
})


@dataclass(frozen=True)
class GPUInfo:
    index: int
    uuid: str
    name: str
    total_bytes: int
    free_bytes: int
    compute_capability: float
    mig_enabled: bool = False


@dataclass(frozen=True)
class ModelResources:
    weights_bytes: int
    kv_bytes_per_token: int
    state_bytes_per_seq: int
    runtime_bytes: int
    runtime_bytes_per_seq: int
    max_model_len: int
    dtype: str
    min_compute_capability: float
    weight_source: str

    def required_bytes(self, seqs: int) -> int:
        kv = max(self.kv_bytes_per_token * self.max_model_len, config.MIN_KV_CACHE_GIB * GIB)
        return math.ceil(self.weights_bytes * config.WEIGHT_MEMORY_MARGIN + kv + self.runtime_bytes
                         + seqs * (self.state_bytes_per_seq + self.runtime_bytes_per_seq))


@dataclass(frozen=True)
class GPUDecision:
    gpu: GPUInfo
    max_model_len: int
    max_num_seqs: int | None
    memory_utilization: float
    reason: str
    routing_weight: float = 1.0

    @property
    def eligible(self) -> bool:
        return self.max_num_seqs is not None


def _nvidia_smi(query: str):
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={query}",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"无法探测 GPU：{exc}") from exc


def discover_gpus() -> list[GPUInfo]:
    fields = "index,uuid,name,memory.total,memory.free,compute_cap"
    result = _nvidia_smi(fields + ",mig.mode.current")
    has_mig = result.returncode == 0
    if not has_mig:
        # 旧驱动可能不支持 mig.mode.current；不要因此失去所有 GPU。
        result = _nvidia_smi(fields)
    if result.returncode:
        raise RuntimeError(f"nvidia-smi 查询失败：{result.stderr.strip()}")
    gpus = []
    try:
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            if not row:
                continue
            values = [s.strip() for s in row]
            if has_mig:
                index, uuid, name, total, free, cap, mig = values
            else:
                index, uuid, name, total, free, cap = values
                mig = "disabled"
            try:
                capability = float(cap)
            except ValueError:
                # 个别虚拟/受限设备不报告 compute capability：保留该卡并让准入跳过，
                # 而不是因为一张卡不可识别就拒绝整个服务。
                capability = 0.0
            gpu = GPUInfo(int(index), uuid, name, int(float(total.split()[0]) * MIB),
                          int(float(free.split()[0]) * MIB), capability,
                          mig.lower() == "enabled" or uuid.startswith("MIG-"))
            if (not uuid.startswith(("GPU-", "MIG-")) or gpu.total_bytes <= 0
                    or not 0 <= gpu.free_bytes <= gpu.total_bytes
                    or not math.isfinite(gpu.compute_capability)):
                raise ValueError("GPU 字段非法")
            gpus.append(gpu)
    except (ValueError, IndexError, OverflowError) as exc:
        raise RuntimeError(f"无法解析 nvidia-smi 输出：{result.stdout!r}") from exc
    if not gpus:
        raise RuntimeError("nvidia-smi 未发现 GPU")
    return gpus


def effective_model_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(model_cfg)

    def merge(dst, src):
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                merge(dst[key], value)
            else:
                dst[key] = copy.deepcopy(value)

    if config.HF_OVERRIDES:
        if not isinstance(config.HF_OVERRIDES, dict):
            raise ValueError("HF_OVERRIDES 必须是 dict")
        merge(result, config.HF_OVERRIDES)
    return result


def _positive(value, name, *, zero=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字")
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} 数值非法：{value}")
    return value


def _dtype(name: str) -> str:
    return {"half": "float16", "fp16": "float16", "bf16": "bfloat16",
            "float": "float32", "fp32": "float32"}.get(str(name), str(name))


def _weight_bytes(path: Path, dtype: str, source_dtype: str, quantized: bool) -> tuple[int, str]:
    """优先读取 safetensors 小文件头，精确处理非量化浮点张量的 dtype 转换。"""
    index = path / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text(encoding="utf-8"))
        files = [path / name for name in sorted(set(data.get("weight_map", {}).values()))]
    else:
        files = sorted(p for p in path.glob("*.safetensors")
                       if not p.name.startswith(model_utils._NON_MODEL_WEIGHT_PREFIXES))
    target_bytes = 4 if dtype == "float32" else 2
    if files:
        total = 0
        for file in files:
            with file.open("rb") as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    raise ValueError(f"无效 safetensors 文件头：{file}")
                length = struct.unpack("<Q", prefix)[0]
                if not 0 < length <= 64 * MIB:
                    raise ValueError(f"无效 safetensors 文件头长度：{file}")
                header = json.loads(stream.read(length))
            for name, tensor in header.items():
                if name == "__metadata__":
                    continue
                start, end = tensor["data_offsets"]
                stored = int(end) - int(start)
                if stored < 0:
                    raise ValueError(f"无效 tensor offsets：{file}:{name}")
                if not quantized and tensor["dtype"] in {"BF16", "F16", "F32", "F64"}:
                    total += math.prod(tensor["shape"]) * target_bytes
                else:
                    total += stored
        return int(_positive(total, "weights_bytes")), "safetensors headers"
    # PyTorch checkpoint 不反序列化（pickle）；采用 index 的体积或权重文件体积上界。
    index = path / "pytorch_model.bin.index.json"
    if index.is_file():
        data = json.loads(index.read_text(encoding="utf-8"))
        total = data.get("metadata", {}).get("total_size")
        if not total:
            total = sum((path / name).stat().st_size for name in set(data["weight_map"].values()))
    else:
        total = sum(p.stat().st_size for p in path.glob("*.bin")
                    if not p.name.startswith(model_utils._NON_MODEL_WEIGHT_PREFIXES))
    _positive(total, "weights_bytes；无法读取权重时请配置 RESOURCE_OVERRIDES.weights_gib")
    # 非 safetensors 路径不因 dtype 下转换减少估计，以免混合精度被低估。
    scale = 1 if quantized else max(1, target_bytes / (4 if source_dtype == "float32" else 2))
    return math.ceil(total * scale), "checkpoint size (conservative)"


def estimate_resources(path: Path, model_cfg: dict[str, Any], *,
                       env: dict[str, str] | None = None) -> ModelResources:
    cfg = effective_model_config(model_cfg)
    env = dict(os.environ) if env is None else env
    batch_invariant = str(env.get("VLLM_BATCH_INVARIANT", "")).strip().lower() in {
        "1", "true", "yes", "on"}
    text = next((cfg[key] for key in ("text_config", "language_config", "llm_config")
                 if isinstance(cfg.get(key), dict)), cfg)
    overrides = dict(model_utils.infer_model_profile(cfg).get("resources", {}))
    overrides.update(config.RESOURCE_OVERRIDES)
    allowed = {"weights_gib", "kv_bytes_per_token", "state_bytes_per_seq", "runtime_memory_gib",
               "runtime_memory_per_seq_mib", "min_compute_capability"}
    if set(overrides) - allowed:
        raise ValueError(f"未知 RESOURCE_OVERRIDES：{set(overrides) - allowed}")
    for name, value in overrides.items():
        _positive(value, name, zero=name in {"state_bytes_per_seq", "kv_bytes_per_token",
                                           "runtime_memory_per_seq_mib", "runtime_memory_gib"})
    source_dtype = _dtype(text.get("dtype", text.get("torch_dtype",
                           cfg.get("dtype", cfg.get("torch_dtype", "float32")))))
    dtype = _dtype(config.DTYPE)
    if dtype == "auto":
        if source_dtype == "float32":
            model_type = str(cfg.get("model_type", ""))
            dtype = ("bfloat16" if model_type in _FLOAT16_UNSUPPORTED_MODEL_TYPES
                     else "float16")
        else:
            dtype = source_dtype
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"无法估算 dtype={dtype}；请显式设置 DTYPE")
    item_bytes = 4 if dtype == "float32" else 2
    qcfg = cfg.get("quantization_config", text.get("quantization_config")) or {}
    quant = config.QUANTIZATION or qcfg.get("quant_method")
    min_cap = 8.0 if dtype == "bfloat16" else 7.5
    if quant:
        # vLLM 0.32 的预量化 per-tensor FP8 可在缺少原生 FP8 运算的卡上走
        # weight-only Marlin（Fp8Config 最低 SM75）。block FP8 与 batch-invariant
        # 路径尚未在本项目验证，继续保守要求 SM89，不据名称放宽。
        prequantized_per_tensor_fp8 = (
            quant == "fp8"
            and qcfg.get("quant_method") == "fp8"
            and qcfg.get("weight_block_size") is None
            and not batch_invariant
        )
        cap = 7.5 if prequantized_per_tensor_fp8 else {
            "awq": 7.5, "gptq": 7.5, "gptq_marlin": 8.0, "awq_marlin": 8.0,
            "fp8": 8.9, "fbgemm_fp8": 8.9, "bitsandbytes": 7.5}.get(quant)
        if cap is None and "min_compute_capability" not in overrides:
            raise ValueError(f"量化 {quant} 需 RESOURCE_OVERRIDES.min_compute_capability")
        min_cap = max(min_cap, cap or 0)
        if not qcfg and "weights_gib" not in overrides:
            raise ValueError("在线量化需 RESOURCE_OVERRIDES.weights_gib，不能用原始权重猜加载体积")
    min_cap = max(min_cap, overrides.get("min_compute_capability", 0))
    kv_dtype = _dtype(config.KV_CACHE_DTYPE)
    if kv_dtype.startswith("fp8"):
        kv_item_bytes = 1
        min_cap = max(min_cap, 8.9)
    elif kv_dtype in {"auto", "float16", "bfloat16", "float32"}:
        kv_item_bytes = item_bytes if kv_dtype == "auto" else (4 if kv_dtype == "float32" else 2)
    else:
        raise ValueError(f"无法估算 KV_CACHE_DTYPE={kv_dtype}")
    max_len = model_utils.clamp_max_model_len(cfg)
    if not max_len:
        raise ValueError("模型无原生上下文长度，请显式设置 MAX_MODEL_LEN")

    layers = text.get("layer_types")
    if layers:
        full = sum(x in {"full_attention", "sliding_attention"} for x in layers)
        linear = sum(x == "linear_attention" for x in layers)
        unknown_layers = len(layers) != full + linear
    else:
        full = text.get("num_hidden_layers")
        linear = 0
        # hybrid 未给出 layer_types，不能把它当普通 Transformer。
        unknown_layers = bool(text.get("full_attention_interval") or "mamba" in str(cfg.get("model_type")))
    if "kv_bytes_per_token" in overrides:
        kv = int(overrides["kv_bytes_per_token"])
    else:
        heads = text.get("num_key_value_heads") or text.get("num_attention_heads")
        dim = text.get("head_dim")
        if not dim and text.get("num_attention_heads") and text.get("hidden_size"):
            dim = text["hidden_size"] // text["num_attention_heads"]
        if (text.get("kv_lora_rank") or unknown_layers
                or not all(isinstance(x, int) and x > 0 for x in (full, heads, dim))):
            raise ValueError("无法估算此 attention 的 KV，请配置 RESOURCE_OVERRIDES.kv_bytes_per_token")
        kv = full * 2 * heads * dim * kv_item_bytes
    if "state_bytes_per_seq" in overrides:
        state = int(overrides["state_bytes_per_seq"])
    elif linear and str(cfg.get("model_type", "")).startswith("qwen3_5"):
        names = ("linear_num_value_heads", "linear_num_key_heads", "linear_key_head_dim",
                 "linear_value_head_dim", "linear_conv_kernel_dim")
        values = [text.get(name) for name in names]
        if not all(isinstance(x, int) and x > 0 for x in values):
            raise ValueError("缺少 linear 状态参数，请配置 RESOURCE_OVERRIDES.state_bytes_per_seq")
        vh, kh, kd, vd, kernel = values
        # recurrent SSM 使用 FP32；conv 状态按 activation dtype，预留整个 kernel。
        state = linear * (vh * kd * vd * 4 + (2 * kh * kd + vh * vd) * kernel * item_bytes)
    elif linear or unknown_layers:
        raise ValueError("未知混合状态，请配置 RESOURCE_OVERRIDES.state_bytes_per_seq")
    else:
        state = 0
    if "weights_gib" in overrides:
        weights, source = math.ceil(overrides["weights_gib"] * GIB), "RESOURCE_OVERRIDES.weights_gib"
    else:
        weights, source = _weight_bytes(path, dtype, source_dtype, bool(quant))
    if config.SPECULATIVE_CONFIG and "weights_gib" not in overrides:
        raise ValueError("投机解码额外资源需显式配置 RESOURCE_OVERRIDES.weights_gib 和运行时预留")
    return ModelResources(weights, kv, state,
        int(overrides.get("runtime_memory_gib", config.RUNTIME_MEMORY_GIB) * GIB),
        int(overrides.get("runtime_memory_per_seq_mib", config.RUNTIME_MEMORY_PER_SEQ_MIB) * MIB),
        max_len, dtype, min_cap, source)


def sequence_candidates(maximum: int) -> list[int]:
    result = [maximum]
    while result[-1] > config.MIN_NUM_SEQS:
        result.append(max(config.MIN_NUM_SEQS, result[-1] // 2))
    return result


def visible_gpu_uuids(gpus: list[GPUInfo], env: dict[str, str] | None = None) -> set[str] | None:
    """CUDA ordinal 不等于 nvidia-smi index；在原环境的隔离子进程中查询真实 UUID。"""
    env = dict(os.environ) if env is None else env
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return None
    if not visible or visible == "-1":
        return set()
    if visible.startswith(("GPU-", "MIG-")):
        # 对齐 PyTorch _parse_visible_devices / _transform_uuid_to_ordinals：
        # 重复 token 清空、首个非前缀项截断、无效/歧义前缀截断、重复卡清空。
        prefix = "GPU-" if visible.startswith("GPU-") else "MIG-"
        tokens = []
        seen_tokens = set()
        for token in visible.split(","):
            if token in seen_tokens:
                return set()
            if not token.startswith(prefix):
                break
            seen_tokens.add(token)
            tokens.append(token)
        result = set()
        seen_cards = set()
        for token in tokens:
            matches = [(index, gpu.uuid) for index, gpu in enumerate(gpus)
                       if gpu.uuid.startswith(token)]
            if len(matches) != 1:
                break
            index, uuid = matches[0]
            if index in seen_cards:
                return set()
            seen_cards.add(index)
            result.add(uuid.lower())
        return result
    tokens = [x.strip() for x in visible.split(",") if x.strip()]
    if not tokens or tokens == ["-1"]:
        return set()
    try:
        result = subprocess.run([sys.executable, "-c",
            "import json,torch; print(json.dumps([str(torch.cuda.get_device_properties(i).uuid) "
            "for i in range(torch.cuda.device_count())]))"],
            env=env, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise ValueError(result.stderr.strip())
        uuids = json.loads(result.stdout.strip().splitlines()[-1])
        if not isinstance(uuids, list) or not all(isinstance(u, str) and u for u in uuids):
            raise ValueError("CUDA 返回无效 UUID 列表")
        # torch 的部分版本返回无 GPU- 前缀的 UUID。
        return {(u if u.startswith(("GPU-", "MIG-")) else f"GPU-{u}").lower() for u in uuids}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise RuntimeError("无法解析 CUDA 可见设备 UUID；请将 CUDA_VISIBLE_DEVICES 改用 GPU UUID，"
                           "或清除此变量并用 config.GPU_IDS 指定物理编号。" + str(exc)) from exc


def plan_replicas(gpus: list[GPUInfo], resources: ModelResources, *,
                  visibility_env: dict[str, str] | None = None) -> list[GPUDecision]:
    visible = visible_gpu_uuids(gpus, visibility_env)
    if config.GPU_IDS != "auto":
        missing = set(config.GPU_IDS) - {gpu.index for gpu in gpus}
        if missing:
            raise ValueError(f"GPU_IDS 包含不存在的物理编号：{sorted(missing)}")
    decisions = []
    for gpu in gpus:
        reason = ""
        if config.GPU_IDS != "auto" and gpu.index not in config.GPU_IDS:
            reason = "不在 GPU_IDS 白名单"
        elif visible is not None and gpu.uuid.lower() not in visible:
            reason = "不在 CUDA_VISIBLE_DEVICES 中"
        elif gpu.mig_enabled:
            reason = "MIG 已启用；不把 MIG 父设备作为完整 GPU 使用"
        elif gpu.compute_capability < resources.min_compute_capability:
            reason = f"计算能力 {gpu.compute_capability} < 要求 {resources.min_compute_capability}"
        budget = max(0, min(gpu.total_bytes * config.GPU_MEMORY_UTILIZATION,
                           gpu.free_bytes - config.GPU_FREE_MEMORY_RESERVE_GIB * GIB))
        # 向下取整，确保传给 vLLM 的预算不高于实际准入预算。
        utilization = math.floor(budget / gpu.total_bytes * 10000) / 10000
        seqs = None
        if not reason:
            for candidate in sequence_candidates(config.MAX_NUM_SEQS):
                if resources.required_bytes(candidate) <= utilization * gpu.total_bytes:
                    seqs = candidate
                    break
            if seqs is None:
                reason = (f"显存不足：预算 {budget / GIB:.2f} GiB，最少需要 "
                          f"{resources.required_bytes(config.MIN_NUM_SEQS) / GIB:.2f} GiB")
        weight = config.GPU_ROUTING_WEIGHTS.get(gpu.index, 1.0)
        _positive(weight, f"GPU_ROUTING_WEIGHTS[{gpu.index}]")
        decisions.append(GPUDecision(gpu, resources.max_model_len, seqs, utilization,
                                     reason or "准入预检通过，等待实际加载", weight))
    return decisions
