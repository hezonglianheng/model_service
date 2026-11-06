"""模型路径解析、config.json 读取、上下文长度钳制等工具。"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import config

GIB = 1024 ** 3
# 首次联网加载至少需要这么多空闲磁盘（模型约 55.6GB，留出余量）。
MIN_FREE_DISK_GIB = 70
# 无 index.json 时，识别 model-00001-of-00004.safetensors 这类分片名。
_SHARD_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.(?P<suffix>safetensors|bin)$"
)


def expand_path(p: str | os.PathLike[str]) -> Path:
    """展开 ~；相对路径以项目根目录为基准（保证可从任意 CWD 启动）。"""
    path = Path(os.path.expanduser(str(p)))
    if not path.is_absolute():
        path = config.ROOT / path
    return path.resolve()


def effective_model_url() -> str:
    """按当前 MODEL_ID / HF_ENDPOINT 计算，避免模块级缓存过期。"""
    return f"{config.HF_ENDPOINT.rstrip('/')}/{config.MODEL_ID}"


def read_model_config(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"模型目录缺少 config.json: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def native_max_model_len(model_cfg: dict[str, Any]) -> int | None:
    """从顶层或常见子配置中读取 max_position_embeddings。"""
    value = model_cfg.get("max_position_embeddings")
    if isinstance(value, int) and value > 0:
        return value
    for key in ("text_config", "language_config", "llm_config"):
        sub = model_cfg.get(key)
        if isinstance(sub, dict):
            value = sub.get("max_position_embeddings")
            if isinstance(value, int) and value > 0:
                return value
    return None


def clamp_max_model_len(model_cfg: dict[str, Any], log=print) -> int | None:
    """把用户配置的 MAX_MODEL_LEN 钳制到模型原生上限，兼容小上下文模型。"""
    native = native_max_model_len(model_cfg)
    requested = config.MAX_MODEL_LEN

    if requested in (None, "", "auto"):
        return native
    if not isinstance(requested, int) or requested <= 0:
        raise ValueError(f"config.MAX_MODEL_LEN 必须是正整数或 'auto'，当前为 {requested!r}")

    if native and requested > native and not config.ALLOW_EXCEED_NATIVE_CONTEXT:
        log(f"[config] MAX_MODEL_LEN={requested} 超过模型原生上限 {native}，已自动钳制")
        return native
    if native and requested > native:
        log(
            f"[config] 警告：MAX_MODEL_LEN={requested} 超过原生上限 {native}，"
            "需要 YaRN/HF_OVERRIDES 才能生效"
        )
    return requested


def infer_model_profile(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """按 model_type / architectures 匹配 MODEL_PROFILES。"""
    model_type = str(model_cfg.get("model_type", "")).lower()
    for key, profile in config.MODEL_PROFILES.items():
        key_l = key.lower()
        if model_type == key_l or model_type.startswith(key_l):
            return profile

    architectures = " ".join(model_cfg.get("architectures") or []).lower()
    for key, profile in config.MODEL_PROFILES.items():
        if key.lower() in architectures:
            return profile
    return {}


def is_multimodal(model_cfg: dict[str, Any]) -> bool:
    """只依据多模态相关字段判断，避免把纯文本 T5 等误判为多模态。"""
    keys = (
        "vision_config",
        "image_token_id",
        "image_token_index",
        "video_token_id",
        "audio_config",
    )
    return any(model_cfg.get(key) is not None for key in keys)


def is_qwen(model_cfg: dict[str, Any]) -> bool:
    model_type = str(model_cfg.get("model_type", "")).lower()
    architectures = " ".join(model_cfg.get("architectures") or []).lower()
    return "qwen" in model_type or "qwen" in architectures or "qwen" in config.MODEL_ID.lower()


def estimate_kv_bytes_per_token(model_cfg: dict[str, Any]) -> int | None:
    """粗略估算每 token 的 KV 字节数（BF16），用于启动日志提示。

    只统计 full attention 层；线性注意力层是每序列的常量状态，不按 token 计。
    """
    text = model_cfg.get("text_config", model_cfg)
    if not isinstance(text, dict):
        return None
    if text.get("kv_lora_rank"):  # MLA 架构不适用该公式
        return None

    layer_types = text.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        n_full = sum(1 for x in layer_types if x == "full_attention")
    else:
        n_full = text.get("num_hidden_layers")

    kv_heads = text.get("num_key_value_heads") or text.get("num_attention_heads")
    n_heads = text.get("num_attention_heads")
    head_dim = text.get("head_dim")
    if not head_dim and isinstance(n_heads, int) and n_heads:
        hidden = text.get("hidden_size")
        if isinstance(hidden, int):
            head_dim = hidden // n_heads

    if not all(isinstance(x, int) and x > 0 for x in (n_full, kv_heads, head_dim)):
        return None
    return n_full * 2 * kv_heads * head_dim * 2  # 2 = K/V, 最后 2 = BF16 字节


def _has_tokenizer(root: Path) -> bool:
    """识别常见 HF tokenizer 布局，避免把“只有权重”的缓存当成完整快照。"""
    for name in (
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
        "vocab.txt",
    ):
        marker = root / name
        if marker.is_file() and marker.stat().st_size > 0:
            return True
    if any(p.is_file() and p.stat().st_size > 0 for p in root.glob("*.tiktoken")):
        return True
    vocab = root / "vocab.json"
    merges = root / "merges.txt"
    return (
        vocab.is_file()
        and vocab.stat().st_size > 0
        and merges.is_file()
        and merges.stat().st_size > 0
    )


def _weights_complete(root: Path) -> bool:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_file = root / index_name
        if not index_file.is_file():
            continue
        try:
            data = json.loads(index_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        shards = set((data.get("weight_map") or {}).values())
        if not shards:
            return False
        return all(
            (root / name).is_file() and (root / name).stat().st_size > 0 for name in shards
        )

    weight_files = [
        p
        for p in list(root.glob("*.safetensors")) + list(root.glob("*.bin"))
        if p.is_file() and p.stat().st_size > 0
    ]
    if not weight_files:
        return False

    # 没有 index.json 时，若文件按分片命名，必须 1..total 全部到齐。
    # 分组键包含扩展名，避免 .bin 与 .safetensors 的残缺分片互相“补齐”。
    shard_groups: dict[tuple[str, str, int], set[int]] = {}
    for path in weight_files:
        match = _SHARD_RE.match(path.name)
        if match:
            key = (match.group("prefix"), match.group("suffix"), int(match.group("total")))
            shard_groups.setdefault(key, set()).add(int(match.group("index")))
    for (_prefix, _suffix, total), indices in shard_groups.items():
        if indices != set(range(1, total + 1)):
            return False
    return True


def _snapshot_complete(
    path: str | os.PathLike[str], *, require_tokenizer: bool = True
) -> bool:
    """检查 snapshot 是否包含 config.json、完整权重，以及（可选）tokenizer。

    require_tokenizer=True 用于“本地缓存能否直接启动”的快速判断；下载完成或
    离线校验时传 False，避免未知但合法的 tokenizer 布局被误判为不完整。
    """
    root = Path(path)
    if not (root / "config.json").is_file():
        return False
    if require_tokenizer and not _has_tokenizer(root):
        return False
    return _weights_complete(root)


def _download_snapshot(local_files_only: bool) -> Path:
    # 延迟导入，保证 config.py / 本模块可在未安装依赖时被读取。
    from huggingface_hub import snapshot_download

    cache_dir = expand_path(config.MODEL_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not local_files_only:
        free_gib = shutil.disk_usage(cache_dir).free / GIB
        if free_gib < MIN_FREE_DISK_GIB:
            print(
                f"[model] 警告：{cache_dir} 所在磁盘仅剩 {free_gib:.1f} GiB，"
                f"模型约需 56GB，下载可能失败",
                file=sys.stderr,
            )

    print(
        f"[model] {'检查本地缓存' if local_files_only else '联网加载'}："
        f"{effective_model_url()} -> {cache_dir}",
        file=sys.stderr,
    )
    return Path(
        snapshot_download(
            repo_id=config.MODEL_ID,
            revision=config.REVISION,
            cache_dir=str(cache_dir),
            endpoint=config.HF_ENDPOINT or None,
            token=config.HF_TOKEN or None,
            local_files_only=local_files_only,
        )
    )


def resolve_model_path() -> Path:
    """返回 vLLM 可以直接加载的本地模型目录。

    优先级：
    1. MODEL_ID 本身是本地目录；
    2. LOCAL_MODEL_PATH（存在时）；
    3. 离线：HF 缓存中的完整 snapshot（local_files_only=True），缺失则报错；
    4. 联网：snapshot_download 到 HF_HUB_CACHE。
    """
    model_id_path = expand_path(config.MODEL_ID)
    if model_id_path.is_dir():
        return model_id_path

    if config.LOCAL_MODEL_PATH:
        local_path = expand_path(config.LOCAL_MODEL_PATH)
        if local_path.is_dir():
            return local_path
        if not config.ONLINE_LOAD:
            raise FileNotFoundError(
                f"ONLINE_LOAD=False 且 LOCAL_MODEL_PATH 不存在：{local_path}"
            )
        print(
            f"[model] 警告：LOCAL_MODEL_PATH 不存在（{local_path}），继续联网加载",
            file=sys.stderr,
        )

    if not config.ONLINE_LOAD:
        try:
            snapshot = _download_snapshot(local_files_only=True)
        except Exception as exc:  # huggingface_hub 的异常类型较多，统一转成可读提示
            raise RuntimeError(
                "离线模式下没有找到可用缓存。请先联网运行一次 "
                "`bash download_model.py`，或把 LOCAL_MODEL_PATH 指向完整模型目录。"
                f"\n原始错误：{exc}"
            ) from exc
        if not _snapshot_complete(snapshot, require_tokenizer=False):
            raise RuntimeError(
                f"缓存 snapshot 不完整：{snapshot}。请联网重新运行 download_model.py。"
            )
        if not _has_tokenizer(snapshot):
            print(
                f"[model] 警告：未在 {snapshot} 中识别到 tokenizer 文件，"
                "如启动失败请联网补齐缓存",
                file=sys.stderr,
            )
        return snapshot

    if config.USE_LOCAL_CACHE_FIRST:
        # 本地已有完整快照时直接启动，省掉 snapshot_download 的远端 revision 查询。
        try:
            cached = _download_snapshot(local_files_only=True)
        except Exception:
            cached = None
        if cached is not None and _snapshot_complete(cached):
            print(
                f"[model] 命中本地完整快照，跳过联网检查：{cached}",
                file=sys.stderr,
            )
            return cached

    snapshot = _download_snapshot(local_files_only=False)
    # 下载刚完成，tokenizer 布局可能超出白名单；这里只校验权重和 config。
    if not _snapshot_complete(snapshot, require_tokenizer=False):
        raise RuntimeError(f"下载后的 snapshot 不完整：{snapshot}")
    return snapshot
