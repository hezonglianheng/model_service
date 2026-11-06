"""Qwen3.8-27B 本地推理服务配置。

所有可调项集中在这里；修改后重启服务即可生效。
默认值：联网从 https://hf-mirror.com 加载，使用物理卡 1，监听 127.0.0.1:8000。
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ============================ 模型 ============================
# HF repo id，或本地目录（本地目录存在时直接使用，不联网）。
MODEL_ID = "Qwen/Qwen3.8-27B"
# 可选 revision（分支/commit/tag）；None 表示 main。
REVISION = None
# /v1/models 返回的模型名，也是客户端 model 字段使用的名字。
SERVED_MODEL_NAME = "Qwen/Qwen3.8-27B"

# True：联网加载（先 snapshot_download 到本地缓存，再交给 vLLM）。
# False：只用本地缓存/LOCAL_MODEL_PATH，完全禁止联网。
ONLINE_LOAD = True
# True：ONLINE_LOAD=True 时也先查完整本地快照，命中就直接启动，跳过 HF 联网检查（重启更快）。
# False：每次启动都向 HF 查询 revision（慢，但能及时发现远端更新）。
USE_LOCAL_CACHE_FIRST = True
# 联网加载点/镜像地址；默认 hf-mirror，可改回 https://huggingface.co。
HF_ENDPOINT = "https://hf-mirror.com"
# 实际加载地址（仅用于日志展示，由上面两项推导）。
MODEL_URL = f"{HF_ENDPOINT.rstrip('/')}/{MODEL_ID}"

# 离线模式下可显式指定本地模型目录；留空则从 HF 缓存里找完整 snapshot。
LOCAL_MODEL_PATH = ""
# HF 缓存目录（snapshot_download 的 cache_dir）。
MODEL_CACHE_DIR = "~/.cache/huggingface/hub"
# HF_HOME（token 等元数据目录）。
HF_HOME = "~/.cache/huggingface"
# vLLM 自己的编译缓存目录（项目内，便于清理）。
VLLM_CACHE_ROOT = ROOT / ".cache" / "vllm"

# gated 模型才需要；也可直接用环境变量 HF_TOKEN。None 表示不覆盖环境变量。
HF_TOKEN = None
# 需要模型仓库自带 modeling 代码时打开；Qwen3.8 本身不需要。
TRUST_REMOTE_CODE = True

# ============================ GPU ============================
# 物理卡号（nvidia-smi 的 0-based 编号）。默认 [1] = 第二张卡（编号 1）。
# 多卡示例：[0, 1]；张量并行数自动等于列表长度。
GPU_IDS = [1]
# "auto" = len(GPU_IDS)；也可显式写整数（必须与 GPU_IDS 数量一致）。
TENSOR_PARALLEL_SIZE = "auto"
# 单卡显存占用上限。A100 80GB + BF16 27B 建议 0.90。
GPU_MEMORY_UTILIZATION = 0.90

# 上下文长度。默认 64K（单卡 A100 80GB 的稳妥值）；
# 128K 可试 131072；原生 262144 在单卡上很紧张，建议两卡 TP2。
MAX_MODEL_LEN = 65536
# 并发序列数 / 单批 token 数；显存紧张时调小。
# MAX_NUM_SEQS 是同时在跑的序列上限：短请求吞吐可试 32；
# 长上下文仍受 KV cache 约束（64K 请求实测约 4 路，vLLM 会自动排队而不是 OOM）。
MAX_NUM_SEQS = 8
MAX_NUM_BATCHED_TOKENS = 8192
# 显存/图捕获失败时打开（禁用 CUDA graph）。
ENFORCE_EAGER = False

# "auto" 由模型 config 决定（Qwen3.8 为 bfloat16）。
DTYPE = "auto"
# A100 是 SM80，不支持 vLLM 的 fp8 KV cache；请保持 auto/bfloat16。
KV_CACHE_DTYPE = "auto"

# ============================ 服务 ============================
HOST = "127.0.0.1"
PORT = 8000
# 设置后 /v1、/v2、/inference 需要 Bearer key；None 表示不鉴权。
API_KEY = None
UVICORN_LOG_LEVEL = "info"
# 官方 recipe 建议开启的 Rust 前端；如遇兼容问题可改 False。
USE_RUST_FRONTEND = True

# ======================= 模型能力自动适配 =======================
# True：按 config.json 的 model_type 自动选择 parser/多模态参数。
# 换非 Qwen 模型时自动跳过 Qwen 专用参数，无需改代码。
AUTO_MODEL_PROFILE = True
# 以 model_type 为键的模型族配置；可自行追加。
MODEL_PROFILES = {
    "qwen3_5": {
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "mm_encoder_tp_mode": "data",
    },
}
# "auto"：从 MODEL_PROFILES 解析；None：不传；也可写死 parser 名。
TOOL_CALL_PARSER = "auto"
REASONING_PARSER = "auto"
MM_ENCODER_TP_MODE = "auto"
# 有 tool_call_parser 时是否同时开启 --enable-auto-tool-choice。
ENABLE_AUTO_TOOL_CHOICE = True
# "auto"/"vllm"/"transformers"；非 auto 时传给 --model-impl。
MODEL_IMPL = "auto"
# 量化方式，例如 "awq"/"gptq"；None 表示由模型 config 决定。
QUANTIZATION = None

# 只加载语言模型、关闭视觉分支（省显存）。
LANGUAGE_MODEL_ONLY = False
# 每个 prompt 最多接受的图片/视频数，防止视觉输入撑爆显存。
LIMIT_MM_PER_PROMPT = {"image": 4, "video": 1}
# 服务器级 chat template 默认参数，例如 {"reasoning_effort": "medium"}。
DEFAULT_CHAT_TEMPLATE_KWARGS = {}
# 投机解码，例如 {"method": "mtp", "num_speculative_tokens": 3}；None 关闭。
SPECULATIVE_CONFIG = None
# 覆盖 generation_config，例如 {"temperature": 0.7}。
OVERRIDE_GENERATION_CONFIG = {}
# 超长上下文（例如 1M）示例：同时把 MAX_MODEL_LEN 设为 1000000、
# ALLOW_EXCEED_NATIVE_CONTEXT 设为 True，并取消下面注释。
# HF_OVERRIDES = {
#     "text_config": {
#         "rope_parameters": {
#             "mrope_interleaved": True,
#             "mrope_section": [11, 11, 10],
#             "rope_type": "yarn",
#             "rope_theta": 10000000,
#             "partial_rotary_factor": 0.25,
#             "factor": 4.0,
#             "original_max_position_embeddings": 262144,
#         }
#     }
# }
HF_OVERRIDES = None
# 前缀缓存（hybrid 模型通常可用）。
ENABLE_PREFIX_CACHING = True

# ============================ 其它 ============================
# 任意额外 vLLM 参数，原样追加。例如 ["--disable-log-requests"]。
EXTRA_VLLM_ARGS = []
# 额外环境变量，原样注入 vLLM 进程。
EXTRA_ENV = {}

# 驱动 550 / CUDA 12.4 的兜底：用 scripts/fetch_cuda_compat.sh 解压
# cuda-compat-12-9 后，把 USE_CUDA_COMPAT 改为 True。
USE_CUDA_COMPAT = False
CUDA_COMPAT_DIR = ROOT / ".cuda-compat"

# 允许请求超过模型原生上下文（需要 YaRN/HF_OVERRIDES；默认关闭）。
ALLOW_EXCEED_NATIVE_CONTEXT = False

# =================== 安装版本（setup.sh 使用） ===================
VLLM_VERSION = "0.32.0"
VLLM_CUDA_VARIANT = "cu129"
TORCH_VERSION = "2.13.0"
TORCHVISION_VERSION = "0.28.0"
TORCHAUDIO_VERSION = "2.11.0"
