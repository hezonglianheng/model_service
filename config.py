"""Qwen3.8-27B 本地推理服务配置。

所有可调项集中在这里；修改后重启服务即可生效。
默认值：联网从 https://hf-mirror.com 加载，每张合格 GPU 一个副本，监听 127.0.0.1:8000。
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
# "auto" 检查所有可见 GPU；也可用物理编号白名单，如 [0, 1]。
# 同时尊重启动环境 CUDA_VISIBLE_DEVICES；实际子进程用 UUID 绑定卡。
GPU_IDS = "auto"
# 每张卡放一份完整模型，固定 TP=1；不再按 GPU_IDS 长度开启 TP。
TENSOR_PARALLEL_SIZE = 1
# 每卡预算上限；空闲不足时自动降低（vLLM 按总显存计算此比例）。
GPU_MEMORY_UTILIZATION = 0.90
GPU_FREE_MEMORY_RESERVE_GIB = 2.0
MIN_NUM_SEQS = 1
WEIGHT_MEMORY_MARGIN = 1.05
MIN_KV_CACHE_GIB = 4.0
# 估算运行时/图捕获预留；不是精确峰值，最终必须通过真实加载。
RUNTIME_MEMORY_GIB = 4.0
RUNTIME_MEMORY_PER_SEQ_MIB = 64
# 模型族 MODEL_PROFILES 中的 resources 可给默认值；这里优先级最高。
# 支持 weights_gib, kv_bytes_per_token, state_bytes_per_seq,
# runtime_memory_gib, runtime_memory_per_seq_mib, min_compute_capability。
# 未知架构/量化需用实测值覆盖，不能按模型名称猜显存。
RESOURCE_OVERRIDES = {}
# 相同 max_num_seqs 下的相对处理能力，按物理 GPU 编号设置；默认 1。
GPU_ROUTING_WEIGHTS = {}

# 上下文长度。默认 64K（单卡 A100 80GB 的稳妥值）；
# 所有副本保持相同上下文，不足的卡减并发或跳过；不会自动缩短上下文。
MAX_MODEL_LEN = 65536
# 并发序列数 / 单批 token 数；显存紧张时调小。
# MAX_NUM_SEQS 是同时在跑的序列上限：短请求吞吐可试 32；
# 长上下文仍受 KV cache 约束（64K 请求实测约 4 路，vLLM 会自动排队而不是 OOM）。
# MAX_NUM_SEQS = 8
MAX_NUM_SEQS = 32
# 单批 token 预算：长 prompt 预填充可试 16384；短筛选请求保持 8192 更稳。
MAX_NUM_BATCHED_TOKENS = 8192
# MAX_NUM_BATCHED_TOKENS = 16384
# 显存/图捕获失败时打开（禁用 CUDA graph）。
ENFORCE_EAGER = False

# "auto" 由模型 config 决定（Qwen3.8 为 bfloat16）。
# 规划器会统一解析成具体精度，并显式下发给所有副本，避免各卡自行选择不同 dtype。
DTYPE = "auto"
# A100 是 SM80，不支持 vLLM 的 fp8 KV cache；请保持 auto/bfloat16。
KV_CACHE_DTYPE = "auto"

# ============================ 服务 ============================
HOST = "127.0.0.1"
PORT = 8000
# 设置后除 /health 外的网关请求需要 Bearer key；None 表示不鉴权。
# 副本 vLLM 使用同一 key，supervisor 的健康探测也会携带它。
API_KEY = None
UVICORN_LOG_LEVEL = "info"
# 外部统一 PORT；内部副本从此端口依次分配，只监听 127.0.0.1。
BACKEND_PORT_START = 8100
REPLICA_STARTUP_TIMEOUT = 1800
REPLICA_HEALTH_INTERVAL = 5.0
REPLICA_HEALTH_TIMEOUT = 3.0
REPLICA_SHUTDOWN_TIMEOUT = 20.0
# supervisor 被强杀时，副本守护进程先等 vLLM 自行退出的秒数，然后强杀整个进程组。
REPLICA_CHILD_GRACE_SECONDS = 15.0
# 连接已建立后的超时不重放请求；流式也使用此空闲读取超时。
PROXY_CONNECT_TIMEOUT = 5.0
PROXY_READ_TIMEOUT = 600.0
PROXY_MAX_BODY_MIB = 64
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
