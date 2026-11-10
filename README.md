# Qwen3.8-27B 本地推理服务

一个配置驱动、可从任意目录启动的本地推理服务：

- 默认从 `https://hf-mirror.com` 加载 `Qwen/Qwen3.8-27B`；
- 启动时探测所有可见 NVIDIA GPU，**每张合格卡一个完整模型副本（TP=1）**；
- 在 `http://127.0.0.1:8000` 提供 **OpenAI 兼容 API**；
- 项目内异步网关按容量和在途请求负载均衡，客户端继续使用同一个地址；
- 换模型、选卡、调整资源估算、离线运行，统一修改 `config.py`。

底层使用 **vLLM 0.32.0（cu129 wheel）**，模型推理、多模态、工具调用和副本内 continuous batching 由 vLLM 处理。副本间无需 GPU 通信。

> 调研信息基于 2026-11：`Qwen/Qwen3.8-27B` 为官方仓库，Apache-2.0，非 gated。请勿使用同名仿冒仓库。

---

## 1. 模型调研摘要

| 项目 | 值 |
| --- | --- |
| Hugging Face 仓库 | `Qwen/Qwen3.8-27B` |
| 架构 | `Qwen3_5ForConditionalGeneration`（`model_type=qwen3_5`） |
| 参数 / 精度 | 27.78B，BF16 safetensors |
| 权重体积 | 18 个分片，约 55.57 GB |
| 层结构 | 64 层：48 层 Gated DeltaNet 线性注意力 + 16 层 full attention |
| 注意力头 | 24 Q heads / 4 KV heads，head_dim 256 |
| 原生上下文 | 262,144 tokens，可用 YaRN 扩展到 1M |
| 输入 | 文本 + 图像 / 视频（文本输出） |
| Thinking | 默认开启，`reasoning_effort` 支持 `xhigh` / `medium` / `low` |
| 推荐采样（thinking） | `temperature=1.0`，`top_p=0.95`，`top_k=20` |
| 推荐采样（non-thinking） | `temperature=0.7`，`top_p=0.80`，`top_k=20`，`presence_penalty=1.5` |
| 官方 vLLM 要求 | vLLM >= 0.17.0 |
| 模型卡 | <https://huggingface.co/Qwen/Qwen3.8-27B>（镜像：<https://hf-mirror.com/Qwen/Qwen3.8-27B>） |

官方 vLLM 启动 recipe（本项目已自动组装等价参数）：

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3.8-27B \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data
```

**A100 注意事项**：A100 是 SM80，不支持 vLLM 的 FP8 KV cache，`KV_CACHE_DTYPE` 请保持 `auto` / `bfloat16`。

---

## 2. 目录结构

```
model_service/
├── config.py                 # 唯一配置入口
├── server.py                 # 读取配置 -> 规划副本 -> 运行 supervisor + 网关
├── gpu_planner.py            # GPU 探测、实际权重/模型配置估算、逐卡准入
├── replica_manager.py        # 独立进程、启动重试、健康检查、进程组退出
├── gateway.py                # 容量加权最少在途请求、流式 HTTP 透传
├── model_utils.py            # 模型路径、HF 缓存、config.json、上下文钳制
├── download_model.py         # 预下载 / 校验模型快照
├── setup.sh                  # 创建 .venv，安装 cu129 torch + vLLM，CUDA 冒烟测试
├── start.sh                  # 前台启动（任意 CWD）
├── service.sh                # 后台 start/stop/restart/status/logs
├── requirements.txt          # 版本参考（安装以 setup.sh 为准）
├── scripts/
│   ├── chat.py               # OpenAI SDK 示例（流式 / 图片 / thinking）
│   ├── batch_chat.py         # 并发批处理（多输入 / 每输入 n 个候选）
│   ├── smoke_test.sh         # /health + /v1/models + 一次 chat
│   └── fetch_cuda_compat.sh  # 可选：无 sudo 解压 cuda-compat-12-9
├── deployment_tests/         # 资源规划、真实 HTTP/子进程集成测试（无需 GPU）
├── tests/                    # 本机历史测试（被 .gitignore 忽略）
├── logs/                     # 运行日志（自动创建）
└── run/                      # PID 文件（自动创建）
```

---

## 3. 环境要求

- Linux + NVIDIA 驱动（本项目在 2× A100 80GB、驱动 550.163.01 上验证）；
- Python 3.12（`setup.sh` 默认用 `python3.12` 创建 venv）；
- 首次下载需要约 60 GB 空闲磁盘；
- 安装 `uv` 可选（有则更快），没有则回退到 `python -m venv` + `pip`。

单个副本在 A100 80GB 上的显存估算（BF16）：

| `MAX_MODEL_LEN` | KV cache | 说明 |
| --- | --- | --- |
| 32K | ≈ 2 GiB | 最稳 |
| 64K（默认） | ≈ 4 GiB | 推荐，余量充足 |
| 128K | ≈ 8 GiB | 可跑，需观察峰值 |
| 262K | ≈ 16 GiB | 单卡余量很小，由资源预检与实际加载决定是否准入 |

权重约 51.75 GiB；A100 80GB 在 `nvidia-smi` 中约为 81920 MiB（80 GiB），`gpu_memory_utilization=0.90` 时预算约 72 GiB，剩余约 20 GiB 给 KV、激活、CUDA graph 和视觉塔。

本项目实测（A100 80GB、`GPU_MEMORY_UTILIZATION=0.90`）：权重加载 51.01 GiB，可用 KV cache 17.77 GiB（270,153 tokens），64K 请求可并发 4.12 路；卡 1 显存占用约 71 GiB。首次启动需要 FlashInfer JIT 编译采样内核（约 1–2 分钟），之后有缓存，重启约 1 分钟即可就绪。

以上为原有单副本测量，不代表改造后的双卡吞吐已实测。现在每卡均加载完整权重；例如 GPU 0 上仍有约 27 GiB 的 LTP 占用，该卡可能被跳过。需要在 LTP 所属项目中将其迁到 CPU，再重启本服务重新探测。

---

## 4. 快速开始

```bash
cd /home/qinyuhang/model_service

# 1) 安装依赖（创建 .venv，安装 cu129 torch + vLLM，跑 CUDA 冒烟测试）
bash setup.sh

# 2) 预下载模型（可选；首次启动也会自动下载，约 55.6GB，支持断点续传）
.venv/bin/python download_model.py

# 3) 前台启动（Ctrl-C 停止）
bash start.sh

# 或者后台启动，并等待 /health 就绪
bash service.sh start --wait
```

首次启动会先下载约 55.6GB 模型（走 `HF_ENDPOINT`，支持断点续传）。`service.sh start --wait` 会一直等到 `/health` 就绪：默认最长 4 小时，且只有日志连续 20 分钟无变化才判定卡住（`STARTUP_TIMEOUT` / `STARTUP_IDLE_TIMEOUT` 可调）。已缓存时启动只需加载权重。

启动后：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/replicas
bash scripts/smoke_test.sh
```

停止 / 查看：

```bash
bash service.sh status
bash service.sh logs
bash service.sh stop
bash service.sh restart --wait
```

`start.sh` / `service.sh` 使用脚本自身路径定位项目，因此可以在任意目录执行：

```bash
bash /home/qinyuhang/model_service/service.sh start --wait
```

---

## 5. config.py 配置说明

默认值：**自动选择合格 GPU、联网、`https://hf-mirror.com`、统一入口 `127.0.0.1:8000`**。修改后重启服务生效。

### 5.1 模型与加载方式

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_ID` | `Qwen/Qwen3.8-27B` | HF repo id，或本地模型目录 |
| `SERVED_MODEL_NAME` | `Qwen/Qwen3.8-27B` | `/v1/models` 与请求里的 `model` 字段 |
| `ONLINE_LOAD` | `True` | `True` 联网；`False` 只用本地缓存/路径 |
| `USE_LOCAL_CACHE_FIRST` | `True` | 有完整本地快照时跳过 HF 联网检查，重启更快 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 联网加载点，可改回 `https://huggingface.co` |
| `LOCAL_MODEL_PATH` | `""` | 离线时可显式指定完整模型目录 |
| `MODEL_CACHE_DIR` | `~/.cache/huggingface/hub` | `snapshot_download` 的缓存目录 |
| `HF_TOKEN` | `None` | gated 模型才需要；不要提交到 git |
| `REVISION` | `None` | 可选分支 / commit / tag |

加载优先级：`MODEL_ID` 是本地目录 → `LOCAL_MODEL_PATH` → HF 缓存/下载。

路径类配置支持 `~`；相对路径以项目根目录为基准（与启动时的当前目录无关），因此可以从任意目录启动。

### 5.2 GPU 与显存

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `GPU_IDS` | `"auto"` | 自动探测；也可用 `[0, 1]` 作为物理卡白名单 |
| `TENSOR_PARALLEL_SIZE` | `1` | 每卡独立副本；不支持在此模式设为 2 |
| `GPU_MEMORY_UTILIZATION` | `0.90` | 每卡预算上限，根据空闲量向下调整 |
| `GPU_FREE_MEMORY_RESERVE_GIB` | `2.0` | 在当前空闲显存之外保留的余量 |
| `MAX_MODEL_LEN` | `65536` | 自动钳制到模型原生上限 |
| `MAX_NUM_SEQS` | `32` | 每副本并发上限；不足时按 32→16→8→4→2→1 降低 |
| `MIN_NUM_SEQS` | `1` | 降低并发的下限 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 单批 token 预算；长 prompt 预填充可试 `16384` |
| `GPU_ROUTING_WEIGHTS` | `{}` | 按物理卡号覆盖路由容量权重，用于异构卡 |
| `RESOURCE_OVERRIDES` | `{}` | 未知架构/量化的显式资源估算，见 7 节 |
| `WEIGHT_MEMORY_MARGIN` | `1.05` | 权重加载余量 |
| `RUNTIME_MEMORY_GIB` | `4` | 每副本固定运行时预留 |
| `RUNTIME_MEMORY_PER_SEQ_MIB` | `64` | 每序列额外预留 |
| `MIN_KV_CACHE_GIB` | `4` | 每副本 KV 预留下限 |
| `DTYPE` | `"auto"` | Qwen3.8 为 bfloat16；`auto` 由规划器统一解析成具体精度后下发所有副本 |
| `KV_CACHE_DTYPE` | `"auto"` | A100 不要用 fp8 |
| `ENFORCE_EAGER` | `False` | 显存/图捕获异常时改 `True` |

指定候选卡示例：

```python
GPU_IDS = [0, 1]          # 两张卡分别评估，合格的各放完整模型，TP 始终为 1
MAX_MODEL_LEN = 65536     # 所有副本相同
```

**准入流程：**

1. 读取本地模型配置（包括 `HF_OVERRIDES`）和 safetensors 文件头；不把 27B/80GB 写成固定阈值。
2. 根据权重精度、完整注意力层数、KV heads/head_dim、线性注意力状态估算需求。Qwen3.8-27B 的 BF16 KV 约 64 KiB/token，64K 约 4 GiB；线性注意力状态另按序列计入。
3. 单卡预算 = `min(总显存 × GPU_MEMORY_UTILIZATION, 当前空闲显存 − 余量)`；传给 vLLM 的比例始终以**总显存**为分母。
4. 保持精度与上下文，尝试降低并发；仍不足、计算能力不兼容、被可见性限制或启用 MIG 的物理父卡会被跳过，输出明确原因。
5. 每个入选 GPU 的 UUID 单独设置为子进程的 `CUDA_VISIBLE_DEVICES`。真实加载、`/health` 成功后才接单；加载发生显存错误时继续降并发重试。其他错误直接移出，全部失败主进程非零退出。

预检不是显存峰值保证。默认权重余量 `WEIGHT_MEMORY_MARGIN=1.05`，运行时固定预留 `RUNTIME_MEMORY_GIB=4`、每序列额外预留 `RUNTIME_MEMORY_PER_SEQ_MIB=64`。KV 下限是 `max(一条完整上下文的 KV, MIN_KV_CACHE_GIB)`，后者默认 4 GiB。**不会要求 32 条请求同时跑满 64K**，超出的实际请求由 vLLM 排队。

预量化 per-tensor FP8 checkpoint（`quantization_config.quant_method=fp8`、无 `weight_block_size`）的量化路径本身最低要求 SM75；BF16 activation 仍要求 SM80，只有 FP16 activation 才可能在 SM75 准入。KV 需保持 `auto` 才适用于这条宽松路径；block FP8、`VLLM_BATCH_INVARIANT=1`、FP8 KV 和未识别的量化方式仍保守要求各自更高门槛。单测准入不代表具体模型/内核已在目标卡实测通过。

资源紧张时，图捕获、多模态、混合 KV 分页等实际开销可能大于估算，需依据副本日志校准以上参数；不会自动改精度、量化、缩短上下文或禁用 CUDA graph。资源评估在每次启动时进行，不动态接管刚空闲的 GPU。

### 5.3 扩展到 1M 上下文

Qwen3.8-27B 原生 262,144 tokens；官方支持用 YaRN 扩到 1M。需要同时修改三处：

```python
MAX_MODEL_LEN = 1000000
ALLOW_EXCEED_NATIVE_CONTEXT = True   # 自动设置 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
HF_OVERRIDES = {
    "text_config": {
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_type": "yarn",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "factor": 4.0,
            "original_max_position_embeddings": 262144,
        }
    }
}
```

1M 上下文的 KV cache 约 64 GiB（BF16），与权重相加已超过单张 A100 80GB。本项目的独立副本模式会拒绝这些卡，不会将两张卡显存相加。需要更大显存的单卡或经验证的低精度模型，并重新估算资源。

### 5.4 服务与兼容性

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | 监听地址 |
| `BACKEND_PORT_START` | `8100` | 副本内部端口依次分配，监听 127.0.0.1 |
| `API_KEY` | `None` | 设置后统一入口和副本健康探测都使用 Bearer token |
| `UVICORN_LOG_LEVEL` | `"info"` | 访问日志级别 |
| `USE_RUST_FRONTEND` | `True` | 官方 recipe 建议；异常时可改 `False` |
| `AUTO_MODEL_PROFILE` | `True` | 按 `model_type` 自动选 parser / 多模态参数 |
| `MODEL_PROFILES` | 见下 | 模型族 → parser 映射 |
| `TOOL_CALL_PARSER` | `"auto"` | `"auto"` / `None` / 显式名字 |
| `REASONING_PARSER` | `"auto"` | 同上 |
| `MM_ENCODER_TP_MODE` | `"auto"` | 多模态编码器并行方式 |
| `TRUST_REMOTE_CODE` | `True` | 兼容性优先 |
| `EXTRA_VLLM_ARGS` | `[]` | 额外 vLLM 参数；禁止覆盖设备/拓扑/精度/容量/入口参数 |
| `EXTRA_ENV` | `{}` | 原样注入任意环境变量（禁止覆盖可见设备） |
| `REPLICA_STARTUP_TIMEOUT` | `1800` | 单次副本加载最长等待秒数 |
| `REPLICA_HEALTH_INTERVAL` / `TIMEOUT` | `5` / `3` | 就绪与存活探测周期/超时 |
| `REPLICA_SHUTDOWN_TIMEOUT` | `20` | 正常退出时每组进程的等待时间 |
| `REPLICA_CHILD_GRACE_SECONDS` | `15` | supervisor 被强杀时，副本守护进程等待 vLLM 自行退出的时间，之后强杀整个进程组 |

`MODEL_PROFILES` 默认：

```python
MODEL_PROFILES = {
    "qwen3_5": {
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "mm_encoder_tp_mode": "data",
    },
}
```

匹配顺序：先精确/前缀匹配 `model_type`，再匹配 `architectures`。**更具体的键要放在前面**（例如以后加 `"qwen3"` 时，`"qwen3_5"` 必须排在它之前）。

### 5.5 负载均衡、健康检查与日志

- 所有外部客户端继续访问 `http://127.0.0.1:8000/v1`，无需感知内部端口。
- 网关选择 `在途请求数 / 容量权重` 最低的健康副本，同负载轮转；默认容量权重为实际 `max_num_seqs × GPU_ROUTING_WEIGHTS[物理卡号]`。例如慢卡可设 `GPU_ROUTING_WEIGHTS={1: 0.5}`。
- 并发容量是初始权重，不等于实测算力比例；异构卡可用同批请求校准性能权重。各副本前缀缓存独立，当前不做前缀粘性路由。
- 请求占用覆盖排队、首 token 等待和整个响应；SSE 原样转发，客户端断开关闭上游并释放计数。客户端断开（下游写失败）不会被误判成 GPU 后端故障；只有 TCP 尚未连接成功的上游失败才切换副本，读超时和已开始生成的请求不会自动重放。请求体按原始编码字节转发，gzip 等 `Content-Encoding`/长度保持一致。
- 启用 `VLLM_ENABLE_RESPONSES_API_STORE=1` 时，网关为 `/v1/responses` 创建出的响应 ID 维护有界 LRU 亲和（最多 10000 条），把查询、取消和带 `previous_response_id` 的后续请求优先送回创建副本；创建响应时会在写出 ID 前登记，JSON 与 SSE 都支持。亲和表只在当前 supervisor 生命周期内有效：网关重启、条目淘汰、副本故障或后端自身重启后可能返回 404；它不是跨副本状态复制。请求体为 gzip 时会在 1 MiB 上限内解压检查 `previous_response_id`，转发字节和头部不变。
- 默认每 5 秒探测健康（`REPLICA_HEALTH_INTERVAL`），探测超时 3 秒；配置 `API_KEY` 时探测也携带同一 Bearer token。健康恢复自动入池。进程崩溃会移出，保留其他健康副本；需要重启服务恢复已退出的进程。
- `/health` 至少一个副本就绪返回 200，否则 503；`service.sh start --wait` 采用这个语义。`/replicas` 查看每张卡的准入原因、实际并发/预算和各副本状态/在途计数，配置 `API_KEY` 后需要 Bearer token。
- 总日志 `logs/server.log`；加载日志 `logs/gpu-<物理编号>-attempt-<尝试次数>.log`。`service.sh status` 显示副本状态。
- 每次加载最长 `REPLICA_STARTUP_TIMEOUT=1800` 秒；网关读取空闲超时 `PROXY_READ_TIMEOUT=600` 秒（长预填充时可提高）。单请求体上限 `PROXY_MAX_BODY_MIB=64`，图像/视频较大时调整。
- `service.sh stop` 通知 supervisor 停止接单、等待请求退出，再终止所有副本进程组。默认每阶段退出等待 `REPLICA_SHUTDOWN_TIMEOUT=20` 秒。
- 每个副本外还有轻量守护进程（`replica_runner.py`）：Linux 上设置 parent-death signal，supervisor 即使被 `SIGKILL`，守护进程也会在 `REPLICA_CHILD_GRACE_SECONDS` 后清理 vLLM 及其 worker，避免显存和端口泄漏。
- `CUDA_VISIBLE_DEVICES` 若用数字编号，启动时会用原环境查询 CUDA 的真实 UUID 映射，再按 `nvidia-smi` 的物理 UUID 选择副本；数字编号不等同于 `nvidia-smi` index。若无法查询，请直接使用 `GPU-...` UUID 或改在 `GPU_IDS` 中写物理编号。

---

## 6. 调用示例

### 6.1 curl

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.8-27B",
    "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.8,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

### 6.2 OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=256,
    temperature=0.7,
    top_p=0.8,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(resp.choices[0].message.content)
```

### 6.3 项目自带客户端

```bash
.venv/bin/python scripts/chat.py "写一个快速排序"
.venv/bin/python scripts/chat.py --stream --no-thinking "写一个快速排序"
.venv/bin/python scripts/chat.py --reasoning-effort high "证明根号 2 是无理数"
.venv/bin/python scripts/chat.py --image ./cat.jpg "描述这张图片"
```

### 6.4 Thinking 控制

- 关闭 thinking：请求体加 `"chat_template_kwargs": {"enable_thinking": false}`；
- 调整强度：请求体顶层加 `"reasoning_effort": "medium"`（`xhigh` / `medium` / `low`）；vLLM 也会把它传入 chat template；
- thinking 内容在响应的 `message.reasoning` 中（vLLM 0.32 的字段名；旧版本为 `reasoning_content`，流式为 `delta.reasoning` / `delta.reasoning_content`），正文仍在 `message.content`；`scripts/chat.py` 已同时兼容两种字段。

### 6.5 工具调用

服务默认已开启 `--enable-auto-tool-choice` 并配置 `qwen3_coder` 解析器：

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询城市天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
resp = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
    tool_choice="auto",
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(resp.choices[0].message.tool_calls)
```

### 6.6 多模态

`/v1/chat/completions` 的 `content` 可以是数组：

```json
[
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
  {"type": "text", "text": "描述这张图片"}
]
```

默认每个 prompt 最多 4 张图 / 1 个视频（`LIMIT_MM_PER_PROMPT`）。只看文本时可设 `LANGUAGE_MODEL_ONLY = True` 省显存。

### 6.7 并发 / 批量 / 多输出

并行推理由 vLLM 的 continuous batching 负责，服务端不需要自建线程池或请求队列：

- **服务端容量**：`MAX_NUM_SEQS`（每副本上限，默认 32，可能逐卡降低）和 `MAX_NUM_BATCHED_TOKENS`（单批 token 预算，默认 8192）；修改后 `bash service.sh restart --wait`。
- **长上下文**：并发受 KV cache 限制，64K 请求单卡约 4 路，超出会自动排队而不是 OOM；短请求可把 `MAX_NUM_SEQS` 调到 32 提升吞吐。
- **单请求多输出**：OpenAI 的 `n` 参数（chat / completions 均支持）；`/v1/completions` 还支持 `prompt` 数组一次提交多个独立输入。
- **多输入批量**：用 `scripts/batch_chat.py`，共享 HTTP 连接池、`--concurrency` 限制在途请求、结果保持输入顺序。

输入 JSONL，每行一个对象（也接受裸字符串）：

```json
{"id": "q1", "prompt": "用一句话解释张量并行"}
{"id": "q2", "messages": [{"role": "user", "content": "写一个快速排序"}], "n": 2}
{"id": "q3", "prompt": "总结这段文字", "system": "你是编辑", "max_tokens": 256}
```

```bash
# 4 路并发；每条输入返回 2 个候选；关闭 thinking
.venv/bin/python scripts/batch_chat.py \
  --input prompts.jsonl --output results.jsonl \
  --concurrency 4 --n 2 --no-thinking
```

结果 JSONL 与输入同序，每条包含全部 `choices`、`usage`、`elapsed_s`；失败项带 `error`，不会中断整批。全部成功退出码为 0，有失败为 1。默认单条重试 2 次（指数退避），`--retries 0` 可关闭。

`messages` 表示完全透传，不能与同一条的 `prompt` / `system` 同时出现；`--system` 只对 `prompt` 输入生效。缓存优先启动时，若需发现 HF 远端更新，把 `USE_LOCAL_CACHE_FIRST` 设为 `False` 再重启。

### 6.8 提速要点

已默认开启、通常不用改：

- continuous batching / chunked prefill（vLLM 调度器）；
- 前缀缓存（`ENABLE_PREFIX_CACHING = True`）；
- CUDA graph（`ENFORCE_EAGER = False`）；
- 本地缓存优先启动（`USE_LOCAL_CACHE_FIRST = True`），重启时跳过 HF 联网检查。

可调项：

| 场景 | 建议 |
| --- | --- |
| 大量短请求 | 提高 `MAX_NUM_SEQS`（如 32），客户端 `--concurrency` 与之匹配 |
| 长 prompt 预填充慢 | 适当提高 `MAX_NUM_BATCHED_TOKENS`（如 16384），观察显存峰值 |
| 客户端串行调用 | 用 `batch_chat.py` 并发 4~8 路，复用连接池 |
| 两个健康副本、批量筛选 | 同一批请求分别测试并发 16、32，观察总 output token/s 和每副本在途数 |
| 启动慢 | 保持 `USE_LOCAL_CACHE_FIRST=True` 和 `VLLM_CACHE_ROOT` 编译缓存 |

用 `batch_chat.py` 输出末尾的“请求/s、输出 token/s”对比调整效果；没有 GPU 的机器只能跑单元测试，吞吐必须在目标机上实测。

---

## 7. 换模型（兼容更多模型）

只改 `config.py`：

```python
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
SERVED_MODEL_NAME = "smollm2-135m"
MAX_MODEL_LEN = 2048          # 会被自动钳制到模型原生上限
GPU_MEMORY_UTILIZATION = 0.90
```

服务启动时会：

1. 读取模型本地 `config.json`；
2. 把 `MAX_MODEL_LEN` 钳制到 `max_position_embeddings`；
3. 按 `MODEL_PROFILES` / `model_type` 决定是否添加 Qwen 专用 parser；
4. 只有模型含 `vision_config` / `image_token_id` / `video_token_id` 等多模态字段时才加多模态参数；
5. 从实际权重和配置重新估算资源，对所有候选卡执行准入；模型架构的最终支持由 vLLM 决定。

常规 dense/GQA 模型可自动读取几何参数；Qwen3.5/3.8 混合线性注意力已额外估算 recurrent 状态。MLA、未知混合结构或未知量化方式无法可靠推导时，会明确报出所需覆盖项。可以在 `MODEL_PROFILES` 的 `resources` 中按模型族设置，也可用全局 `RESOURCE_OVERRIDES` 覆盖（后者优先）：

```python
# 以下只展示字段格式，必须填入目标模型的配置推导/实测值，不能照抄！
# RESOURCE_OVERRIDES = {
#     "weights_gib": ...,                 # 加载权重及投机模型的总占用
#     "kv_bytes_per_token": ...,          # 已考虑运行精度的全模型每 token KV
#     "state_bytes_per_seq": ...,         # 混合/SSM 架构每序列固定状态
#     "runtime_memory_gib": ...,
#     "runtime_memory_per_seq_mib": ...,
#     "min_compute_capability": ...,
# }
```

预量化 safetensors 使用其实际存储体积做估算；在线量化和投机解码需显式覆盖权重预算。估算默认保留所有 checkpoint 权重（包括视觉/MTP），因此可能偏保守；已知实际加载体积可覆盖。模型换成 MoE 时每卡仍需容纳完整权重，而不是只算激活专家。小模型/小显存卡若被默认 4 GiB 的 KV 下限错误跳过，可降低 `MIN_KV_CACHE_GIB`，这不会改变对外上下文。

`DTYPE="auto"` 时，规划器会按模型 config 统一解析出具体精度：FP32 模型默认 FP16，但 vLLM 0.32.0 禁止 FP16 的 `gemma2` / `gemma3` / `gemma3_text` / `glm4` 会改用 BF16；FP16 模型保持 FP16，BF16 模型保持 BF16。随后把同一个 `--dtype` 显式传给每张卡的副本，避免异构卡各自选择不同精度。其他实际禁止 FP16 的模型请显式设置 `DTYPE` 为可用精度。

官方配置和权重索引依据：
[config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)、
[weight index](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/model.safetensors.index.json)、
[vLLM DP 文档](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/)。启动时使用下载到本地的实际文件，离线模式同样可估算。

新模型族需要 parser 时，在 `MODEL_PROFILES` 里追加一条即可，例如：

```python
MODEL_PROFILES = {
    "qwen3_5": {"tool_call_parser": "qwen3_coder", "reasoning_parser": "qwen3",
                "mm_encoder_tp_mode": "data"},
    "qwen3":   {"tool_call_parser": "hermes", "reasoning_parser": "qwen3"},
}
```

没有 chat template 的模型仍可使用 `/v1/completions`。需要额外 vLLM 参数时用 `EXTRA_VLLM_ARGS`，例如：

```python
EXTRA_VLLM_ARGS = ["--disable-log-requests"]
```

---

## 8. 离线 / 内网运行

```python
ONLINE_LOAD = False
# 可选：显式指定完整模型目录；留空则从 HF 缓存找完整 snapshot
LOCAL_MODEL_PATH = ""
```

步骤：

```bash
# 联网环境、config 中保持 ONLINE_LOAD=True 时，先下载一次
.venv/bin/python download_model.py
# 把 config.py 改成 ONLINE_LOAD=False 后：
bash service.sh restart --wait
.venv/bin/python download_model.py --check-only       # 校验本地快照（不联网）
```

离线模式下会设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，不会发起任何 HF 网络请求。缓存缺失或分片不完整时会给出明确错误，提示先联网下载或设置 `LOCAL_MODEL_PATH`。

---

## 9. 驱动兼容：cuda-compat（可选）

驱动 550.163.01 最高支持 CUDA 12.4，而 vLLM 只提供 cu129/cu130 wheel。CUDA 12.x 的 minor-version compatibility 通常允许 cu129 应用运行在 550 驱动上；如果启动时报：

```
CUDA driver version is insufficient for CUDA runtime version
```

用 `fetch_cuda_compat.sh` 获取 **用户态** 前向兼容库（A100 属数据中心卡，支持该机制，无需 sudo）：

```bash
bash scripts/fetch_cuda_compat.sh
# 然后编辑 config.py：USE_CUDA_COMPAT = True
bash service.sh restart --wait
```

脚本只下载 `.deb` 并用 `dpkg-deb -x` 解压，不执行其中任何内容；`server.py` 会自动设置 `VLLM_ENABLE_CUDA_COMPATIBILITY=1`、`VLLM_CUDA_COMPATIBILITY_PATH` 和 `LD_LIBRARY_PATH`。若机器不能访问 NVIDIA 下载站，可先手动下载 `cuda-compat-12-9_575.57.08-0ubuntu1_amd64.deb`，再：

```bash
CUDA_COMPAT_URL=file:///path/to/cuda-compat-12-9_575.57.08-0ubuntu1_amd64.deb \
  bash scripts/fetch_cuda_compat.sh
```

长期方案仍是请管理员升级驱动到 580+（或安装系统级 `cuda-compat-12-9`）。

---

## 10. systemd user service（可选）

`~/.config/systemd/user/qwen38.service`：

```ini
[Unit]
Description=Qwen3.8-27B vLLM service
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/qinyuhang/model_service
ExecStart=/home/qinyuhang/model_service/.venv/bin/python /home/qinyuhang/model_service/server.py
Restart=on-failure
RestartSec=10
TimeoutStopSec=120
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now qwen38
systemctl --user status qwen38
journalctl --user -u qwen38 -f
```

需要在未登录时也运行：`sudo loginctl enable-linger $USER`（需要管理员权限）。

---

## 11. 排障

| 现象 | 处理 |
| --- | --- |
| `CUDA driver version is insufficient` | 见第 9 节 cuda-compat；或升级驱动 |
| 某张卡被跳过 / 加载失败 | 查看 `/replicas`、总日志和逐卡日志；检查空闲显存、计算能力和资源预留。真实加载 OOM 会自动降低该副本并发 |
| `fp8` KV 相关报错 | A100 不支持，`KV_CACHE_DTYPE="auto"` |
| 端口被占用 | `bash service.sh stop`，或修改 `PORT` |
| 模型下载慢 / 失败 | 检查 `HF_ENDPOINT`；`snapshot_download` 支持断点续传，重跑即可；gated 模型设 `HF_TOKEN` |
| `Using SOCKS proxy, but the 'socksio' package is not installed` | 环境里设置了 `ALL_PROXY=socks5://...`；`setup.sh` 已安装 `socksio`，也可 `uv pip install --python .venv/bin/python socksio` |
| `No chat template` | 该模型用 `/v1/completions`，或在 `EXTRA_VLLM_ARGS` 里指定 `--chat-template` |
| 请求超过上下文 | 调大 `MAX_MODEL_LEN`（不超过模型原生上限），或压缩输入 |
| Rust 前端异常 | `USE_RUST_FRONTEND = False` 后重启 |
| `libnvrtc.so.13: cannot open shared object file`（torchcodec） | PyPI 的 torchcodec 是 CUDA 13 构建；`setup.sh` 会自动卸载它，音频/视频回退 PyAV/OpenCV。也可手动 `uv pip uninstall --python .venv/bin/python torchcodec` |
| `FileNotFoundError: 'ninja'`（FlashInfer 采样 JIT） | venv 的 `bin` 不在 PATH；`server.py` 会自动把 `.venv/bin` 加进 PATH，`setup.sh` 会安装 ninja。手工运行 `vllm` 时请先 `source .venv/bin/activate` |
| 想看最终命令 | `.venv/bin/python server.py --dry-run`（会先确保模型已缓存/下载） |

### 验证改造

无需 GPU 的回归与真实本机 HTTP/子进程测试：

```bash
.venv/bin/python -m unittest discover -s deployment_tests -v
```

目标服务器维护窗口：

```bash
.venv/bin/python server.py --dry-run  # 每卡原因、预算、TP=1 独立命令
bash service.sh restart --wait
bash service.sh status
curl http://127.0.0.1:8000/replicas   # 启用 API_KEY 时加 Authorization
.venv/bin/python scripts/batch_chat.py --input prompts.jsonl --output dp-16.jsonl --concurrency 16 --no-thinking
.venv/bin/python scripts/batch_chat.py --input prompts.jsonl --output dp-32.jsonl --concurrency 32 --no-thinking
```

以同一请求集比较 aggregate output tokens/s、结果中的 elapsed_s 分位数，以及 `/replicas` 的分布；GPU/KV 利用率从 nvidia-smi 和各实例的 vLLM 日志/内部 `/metrics` 查看。网关 `/metrics` 不是多副本指标聚合器。

---

## 12. 安全说明

- 默认只监听 `127.0.0.1`，不对外网开放；
- 如需跨机器访问，建议用 SSH 隧道或反向代理，不要在公网直接暴露；
- 设置 `API_KEY` 后所有 `/v1` 请求都需要 Bearer token；
- 不要把 `HF_TOKEN`、`API_KEY` 提交到 git（`.gitignore` 已忽略日志、缓存和 venv）。
