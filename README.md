# Qwen3.8-27B 本地推理服务

一个配置驱动、可从任意目录启动的本地推理服务：

- 默认从 `https://hf-mirror.com` 加载 `Qwen/Qwen3.8-27B`；
- 使用 **物理卡 1**（`nvidia-smi` 的 0-based 编号，即第二张卡）；
- 在 `http://127.0.0.1:8000` 提供 **OpenAI 兼容 API**；
- 换模型、换卡、换加载地址、离线运行，只改 `config.py`，不改代码。

底层使用 **vLLM 0.32.0（cu129 wheel）**，模型架构、多模态、工具调用、张量并行都由 vLLM 处理。

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
├── server.py                 # 读配置 -> 解析模型 -> 组装 vllm serve -> exec
├── model_utils.py            # 模型路径、HF 缓存、config.json、上下文钳制
├── download_model.py         # 预下载 / 校验模型快照
├── setup.sh                  # 创建 .venv，安装 cu129 torch + vLLM，CUDA 冒烟测试
├── start.sh                  # 前台启动（任意 CWD）
├── service.sh                # 后台 start/stop/restart/status/logs
├── requirements.txt          # 版本参考（安装以 setup.sh 为准）
├── scripts/
│   ├── chat.py               # OpenAI SDK 示例（流式 / 图片 / thinking）
│   ├── smoke_test.sh         # /health + /v1/models + 一次 chat
│   └── fetch_cuda_compat.sh  # 可选：无 sudo 解压 cuda-compat-12-9
├── logs/                     # 运行日志（自动创建）
└── run/                      # PID 文件（自动创建）
```

---

## 3. 环境要求

- Linux + NVIDIA 驱动（本项目在 2× A100 80GB、驱动 550.163.01 上验证）；
- Python 3.12（`setup.sh` 默认用 `python3.12` 创建 venv）；
- 首次下载需要约 60 GB 空闲磁盘；
- 安装 `uv` 可选（有则更快），没有则回退到 `python -m venv` + `pip`。

单卡 A100 80GB 的显存估算（BF16）：

| `MAX_MODEL_LEN` | KV cache | 说明 |
| --- | --- | --- |
| 32K | ≈ 2 GiB | 最稳 |
| 64K（默认） | ≈ 4 GiB | 推荐，余量充足 |
| 128K | ≈ 8 GiB | 可跑，需观察峰值 |
| 262K | ≈ 16 GiB | 单卡余量很小，建议 `GPU_IDS=[0,1]` 开 TP2 |

权重约 51.75 GiB；A100 80GB 在 `nvidia-smi` 中约为 81920 MiB（80 GiB），`gpu_memory_utilization=0.90` 时预算约 72 GiB，剩余约 20 GiB 给 KV、激活、CUDA graph 和视觉塔。

本项目实测（A100 80GB、`GPU_MEMORY_UTILIZATION=0.90`）：权重加载 51.01 GiB，可用 KV cache 17.77 GiB（270,153 tokens），64K 请求可并发 4.12 路；卡 1 显存占用约 71 GiB。首次启动需要 FlashInfer JIT 编译采样内核（约 1–2 分钟），之后有缓存，重启约 1 分钟即可就绪。

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

默认值已满足需求：**卡 1、联网、`https://hf-mirror.com`、`127.0.0.1:8000`**。修改后重启服务生效。

### 5.1 模型与加载方式

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_ID` | `Qwen/Qwen3.8-27B` | HF repo id，或本地模型目录 |
| `SERVED_MODEL_NAME` | `Qwen/Qwen3.8-27B` | `/v1/models` 与请求里的 `model` 字段 |
| `ONLINE_LOAD` | `True` | `True` 联网；`False` 只用本地缓存/路径 |
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
| `GPU_IDS` | `[1]` | 物理卡号列表；默认第二张卡 |
| `TENSOR_PARALLEL_SIZE` | `"auto"` | 自动等于 `len(GPU_IDS)` |
| `GPU_MEMORY_UTILIZATION` | `0.90` | 显存占用上限 |
| `MAX_MODEL_LEN` | `65536` | 自动钳制到模型原生上限 |
| `MAX_NUM_SEQS` | `8` | 并发序列数 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 单批 token 数 |
| `DTYPE` | `"auto"` | Qwen3.8 为 bfloat16 |
| `KV_CACHE_DTYPE` | `"auto"` | A100 不要用 fp8 |
| `ENFORCE_EAGER` | `False` | 显存/图捕获异常时改 `True` |

多卡示例：

```python
GPU_IDS = [0, 1]          # 两张 A100，自动 TP=2
MAX_MODEL_LEN = 262144    # 此时才建议开原生 262K
```

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

1M 上下文的 KV cache 约 64 GiB（BF16）。单张 A100 80GB 放不下（权重 + KV 已超过 80 GiB），至少需要 `GPU_IDS=[0,1]` 开 TP2，并视显存把 `GPU_MEMORY_UTILIZATION` 调到 0.95 左右；显存不足时 vLLM 会拒绝启动并给出所需 KV 大小。

### 5.4 服务与兼容性

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | 监听地址 |
| `API_KEY` | `None` | 设置后需 `Authorization: Bearer ...` |
| `UVICORN_LOG_LEVEL` | `"info"` | 访问日志级别 |
| `USE_RUST_FRONTEND` | `True` | 官方 recipe 建议；异常时可改 `False` |
| `AUTO_MODEL_PROFILE` | `True` | 按 `model_type` 自动选 parser / 多模态参数 |
| `MODEL_PROFILES` | 见下 | 模型族 → parser 映射 |
| `TOOL_CALL_PARSER` | `"auto"` | `"auto"` / `None` / 显式名字 |
| `REASONING_PARSER` | `"auto"` | 同上 |
| `MM_ENCODER_TP_MODE` | `"auto"` | 多模态编码器并行方式 |
| `TRUST_REMOTE_CODE` | `True` | 兼容性优先 |
| `EXTRA_VLLM_ARGS` | `[]` | 原样追加任意 vLLM 参数 |
| `EXTRA_ENV` | `{}` | 原样注入任意环境变量 |

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

---

## 7. 换模型（兼容更多模型）

只改 `config.py`：

```python
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
SERVED_MODEL_NAME = "smollm2-135m"
MAX_MODEL_LEN = 2048          # 会被自动钳制到模型原生上限
GPU_MEMORY_UTILIZATION = 0.30
```

服务启动时会：

1. 读取模型本地 `config.json`；
2. 把 `MAX_MODEL_LEN` 钳制到 `max_position_embeddings`；
3. 按 `MODEL_PROFILES` / `model_type` 决定是否添加 Qwen 专用 parser；
4. 只有模型含 `vision_config` / `image_token_id` / `video_token_id` 等多模态字段时才加多模态参数；
5. 其他参数原样交给 vLLM，由 vLLM 负责架构支持。

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
| 启动 OOM / KV cache 不足 | 调低 `MAX_MODEL_LEN`、`GPU_MEMORY_UTILIZATION`、`MAX_NUM_SEQS`；确认卡 1 没有被其他进程占用 |
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

---

## 12. 安全说明

- 默认只监听 `127.0.0.1`，不对外网开放；
- 如需跨机器访问，建议用 SSH 隧道或反向代理，不要在公网直接暴露；
- 设置 `API_KEY` 后所有 `/v1` 请求都需要 Bearer token；
- 不要把 `HF_TOKEN`、`API_KEY` 提交到 git（`.gitignore` 已忽略日志、缓存和 venv）。
