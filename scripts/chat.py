#!/usr/bin/env python3
"""OpenAI 兼容客户端示例：文本 / 流式 / 图片 / thinking 开关。

示例：
    .venv/bin/python scripts/chat.py "用一句话解释什么是张量并行"
    .venv/bin/python scripts/chat.py --stream --no-thinking "写一个快速排序"
    .venv/bin/python scripts/chat.py --image ./cat.jpg "描述这张图片"
    .venv/bin/python scripts/chat.py --reasoning-effort high "证明根号2是无理数"
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import sys
from pathlib import Path


def get_reasoning(message) -> str | None:
    """兼容 vLLM 新旧字段：0.32 起用 reasoning，旧版本用 reasoning_content。"""
    return getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)


def build_content(prompt: str, images: list[str]) -> str | list[dict]:
    if not images:
        return prompt
    content: list[dict] = [{"type": "text", "text": prompt}]
    for image in images:
        path = Path(image).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"图片不存在：{path}")
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
        )
    return content


def main() -> int:
    parser = argparse.ArgumentParser(description="调用本地 vLLM 服务")
    parser.add_argument("prompt", nargs="*", help="用户问题")
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3.8-27B"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--image", action="append", default=[], help="本地图片路径，可重复")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--no-thinking", action="store_true", help="关闭 thinking")
    parser.add_argument("--reasoning-effort", choices=["xhigh", "medium", "low"])
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--show-reasoning", action="store_true", help="打印 thinking 内容")
    parser.add_argument("--system", default=None)
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("缺少 openai 包，请运行：.venv/bin/python -m pip install openai")

    prompt = " ".join(args.prompt) or "你好，请用一句话介绍你自己。"
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": build_content(prompt, args.image)})

    template_kwargs: dict = {}
    if args.no_thinking:
        template_kwargs["enable_thinking"] = False

    temperature = args.temperature
    top_p = args.top_p
    if temperature is None:
        temperature = 0.7 if args.no_thinking else 1.0
    if top_p is None:
        top_p = 0.80 if args.no_thinking else 0.95

    extra_body: dict = {
        "top_k": 20,
        "chat_template_kwargs": template_kwargs,
    }
    if args.no_thinking:
        extra_body["presence_penalty"] = 1.5

    kwargs = dict(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body=extra_body,
    )
    if args.reasoning_effort:
        # vLLM 0.32 支持顶层的 reasoning_effort，并会同步传入 chat template。
        kwargs["reasoning_effort"] = args.reasoning_effort

    if args.stream:
        kwargs["stream"] = True
        reasoning_started = False
        content_started = False
        for chunk in client.chat.completions.create(**kwargs):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = get_reasoning(delta)
            if reasoning and args.show_reasoning:
                if not reasoning_started:
                    print("--- thinking ---")
                    reasoning_started = True
                print(reasoning, end="", flush=True)
            if delta.content:
                if reasoning_started and not content_started:
                    print("\n--- answer ---")
                content_started = True
                print(delta.content, end="", flush=True)
        print()
        return 0

    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message
    reasoning = get_reasoning(message)
    if reasoning and args.show_reasoning:
        print("--- thinking ---")
        print(reasoning)
        print("--- answer ---")
    print(message.content or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
