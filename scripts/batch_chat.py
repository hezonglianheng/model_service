#!/usr/bin/env python3
"""并发批处理客户端：多个输入并行请求，每个输入可返回多个候选。

服务端由 vLLM 的 continuous batching 负责真正的并行推理；本脚本负责用同一个
HTTP 连接池并发提交请求、限制在途数量，并按输入顺序落盘结果。

输入 JSONL，每行一个对象（也接受裸字符串）：
    {"id": "q1", "prompt": "用一句话解释张量并行"}
    {"id": "q2", "messages": [{"role": "user", "content": "写一个快排"}], "n": 2}
    {"id": "q3", "prompt": "...", "system": "...", "max_tokens": 256}

示例：
    .venv/bin/python scripts/batch_chat.py --input prompts.jsonl --concurrency 4
    .venv/bin/python scripts/batch_chat.py --input prompts.jsonl --n 2 --no-thinking
    .venv/bin/python scripts/batch_chat.py --input prompts.jsonl --output out.jsonl

结果 JSONL 每行：
    {"id": "...", "ok": true, "choices": [...], "usage": {...}, "elapsed_s": 1.23}
    {"id": "...", "ok": false, "error": "..."}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Callable, IO

DEFAULT_BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_MODEL = os.environ.get("MODEL", "Qwen/Qwen3.8-27B")


def get_reasoning(message: Any) -> str | None:
    """兼容 vLLM 新旧字段：0.32 起用 reasoning，旧版本用 reasoning_content。"""
    return getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)


def load_items(stream: IO[str], source: str = "<stdin>") -> list[dict[str, Any]]:
    """读取 JSONL；每项补充唯一可读的 id（未显式给出时用行号）。"""
    items: list[dict[str, Any]] = []
    for lineno, line in enumerate(stream, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{lineno} JSON 解析失败：{exc}") from exc

        if isinstance(item, str):
            item = {"prompt": item}
        if not isinstance(item, dict):
            raise ValueError(f"{source}:{lineno} 每行必须是 JSON 对象或字符串")
        if "messages" not in item and "prompt" not in item:
            raise ValueError(f"{source}:{lineno} 缺少 prompt 或 messages 字段")
        if "messages" in item and not isinstance(item["messages"], list):
            raise ValueError(f"{source}:{lineno} messages 必须是数组")
        if "prompt" in item and not isinstance(item["prompt"], str):
            raise ValueError(f"{source}:{lineno} prompt 必须是字符串")

        item = dict(item)
        item.setdefault("id", f"line-{lineno}")
        items.append(item)
    return items


def build_messages(item: dict[str, Any], default_system: str | None = None) -> list[dict[str, Any]]:
    """把 prompt/messages 统一成 OpenAI messages；不修改输入对象。

    messages 表示完全透传，不能同时再给 prompt/system；default_system 只用于 prompt 输入。
    """
    if "messages" in item:
        if "prompt" in item:
            raise ValueError(f"id={item.get('id')!r}: prompt 与 messages 不能同时出现")
        if item.get("system"):
            raise ValueError(
                f"id={item.get('id')!r}: 使用 messages 时请把 system 写进 messages，"
                "不要再给 system 字段"
            )
        return [dict(message) for message in item["messages"]]

    system = item.get("system", default_system)
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": item["prompt"]})
    return messages


def build_kwargs(
    item: dict[str, Any],
    *,
    model: str,
    n: int = 1,
    max_tokens: int = 1024,
    temperature: float | None = None,
    top_p: float | None = None,
    no_thinking: bool = False,
    system: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """合并全局默认值与单条输入覆盖项，返回 chat.completions.create 参数。"""
    item_no_thinking = bool(item.get("no_thinking", no_thinking))
    item_n = int(item.get("n", n))
    item_max_tokens = int(item.get("max_tokens", max_tokens))
    if item_n < 1:
        raise ValueError(f"id={item.get('id')!r}: n 必须 >= 1")
    if item_max_tokens < 1:
        raise ValueError(f"id={item.get('id')!r}: max_tokens 必须 >= 1")

    item_temperature = item.get("temperature", temperature)
    item_top_p = item.get("top_p", top_p)
    if item_temperature is None:
        item_temperature = 0.7 if item_no_thinking else 1.0
    if item_top_p is None:
        item_top_p = 0.80 if item_no_thinking else 0.95

    extra_body: dict[str, Any] = {"top_k": 20, "chat_template_kwargs": {}}
    if item_no_thinking:
        extra_body["chat_template_kwargs"]["enable_thinking"] = False
        extra_body["presence_penalty"] = 1.5

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": build_messages(item, system),
        "max_tokens": item_max_tokens,
        "temperature": item_temperature,
        "top_p": item_top_p,
        "n": item_n,
        "extra_body": extra_body,
    }
    effort = item.get("reasoning_effort", reasoning_effort)
    if effort:
        kwargs["reasoning_effort"] = effort
    return kwargs


def make_result(item: dict[str, Any], response: Any, elapsed: float) -> dict[str, Any]:
    """把 SDK 响应转成可落盘的普通 dict，保留全部候选。"""
    choices: list[dict[str, Any]] = []
    for position, choice in enumerate(getattr(response, "choices", None) or []):
        message = getattr(choice, "message", None)
        choices.append(
            {
                "index": getattr(choice, "index", position),
                "content": getattr(message, "content", None) if message is not None else None,
                "reasoning": get_reasoning(message) if message is not None else None,
                "finish_reason": getattr(choice, "finish_reason", None),
            }
        )

    usage = getattr(response, "usage", None)
    usage_dict = None
    if usage is not None:
        usage_dict = {
            key: getattr(usage, key, None)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }

    return {
        "id": item["id"],
        "ok": True,
        "choices": choices,
        "usage": usage_dict,
        "elapsed_s": round(elapsed, 4),
    }


async def run_batch(
    client: Any,
    items: list[dict[str, Any]],
    *,
    concurrency: int = 4,
    model: str = DEFAULT_MODEL,
    n: int = 1,
    max_tokens: int = 1024,
    temperature: float | None = None,
    top_p: float | None = None,
    no_thinking: bool = False,
    system: str | None = None,
    reasoning_effort: str | None = None,
    retries: int = 2,
    retry_backoff: float = 1.0,
) -> list[dict[str, Any]]:
    """并发执行全部输入；结果顺序与输入一致，单条失败不影响其它请求。"""
    if concurrency < 1:
        raise ValueError("concurrency 必须 >= 1")
    if retries < 0:
        raise ValueError("retries 必须 >= 0")

    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(item: dict[str, Any]) -> dict[str, Any]:
        try:
            kwargs = build_kwargs(
                item,
                model=model,
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                no_thinking=no_thinking,
                system=system,
                reasoning_effort=reasoning_effort,
            )
        except (TypeError, ValueError) as exc:
            return {"id": item.get("id"), "ok": False, "error": str(exc)}

        async with semaphore:
            last_error: Exception | None = None
            for attempt in range(retries + 1):
                started = time.perf_counter()
                try:
                    response = await client.chat.completions.create(**kwargs)
                    return make_result(item, response, time.perf_counter() - started)
                except Exception as exc:  # 单条请求失败不应中断整批
                    last_error = exc
                    if attempt < retries:
                        await asyncio.sleep(retry_backoff * (2**attempt))

            return {
                "id": item["id"],
                "ok": False,
                "error": f"{type(last_error).__name__}: {last_error}"[:2000],
            }

    return list(await asyncio.gather(*(run_one(item) for item in items)))


def write_results(results: list[dict[str, Any]], output: IO[str]) -> None:
    for result in results:
        output.write(json.dumps(result, ensure_ascii=False) + "\n")
    output.flush()


def summarize(results: list[dict[str, Any]], wall_seconds: float) -> str:
    ok = sum(1 for r in results if r.get("ok"))
    failed = len(results) - ok
    completion_tokens = sum(
        int((r.get("usage") or {}).get("completion_tokens") or 0) for r in results if r.get("ok")
    )
    rate = len(results) / wall_seconds if wall_seconds > 0 else 0.0
    token_rate = completion_tokens / wall_seconds if wall_seconds > 0 else 0.0
    return (
        f"[batch] 完成 {len(results)} 条：成功 {ok}，失败 {failed}；"
        f"耗时 {wall_seconds:.2f}s，{rate:.2f} 请求/s，{token_rate:.1f} 输出 token/s"
    )


def open_input(path: str) -> tuple[IO[str], Callable[[], None]]:
    if path == "-":
        return sys.stdin, lambda: None
    handle = open(path, "r", encoding="utf-8")
    return handle, handle.close


def open_output(path: str) -> tuple[IO[str], Callable[[], None]]:
    if path == "-":
        return sys.stdout, lambda: None
    handle = open(path, "w", encoding="utf-8")
    return handle, handle.close


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="并发调用本地 vLLM 服务（多输入 / 多输出）")
    parser.add_argument("--input", required=True, help="输入 JSONL；- 表示 stdin")
    parser.add_argument("--output", default="-", help="输出 JSONL；- 表示 stdout（默认）")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--concurrency", type=int, default=4, help="在途请求上限（默认 4）")
    parser.add_argument("--n", type=int, default=1, help="每个输入返回的候选数（默认 1）")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--no-thinking", action="store_true", help="关闭 thinking")
    parser.add_argument("--reasoning-effort", choices=["xhigh", "medium", "low"])
    parser.add_argument("--system", default=None, help="默认 system 提示，仅对 prompt 输入生效")
    parser.add_argument("--retries", type=int, default=2, help="单条请求重试次数（默认 2）")
    parser.add_argument("--retry-backoff", type=float, default=1.0, help="重试退避基数秒")
    parser.add_argument("--timeout", type=float, default=600.0, help="单请求超时秒数")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.concurrency < 1:
        raise SystemExit("--concurrency 必须 >= 1")
    if args.n < 1:
        raise SystemExit("--n 必须 >= 1")

    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise SystemExit("缺少 openai 包，请运行：.venv/bin/python -m pip install openai")

    try:
        input_handle, close_input = open_input(args.input)
        try:
            items = load_items(input_handle, args.input)
        finally:
            close_input()
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))

    if not items:
        print("[batch] 输入为空", file=sys.stderr)
        return 0

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=0,  # 重试由 run_batch 控制，避免 SDK 与脚本重复放大
    )

    started = time.perf_counter()
    results = asyncio.run(
        run_batch(
            client,
            items,
            concurrency=args.concurrency,
            model=args.model,
            n=args.n,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            no_thinking=args.no_thinking,
            system=args.system,
            reasoning_effort=args.reasoning_effort,
            retries=args.retries,
            retry_backoff=args.retry_backoff,
        )
    )
    wall_seconds = time.perf_counter() - started

    output_handle, close_output = open_output(args.output)
    try:
        write_results(results, output_handle)
    finally:
        close_output()

    print(summarize(results, wall_seconds), file=sys.stderr)
    return 0 if all(result.get("ok") for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
