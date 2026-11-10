"""单入口异步代理；只按请求负载路由，模型推理由独立 vLLM 负责。"""
from __future__ import annotations

import asyncio
import hmac
import json
import re
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import aiohttp
from aiohttp import web
from multidict import CIMultiDict
from yarl import URL


@dataclass
class Backend:
    name: str
    url: str
    capacity: float
    healthy: bool = False
    inflight: int = 0
    completed: int = 0
    failures: int = 0
    state: str = "starting"
    detail: str = ""


class BackendPool:
    """acquire/release 不 await；单事件循环内选择和占位是原子操作。"""
    AFFINITY_LIMIT = 10_000

    def __init__(self, backends: list[Backend]):
        self.backends = backends
        self.cursor = 0
        self.accepting = True
        self._affinity: OrderedDict[str, str] = OrderedDict()

    def bind(self, key: str | None, backend: Backend) -> None:
        """把有状态资源（例如 Responses API 的 response id）绑定到创建副本。"""
        if not key:
            return
        self._affinity[key] = backend.name
        self._affinity.move_to_end(key)
        while len(self._affinity) > self.AFFINITY_LIMIT:
            self._affinity.popitem(last=False)

    def preferred(self, key: str | None) -> str | None:
        if not key:
            return None
        name = self._affinity.get(key)
        if name is None:
            return None
        self._affinity.move_to_end(key)
        return name

    def acquire(self, exclude: set[str] | None = None,
                preferred: str | None = None) -> Backend | None:
        if not self.accepting:
            return None
        ordered = self.backends[self.cursor:] + self.backends[:self.cursor]
        ready = [b for b in ordered if b.healthy and b.name not in (exclude or set())]
        if not ready:
            return None
        if preferred:
            chosen = next((b for b in ready if b.name == preferred), None)
        else:
            chosen = None
        backend = chosen or min(ready, key=lambda b: b.inflight / b.capacity)
        backend.inflight += 1
        self.cursor = (self.backends.index(backend) + 1) % len(self.backends)
        return backend

    def release(self, backend: Backend, *, success: bool = True) -> None:
        backend.inflight -= 1
        if success:
            backend.completed += 1
        else:
            backend.failures += 1

    def status(self) -> list[dict]:
        return [{"name": b.name, "healthy": b.healthy, "state": b.state, "detail": b.detail,
                 "capacity_weight": b.capacity, "inflight": b.inflight,
                 "completed": b.completed, "failures": b.failures} for b in self.backends]


_HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailer", "transfer-encoding", "upgrade", "host"}


def _headers(headers) -> CIMultiDict:
    blocked = _HOP_HEADERS | {h.strip().lower() for h in headers.get("Connection", "").split(",")}
    return CIMultiDict((key, value) for key, value in headers.items() if key.lower() not in blocked)


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": {"message": message, "type": "service_unavailable",
                                         "code": status}}, status=status)


_JSON_INSPECT_LIMIT = 1024 * 1024
_AFFINITY_CAPTURE_LIMIT = 64 * 1024
_RESPONSES_PATH = re.compile(r"^/v1/responses/([^/]+)(?:/cancel)?/?$")


def _json_payload(body: bytes, content_encoding: str | None) -> dict | None:
    encoding = (content_encoding or "").strip().lower()
    raw = body
    if encoding == "gzip":
        try:
            raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(body, _JSON_INSPECT_LIMIT)
        except zlib.error:
            return None
    elif encoding not in ("", "identity"):
        return None
    if len(raw) > _JSON_INSPECT_LIMIT:
        return None
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _request_affinity_key(request: web.Request, body: bytes) -> str | None:
    path = request.path.rstrip("/") or "/"
    if request.method == "POST" and path == "/v1/responses":
        payload = _json_payload(body, request.headers.get("Content-Encoding"))
        key = payload.get("previous_response_id") if payload else None
        return key if isinstance(key, str) and key else None
    match = _RESPONSES_PATH.match(path)
    return match.group(1) if match else None


def _extract_response_id(data: bytes) -> str | None:
    """从 JSON 响应或 SSE response.created 事件中提取顶层响应 ID。"""
    text = data.decode("utf-8", errors="ignore")
    for block in text.split("\n\n"):
        event = None
        payload = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                payload = line[5:].strip()
        if event != "response.created" or not payload:
            continue
        try:
            created = json.loads(payload)
        except ValueError:
            continue
        response = created.get("response") if isinstance(created, dict) else None
        if isinstance(response, dict) and isinstance(response.get("id"), str):
            return response["id"]
    match = re.search(r'"id"\s*:\s*"(resp_[^"]+)"', text)
    return match.group(1) if match else None


def create_app(pool: BackendPool, *, api_key: str | None = None,
               status_provider: Callable[[], list[dict]] | None = None,
               connect_timeout: float = 5, read_timeout: float = 600,
               max_body_mib: int = 64) -> web.Application:
    @web.middleware
    async def auth(request, handler):
        if api_key and request.path != "/health":
            actual = request.headers.get("Authorization", "").encode()
            if not hmac.compare_digest(actual, f"Bearer {api_key}".encode()):
                return _error(401, "Invalid API key")
        return await handler(request)

    # 入站和出站都保持原始编码字节，Content-Encoding/Length 才与转发体一致。
    app = web.Application(middlewares=[auth], client_max_size=max_body_mib * 1024 ** 2,
                          handler_args={"auto_decompress": False})
    session: aiohttp.ClientSession

    async def client_context(app):
        nonlocal session
        # total=None 允许长流式响应，sock_read 限制无数据等待；本机不走环境代理。
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=connect_timeout, sock_read=read_timeout)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False, auto_decompress=False,
                connector=aiohttp.TCPConnector(limit=0),
                skip_auto_headers={"Accept-Encoding", "User-Agent"}) as session:
            yield

    app.cleanup_ctx.append(client_context)

    async def health(request):
        ready = sum(b.healthy for b in pool.backends)
        return web.json_response({"status": "ok" if ready and pool.accepting else "unavailable",
                                  "ready_replicas": ready, "total_replicas": len(pool.backends)},
                                 status=200 if ready and pool.accepting else 503)

    async def replicas(request):
        return web.json_response({"backends": pool.status(),
                                  "gpus": status_provider() if status_provider else []})

    async def proxy(request):
        body = await request.read()
        affinity_key = _request_affinity_key(request, body)
        creates_response = (request.method == "POST"
                            and (request.path.rstrip("/") or "/") == "/v1/responses")
        tried: set[str] = set()
        while (backend := pool.acquire(tried, preferred=pool.preferred(affinity_key))) is not None:
            tried.add(backend.name)
            downstream = None
            writing_downstream = False
            success = False
            capture = bytearray()
            capture_enabled = creates_response
            try:
                # raw_path 保留 URL 编码和查询；禁止重定向及代理本机以外的目标。
                url = URL(backend.url.rstrip("/") + request.raw_path, encoded=True)
                async with session.request(request.method, url, data=body, headers=_headers(request.headers),
                                           allow_redirects=False) as upstream:
                    if upstream.status in {502, 503, 504}:
                        backend.healthy = False
                        backend.detail = f"upstream HTTP {upstream.status}"
                    if upstream.status >= 400:
                        capture_enabled = False
                    downstream = web.StreamResponse(status=upstream.status, reason=upstream.reason,
                                                    headers=_headers(upstream.headers))
                    writing_downstream = True
                    await downstream.prepare(request)
                    writing_downstream = False
                    async for chunk in upstream.content.iter_any():
                        if capture_enabled:
                            capture.extend(chunk)
                            response_id = _extract_response_id(bytes(capture))
                            if response_id:
                                # 在把 ID 写给客户端前登记，避免紧随其后的查询竞态。
                                pool.bind(response_id, backend)
                                capture_enabled = False
                                capture.clear()
                            elif len(capture) > _AFFINITY_CAPTURE_LIMIT:
                                capture_enabled = False
                                capture.clear()
                        writing_downstream = True
                        await downstream.write(chunk)
                        writing_downstream = False
                    writing_downstream = True
                    await downstream.write_eof()
                    writing_downstream = False
                    success = upstream.status < 500
                    return downstream
            except aiohttp.ClientConnectorError as exc:
                # 只在 TCP 连接尚未建立时重试；读超时/断流可能已执行，绝不重放。
                backend.healthy = False
                backend.detail = f"connect failed: {exc}"
                continue
            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as exc:
                if not writing_downstream:
                    backend.healthy = False
                    backend.detail = f"upstream failed: {type(exc).__name__}"
                if downstream is not None and downstream.prepared:
                    if request.transport:
                        request.transport.close()  # 中断响应，而不是伪造成功 EOF/第二个 HTTP 响应
                    return downstream
                return _error(504 if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) else 502,
                              "Upstream request failed")
            finally:
                # AppRunner(handler_cancellation=True) 使客户端断开时也走这里。
                pool.release(backend, success=success)
        return _error(503, "No healthy model replicas")

    app.router.add_get("/health", health)
    app.router.add_get("/replicas", replicas)
    app.router.add_route("*", "/{path:.*}", proxy)
    return app
