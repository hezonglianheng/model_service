"""独立 vLLM 副本的启动验证、健康探测和进程组回收。"""
from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, IO

import aiohttp

import config
from gateway import Backend, BackendPool
from gpu_planner import GPUDecision, sequence_candidates


def log(message: str) -> None:
    print(f"[replicas] {message}", file=sys.stderr, flush=True)


@dataclass
class Replica:
    decision: GPUDecision
    backend: Backend
    port: int
    process: subprocess.Popen | None = None
    attempts: int = 0
    log_file: IO | None = None
    log_path: Path | None = None
    log_offset: int = 0


class ReplicaManager:
    def __init__(self, decisions: list[GPUDecision], command: Callable[[GPUDecision, int], list[str]],
                 environment: Callable[[GPUDecision], dict[str, str]], *, log_dir: Path,
                 port_start: int = 8100, startup_timeout: float = 1800,
                 health_interval: float = 5, health_timeout: float = 3,
                 shutdown_timeout: float = 20, api_key: str | None = None):
        self.decisions = decisions
        self.command = command
        self.environment = environment
        self.log_dir = log_dir
        self.startup_timeout = startup_timeout
        self.health_interval = health_interval
        self.health_timeout = health_timeout
        self.shutdown_timeout = shutdown_timeout
        self.api_key = api_key
        self.replicas: list[Replica] = []
        for d in decisions:
            if not d.eligible:
                continue
            port = port_start + len(self.replicas) if port_start else 0
            backend = Backend(f"gpu{d.gpu.index}", f"http://127.0.0.1:{port}",
                              d.max_num_seqs * d.routing_weight)
            self.replicas.append(Replica(d, backend, port))
        self.pool = BackendPool([r.backend for r in self.replicas])
        self.all_failed = asyncio.Event()
        self.tasks: list[asyncio.Task] = []
        self.client: aiohttp.ClientSession | None = None
        self.closed = False

    async def start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.client = aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=self.health_timeout))
        for replica in self.replicas:
            self.tasks.append(asyncio.create_task(self._watch(replica)))
        if not self.replicas:
            self.all_failed.set()

    def status(self) -> list[dict]:
        active = {r.decision.gpu.index: r for r in self.replicas}
        result = []
        for decision in self.decisions:
            replica = active.get(decision.gpu.index)
            d = replica.decision if replica else decision
            result.append({"gpu": d.gpu.index, "uuid": d.gpu.uuid, "name": d.gpu.name,
                           "eligible": d.eligible, "reason": d.reason,
                           "memory_utilization": d.memory_utilization,
                           "max_model_len": d.max_model_len, "max_num_seqs": d.max_num_seqs,
                           "port": replica.port if replica else None,
                           "attempts": replica.attempts if replica else 0,
                           "log": str(replica.log_path) if replica else None})
        return result

    def _spawn(self, replica: Replica) -> None:
        # 检测已有监听，避免将无关服务的 /health 当成该副本就绪。
        # POSIX 允许复用 TIME_WAIT，否则正常重启会被旧连接误判为端口占用；
        # Windows 的 SO_REUSEADDR 会抢占活跃监听，改用排他绑定。
        with socket.socket() as sock:
            if os.name == "posix":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            elif hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            sock.bind(("127.0.0.1", replica.port))
            replica.port = sock.getsockname()[1]
        replica.backend.url = f"http://127.0.0.1:{replica.port}"
        replica.attempts += 1
        replica.log_path = self.log_dir / f"gpu-{replica.decision.gpu.index}-attempt-{replica.attempts}.log"
        replica.log_file = replica.log_path.open("ab", buffering=0)
        replica.log_offset = replica.log_file.tell()
        # 中间守护进程设置 parent-death signal；supervisor 被强杀时仍会回收 GPU 进程组。
        wrapped = [sys.executable, str(Path(__file__).with_name("replica_runner.py"))]
        wrapped += self.command(replica.decision, replica.port)
        env = self.environment(replica.decision)
        env.setdefault("REPLICA_CHILD_GRACE_SECONDS", str(config.REPLICA_CHILD_GRACE_SECONDS))
        replica.process = subprocess.Popen(wrapped,
            env=env, stdin=subprocess.DEVNULL,
            stdout=replica.log_file, stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"))
        log(f"{replica.backend.name} PID={replica.process.pid} port={replica.port} "
            f"seqs={replica.decision.max_num_seqs}，日志：{replica.log_path}")

    def _tail(self, replica: Replica) -> str:
        if not replica.log_path:
            return ""
        with replica.log_path.open("rb") as stream:
            stream.seek(max(replica.log_offset, replica.log_path.stat().st_size - 16384))
            return stream.read().decode("utf-8", errors="replace")

    async def _health(self, replica: Replica) -> bool:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        try:
            async with self.client.get(replica.backend.url + "/health", headers=headers,
                                       allow_redirects=False) as response:
                return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def _terminate(self, replica: Replica) -> None:
        process = replica.process
        if process is not None:
            if os.name != "posix":
                # Windows 没有进程组信号：taskkill /T 同时结束守护进程和 vLLM 子进程。
                if process.poll() is None:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True, check=False)
                await asyncio.to_thread(process.wait)
            else:
                def send(sig):
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        pass

                def alive():
                    process.poll()  # 回收 leader；同组子进程可能仍然存活。
                    try:
                        os.killpg(process.pid, 0)
                        return True
                    except ProcessLookupError:
                        return False

                send(signal.SIGTERM)
                deadline = time.monotonic() + self.shutdown_timeout
                while alive() and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                if alive():
                    send(signal.SIGKILL)
                await asyncio.to_thread(process.wait)
        if replica.log_file:
            replica.log_file.close()
            replica.log_file = None

    async def _watch(self, replica: Replica) -> None:
        backend = replica.backend
        try:
            for seqs in sequence_candidates(replica.decision.max_num_seqs):
                replica.decision = replace(replica.decision, max_num_seqs=seqs)
                backend.capacity = seqs * replica.decision.routing_weight
                backend.state = "starting"
                self._spawn(replica)
                started = heartbeat = time.monotonic()
                was_ready = False
                while replica.process.poll() is None:
                    ready = await self._health(replica)
                    if ready and replica.process.poll() is None:
                        if not backend.healthy:
                            log(f"{backend.name} 就绪，seqs={seqs}")
                        was_ready = backend.healthy = True
                        backend.state, backend.detail = "ready", ""
                    else:
                        backend.healthy = False
                        backend.state = "unhealthy" if was_ready else "starting"
                    now = time.monotonic()
                    if not was_ready and now - started >= self.startup_timeout:
                        backend.detail = f"启动健康检查超时\n{self._tail(replica)[-2000:]}"
                        break
                    if now - heartbeat >= 30:
                        log(f"{backend.name} state={backend.state}，日志：{replica.log_path}")
                        heartbeat = now
                    await asyncio.sleep(self.health_interval)
                backend.healthy = False
                tail = self._tail(replica)
                backend.detail = backend.detail or f"进程退出：{replica.process.poll()}\n{tail[-2000:]}"
                await self._terminate(replica)
                memory_failure = any(marker in tail.lower() for marker in (
                    "out of memory", "outofmemory", "not enough memory", "free memory",
                    "gpu memory utilization", "no available memory for the cache blocks",
                    "larger than the available kv cache memory", "kv cache is needed"))
                if was_ready or not memory_failure or seqs == sequence_candidates(seqs)[-1]:
                    break
                log(f"{backend.name} 加载显存不足，降低并发重试（上下文保持不变）")
                backend.detail = ""
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            backend.detail = f"启动/监控失败：{type(exc).__name__}: {exc!r}"
            await self._terminate(replica)
        backend.healthy = False
        backend.state = "failed"
        log(f"{backend.name} 已移出：{backend.detail}")
        if all(b.state == "failed" for b in self.pool.backends):
            self.all_failed.set()

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.pool.accepting = False
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await asyncio.gather(*(self._terminate(r) for r in self.replicas))
        for backend in self.pool.backends:
            backend.healthy = False
            backend.state = "stopped"
        if self.client:
            await self.client.close()
