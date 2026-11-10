"""vLLM 副本守护进程：让父进程崩溃/被强杀时也能回收整个 GPU 进程组。

由 ReplicaManager 以独立 session 启动；vLLM 子进程继承同一进程组。
Linux 使用 PR_SET_PDEATHSIG。正常的 SIGTERM 仍由 manager 发给整个进程组。
"""
from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time

PR_SET_PDEATHSIG = 1
GRACE_SECONDS = float(os.environ.get("REPLICA_CHILD_GRACE_SECONDS", "15"))


def _set_parent_death_signal() -> bool:
    if sys.platform != "linux":
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM) != 0:
            return False
    except (OSError, AttributeError):
        return False
    # prctl 与 exec 之间存在竞态：父进程可能已经退出。
    return os.getppid() != 1


def _kill_group_after_stop(process: subprocess.Popen) -> None:
    """先让直接子进程优雅退出，再清理它可能遗留的 worker。

    vLLM 的 worker 可能忽略 SIGTERM；即使直接子进程已退出，也必须清理同组残员。
    只有本进程是 session/group leader 时才 killpg，避免误杀 manager 的进程组。
    """
    if process.poll() is None:
        try:
            process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + GRACE_SECONDS
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
    if os.getpgrp() == os.getpid():
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
        process.wait()


def main() -> int:
    if len(sys.argv) < 2:
        print("[replica-runner] 缺少子进程命令", file=sys.stderr)
        return 2
    inherited = _set_parent_death_signal()
    stopping = False

    def stop(signum, _frame):
        nonlocal stopping
        if stopping:
            return
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    process = subprocess.Popen(sys.argv[1:], stdin=subprocess.DEVNULL,
                               stdout=sys.stdout, stderr=sys.stderr)
    if not inherited and sys.platform == "linux" and os.getppid() == 1:
        _kill_group_after_stop(process)
        return 1
    while True:
        try:
            code = process.wait(timeout=0.2)
            if stopping:
                _kill_group_after_stop(process)
                return process.poll() or 0
            return code
        except subprocess.TimeoutExpired:
            if stopping:
                _kill_group_after_stop(process)
                return process.poll() or 0


if __name__ == "__main__":
    raise SystemExit(main())
