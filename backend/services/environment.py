"""Process-local offline defaults and cache locations."""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]


def configure_environment(*, offline: bool = True) -> None:
    """Configure this process without changing Windows or system settings."""
    os.environ.update(
        {
            "HF_HOME": str(ROOT / ".cache/huggingface"),
            "XDG_CACHE_HOME": str(ROOT / ".cache"),
            "TRITON_CACHE_DIR": str(ROOT / ".cache/triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(ROOT / ".cache/torchinductor"),
            "UNSLOTH_COMPILE_LOCATION": str(
                ROOT / ".cache/unsloth_compiled_cache"
            ),
            "HF_HUB_OFFLINE": "1" if offline else "0",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "UNSLOTH_DISABLE_STATISTICS": "1",
            "DO_NOT_TRACK": "1",
            "WANDB_MODE": "disabled",
            "TOKENIZERS_PARALLELISM": "false",
            "IPYTHONDIR": str(ROOT / ".cache/ipython"),
        }
    )


@contextmanager
def block_python_network() -> Iterator[list[str]]:
    """Record and reject Python network access during offline verification."""
    attempts: list[str] = []
    originals = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.getaddrinfo,
    )

    def blocked(*args: Any, **kwargs: Any) -> Any:
        attempts.append("Python socket connection or DNS lookup")
        raise OSError("Network access disabled for local model verification")

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.getaddrinfo = blocked
    try:
        yield attempts
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = (
            originals
        )
