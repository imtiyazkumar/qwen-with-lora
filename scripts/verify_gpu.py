"""Verify PyTorch CUDA computation without downloading a model."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def verify() -> dict[str, Any]:
    """Run a small, deterministic forward and backward pass on the GPU."""
    import torch

    result: dict[str, Any] = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "interpreter": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not result["cuda_available"]:
        raise RuntimeError(f"PyTorch cannot access CUDA: {result}")

    properties = torch.cuda.get_device_properties(0)
    result.update(
        gpu=properties.name,
        compute_capability=f"{properties.major}.{properties.minor}",
        vram_total_mib=round(properties.total_memory / 1024**2, 1),
    )
    torch.cuda.reset_peak_memory_stats()
    # Keep this test small even when Windows has other applications open.
    x = torch.ones((64, 64), device="cuda", requires_grad=True)
    y = x @ x
    torch.testing.assert_close(y, torch.full_like(y, 64))
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 128))
    torch.cuda.synchronize()
    result["forward_pass"] = "passed"
    result["backward_pass"] = "passed"
    result["peak_torch_allocated_mib"] = round(
        torch.cuda.max_memory_allocated() / 1024**2, 2
    )
    free, _total = torch.cuda.mem_get_info()
    result["vram_free_mib"] = round(free / 1024**2, 1)
    result["status"] = "passed"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Save a JSON verification report."
    )
    args = parser.parse_args()
    try:
        result = verify()
    except Exception as exc:
        result = {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
