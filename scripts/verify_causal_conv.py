"""Compare the optional CUDA causal-convolution kernels with PyTorch."""

from __future__ import annotations

import inspect
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.environment import (  # noqa: E402
    ROOT,
    block_python_network,
    configure_environment,
)


def main() -> int:
    configure_environment()
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": [],
    }
    try:
        with block_python_network() as attempts:
            import causal_conv1d
            import causal_conv1d_cuda
            import torch
            import torch.nn.functional as functional

            report.update(
                package_version=causal_conv1d.__version__,
                native_extension=causal_conv1d_cuda.__file__,
                torch=torch.__version__,
                cuda_runtime=torch.version.cuda,
                cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                gpu=torch.cuda.get_device_name(0),
            )
            torch.manual_seed(42)
            torch.backends.cudnn.allow_tf32 = False
            for dtype in (torch.float32, torch.bfloat16):
                x = torch.randn(
                    1,
                    64,
                    32,
                    device="cuda",
                    dtype=dtype,
                    requires_grad=True,
                )
                weight = torch.randn(
                    64,
                    4,
                    device="cuda",
                    dtype=dtype,
                    requires_grad=True,
                )
                bias = torch.randn(
                    64,
                    device="cuda",
                    dtype=dtype,
                    requires_grad=True,
                )
                reference_inputs = [
                    value.detach().clone().requires_grad_()
                    for value in (x, weight, bias)
                ]
                rx, rw, rb = reference_inputs
                actual = causal_conv1d.causal_conv1d_fn(
                    x, weight, bias, activation="silu"
                )
                expected = functional.silu(
                    functional.conv1d(
                        rx, rw.unsqueeze(1), rb, padding=3, groups=64
                    )[..., :32]
                )
                tolerance = (
                    {"rtol": 1e-4, "atol": 1e-5}
                    if dtype == torch.float32
                    else {"rtol": 0.03, "atol": 0.03}
                )
                torch.testing.assert_close(actual, expected, **tolerance)
                actual.float().square().mean().backward()
                expected.float().square().mean().backward()
                for value, reference in zip(
                    (x, weight, bias), reference_inputs, strict=True
                ):
                    torch.testing.assert_close(
                        value.grad, reference.grad, **tolerance
                    )

                # Incremental decoding updates the state buffer in place.
                state = torch.randn(1, 64, 4, device="cuda", dtype=dtype)
                step = torch.randn(1, 64, device="cuda", dtype=dtype)
                expected_state = torch.cat(
                    (state, step.unsqueeze(-1)), dim=-1
                )[..., -4:]
                with torch.no_grad():
                    expected_update = functional.silu(
                        functional.conv1d(
                            expected_state,
                            weight.unsqueeze(1),
                            bias,
                            groups=64,
                        )
                    ).squeeze(-1)
                    actual_update = causal_conv1d.causal_conv1d_update(
                        step, state, weight, bias, activation="silu"
                    )
                torch.testing.assert_close(
                    actual_update, expected_update, **tolerance
                )
                torch.testing.assert_close(
                    state, expected_state, rtol=0, atol=0
                )
                report["checks"].append(
                    {
                        "dtype": str(dtype),
                        "forward": "passed",
                        "backward": "passed",
                        "incremental_update": "passed",
                        "reference": "PyTorch grouped conv1d plus SiLU",
                        "tolerance": tolerance,
                    }
                )
            torch.cuda.synchronize()
            from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

            dispatch = {}
            for name in ("causal_conv1d_fn", "causal_conv1d_update"):
                closure = inspect.getclosurevars(getattr(qwen, name)).nonlocals
                native = closure.get("is_new_implementation") is True
                if not native:
                    raise RuntimeError(f"{name} still uses the reference path")
                dispatch[name] = {
                    "native_selected": native,
                    "implementation_module": closure[
                        "implementation"
                    ].__module__,
                }
            report["transformers_dispatch"] = dispatch
            report["python_network_attempts"] = attempts
            if attempts:
                raise RuntimeError("A network connection was attempted")
            report["status"] = "passed"
    except Exception as exc:
        report.update(
            status="failed", error_type=type(exc).__name__, error=str(exc)
        )
        traceback.print_exc()
    output = ROOT / "evaluation/reports/phase3-causal-conv.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
