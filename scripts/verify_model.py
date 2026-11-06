"""Verify the downloaded model in a fresh, offline-configured process."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.inference.runtime import LocalModel, ModelSettings  # noqa: E402
from backend.services.environment import (  # noqa: E402
    ROOT,
    block_python_network,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation/reports/phase3-model.json",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    settings = ModelSettings.read()
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": settings.repo_id,
        "revision": settings.revision,
        "runtime": settings.runtime,
        "tests": [],
    }
    try:
        manifest = json.loads((ROOT / "config/model-source.json").read_text())
        if manifest["revision"] != settings.revision:
            raise RuntimeError(
                "Source manifest does not match the pinned model"
            )
        print("Verifying checkpoint sizes and SHA256...", flush=True)
        for item in manifest["files"]:
            if item["path"] == ".gitattributes":
                continue
            path = settings.path / item["path"]
            if path.stat().st_size != item["bytes"]:
                raise RuntimeError(f"File size mismatch: {path.name}")
            if item["sha256"]:
                with path.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != item["sha256"]:
                    raise RuntimeError(f"SHA256 mismatch: {path.name}")
        report["checkpoint_integrity"] = "passed"
        with block_python_network() as attempts:
            runtime = LocalModel(settings)
            report.update(
                model_class=type(runtime.model).__name__,
                model_type=runtime.model.config.model_type,
                tokenizer_class=type(runtime.tokenizer).__name__,
                tokenizer_vocab_size=len(runtime.tokenizer),
                parameters=sum(p.numel() for p in runtime.model.parameters()),
                load_seconds=round(runtime.load_seconds, 3),
                after_load_memory=runtime.memory(),
                dtype=str(next(runtime.model.parameters()).dtype),
                devices=sorted(
                    {str(p.device) for p in runtime.model.parameters()}
                ),
                missing_keys=sorted(
                    runtime.loading_info.get("missing_keys", [])
                ),
            )
            print("Warming up the local model...", flush=True)
            warmup = runtime.generate(
                [{"role": "user", "content": "Say hello briefly."}],
                max_new_tokens=8,
            )
            report["warmup_seconds"] = warmup["elapsed_seconds"]
            cases = json.loads(
                (ROOT / "evaluation/datasets/model-smoke.json").read_text()
            )
            for case in cases:
                print(f"Checking {case['id']}...", flush=True)
                result = runtime.generate(case["messages"], max_new_tokens=64)
                result["id"] = case["id"]
                result["expected_pattern"] = case["expected_pattern"]
                result["passed"] = bool(
                    re.search(case["expected_pattern"], result["answer"])
                )
                report["tests"].append(result)
                print(json.dumps(result), flush=True)
            print("Benchmarking a longer response...", flush=True)
            report["benchmark"] = runtime.generate(
                [
                    {
                        "role": "user",
                        "content": (
                            "Write five short practical tips for organizing "
                            "a personal collection of documents."
                        ),
                    }
                ],
                max_new_tokens=128,
            )
            print("Checking an approximately 512-token prompt...", flush=True)
            context = "The Cedar workshop starts at 09:30. " * 45
            report["context_probe"] = runtime.generate(
                [
                    {
                        "role": "user",
                        "content": (
                            context
                            + "At what time does the Cedar workshop start? "
                            "Reply with the time only."
                        ),
                    }
                ],
                max_new_tokens=32,
            )
            if "09:30" not in report["context_probe"]["answer"]:
                raise RuntimeError("Longer-context smoke check failed")
            try:
                runtime.generate(
                    [
                        {
                            "role": "user",
                            "content": "word " * 2000,
                        }
                    ]
                )
            except ValueError:
                report["input_budget_guard"] = "passed"
            else:
                raise RuntimeError("Input token budget was not enforced")
            report["python_network_attempts"] = attempts
            if attempts:
                raise RuntimeError("A network connection was attempted")
            if not all(case["passed"] for case in report["tests"]):
                raise RuntimeError(
                    "One or more basic conversation checks failed"
                )
            report["status"] = "passed"
            report["offline_scope"] = (
                "local-only assets and HF offline flags; "
                "Python socket/DNS calls blocked"
            )
    except Exception as exc:
        report.update(
            status="failed", error_type=type(exc).__name__, error=str(exc)
        )
        traceback.print_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
