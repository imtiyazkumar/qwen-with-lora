"""Download the one pinned official model into the project model directory."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.inference.runtime import ModelSettings  # noqa: E402
from backend.services.environment import (  # noqa: E402
    ROOT,
    configure_environment,
)


def main() -> None:
    configure_environment(offline=False)
    from huggingface_hub import HfApi, snapshot_download

    settings = ModelSettings.read()
    info = HfApi().model_info(
        settings.repo_id, revision=settings.revision, files_metadata=True
    )
    if info.card_data.license != settings.license:
        raise RuntimeError(
            "Model license differs from the reviewed configuration"
        )
    snapshot_download(
        repo_id=settings.repo_id,
        revision=settings.revision,
        local_dir=settings.path,
        max_workers=2,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.jinja",
            "merges.txt",
            "LICENSE",
            "README.md",
        ],
    )
    files = []
    for item in info.siblings:
        path = settings.path / item.rfilename
        if not path.is_file():
            continue
        if path.stat().st_size != item.size:
            raise RuntimeError(f"File size mismatch: {item.rfilename}")
        expected = item.lfs.get("sha256") if item.lfs else None
        if expected:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise RuntimeError(f"SHA256 mismatch: {item.rfilename}")
        files.append(
            {"path": item.rfilename, "bytes": item.size, "sha256": expected}
        )
    manifest = {
        "repo_id": settings.repo_id,
        "revision": settings.revision,
        "license": settings.license,
        "format": "safetensors",
        "precision": settings.dtype,
        "files": files,
    }
    (ROOT / "config/model-source.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Downloaded and verified {settings.repo_id} at {settings.revision}")


if __name__ == "__main__":
    main()
