"""Regenerate ARTIFACT_MANIFEST.json after an intentional repository update."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "ARTIFACT_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included_files() -> list[Path]:
    # Honor .gitignore: downloaded faces, generated features/model weights,
    # local environments, and resume outputs must never enter the manifest.
    listed = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPO,
    ).decode("utf-8").split("\0")
    files = [REPO / name for name in set(listed) if name]
    files = [path for path in files if path.is_file() and path != OUTPUT]
    return sorted(files, key=lambda path: path.relative_to(REPO).as_posix())


def main() -> None:
    files = included_files()
    payload = {
        "schema_version": 1,
        "artifact": "Noisy Anchor Alignment reproducibility repository",
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files": [
            {
                "path": path.relative_to(REPO).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    OUTPUT.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    print(
        f"Wrote {OUTPUT.name}: {payload['file_count']} files, "
        f"{payload['total_bytes']} bytes"
    )


if __name__ == "__main__":
    main()
