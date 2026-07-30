"""Regenerate ARTIFACT_MANIFEST.json after an intentional repository update."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "ARTIFACT_MANIFEST.json"
EXCLUDED_TOP_LEVEL = {".git", "build"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included_files() -> list[Path]:
    files = []
    for path in REPO.rglob("*"):
        if not path.is_file() or path == OUTPUT:
            continue
        relative = path.relative_to(REPO)
        if relative.parts[0] in EXCLUDED_TOP_LEVEL:
            continue
        if "__pycache__" in relative.parts:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(REPO).as_posix())


def main() -> None:
    files = included_files()
    payload = {
        "schema_version": 1,
        "artifact": "Anchor-Side Perturbation reproducibility repository",
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
