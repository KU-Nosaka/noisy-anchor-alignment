"""Validate frozen artifacts before analysis or figure rendering."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile


REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gzip_json(relative: str) -> dict:
    path = REPO / relative
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"{relative} is not a result object with a records array")
    return payload


def require_count(relative: str, expected: int, participant_count: int | None = None) -> list[dict]:
    payload = load_gzip_json(relative)
    records = payload["records"]
    if len(records) != expected:
        raise ValueError(f"{relative}: expected {expected} records, found {len(records)}")
    declared = payload.get("record_count")
    if declared is not None and int(declared) != expected:
        raise ValueError(f"{relative}: declared record_count={declared}, expected {expected}")
    if participant_count is not None:
        counts = {int(record["n_participants"]) for record in records}
        if counts != {participant_count}:
            raise ValueError(f"{relative}: participant counts are {sorted(counts)}")
    return records


def validate_results() -> None:
    mnist = require_count("results/mnist/mnist_combined_609_results.json.gz", 609)
    mnist_schedule = Counter(
        (str(row["protocol"]), str(row["run_role"])) for row in mnist
    )
    expected_mnist = Counter(
        {
            ("c_gdp", "random"): 1,
            ("aa_i_gdp", "random"): 1,
            ("i_gdp", "random"): 1,
            ("c_gdp_an", "random"): 100,
            ("aa_i_gdp_an", "random"): 100,
            ("pa_i_gdp", "random"): 100,
            ("c_gdp_an", "supplemental_private_sigma_35_50_v1"): 100,
            ("aa_i_gdp_an", "supplemental_private_sigma_35_50_v1"): 100,
            ("pa_i_gdp", "supplemental_anchor_v_0_0p2_v1"): 100,
            ("c_gdp_an", "endpoint_zero"): 1,
            ("aa_i_gdp_an", "endpoint_zero"): 1,
            ("pa_i_gdp", "endpoint_zero"): 1,
            ("c_gdp_an", "endpoint_upper"): 1,
            ("aa_i_gdp_an", "endpoint_upper"): 1,
            ("pa_i_gdp", "endpoint_upper"): 1,
        }
    )
    if mnist_schedule != expected_mnist:
        raise ValueError(f"MNIST schedule differs: {mnist_schedule}")

    p10 = require_count(
        "results/celeba/p010_combined_407_results.json.gz", 407, 10
    )
    p10_schedule = Counter(
        (str(row["protocol"]), str(row["run_role"])) for row in p10
    )
    expected_p10 = Counter(
        {
            ("c_gdp", "clean"): 1,
            ("i_gdp", "clean"): 1,
            ("c_gdp_an", "random"): 100,
            ("pa_i_gdp", "random"): 100,
            ("c_gdp_an", "supplemental_private_sigma_1p5_3p0_v1"): 100,
            ("pa_i_gdp", "supplemental_anchor_v_0_0p05_v1"): 100,
            ("c_gdp_an", "endpoint_zero"): 1,
            ("pa_i_gdp", "endpoint_zero"): 1,
            ("c_gdp_an", "endpoint_upper"): 1,
            ("pa_i_gdp", "endpoint_upper"): 1,
            ("pa_i_gdp", "i_gdp_equivalence_control"): 1,
        }
    )
    if p10_schedule != expected_p10:
        raise ValueError(f"CelebA p=10 schedule differs: {p10_schedule}")

    require_count("results/celeba/p030_core_207_results.json.gz", 207, 30)
    require_count("results/celeba/p050_core_207_results.json.gz", 207, 50)
    for p in (30, 50):
        records = require_count(
            f"results/celeba/p{p:03d}_low_v_100_results.json.gz", 100, p
        )
        if {
            (str(row["protocol"]), str(row["run_role"])) for row in records
        } != {("pa_i_gdp", "supplemental_anchor_v_0_0p05_v1")}:
            raise ValueError(f"CelebA p={p} low-v schedule differs")


def validate_archives() -> None:
    expectations = {
        "figure_inputs/mnist_reconstruction_cells.zip": (
            803,
            "cells/digit_0/c01_original.png",
        ),
        "figure_inputs/celeba_mixed_reconstructions.zip": (
            78,
            "reconstruction_leakage_ordered_v2/metadata.json",
        ),
        "figure_inputs/celeba_identity_reconstructions.zip": (
            708,
            "reconstruction_leakage_ordered_by_identity_v1/metadata.json",
        ),
    }
    for relative, (expected_count, required_member) in expectations.items():
        path = REPO / relative
        with zipfile.ZipFile(path) as handle:
            names = [member.filename for member in handle.infolist()]
        if len(names) != expected_count:
            raise ValueError(
                f"{relative}: expected {expected_count} entries, found {len(names)}"
            )
        if required_member not in names:
            raise ValueError(f"{relative}: missing {required_member}")


def validate_manifest() -> None:
    manifest_path = REPO / "ARTIFACT_MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for item in payload["files"]:
        path = REPO / item["path"]
        if not path.is_file():
            failures.append(f"missing: {item['path']}")
            continue
        if path.stat().st_size != int(item["bytes"]):
            failures.append(f"size: {item['path']}")
            continue
        if sha256(path) != item["sha256"]:
            failures.append(f"sha256: {item['path']}")
    if failures:
        raise ValueError("Artifact-manifest failures:\n  " + "\n  ".join(failures))


def main() -> None:
    validate_manifest()
    validate_results()
    validate_archives()
    print("Repository validation passed: hashes, schedules, counts, and archives.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        raise
