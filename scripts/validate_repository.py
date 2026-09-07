# Historical MNIST/CelebA package; see current/ for the current manuscript.
"""Validate frozen artifacts before analysis or figure rendering."""

from __future__ import annotations

from collections import Counter
import csv
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


def load_gzip_csv(relative: str) -> list[dict[str, str]]:
    path = REPO / relative
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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

    replay_grid = (2, 5, 10, 20, 50)
    expected_replay_schedule = Counter(
        {
            ("c_gdp", "clean"): 1,
            ("i_gdp", "clean"): 1,
            ("pa_i_gdp", "endpoint_zero"): 1,
            ("pa_i_gdp", "random"): 100,
        }
    )
    expected_convergence = {
        2: (23, 77),
        5: (26, 74),
        10: (32, 68),
        20: (34, 66),
        50: (40, 60),
    }
    expected_anchor_hash = (
        "b0e6fe305579175c80a9b9e5ec33f0969b461814a93886541147b25fb69347a5"
    )
    expected_bank_hash = (
        "32d672b0b58fcdd38da68cd4669bd940146b88205893a5749f88c64a61198afa"
    )
    paired_v: tuple[float, ...] | None = None
    for p in replay_grid:
        relative = f"results/celeba/participant_replay/p{p:03d}_results.csv.gz"
        records = load_gzip_csv(relative)
        if len(records) != 103:
            raise ValueError(f"{relative}: expected 103 records, found {len(records)}")
        if {int(row["n_participants"]) for row in records} != {p}:
            raise ValueError(f"{relative}: participant-count field mismatch")
        if len({row["run_id"] for row in records}) != 103:
            raise ValueError(f"{relative}: run IDs are not unique")
        schedule = Counter(
            (row["protocol"], row["run_role"]) for row in records
        )
        if schedule != expected_replay_schedule:
            raise ValueError(f"{relative}: schedule differs: {schedule}")
        random_rows = sorted(
            (row for row in records if row["run_role"] == "random"),
            key=lambda row: int(row["draw"]),
        )
        if [int(row["draw"]) for row in random_rows] != list(range(100)):
            raise ValueError(f"{relative}: random draws are not 0,...,99")
        v = tuple(float(row["anchor_v"]) for row in random_rows)
        if len(set(v)) != 100 or not all(0.0 < value < 0.1 for value in v):
            raise ValueError(f"{relative}: invalid Uniform(0, 0.1) realization")
        if paired_v is None:
            paired_v = v
        elif v != paired_v:
            raise ValueError(f"{relative}: anchor-noise draws are not paired")
        if {
            row["anchor.tensor_sha256_float64_c_order"] for row in records
        } != {expected_anchor_hash}:
            raise ValueError(f"{relative}: anchor hash differs")
        if {row["deployment.tensor_file_sha256"] for row in records} != {
            expected_bank_hash
        }:
            raise ValueError(f"{relative}: deployment-bank hash differs")
        convergence = sum(
            row["gpm.converged"].strip().lower() == "true"
            for row in random_rows
        )
        expected_converged, expected_capped = expected_convergence[p]
        if (convergence, len(random_rows) - convergence) != (
            expected_converged,
            expected_capped,
        ):
            raise ValueError(f"{relative}: GPM stopping counts differ")

    assert paired_v is not None
    validate_supplemental_replay(paired_v)


def validate_supplemental_replay(published_v: tuple[float, ...]) -> None:
    """Check the paired low-noise stratum of the participant-count replay.

    Supplemental draw ``j`` of every participant count reuses the uniform
    quantile, noise tensors and seeds of published draw ``j`` at exactly half
    the anchor-noise amplitude, so ``anchor_v`` must equal ``published_v[j] / 2``
    bit for bit.  Until the complete records are frozen, the distributed files
    are the compact per-draw tables exported by Notebook 09.
    """

    expected_convergence = {2: (48, 52), 5: (58, 42), 10: (71, 29), 20: (81, 19), 50: (90, 10)}
    required_columns = {
        "p",
        "draw",
        "anchor_v",
        "test_balanced_accuracy",
        "op_cross_image_identity_top1",
        "gpm_converged",
        "gpm_iterations",
    }
    for p in (2, 5, 10, 20, 50):
        relative = (
            "results/celeba/participant_replay/supplemental/"
            f"p{p:03d}_supplemental_compact.csv"
        )
        path = REPO / relative
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or not required_columns <= set(rows[0].keys()):
            raise ValueError(f"{relative}: missing columns")
        if len(rows) != 100:
            raise ValueError(f"{relative}: expected 100 records, found {len(rows)}")
        if {int(row["p"]) for row in rows} != {p}:
            raise ValueError(f"{relative}: participant-count field mismatch")
        rows.sort(key=lambda row: int(row["draw"]))
        if [int(row["draw"]) for row in rows] != list(range(100)):
            raise ValueError(f"{relative}: supplemental draws are not 0,...,99")
        v = [float(row["anchor_v"]) for row in rows]
        if any(value != 0.5 * source for value, source in zip(v, published_v)):
            raise ValueError(f"{relative}: anchor_v is not half of the paired published draw")
        if not all(0.0 < value < 0.05 for value in v):
            raise ValueError(f"{relative}: invalid (0, 0.05) realization")
        for column in ("test_balanced_accuracy", "op_cross_image_identity_top1"):
            if not all(0.0 <= float(row[column]) <= 1.0 for row in rows):
                raise ValueError(f"{relative}: {column} outside [0, 1]")
        convergence = sum(row["gpm_converged"].strip().lower() == "true" for row in rows)
        if (convergence, 100 - convergence) != expected_convergence[p]:
            raise ValueError(f"{relative}: GPM stopping counts differ")


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


def validate_replay_notebooks() -> None:
    expected_shards = {
        "notebooks/06_celeba_participant_replay_p050.ipynb": (50,),
        "notebooks/07_celeba_participant_replay_p002_p020.ipynb": (2, 20),
        "notebooks/08_celeba_participant_replay_p005_p010.ipynb": (5, 10),
    }
    runner = (REPO / "scripts" / "celeba_participant_replay_runner.py").read_text(
        encoding="utf-8"
    )
    normalized_runner = runner.replace("\r\n", "\n").rstrip() + "\n"
    writefile_header = "%%writefile celeba_participant_replay_runner.py\n"
    for relative, shard in expected_shards.items():
        payload = json.loads((REPO / relative).read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in payload["cells"]]
        controls = [source for source in sources if "SHARD_PARTICIPANT_COUNTS" in source]
        expected_assignment = f"SHARD_PARTICIPANT_COUNTS = {shard!r}"
        if len(controls) < 1 or expected_assignment not in controls[0]:
            raise ValueError(f"{relative}: shard assignment differs")
        embedded = [source for source in sources if source.startswith(writefile_header)]
        if len(embedded) != 1:
            raise ValueError(f"{relative}: expected one embedded replay runner")
        normalized_embedded = (
            embedded[0][len(writefile_header):]
            .replace("\r\n", "\n")
            .rstrip()
            + "\n"
        )
        if normalized_embedded != normalized_runner:
            raise ValueError(f"{relative}: embedded replay runner differs")

    # Notebook 09 rebuilds the reviewed modules from the pinned commit of this
    # repository (Notebook 06's %%writefile cells) and embeds only the
    # supplemental overlay, which is mirrored in scripts/ for review.
    relative = "notebooks/09_celeba_participant_replay_supplemental_v0_0p05.ipynb"
    payload = json.loads((REPO / relative).read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in payload["cells"]]
    pinned = 'REPO_COMMIT = "20383a6fcd87f83eb615397e84a7e6f4af996210"'
    if not any(pinned in source for source in sources):
        raise ValueError(f"{relative}: pinned repository commit differs")
    overlay = (
        REPO / "scripts" / "celeba_participant_replay_supplemental_runner.py"
    ).read_text(encoding="utf-8")
    normalized_overlay = overlay.replace("\r\n", "\n").rstrip() + "\n"
    header = "%%writefile celeba_participant_replay_supplemental_runner.py\n"
    embedded = [source for source in sources if source.startswith(header)]
    others = [source for source in sources if source.startswith("%%writefile ") and not source.startswith(header)]
    if len(embedded) != 1 or others:
        raise ValueError(f"{relative}: expected exactly one embedded module, the supplemental overlay")
    normalized_embedded = embedded[0][len(header):].replace("\r\n", "\n").rstrip() + "\n"
    if normalized_embedded != normalized_overlay:
        raise ValueError(f"{relative}: embedded supplemental overlay differs")


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
    validate_replay_notebooks()
    print("Repository validation passed: hashes, schedules, counts, and archives.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        raise

