"""Resumable participant-count replay for CelebA anchor-side noise.

This overlay reuses the reviewed CelebA data preparation, training procedure,
privacy auditor, attacks, and GPM implementation. It independently creates a
deterministic 50-participant GDP parameter bank and centered unit-isometric
anchor in every shard, so it has no dependency on a previous participant-
sweep result tree. Every noisy stage contains 100 paired draws
v ~ Uniform(0, 0.1), and every fit is committed atomically to Drive.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import celeba_public_svd_identity100_runner as _base
import celeba_public_svd_identity100_unit_isometric_runner as _unit


SCHEMA_VERSION = 1
SCHEDULE_VERSION = "celeba-participant-replay-standalone-anchor-v0-0p1-100-v1"
ALLOWED_PARTICIPANTS = (2, 5, 10, 20, 50)
MASTER_PARTICIPANTS = 50
RANDOM_DRAWS = 100
ANCHOR_V_LOWER = 0.0
ANCHOR_V_UPPER = 0.1
CONTROL_COUNT = 3
PLANNED_FITS_PER_PARTICIPANT = RANDOM_DRAWS + CONTROL_COUNT
STAGE_DIRECTORY = "anchor_v_0_0p1_100_standalone_v1"
STORE_STEM = "celeba_participant_replay_results"


_base.MAX_PARTICIPANTS = MASTER_PARTICIPANTS
_base.SCHEMA_VERSION = SCHEMA_VERSION
_unit.SCHEMA_VERSION = SCHEMA_VERSION


@dataclass(frozen=True)
class ExperimentConfig(_unit.ExperimentConfig):
    max_participants: int = MASTER_PARTICIPANTS

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_participants != MASTER_PARTICIPANTS:
            raise ValueError("The replay fixes a 50-participant master")
        if self.n_participants not in (*ALLOWED_PARTICIPANTS, MASTER_PARTICIPANTS):
            raise ValueError(
                "n_participants must be a replay value or the 50-participant bank"
            )
_base.ExperimentConfig = ExperimentConfig


if not hasattr(_base, "_participant_replay_original_master_loader"):
    _base._participant_replay_original_master_loader = (
        _base._load_or_create_master_manifest
    )
_ORIGINAL_MASTER_LOADER = _base._participant_replay_original_master_loader


def _validate_frozen_master_manifest(
    manifest: Mapping[str, Any],
    frame: Any,
    cfg: ExperimentConfig,
    archive_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and reuse the authoritative identity allocation on Drive.

    The allocation depends on the metadata, seed, and participant design.  It
    does not depend on byte-level ZIP-container metadata, which can differ when
    identical Kaggle files are archived concurrently by separate Colab
    runtimes.  Once created, the self-hashed allocation is therefore loaded as
    the authority rather than rebuilt and compared through the archive hash.
    """

    loaded = dict(manifest)
    claimed_hash = loaded.get("manifest_hash")
    content = {key: value for key, value in loaded.items() if key != "manifest_hash"}
    if claimed_hash != _base.stable_hash(content):
        raise RuntimeError("Frozen CelebA master manifest self-hash mismatch")
    scalar_checks = {
        "schema_version": _base.DATA_CACHE_SCHEMA_VERSION,
        "dataset": _base.DATASET_NAME,
        "seed": int(cfg.seed),
    }
    for key, expected in scalar_checks.items():
        if loaded.get(key) != expected:
            raise RuntimeError(
                f"Frozen CelebA master manifest has incompatible {key}"
            )

    expected_metadata = {
        "identity": frame.attrs["identity_metadata_sha256"],
        "attributes": frame.attrs["attribute_metadata_sha256"],
    }
    if loaded.get("metadata_hashes") != expected_metadata:
        raise RuntimeError("Frozen CelebA master metadata hashes differ")

    frozen_archive = loaded.get("archive", {})
    for key in ("source", "file_count"):
        if key in frozen_archive and key in archive_manifest:
            if frozen_archive[key] != archive_manifest[key]:
                raise RuntimeError(
                    f"Frozen CelebA archive provenance differs for {key}"
                )

    design = loaded.get("participant_design", {})
    expected_splits = dict(_base.SPLIT_IDENTITY_COUNTS)
    if int(design.get("master_capacity_participants", -1)) != int(
        cfg.max_participants
    ):
        raise RuntimeError("Frozen CelebA master capacity differs")
    if int(design.get("identities_per_participant", -1)) != int(
        cfg.identities_per_participant
    ):
        raise RuntimeError("Frozen CelebA identities-per-participant differs")
    if design.get("split_identity_counts") != expected_splits:
        raise RuntimeError("Frozen CelebA identity split counts differ")
    if not bool(design.get("globally_participant_disjoint", False)):
        raise RuntimeError("Frozen CelebA participants are not declared disjoint")
    if not bool(design.get("reserved_references_excluded_from_protocol", False)):
        raise RuntimeError("Frozen CelebA references are not excluded from protocol data")

    public = loaded.get("public_svd", {})
    public_filenames = [str(value) for value in public.get("filenames", [])]
    public_identities = {int(value) for value in public.get("identity_ids", [])}
    if int(public.get("image_count", -1)) != int(cfg.public_svd_images):
        raise RuntimeError("Frozen CelebA public-SVD image count differs")
    if len(public_filenames) != int(cfg.public_svd_images):
        raise RuntimeError("Frozen CelebA public-SVD filename count differs")
    if len(public_filenames) != len(set(public_filenames)):
        raise RuntimeError("Frozen CelebA public-SVD filenames are not unique")

    assignments = loaded.get("assignments")
    if not isinstance(assignments, list):
        raise RuntimeError("Frozen CelebA assignments are missing")
    expected_assignment_count = int(
        cfg.max_participants * cfg.identities_per_participant
    )
    if len(assignments) != expected_assignment_count:
        raise RuntimeError("Frozen CelebA assignment count differs")

    participant_counts: dict[int, int] = {}
    split_counts: dict[tuple[int, str], int] = {}
    identities: set[int] = set()
    reserved_filenames: set[str] = set()
    protocol_filenames: set[str] = set()
    for item in assignments:
        participant = int(item["participant"])
        split = str(item["split"])
        identity = int(item["identity"])
        reserved = str(item["reserved_filename"])
        protocol = [str(value) for value in item["protocol_filenames"]]
        if participant not in range(int(cfg.max_participants)):
            raise RuntimeError("Frozen CelebA participant index is outside the master")
        if split not in expected_splits:
            raise RuntimeError("Frozen CelebA assignment has an unknown split")
        if identity in identities:
            raise RuntimeError("Frozen CelebA participant identities are not unique")
        identities.add(identity)
        participant_counts[participant] = participant_counts.get(participant, 0) + 1
        key = (participant, split)
        split_counts[key] = split_counts.get(key, 0) + 1
        if reserved in reserved_filenames:
            raise RuntimeError("Frozen CelebA reserved filenames are not unique")
        reserved_filenames.add(reserved)
        protocol_filenames.update(protocol)

    for participant in range(int(cfg.max_participants)):
        if participant_counts.get(participant, 0) != int(
            cfg.identities_per_participant
        ):
            raise RuntimeError("Frozen CelebA participant identity count differs")
        for split, expected in expected_splits.items():
            if split_counts.get((participant, split), 0) != int(expected):
                raise RuntimeError("Frozen CelebA participant split count differs")

    quarantine = {
        int(value)
        for value in loaded.get("legacy_test_quarantine", {}).get(
            "identity_ids", []
        )
    }
    if identities & public_identities:
        raise RuntimeError("Frozen CelebA public and participant identities overlap")
    if identities & quarantine:
        raise RuntimeError("Frozen CelebA quarantine and participant identities overlap")
    if reserved_filenames & protocol_filenames:
        raise RuntimeError("Frozen CelebA reserved files entered protocol data")
    if set(public_filenames) & (reserved_filenames | protocol_filenames):
        raise RuntimeError("Frozen CelebA public files entered participant data")
    return loaded


def _load_or_create_frozen_master_manifest(
    frame: Any,
    cfg: ExperimentConfig,
    archive_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(cfg.cache_dir) / "master" / (
        f"master_pmax{cfg.max_participants}_i100_seed{cfg.seed}.json"
    )
    if not path.is_file():
        return _ORIGINAL_MASTER_LOADER(frame, cfg, archive_manifest)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return _validate_frozen_master_manifest(
        loaded, frame, cfg, archive_manifest
    )


_base._load_or_create_master_manifest = _load_or_create_frozen_master_manifest


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _base._atomic_json(path, dict(payload))


def _schedule(cfg: ExperimentConfig) -> list[dict[str, Any]]:
    controls = [
        {
            "method": "c_gdp",
            "outer": 0,
            "draw": 0,
            "run_role": "clean",
            "level": 0.0,
        },
        {
            "method": "i_gdp",
            "outer": 0,
            "draw": 0,
            "run_role": "clean",
            "level": 0.0,
        },
        {
            "method": "pa_i_gdp",
            "outer": 0,
            "draw": RANDOM_DRAWS,
            "run_role": "endpoint_zero",
            "level": 0.0,
        },
    ]
    noisy = [
        {
            "method": "pa_i_gdp",
            "outer": 0,
            "draw": draw,
            "run_role": "random",
            "level": _base._lambda_draw(
                cfg, ANCHOR_V_UPPER, draw, "anchor"
            ),
        }
        for draw in range(RANDOM_DRAWS)
    ]
    schedule = [*controls, *noisy]
    if len(schedule) != PLANNED_FITS_PER_PARTICIPANT:
        raise AssertionError("The replay schedule has the wrong length")
    levels = [float(item["level"]) for item in noisy]
    if not all(ANCHOR_V_LOWER < value < ANCHOR_V_UPPER for value in levels):
        raise AssertionError("A random anchor-noise draw is outside (0, 0.1)")
    if len(set(levels)) != RANDOM_DRAWS:
        raise AssertionError("The 100 deterministic uniform draws are not unique")
    return schedule


def paired_anchor_levels() -> list[float]:
    cfg = ExperimentConfig.for_mode(
        "full",
        n_participants=2,
        max_participants=MASTER_PARTICIPANTS,
        private_lambda_upper=0.5,
        anchor_lambda_upper=ANCHOR_V_UPPER,
        anchor_v_upper=ANCHOR_V_UPPER,
        device="cpu",
    )
    return [
        float(item["level"])
        for item in _schedule(cfg)
        if item["run_role"] == "random"
    ]


def _runtime_lock(cfg: ExperimentConfig, out: Path) -> dict[str, Any]:
    lock = {
        **dict(_unit._runtime_environment_lock(cfg)),
        "replay_schema_version": SCHEMA_VERSION,
        "schedule_version": SCHEDULE_VERSION,
        "deterministic": bool(cfg.deterministic),
        "module_sha256": dict(cfg.module_sha256),
    }
    _base._write_or_validate_json(
        out / "runtime_environment_lock.json",
        lock,
        "replay runtime environment lock",
    )
    return lock


def _stage_config(cfg: ExperimentConfig, out: Path) -> ExperimentConfig:
    out.mkdir(parents=True, exist_ok=True)
    _base._write_or_validate_json(
        out / "config.json", cfg.to_dict(), "replay configuration"
    )
    return cfg


def _expected_run_id(
    ctx: Any, specification: Mapping[str, Any]
) -> str:
    identity = {
        "config": _base.stable_hash(ctx.cfg.to_dict()),
        "data": _base.stable_hash(ctx.data.manifest),
        "method": specification["method"],
        "outer": int(specification["outer"]),
        "draw": int(specification["draw"]),
        "run_role": specification["run_role"],
        "lambda": float(specification["level"]),
    }
    return _base.stable_hash(identity, 24)


def _expected_run_ids(
    ctx: Any, schedule: Sequence[Mapping[str, Any]]
) -> list[str]:
    run_ids = [_expected_run_id(ctx, item) for item in schedule]
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError("The standalone schedule contains duplicate run IDs")
    return run_ids


def _scheduled_records(
    store: Any,
    expected_ids: Sequence[str],
    *,
    require_complete: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for run_id in expected_ids:
        path = store.path(str(run_id))
        if not path.is_file():
            missing.append(str(run_id))
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if str(record.get("run_id")) != str(run_id):
            raise RuntimeError(f"Run file {path} contains the wrong run ID")
        records.append(record)
    if require_complete and missing:
        raise RuntimeError(
            f"Missing {len(missing)} scheduled records; "
            f"first missing ID is {missing[0]}"
        )
    return records, missing


def _completed_records(
    store: Any, expected_ids: Sequence[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    return _scheduled_records(
        store, expected_ids, require_complete=False
    )


def _aggregate_scheduled_records(
    store: Any, records: Sequence[Mapping[str, Any]]
) -> tuple[Path, Path]:
    selected = [dict(row) for row in records]
    json_path = store.output_dir / f"{store.stem}.json"
    csv_path = store.output_dir / f"{store.stem}.csv"
    _atomic_json(
        json_path,
        {
            "schema_version": SCHEMA_VERSION,
            "record_count": len(selected),
            "record_ids": [str(row["run_id"]) for row in selected],
            "records": selected,
        },
    )

    def flatten(
        value: Mapping[str, Any], prefix: str = ""
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, Mapping):
                output.update(flatten(item, name))
            elif isinstance(item, (list, tuple)):
                output[name] = json.dumps(item, separators=(",", ":"))
            else:
                output[name] = item
        return output

    flattened = [flatten(row) for row in selected]
    fields = sorted({key for row in flattened for key in row})
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=store.output_dir,
        suffix=".tmp",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)
        temporary = Path(handle.name)
    os.replace(temporary, csv_path)
    return json_path, csv_path


def _design_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "anchor": root / "anchors" / "anchor_outer0.pt",
        "anchor_manifest": root / "anchors" / "anchor_manifest.json",
        "deployment": root / "deployment" / "gdp_parameters_outer0.pt",
        "deployment_manifest": (
            root / "deployment" / "gdp_parameters_manifest.json"
        ),
    }


def _load_or_create_design(
    bank_cfg: ExperimentConfig, design_root: Path
) -> dict[str, Any]:
    """Create or validate one shard-local deterministic master design."""

    paths = _design_paths(design_root)
    required = (
        paths["anchor"],
        paths["anchor_manifest"],
        paths["deployment"],
        paths["deployment_manifest"],
    )
    if all(path.is_file() for path in required):
        anchor_expected = json.loads(
            paths["anchor_manifest"].read_text(encoding="utf-8")
        )
        deployment_expected = json.loads(
            paths["deployment_manifest"].read_text(encoding="utf-8")
        )
        for artifact, expected, label in (
            (paths["anchor"], anchor_expected, "anchor"),
            (
                paths["deployment"],
                deployment_expected,
                "GDP parameter bank",
            ),
        ):
            if Path(str(expected.get("tensor_path", ""))) != artifact:
                raise RuntimeError(
                    f"Persisted standalone {label} path differs"
                )
            if _base.file_sha256(artifact) != expected.get(
                "tensor_file_sha256"
            ):
                raise RuntimeError(
                    f"Persisted standalone {label} hash differs"
                )
        anchor_payload = torch.load(
            paths["anchor"], map_location="cpu", weights_only=False
        )
        deployment_payload = torch.load(
            paths["deployment"],
            map_location="cpu",
            weights_only=False,
        )
        anchor = anchor_payload["anchor"].to(torch.float64)
        rotations = [
            item.to(torch.float64)
            for item in deployment_payload["rotations"]
        ]
        translations = [
            item.to(torch.float64)
            for item in deployment_payload["translations"]
        ]
        anchor_spec = _unit._validate_loaded_anchor(
            anchor, bank_cfg, 0, anchor_expected
        )
        deployment_spec = _unit._validate_loaded_gdp_parameters(
            rotations,
            translations,
            bank_cfg,
            0,
            deployment_expected,
        )
    else:
        generator = _base.cp.make_generator(
            _base.named_seed(bank_cfg.seed, "gdp-parameters", 0),
            "cpu",
        )
        rotations, translations = _base.cp.sample_gdp_parameters(
            MASTER_PARTICIPANTS,
            bank_cfg.reduced_dim,
            generator=generator,
            device="cpu",
        )
        raw_deployment_spec = _unit._gdp_parameter_diagnostics(
            rotations, translations, bank_cfg, 0
        )
        deployment_spec = _unit._persist_gdp_parameters(
            rotations,
            translations,
            raw_deployment_spec,
            design_root,
            bank_cfg,
        )
        anchor, raw_anchor_spec = (
            _unit._make_centered_unit_isometric_anchor(bank_cfg, 0)
        )
        anchor_spec = _unit._persist_anchor(
            anchor,
            raw_anchor_spec,
            design_root,
            bank_cfg,
        )
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "anchor": anchor,
        "anchor_spec": dict(anchor_spec),
        "rotations": rotations,
        "translations": translations,
        "deployment_spec": dict(deployment_spec),
    }


def _active_deployment(
    ctx: Any,
    cfg: ExperimentConfig,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    p = int(cfg.n_participants)
    active_rotations = list(design["rotations"][:p])
    active_translations = list(design["translations"][:p])
    active_spec = {
        **dict(design["deployment_spec"]),
        "bank_participants": MASTER_PARTICIPANTS,
        "active_participants": p,
        "active_rotations_sha256_float64_c_order": (
            _unit._tensor_sequence_sha256(active_rotations)
        ),
        "active_translations_sha256_float64_c_order": (
            _unit._tensor_sequence_sha256(active_translations)
        ),
        "prefix_invariant": True,
        "standalone_regeneration": True,
    }
    device = _base.resolve_device(cfg.device)
    return {
        "rotations": [item.to(device) for item in active_rotations],
        "translations": [item.to(device) for item in active_translations],
        "anchor": design["anchor"].to(device),
        "private_rms": float(
            torch.sqrt(ctx.data.train.reduced.double().square().mean())
        ),
        "anchor_rms": float(design["anchor_spec"]["entrywise_rms"]),
        "anchor_spec": dict(design["anchor_spec"]),
        "deployment_spec": active_spec,
    }


def _annotate_record(
    record: Mapping[str, Any],
    store: Any,
    cfg: ExperimentConfig,
) -> dict[str, Any]:
    annotated = dict(record)
    v = float(annotated.get("anchor_v", 0.0))
    annotated.update(
        {
            "schema_version": SCHEMA_VERSION,
            "n_participants": int(cfg.n_participants),
            "participant_grid": list(ALLOWED_PARTICIPANTS),
            "anchor_mean_translation_sd": (
                v / math.sqrt(cfg.anchor_rows)
            ),
            "anchor_noise_to_signal_frobenius": (
                v * math.sqrt(cfg.anchor_rows - 1)
            ),
        }
    )
    _atomic_json(store.path(str(annotated["run_id"])), annotated)
    return annotated


def _execute(
    ctx: Any,
    store: Any,
    deployment: Mapping[str, Any],
    **specification: Any,
) -> dict[str, Any]:
    record = _unit._execute_fit(
        ctx, store, deployment, **specification
    )
    return _annotate_record(record, store, ctx.cfg)


def _completion_if_valid(
    out: Path,
    store: Any,
    schedule: Sequence[Mapping[str, Any]],
    cfg: ExperimentConfig,
    runtime_lock: Mapping[str, Any],
) -> dict[str, Any] | None:
    manifest_path = out / "replay_manifest.json"
    completion_path = out / "completion_summary.json"
    if not manifest_path.is_file() or not completion_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    immutable = {
        "schedule_version": SCHEDULE_VERSION,
        "participant_count": int(cfg.n_participants),
        "schedule": list(schedule),
        "config_hash": _base.stable_hash(cfg.to_dict()),
        "runtime_environment_hash": _base.stable_hash(runtime_lock),
        "module_sha256": dict(cfg.module_sha256),
    }
    for key, expected in immutable.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"Completed replay metadata differs for {key}: "
                f"{manifest.get(key)!r} != {expected!r}"
            )
    expected_ids = [str(value) for value in manifest["expected_run_ids"]]
    records, missing = _completed_records(store, expected_ids)
    if missing:
        return None
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("record_hash") != _base.stable_hash(records):
        raise RuntimeError("Completed replay record hash differs")
    return {**completion, "resumed_from_completed_stage": True}


def run_for_p(
    participant_count: int,
    results_root: str | Path,
    design: Mapping[str, Any],
    **overrides: Any,
) -> dict[str, Any]:
    p = int(participant_count)
    if p not in ALLOWED_PARTICIPANTS:
        raise ValueError(f"p must be one of {ALLOWED_PARTICIPANTS}")
    results_root = Path(results_root)
    out = results_root / f"p{p:03d}" / STAGE_DIRECTORY
    cfg = ExperimentConfig.for_mode(
        "full",
        n_participants=p,
        max_participants=MASTER_PARTICIPANTS,
        output_dir=str(out),
        private_lambda_upper=0.5,
        anchor_lambda_upper=ANCHOR_V_UPPER,
        anchor_v_upper=ANCHOR_V_UPPER,
        **overrides,
    )
    if cfg.anchor_v_upper is None or not math.isclose(
        float(cfg.anchor_v_upper),
        ANCHOR_V_UPPER,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError(
            "The active participant replay must use anchor_v_upper=0.1"
        )
    cfg = _stage_config(cfg, out)
    runtime_lock = _runtime_lock(cfg, out)
    schedule = _schedule(cfg)
    store = _base.AtomicRunStore(out, stem=STORE_STEM)
    completed = _completion_if_valid(
        out, store, schedule, cfg, runtime_lock
    )
    if completed is not None:
        return completed

    runtime_path = out / "stage_runtime.json"
    manifest_path = out / "replay_manifest.json"
    completion_path = out / "completion_summary.json"
    previous = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.is_file()
        else {}
    )
    runtime: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schedule_version": SCHEDULE_VERSION,
        "participant_count": p,
        "first_started_at_utc": previous.get(
            "first_started_at_utc", _utc_now()
        ),
        "latest_attempt_started_at_utc": _utc_now(),
        "latest_attempt_completed_at_utc": None,
        "attempts": int(previous.get("attempts", 0)) + 1,
        "phase": "preparing-context",
        "completed": False,
        "planned_records": len(schedule),
        "completed_records": 0,
    }
    _atomic_json(runtime_path, runtime)
    started = time.perf_counter()

    try:
        ctx = _base._make_context(cfg, out, include_test=True)
        if len(ctx.linkage_manifest.query_identities) != 100:
            raise AssertionError(
                "Replay must evaluate all 100 target identity clusters"
            )
        deployment = _active_deployment(ctx, cfg, design)
        expected_ids = _expected_run_ids(ctx, schedule)
        provenance = {
            "master_manifest_hash": ctx.data.manifest[
                "master_manifest_hash"
            ],
            "basis_matrix_sha256": ctx.data.manifest["projection"][
                "basis_matrix_sha256"
            ],
            "data_manifest_hash": _base.stable_hash(ctx.data.manifest),
            "linkage_manifest_hash": _base.stable_hash(
                ctx.linkage_manifest.to_dict()
            ),
            "config_hash": _base.stable_hash(cfg.to_dict()),
            "runtime_environment_hash": _base.stable_hash(runtime_lock),
            "module_sha256": dict(cfg.module_sha256),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "schedule_version": SCHEDULE_VERSION,
            "participant_count": p,
            "participant_grid": list(ALLOWED_PARTICIPANTS),
            "random_draws": RANDOM_DRAWS,
            "anchor_v_interval": [
                ANCHOR_V_LOWER,
                ANCHOR_V_UPPER,
            ],
            "schedule": schedule,
            "expected_run_ids": expected_ids,
            "uniform_noise_seed_rule": (
                "named_seed(seed,uniform-noise,anchor,draw)"
            ),
            "participant_noise_seed_rule": (
                "named_seed(seed,noise,family,outer,draw,split,participant)"
            ),
            "config_hash": provenance["config_hash"],
            "runtime_environment_hash": provenance[
                "runtime_environment_hash"
            ],
            "module_sha256": provenance["module_sha256"],
            "provenance": provenance,
            "standalone_anchor": dict(deployment["anchor_spec"]),
            "standalone_deployment": dict(
                deployment["deployment_spec"]
            ),
        }
        _base._write_or_validate_json(
            manifest_path, manifest, "replay schedule"
        )
        runtime["phase"] = "scheduled-fits"
        runtime["expected_run_ids_hash"] = _base.stable_hash(expected_ids)
        _atomic_json(runtime_path, runtime)

        for specification, run_id in zip(schedule, expected_ids):
            if store.contains(run_id):
                continue
            _execute(ctx, store, deployment, **specification)
            records, _ = _completed_records(store, expected_ids)
            runtime.update(
                {
                    "completed_records": len(records),
                    "last_completed_run_id": run_id,
                    "latest_progress_at_utc": _utc_now(),
                }
            )
            _atomic_json(runtime_path, runtime)
    except BaseException as exc:
        completed_count = 0
        if "expected_ids" in locals():
            completed_count = len(
                _completed_records(store, expected_ids)[0]
            )
        runtime.update(
            {
                "latest_attempt_completed_at_utc": _utc_now(),
                "latest_attempt_wall_seconds": (
                    time.perf_counter() - started
                ),
                "completed_records": completed_count,
                "phase": "failed",
                "last_error": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_json(runtime_path, runtime)
        raise

    records, missing = _scheduled_records(
        store, expected_ids, require_complete=True
    )
    if missing:
        raise AssertionError("A completed replay stage has missing records")
    json_path, csv_path = _aggregate_scheduled_records(
        store, records
    )
    noisy = [
        row
        for row in records
        if row.get("protocol") == "pa_i_gdp"
        and row.get("run_role") == "random"
    ]
    converged = sum(
        bool(row.get("gpm", {}).get("converged", False))
        for row in noisy
    )
    runtime.update(
        {
            "latest_attempt_completed_at_utc": _utc_now(),
            "latest_attempt_wall_seconds": time.perf_counter() - started,
            "completed_records": len(records),
            "phase": "completed",
            "completed": True,
        }
    )
    _atomic_json(runtime_path, runtime)
    result = {
        "mode": "participant-count-anchor-replay",
        "participant_count": p,
        "status": {
            "completed": len(records),
            "planned": len(expected_ids),
        },
        "random_anchor_records": len(noisy),
        "gpm_converged_random_records": converged,
        "gpm_capped_random_records": len(noisy) - converged,
        "anchor_v_min_observed": min(float(row["anchor_v"]) for row in noisy),
        "anchor_v_max_observed": max(float(row["anchor_v"]) for row in noisy),
        "output_dir": str(out),
        "manifest_path": str(manifest_path),
        "runtime_path": str(runtime_path),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "record_hash": _base.stable_hash(records),
        "completed_at_utc": _utc_now(),
    }
    _atomic_json(completion_path, result)
    del ctx
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_shard(
    participant_counts: Sequence[int],
    results_root: str | Path,
    **overrides: Any,
) -> dict[str, Any]:
    selected = tuple(int(value) for value in participant_counts)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("A shard requires unique participant counts")
    if any(value not in ALLOWED_PARTICIPANTS for value in selected):
        raise ValueError(f"Unsupported shard: {selected}")
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    shard_name = "_".join(f"p{value:03d}" for value in selected)
    design_root = (
        results_root / "shards" / shard_name / "frozen_design"
    )
    bank_cfg = ExperimentConfig.for_mode(
        "full",
        n_participants=MASTER_PARTICIPANTS,
        max_participants=MASTER_PARTICIPANTS,
        output_dir=str(design_root),
        private_lambda_upper=0.5,
        anchor_lambda_upper=ANCHOR_V_UPPER,
        anchor_v_upper=ANCHOR_V_UPPER,
        **overrides,
    )
    design = _load_or_create_design(bank_cfg, design_root)
    status_path = results_root / f"shard_status_{shard_name}.json"
    previous = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.is_file()
        else {}
    )
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schedule_version": SCHEDULE_VERSION,
        "participant_counts": list(selected),
        "first_started_at_utc": previous.get(
            "first_started_at_utc", _utc_now()
        ),
        "latest_attempt_started_at_utc": _utc_now(),
        "completed_participant_counts": list(
            previous.get("completed_participant_counts", [])
        ),
        "active_participant_count": None,
        "completed": False,
    }
    _atomic_json(status_path, status)
    summaries: dict[str, Any] = {}
    for p in selected:
        status["active_participant_count"] = p
        _atomic_json(status_path, status)
        summary = run_for_p(
            p,
            results_root,
            design,
            **overrides,
        )
        summaries[str(p)] = summary
        if summary["status"]["completed"] == summary["status"]["planned"]:
            status["completed_participant_counts"] = sorted(
                set(status["completed_participant_counts"]) | {p}
            )
        status["active_participant_count"] = None
        _atomic_json(status_path, status)
    status.update(
        {
            "latest_attempt_completed_at_utc": _utc_now(),
            "completed": set(status["completed_participant_counts"])
            >= set(selected),
        }
    )
    _atomic_json(status_path, status)
    return {
        "mode": "participant-count-anchor-replay-shard",
        "participant_counts": list(selected),
        "status": status,
        "summaries": summaries,
        "standalone_design": {
            "root": str(design_root),
            "anchor_sha256": design["anchor_spec"][
                "tensor_sha256_float64_c_order"
            ],
            "deployment_sha256": design["deployment_spec"][
                "rotations_sha256_float64_c_order"
            ],
        },
        "status_path": str(status_path),
    }


def run_static_self_tests() -> dict[str, Any]:
    levels = paired_anchor_levels()
    cfg_by_p = {
        p: ExperimentConfig.for_mode(
            "full",
            n_participants=p,
            max_participants=MASTER_PARTICIPANTS,
            private_lambda_upper=0.5,
            anchor_lambda_upper=ANCHOR_V_UPPER,
            anchor_v_upper=ANCHOR_V_UPPER,
            device="cpu",
        )
        for p in ALLOWED_PARTICIPANTS
    }
    schedules = {p: _schedule(cfg) for p, cfg in cfg_by_p.items()}
    checks = {
        "participant_grid": ALLOWED_PARTICIPANTS == (2, 5, 10, 20, 50),
        "one_hundred_draws": len(levels) == RANDOM_DRAWS,
        "range_is_0_to_0p1": all(0.0 < value < 0.1 for value in levels),
        "levels_are_paired": all(
            [
                float(item["level"])
                for item in schedule
                if item["run_role"] == "random"
            ]
            == levels
            for schedule in schedules.values()
        ),
        "three_controls": all(
            len(schedule) == RANDOM_DRAWS + CONTROL_COUNT
            for schedule in schedules.values()
        ),
        "target_exists_at_p2": cfg_by_p[2].target_participant == 1,
        "master_capacity_50": all(
            cfg.max_participants == MASTER_PARTICIPANTS
            for cfg in cfg_by_p.values()
        ),
        "frozen_master_loader_bound": (
            _base._load_or_create_master_manifest
            is _load_or_create_frozen_master_manifest
        ),
    }
    report = {"checks": checks, "all_passed": all(checks.values())}
    if not report["all_passed"]:
        raise AssertionError(report)
    return report


resolve_device = _base.resolve_device

__all__ = [
    "ALLOWED_PARTICIPANTS",
    "ANCHOR_V_UPPER",
    "RANDOM_DRAWS",
    "ExperimentConfig",
    "paired_anchor_levels",
    "run_for_p",
    "run_shard",
    "run_static_self_tests",
    "resolve_device",
]
