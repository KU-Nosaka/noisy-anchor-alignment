"""Paired supplemental low-noise anchor draws for the CelebA participant-count replay.

This module overlays, rather than edits, the provenance-locked participant-count
replay implementation (``celeba_participant_replay_runner``).  For every
participant count ``p`` in {2, 5, 10, 20, 50} it adds one stratum of 100
NAA-GDP (``pa_i_gdp``) fits whose absolute anchor-noise scale is

    v_j = 0.05 * u_j,      j = 0, ..., 99,

where ``u_j`` is the normalized uniform quantile that produced the published
draw ``j`` of the original replay (``v_j^orig = 0.1 * u_j``).  The draw index
``j`` is reused, so the participant-indexed standard-normal noise tensors, the
model seed, the minibatch order and the GPM restart seed are those of original
draw ``j``; only the amplitude changes, exactly as in the p=10 supplemental
noise stage.  Because ``_lambda_draw`` returns ``upper * u`` from a seed that
depends only on the draw index, the supplemental level is exactly half of the
published level, which this module asserts before any fit is executed.

The original 103-record stages are never touched: the supplemental records are
written to a sibling stage directory inside a fresh results root, and the
regenerated anchor matrix and GDP parameter bank are verified against the
hashes recorded in the published result tables.
"""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import celeba_public_svd_identity100_runner as _base
import celeba_public_svd_identity100_unit_isometric_runner as _unit
import celeba_participant_replay_runner as _replay
from celeba_participant_replay_runner import ExperimentConfig


SCHEMA_VERSION = 1
SUPPLEMENT_SCHEDULE_VERSION = (
    "celeba-participant-replay-supplemental-anchor-v0-0p05-100-v1"
)
ALLOWED_PARTICIPANTS = _replay.ALLOWED_PARTICIPANTS
MASTER_PARTICIPANTS = _replay.MASTER_PARTICIPANTS
SAMPLES = 100
ANCHOR_V_LOWER = 0.0
ANCHOR_V_UPPER = 0.05
SOURCE_V_UPPER = _replay.ANCHOR_V_UPPER  # 0.1 in the published replay
ANCHOR_ROLE = "supplemental_anchor_v_0_0p05_v1"
SAMPLING_STRATUM = "anchor_v_dense_low"
STAGE_DIRECTORY = "anchor_v_0_0p05_100_supplemental_v1"
STORE_STEM = "celeba_participant_replay_supplemental_results"

# Provenance of the published replay (results/celeba/participant_replay/*.csv.gz
# in the reproducibility repository).  The regenerated design must reproduce
# these hashes, and the regenerated source levels must reproduce these draws.
PUBLISHED_ANCHOR_SHA256 = (
    "b0e6fe305579175c80a9b9e5ec33f0969b461814a93886541147b25fb69347a5"
)
PUBLISHED_ROTATIONS_SHA256 = (
    "a1209d1455c56af042cbae50c8f8f6e09bfd454b5ef2d41743ce7d916b2539aa"
)
PUBLISHED_TRANSLATIONS_SHA256 = (
    "51e2a019c876dd2de403dd6951b0c4dba9bbc3286983b8b337af569858bcc2a1"
)
PUBLISHED_ACTIVE_ROTATIONS_SHA256 = {
    2: "daba2a215826accf",
    5: "3192d6b59b1f1996",
    10: "3cb43b93ccc3eb04",
    20: "a744be7824985f8a",
    50: "a1209d1455c56af0",
}
# The 100 published draws v_j = 0.1 * u_j of the replay (identical for every p).
PUBLISHED_SOURCE_V = (
    0.099118925649949, 0.08792999835433495, 0.038525254557954164, 0.07647020528933253,
    0.02789945711831271, 0.03735196146165041, 0.08659290247476965, 0.04960084287501074,
    0.07871073740219854, 0.010218691066049002, 0.09565680397584862, 0.020805473271928023,
    0.05797032749139372, 0.045759964465254536, 0.0935535763408524, 0.08038850020641401,
    0.06701655683156225, 0.09538469552832979, 0.027257192391108544, 0.0568720653791088,
    0.06881025679662807, 0.07423694025060126, 0.012850343541073773, 0.09842852545240932,
    0.08481580502642115, 0.05644340024331395, 0.0702525427880074, 0.0019770103210860145,
    0.038867120731537, 0.04762717742617896, 0.07895355473604097, 0.026271912730994607,
    0.08512809847976535, 0.08020792406881458, 0.0550012417977439, 0.08819904317260024,
    0.04212801572922758, 0.016857108306307902, 0.056343410052564206, 0.09417145634954106,
    0.09325060293430838, 0.048273144154011786, 0.07364688827187348, 0.08556307935291256,
    0.0016054705157957927, 0.06599017163253494, 0.06421691612490191, 0.0782951866534592,
    0.053916275957916926, 0.02212388587984875, 0.06162461092540982, 0.06210300039552197,
    0.07618191259464892, 0.036650298352211744, 0.08964568117432625, 0.09150611633060596,
    0.013007881160401115, 0.04654874612197089, 0.017060481621001468, 0.043365788909983286,
    0.05807593670259843, 0.07040241359071826, 0.025638148674904684, 0.005069325918935341,
    0.035131528505747245, 0.023743683863911725, 0.0010480691991081815, 0.0661119495388775,
    0.09134218310219962, 0.05238855205294762, 0.0857871284824002, 0.03982673356030678,
    0.020030381265356424, 0.09871213526268532, 0.08625892773829688, 0.08746506914725001,
    0.03117030897392439, 0.03712714448093474, 0.06566766446721117, 0.09855063319348357,
    0.009181245738348344, 0.05288958319328475, 0.05477890963891716, 0.029037714418139585,
    0.07557169395728687, 0.09858815156588266, 0.07743569381941, 0.020870657323210064,
    0.09486424841671359, 0.007754039815766301, 0.006959376437943499, 0.07809473192326351,
    0.07612860860531621, 0.056001334757985646, 0.09693352147797886, 0.004931895696543742,
    0.009162168782255287, 0.052886663699241214, 0.04361283929310427, 0.008759775606103715,
)


def _utc_now() -> str:
    return _replay._utc_now()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _replay._atomic_json(path, payload)


def _levels_sha256(levels: Sequence[float]) -> str:
    return _base.stable_hash([float(value) for value in levels], 64)


def _config_for(p: int, out: Path, **overrides: Any) -> ExperimentConfig:
    """The published replay configuration for ``p``; only output_dir differs."""

    return ExperimentConfig.for_mode(
        "full",
        n_participants=int(p),
        max_participants=MASTER_PARTICIPANTS,
        output_dir=str(out),
        private_lambda_upper=0.5,
        anchor_lambda_upper=SOURCE_V_UPPER,
        anchor_v_upper=SOURCE_V_UPPER,
        **overrides,
    )


def _schedule(cfg: ExperimentConfig) -> list[dict[str, Any]]:
    """One hundred paired low-noise draws with the published draw quantiles."""

    schedule: list[dict[str, Any]] = []
    for draw in range(SAMPLES):
        source_level = _base._lambda_draw(cfg, SOURCE_V_UPPER, draw, "anchor")
        level = _base._lambda_draw(cfg, ANCHOR_V_UPPER, draw, "anchor")
        quantile = source_level / SOURCE_V_UPPER
        if not 0.0 < quantile < 1.0:
            raise RuntimeError(f"Paired quantile for draw {draw} is not in (0, 1)")
        if not math.isclose(level, ANCHOR_V_UPPER * quantile, rel_tol=1.0e-12, abs_tol=0.0):
            raise RuntimeError(f"Draw {draw} does not rescale the published quantile")
        if not ANCHOR_V_LOWER < level < ANCHOR_V_UPPER:
            raise RuntimeError(f"Supplemental v for draw {draw} is outside (0, 0.05)")
        schedule.append(
            {
                "method": "pa_i_gdp",
                "outer": 0,
                "draw": draw,
                "run_role": ANCHOR_ROLE,
                "level": level,
                "sampling_stratum": SAMPLING_STRATUM,
                "supplement_sample_index": draw,
                "paired_draw": draw,
                "paired_uniform_quantile": quantile,
                "paired_source_level": source_level,
                "requested_anchor_v": level,
            }
        )
    levels = [float(item["level"]) for item in schedule]
    if len(set(levels)) != SAMPLES:
        raise AssertionError("The 100 supplemental draws are not unique")
    return schedule


def _fit_specification(item: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item[key] for key in ("method", "outer", "draw", "run_role", "level")}


def _validate_source_levels(cfg: ExperimentConfig, schedule: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The regenerated source draws must be exactly the published v ~ U(0, 0.1) draws."""

    source = [float(item["paired_source_level"]) for item in schedule]
    if len(source) != len(PUBLISHED_SOURCE_V):
        raise RuntimeError("The published replay lists a different number of draws")
    for index, (observed, expected) in enumerate(zip(source, PUBLISHED_SOURCE_V)):
        if observed != float(expected):
            raise RuntimeError(
                f"Regenerated source draw {index} is {observed!r}, published is {expected!r}; "
                "the configuration seed does not reproduce the published replay"
            )
    return {
        "source_v_upper": SOURCE_V_UPPER,
        "source_levels_sha256": _levels_sha256(source),
        "supplemental_levels_sha256": _levels_sha256([float(item["level"]) for item in schedule]),
        "seed": int(cfg.seed),
    }


def _validate_design(design: Mapping[str, Any]) -> dict[str, str]:
    anchor_sha = str(design["anchor_spec"].get("tensor_sha256_float64_c_order"))
    rotations_sha = str(design["deployment_spec"].get("rotations_sha256_float64_c_order"))
    translations_sha = str(design["deployment_spec"].get("translations_sha256_float64_c_order"))
    mismatches = {
        label: {"expected": expected, "actual": actual}
        for label, expected, actual in (
            ("anchor", PUBLISHED_ANCHOR_SHA256, anchor_sha),
            ("rotations", PUBLISHED_ROTATIONS_SHA256, rotations_sha),
            ("translations", PUBLISHED_TRANSLATIONS_SHA256, translations_sha),
        )
        if expected != actual
    }
    if mismatches:
        raise RuntimeError(f"Regenerated design differs from the published replay: {mismatches}")
    return {
        "anchor_sha256": anchor_sha,
        "rotations_sha256": rotations_sha,
        "translations_sha256": translations_sha,
    }


def _validate_active_deployment(p: int, deployment: Mapping[str, Any]) -> str:
    actual = str(deployment["deployment_spec"].get("active_rotations_sha256_float64_c_order"))
    expected_prefix = PUBLISHED_ACTIVE_ROTATIONS_SHA256[int(p)]
    if not actual.startswith(expected_prefix):
        raise RuntimeError(
            f"Active deployment for p={p} ({actual[:16]}) differs from the published prefix {expected_prefix}"
        )
    return actual


def _annotate_supplemental_record(
    record: Mapping[str, Any], item: Mapping[str, Any], store: Any, p: int
) -> dict[str, Any]:
    annotated = dict(record)
    observed = float(annotated["anchor_v"])
    requested = float(item["requested_anchor_v"])
    if not math.isclose(observed, requested, rel_tol=0.0, abs_tol=1.0e-15):
        raise RuntimeError("A supplemental fit used the wrong anchor-noise scale v")
    if not ANCHOR_V_LOWER < observed < ANCHOR_V_UPPER:
        raise RuntimeError("A supplemental anchor-noise scale is outside (0, 0.05)")
    annotated.update(
        {
            "supplement_schema_version": SCHEMA_VERSION,
            "supplement_schedule_version": SUPPLEMENT_SCHEDULE_VERSION,
            "source_stage": f"p{int(p):03d}/{STAGE_DIRECTORY}",
            "sampling_stratum": SAMPLING_STRATUM,
            "supplement_sample_index": int(item["supplement_sample_index"]),
            "paired_draw": int(item["paired_draw"]),
            "paired_uniform_quantile": float(item["paired_uniform_quantile"]),
            "paired_source_level": float(item["paired_source_level"]),
            "paired_source_v_upper": SOURCE_V_UPPER,
            "requested_anchor_v": requested,
            "paired_scale_only_design": True,
        }
    )
    _atomic_json(store.path(str(annotated["run_id"])), annotated)
    return annotated


def _runtime_lock(cfg: ExperimentConfig, out: Path) -> dict[str, Any]:
    lock = {
        **dict(_unit._runtime_environment_lock(cfg)),
        "replay_schema_version": _replay.SCHEMA_VERSION,
        "supplement_schema_version": SCHEMA_VERSION,
        "schedule_version": SUPPLEMENT_SCHEDULE_VERSION,
        "deterministic": bool(cfg.deterministic),
        "module_sha256": dict(cfg.module_sha256),
    }
    _base._write_or_validate_json(
        out / "runtime_environment_lock.json", lock, "supplemental runtime environment lock"
    )
    return lock


def _completion_if_valid(
    out: Path,
    store: Any,
    schedule: Sequence[Mapping[str, Any]],
    cfg: ExperimentConfig,
    runtime_lock: Mapping[str, Any],
) -> dict[str, Any] | None:
    manifest_path = out / "supplemental_replay_manifest.json"
    completion_path = out / "completion_summary.json"
    if not manifest_path.is_file() or not completion_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    immutable = {
        "schedule_version": SUPPLEMENT_SCHEDULE_VERSION,
        "participant_count": int(cfg.n_participants),
        "schedule": [dict(item) for item in schedule],
        "config_hash": _base.stable_hash(cfg.to_dict()),
        "runtime_environment_hash": _base.stable_hash(runtime_lock),
        "module_sha256": dict(cfg.module_sha256),
    }
    for key, expected in immutable.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"Completed supplemental metadata differs for {key}: "
                f"{manifest.get(key)!r} != {expected!r}"
            )
    expected_ids = [str(value) for value in manifest["expected_run_ids"]]
    records, missing = _replay._completed_records(store, expected_ids)
    if missing:
        return None
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("record_hash") != _base.stable_hash(records):
        raise RuntimeError("Completed supplemental record hash differs")
    return {**completion, "resumed_from_completed_stage": True}


def run_for_p(
    participant_count: int,
    results_root: str | Path,
    design: Mapping[str, Any],
    supplemental_module_sha256: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    p = int(participant_count)
    if p not in ALLOWED_PARTICIPANTS:
        raise ValueError(f"p must be one of {ALLOWED_PARTICIPANTS}")
    results_root = Path(results_root)
    out = results_root / f"p{p:03d}" / STAGE_DIRECTORY
    cfg = _config_for(p, out, **overrides)
    cfg = _replay._stage_config(cfg, out)
    runtime_lock = _runtime_lock(cfg, out)
    schedule = _schedule(cfg)
    level_locks = _validate_source_levels(cfg, schedule)
    design_locks = _validate_design(design)
    store = _base.AtomicRunStore(out, stem=STORE_STEM)
    completed = _completion_if_valid(out, store, schedule, cfg, runtime_lock)
    if completed is not None:
        print(f"[supplement] p={p}: stage already complete, skipping", flush=True)
        return completed

    runtime_path = out / "stage_runtime.json"
    manifest_path = out / "supplemental_replay_manifest.json"
    completion_path = out / "completion_summary.json"
    previous = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else {}
    runtime: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schedule_version": SUPPLEMENT_SCHEDULE_VERSION,
        "participant_count": p,
        "first_started_at_utc": previous.get("first_started_at_utc", _utc_now()),
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
    expected_ids: list[str] = []

    try:
        ctx = _base._make_context(cfg, out, include_test=True)
        if len(ctx.linkage_manifest.query_identities) != 100:
            raise AssertionError("The supplement must evaluate all 100 target identity clusters")
        deployment = _replay._active_deployment(ctx, cfg, design)
        active_sha = _validate_active_deployment(p, deployment)
        expected_ids = _replay._expected_run_ids(ctx, [_fit_specification(item) for item in schedule])
        provenance = {
            "master_manifest_hash": ctx.data.manifest["master_manifest_hash"],
            "basis_matrix_sha256": ctx.data.manifest["projection"]["basis_matrix_sha256"],
            "data_manifest_hash": _base.stable_hash(ctx.data.manifest),
            "linkage_manifest_hash": _base.stable_hash(ctx.linkage_manifest.to_dict()),
            "config_hash": _base.stable_hash(cfg.to_dict()),
            "runtime_environment_hash": _base.stable_hash(runtime_lock),
            "module_sha256": dict(cfg.module_sha256),
            "supplemental_module_sha256": supplemental_module_sha256,
            "published_design": design_locks,
            "published_active_rotations_sha256": active_sha,
            "published_source_levels": level_locks,
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "schedule_version": SUPPLEMENT_SCHEDULE_VERSION,
            "source_schedule_version": _replay.SCHEDULE_VERSION,
            "participant_count": p,
            "participant_grid": list(ALLOWED_PARTICIPANTS),
            "samples": SAMPLES,
            "run_role": ANCHOR_ROLE,
            "anchor_v_interval": {
                "lower": ANCHOR_V_LOWER,
                "upper": ANCHOR_V_UPPER,
                "lower_closed": False,
                "upper_closed": False,
            },
            "paired_design": (
                "reuse published draw index and quantile, participant noise tensors, "
                "model seed, minibatch order and GPM restart; halve the amplitude"
            ),
            "schedule": schedule,
            "expected_run_ids": expected_ids,
            "uniform_noise_seed_rule": "named_seed(seed,uniform-noise,anchor,draw)",
            "participant_noise_seed_rule": "named_seed(seed,noise,family,outer,draw,split,participant)",
            "config_hash": provenance["config_hash"],
            "runtime_environment_hash": provenance["runtime_environment_hash"],
            "module_sha256": provenance["module_sha256"],
            "provenance": provenance,
            "standalone_anchor": dict(deployment["anchor_spec"]),
            "standalone_deployment": dict(deployment["deployment_spec"]),
        }
        _base._write_or_validate_json(manifest_path, manifest, "supplemental replay schedule")
        runtime["phase"] = "scheduled-fits"
        runtime["expected_run_ids_hash"] = _base.stable_hash(expected_ids)
        _atomic_json(runtime_path, runtime)
        print(f"[supplement] p={p}: context ready after {time.perf_counter() - started:.0f}s; {len(schedule)} fits scheduled", flush=True)

        for item, run_id in zip(schedule, expected_ids):
            if store.contains(run_id):
                continue
            fit_started = time.perf_counter()
            record = _replay._execute(ctx, store, deployment, **_fit_specification(item))
            _annotate_supplemental_record(record, item, store, p)
            records, _ = _replay._completed_records(store, expected_ids)
            runtime.update(
                {
                    "completed_records": len(records),
                    "last_completed_run_id": run_id,
                    "latest_progress_at_utc": _utc_now(),
                }
            )
            _atomic_json(runtime_path, runtime)
            gpm = record.get("gpm") or {}
            print(
                f"[supplement] p={p} draw={int(item['draw']):3d} v={float(item['level']):.5f} "
                f"done {len(records):3d}/{len(schedule)} in {time.perf_counter() - fit_started:.0f}s "
                f"(GPM {'converged' if gpm.get('converged') else 'capped'}, "
                f"{int(gpm.get('iterations', 0))} it) elapsed {(time.perf_counter() - started) / 60:.1f} min",
                flush=True,
            )
    except BaseException as exc:
        completed_count = len(_replay._completed_records(store, expected_ids)[0]) if expected_ids else 0
        runtime.update(
            {
                "latest_attempt_completed_at_utc": _utc_now(),
                "latest_attempt_wall_seconds": time.perf_counter() - started,
                "completed_records": completed_count,
                "phase": "failed",
                "last_error": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_json(runtime_path, runtime)
        raise

    records, missing = _replay._scheduled_records(store, expected_ids, require_complete=True)
    if missing:
        raise AssertionError("A completed supplemental stage has missing records")
    json_path, csv_path = _replay._aggregate_scheduled_records(store, records)
    converged = sum(bool((row.get("gpm") or {}).get("converged", False)) for row in records)
    runtime.update(
        {
            "latest_attempt_completed_at_utc": _utc_now(),
            "latest_attempt_wall_seconds": time.perf_counter() - started,
            "completed_records": len(records),
            "summed_fit_runtime_seconds": float(sum(float(row.get("runtime_seconds", 0.0)) for row in records)),
            "phase": "completed",
            "completed": True,
        }
    )
    _atomic_json(runtime_path, runtime)
    result = {
        "mode": "participant-count-anchor-replay-supplement",
        "participant_count": p,
        "status": {"completed": len(records), "planned": len(expected_ids)},
        "run_role": ANCHOR_ROLE,
        "gpm_converged_records": converged,
        "gpm_capped_records": len(records) - converged,
        "anchor_v_min_observed": min(float(row["anchor_v"]) for row in records),
        "anchor_v_max_observed": max(float(row["anchor_v"]) for row in records),
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
    print(f"[supplement] p={p}: completed {len(records)} fits, {converged} converged, csv {csv_path}", flush=True)
    return result


def run_shard(
    participant_counts: Sequence[int],
    results_root: str | Path,
    supplemental_module_sha256: str | None = None,
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
    design_root = results_root / "shards" / shard_name / "frozen_design"
    bank_cfg = _config_for(MASTER_PARTICIPANTS, design_root, **overrides)
    design = _replay._load_or_create_design(bank_cfg, design_root)
    design_locks = _validate_design(design)
    print(f"[supplement] shard {shard_name}: design verified against the published replay {design_locks}", flush=True)
    status_path = results_root / f"supplemental_shard_status_{shard_name}.json"
    previous = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schedule_version": SUPPLEMENT_SCHEDULE_VERSION,
        "participant_counts": list(selected),
        "first_started_at_utc": previous.get("first_started_at_utc", _utc_now()),
        "latest_attempt_started_at_utc": _utc_now(),
        "completed_participant_counts": list(previous.get("completed_participant_counts", [])),
        "active_participant_count": None,
        "completed": False,
    }
    _atomic_json(status_path, status)
    summaries: dict[str, Any] = {}
    for p in selected:
        status["active_participant_count"] = p
        _atomic_json(status_path, status)
        summary = run_for_p(
            p, results_root, design,
            supplemental_module_sha256=supplemental_module_sha256, **overrides,
        )
        summaries[str(p)] = summary
        if summary["status"]["completed"] == summary["status"]["planned"]:
            status["completed_participant_counts"] = sorted(set(status["completed_participant_counts"]) | {p})
        status["active_participant_count"] = None
        _atomic_json(status_path, status)
    status.update(
        {
            "latest_attempt_completed_at_utc": _utc_now(),
            "completed": set(status["completed_participant_counts"]) >= set(selected),
        }
    )
    _atomic_json(status_path, status)
    return {
        "mode": "participant-count-anchor-replay-supplement-shard",
        "participant_counts": list(selected),
        "status": status,
        "summaries": summaries,
        "standalone_design": {"root": str(design_root), **design_locks},
        "status_path": str(status_path),
    }


def run_static_self_tests() -> dict[str, Any]:
    cfg = ExperimentConfig.for_mode(
        "full",
        n_participants=2,
        max_participants=MASTER_PARTICIPANTS,
        private_lambda_upper=0.5,
        anchor_lambda_upper=SOURCE_V_UPPER,
        anchor_v_upper=SOURCE_V_UPPER,
        device="cpu",
    )
    schedule = _schedule(cfg)
    locks = _validate_source_levels(cfg, schedule)
    published = _replay.paired_anchor_levels()
    levels = [float(item["level"]) for item in schedule]
    source = [float(item["paired_source_level"]) for item in schedule]
    identities = [(row["method"], row["draw"], row["run_role"], row["level"]) for row in schedule]
    checks = {
        "one_hundred_draws": len(schedule) == SAMPLES,
        "draw_indices_0_to_99": [row["draw"] for row in schedule] == list(range(SAMPLES)),
        "range_is_0_to_0p05": all(ANCHOR_V_LOWER < value < ANCHOR_V_UPPER for value in levels),
        "source_is_published_replay_schedule": source == published,
        "exactly_half_of_published_draw": all(
            math.isclose(new, 0.5 * old, rel_tol=1.0e-12, abs_tol=0.0) for new, old in zip(levels, source)
        ),
        "published_draws_reproduced": source == [float(value) for value in PUBLISHED_SOURCE_V],
        "unique_schedule_identities": len(identities) == len(set(identities)),
        "role_is_new": ANCHOR_ROLE not in {"random", "clean", "endpoint_zero", "endpoint_upper", "smoke"},
        "stage_directory_is_new": STAGE_DIRECTORY != _replay.STAGE_DIRECTORY,
    }
    report = {"checks": checks, "locks": locks, "all_passed": all(checks.values())}
    if not report["all_passed"]:
        raise AssertionError(report)
    return report


__all__ = [
    "run_shard",
    "run_for_p",
    "run_static_self_tests",
    "STAGE_DIRECTORY",
    "SUPPLEMENT_SCHEDULE_VERSION",
    "ANCHOR_ROLE",
]
