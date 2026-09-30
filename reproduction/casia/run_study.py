"""Single-worker all-p CASIA h400 CNN study with user-approved revised bins."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import math
from pathlib import Path
import re
import socket
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
PARTICIPANTS = (2, 5, 10, 20, 50)
VERSION = "casia-allp-h400-cnn-spectral-rebinned-20260928-v2"
PLANNED_FITS = sum(202 + p for p in PARTICIPANTS)
BINS = (0, 15, 25, 35, 45, 100)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def immutable(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise RuntimeError(f"Resume identity changed: {path}")
    else:
        write(path, value)


@contextmanager
def single_worker(output):
    path = Path(output) / ".worker.lock"
    token = uuid.uuid4().hex
    value = {"pid": os.getpid(), "host": socket.gethostname(), "token": token,
             "started_utc": utc(), "output": str(Path(output).resolve())}
    try:
        with path.open("x", encoding="utf-8") as f:
            json.dump(value, f)
    except FileExistsError as exc:
        raise RuntimeError(f"Worker lock exists: {path}. Inspect its owner before resuming; "
                           "never launch a duplicate or automatically clear a foreign-runtime lock.") from exc
    try:
        yield value
    finally:
        if path.exists() and read(path).get("token") == token:
            path.unlink()


def verify_record(directory, signature):
    directory = Path(directory).resolve()
    path = directory / "record.json"
    if not path.exists():
        return None
    row = read(path)
    if row["condition_signature"] != signature:
        raise RuntimeError(f"Completed condition signature changed: {directory}")
    if not row.get("artifact_hashes"):
        raise RuntimeError(f"Completed condition has no committed artifacts: {directory}")
    for relative, expected in row["artifact_hashes"].items():
        artifact = (directory / relative).resolve()
        if directory not in artifact.parents:
            raise RuntimeError(f"Completed artifact escapes condition directory: {relative}")
        if not artifact.is_file() or digest(artifact) != expected:
            raise RuntimeError(f"Completed artifact missing or changed: {artifact}")
    return row


def validate_schedule(conditions, p, complete=True):
    if p not in PARTICIPANTS:
        raise RuntimeError("Unsupported participant count")
    counts = Counter(c["family"] for c in conditions)
    expected = Counter(anchor=100, private=100, local=p, i_gdp=1, c_gdp=1)
    if len({c["id"] for c in conditions}) != len(conditions):
        raise RuntimeError("Duplicate condition IDs")
    if (any(f not in expected for f in counts)
            or any(counts[f] != expected[f] for f in ("local", "i_gdp", "c_gdp"))
            or (complete and counts != expected)):
        raise RuntimeError(f"Wrong p={p} condition composition: {counts}")
    if {c.get("participant") for c in conditions if c["family"] == "local"} != set(range(p)):
        raise RuntimeError("Local participants differ")
    for condition in conditions:
        if condition.get("p") != p or not re.fullmatch(r"[A-Za-z0-9_-]+", str(condition["id"])):
            raise RuntimeError("Invalid condition identity")
    for family in ("anchor", "private"):
        selected = [c for c in conditions if c["family"] == family]
        bins = Counter(c["bin"] for c in selected)
        if any(b not in range(5) or n > 20 for b, n in bins.items()) or (complete and bins != Counter({i: 20 for i in range(5)})):
            raise RuntimeError(f"Wrong {family} bin quotas: {bins}")
        for condition in selected:
            value, scale = condition["linkage_percent"], condition["scale"]
            if (not math.isfinite(value) or not 0 <= value <= 100 or not math.isfinite(scale)
                    or not 0 < scale <= {"anchor": .08, "private": 3.}[family]):
                raise RuntimeError("Invalid linkage/noise scale")
            observed_bin = next(i for i in range(5) if value < BINS[i + 1] or i == 4)
            if condition["bin"] != observed_bin:
                raise RuntimeError("Condition linkage disagrees with declared bin")


def validate_spec(spec):
    required = {"participants": list(PARTICIPANTS), "sampling_mode": "quota_bins", "h": 400,
                "linkage_bin_edges_percent": list(BINS), "conditions_per_bin_per_family": 20,
                "max_proposals_per_family": 1500, "planned_training_fits": PLANNED_FITS,
                "gpm_iterations": 0, "paired_gpm_comparison": False,
                "epochs": 15, "batch_size": 256, "learning_rate": .001, "weight_decay": 0,
                "master_participants": 50, "identities_per_participant": 100,
                "identity_split_counts": {"train": 70, "val": 15, "test": 15},
                "public_svd_images": 10000, "raw_shape": [3, 64, 64],
                "cnn_input_shape": [1, 20, 20], "anchor_rows": 1600,
                "seed": 20260928, "test_threshold": .5,
                "reference_treatment": "projected_decoded", "anchor_attack": "PA-OP",
                "private_attack": "known-secret", "audit_query_identities": 100,
                "audit_episodes_per_identity": 10, "audit_candidates_per_episode": 10,
                "noise_scale_upper_bounds": {"anchor": .08, "private": 3.}}
    for key, expected in required.items():
        if spec.get(key) != expected:
            raise RuntimeError(f"Authorized fixed recipe differs: {key}")
    if str(spec.get("dataset", "")).lower() not in ("celeba", "vggface2"):
        raise RuntimeError("Unsupported dataset")
    dataset_seed = {"celeba": 20260713, "vggface2": 20260928}[spec["dataset"].lower()]
    if spec.get("dataset_seed") != dataset_seed:
        raise RuntimeError("Authorized fixed recipe differs: dataset_seed")
    if not isinstance(spec.get("seed"), int) or not 1 <= spec.get("cpu_threads", 0) <= 8:
        raise RuntimeError("Invalid deterministic seed/thread configuration")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", spec.get("input_adapter", "")):
        raise RuntimeError("Invalid input adapter module")
    return spec


def source_hashes(config_path, module_dir):
    """Hash shipped code and selected configuration, never mutable output directories."""
    result = {path.name: digest(path) for path in sorted(HERE.glob("*.py"))
              if not path.name.startswith("test_")}
    result["vgg_asset_pins.json"] = digest(HERE / "vgg_asset_pins.json")
    result["selected_config"] = digest(config_path)
    return result


def local_pooled_summary(rows, p):
    """Pool held-out local confusion counts before calculating balanced accuracy."""
    local = [row for row in rows if row["condition"]["family"] == "local"]
    if len(local) != p or {row["condition"]["participant"] for row in local} != set(range(p)):
        raise RuntimeError("Local summary requires exactly one model per participant")
    totals = [[0, 0], [0, 0]]
    for row in local:
        test = row["result"]["test"]
        matrix = test["confusion_matrix"]
        if (len(matrix) != 2 or any(len(values) != 2 for values in matrix)
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                       for values in matrix for value in values)
                or sum(sum(values) for values in matrix) != test["n"]):
            raise RuntimeError("Invalid local test confusion counts")
        for i in range(2):
            for j in range(2):
                totals[i][j] += matrix[i][j]
    (tn, fp), (fn, tp) = totals
    if tn + fp == 0 or fn + tp == 0:
        raise RuntimeError("Pooled local test set must contain both classes")
    return {"n_models": p, "n": tn + fp + fn + tp,
            "confusion_matrix": totals,
            "test_balanced_accuracy": 0.5 * (tn / (tn + fp) + tp / (tp + fn)),
            "test_accuracy": (tn + tp) / (tn + fp + fn + tp),
            "privacy": None, "aggregation": "pooled test confusion counts"}


def resolve_execution_identity(output, current_identity, migration_sha256=None):
    if migration_sha256 is not None:
        raise RuntimeError("This isolated study never imports an earlier study identity")
    return current_identity, None


def save_summary(path, summary, execution_migration=None):
    if execution_migration is not None:
        raise RuntimeError("Cross-study migration is forbidden")
    write(path, summary)
    return summary


def train_one(ctx, condition, root_signature, output, seed, device, report,
              execution_migration=None):
    import data_pipeline
    from training import train_condition
    directory = Path(output) / "conditions" / condition["id"]
    signature = object_hash({"study_signature": root_signature, "condition": condition,
                             "training_seed": seed})
    existing = verify_record(directory, signature)
    if existing is not None:
        data_pipeline.release_arrays(ctx, condition)
        report(stage="condition_reused", condition=condition["id"])
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    immutable(directory / "condition.json", {"condition": condition, "signature": signature})
    report(stage="building_collaboration", condition=condition["id"])
    started = time.perf_counter()
    arrays = data_pipeline.build_arrays(ctx, condition)
    preparation_seconds = time.perf_counter() - started
    metadata = arrays.get("metadata", {})
    metadata["alignment_seconds"] = float(arrays.get("alignment_seconds", 0.0))
    metadata["collaboration_preparation_seconds"] = preparation_seconds
    # Timings legitimately differ when resuming an unfinished fit. The trainer
    # separately locks hashes of the exact input arrays and scientific recipe.
    write(directory / "input_metadata.json", metadata)
    result = train_condition(*(arrays[key] for key in
        ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y")), directory,
        seed=seed, condition_signature=signature, device=device,
        progress=lambda event: report(condition=condition["id"], **event))
    row = {"version": VERSION, "condition": condition, "condition_signature": signature,
           "training_seed": seed, "result": result, "input_metadata": metadata,
           "completed_utc": utc(), "attempt_seconds": time.perf_counter() - started}
    if execution_migration is not None:
        row["execution_migration"] = execution_migration
    # The trainer owns portable model, optimizer/checkpoint and predictions.
    # The outer receipt verifies every committed file before skipping a fit.
    row["artifact_hashes"] = {str(f.relative_to(directory)): digest(f)
                             for f in sorted(directory.rglob("*")) if f.is_file()
                             and f.name not in ("record.json", "record.json.tmp")
                             and not f.name.endswith(".tmp")}
    write(directory / "record.json", row)
    verify_record(directory, signature)
    # Memmap views must be dropped before removing only the adapter-owned,
    # regenerable condition cache. Failed fits retain it for checkpoint resume.
    del arrays
    gc.collect()
    data_pipeline.release_arrays(ctx, condition)
    report(stage="condition_complete", condition=condition["id"], family=condition["family"])
    return row


def run(args):
    import data_pipeline
    import training
    output = Path(args.output).resolve()
    config_path = Path(args.config).resolve()
    spec = validate_spec(read(config_path))
    for path in (config_path.parent, HERE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    frozen_root = Path(args.source_root or spec["source_root"]).resolve()
    module_dir = Path(args.module_dir or spec.get("module_dir", HERE)).resolve()
    cache_root = Path(args.cache_root or spec.get("cache_root") or output.parent / "prepared").resolve()
    protocol = {**spec, "study_root": str(output.parent), "module_dir": str(module_dir),
                "model_cache_dir": str(output.parent / "model_assets"),
                "design_root": str(output.parent / "shared_design"),
                "array_cache_root": spec.get("array_cache_root", str(output.parent / "working_arrays"))}
    data_pipeline.configure(protocol)
    if not frozen_root.is_dir() or output == frozen_root or frozen_root in output.parents:
        raise RuntimeError("Use the downloaded raw source and a separate new output root")
    output.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(spec["cpu_threads"])
    sources = source_hashes(config_path, module_dir)
    runtime = training.runtime_probe(args.device)
    if args.device == "cuda" and "T4" not in runtime.get("gpu", ""):
        raise RuntimeError("This notebook was authorized for a T4; select a T4 runtime")
    current_identity = {"version": VERSION, "specification": spec, "source_hashes": sources,
                    "effective_protocol": protocol, "runtime": runtime, "frozen_root": str(frozen_root),
                    "cache_root": str(cache_root), "output": str(output)}
    state = {"version": VERSION, "dataset": spec["dataset"], "planned_fits": PLANNED_FITS, "started_utc": utc(),
             "worker_pid": os.getpid(), "worker_host": socket.gethostname()}

    def report(**event):
        state.update(event, updated_utc=utc())
        write(output / "status.json", state)
        print("CASIA_SPECTRAL", json.dumps(state, allow_nan=False), flush=True)

    with single_worker(output):
        try:
            run_identity, execution_migration = resolve_execution_identity(
                output, current_identity, getattr(args, "validation_migration_sha256", None))
            study_signature = object_hash(run_identity)
            immutable(output / "run_identity.json", run_identity)
            report(state="running", stage="runtime_pilot")
            pilot_path = output / "runtime_pilot.json"
            if not pilot_path.exists():
                pilot_result = training.pilot(output / "runtime_pilot", device=args.device)
                write(pilot_path, {"study_signature": study_signature, "result": pilot_result})
            elif read(pilot_path)["study_signature"] != study_signature:
                raise RuntimeError("Runtime pilot signature differs")
            summaries = []
            for p in PARTICIPANTS:
                directory = output / f"p{p:03d}"
                directory.mkdir(exist_ok=True)
                report(p=p, stage="loading_frozen_context", completed_at_p=0, calibration_complete=None)
                ctx = data_pipeline.load_context(frozen_root, directory, p, module_dir,
                         audit_device="cuda" if args.device == "cuda" else "cpu", protocol_config=protocol,
                         cache_root=cache_root, progress=lambda event=None, **fields: report(**(event or {}), **fields))
                provenance = data_pipeline.context_provenance(ctx)
                immutable(directory / "input_provenance.json", provenance)
                p_signature = object_hash({"study_signature": study_signature, "p": p,
                                           "input_provenance": provenance})
                controls = data_pipeline.control_conditions(p)
                # A real full 15-epoch fit validates workload timing before calibration.
                c_control = next(c for c in controls if c["family"] == "c_gdp")
                first = train_one(ctx, c_control, p_signature, directory, spec["seed"], args.device, report,
                                  execution_migration=execution_migration)
                report(completed_at_p=1, planned_at_p=202 + p,
                       completed_fits=sum(s["completed_conditions"] for s in summaries) + 1)
                if args.pilot_only:
                    write(output / "real_pilot.json", {"p": p, "record": first,
                         "note": "Production C-GDP fit; reused in the full sweep. No A100 speedup claim."})
                    report(state="pilot_complete", stage="awaiting_full_sweep", completed_fits=1)
                    return
                report(stage="linkage_calibration", condition=None)
                calibration_complete = True
                try:
                    noisy = data_pipeline.calibrate(ctx, directory, progress=lambda event: report(**event))
                except data_pipeline.CalibrationIncomplete as exc:
                    calibration_complete = False
                    noisy = exc.conditions
                    report(stage="limited_bins", calibration_complete=False, accepted_noisy=len(noisy))
                conditions = controls + noisy
                validate_schedule(conditions, p, complete=calibration_complete)
                report(stage="fitting_scheduled_conditions", calibration_complete=calibration_complete,
                       accepted_noisy=len(noisy))
                immutable(directory / "schedule.json", {"p": p, "conditions": conditions,
                          "signature": p_signature, "sampling_mode": "quota_bins",
                          "calibration_complete": calibration_complete})
                rows = []
                for condition in conditions:
                    # Same seed across noisy conditions; local seeds remain participant-specific.
                    seed = spec["seed"] + int(condition.get("participant", 0)) if condition["family"] == "local" else spec["seed"]
                    rows.append(train_one(ctx, condition, p_signature, directory, seed, args.device, report,
                                          execution_migration=execution_migration))
                    report(completed_at_p=len(rows), planned_at_p=202 + p,
                           completed_fits=sum(s["completed_conditions"] for s in summaries) + len(rows))
                summary = {"state": "complete" if calibration_complete else "incomplete_bins",
                           "calibration_complete": calibration_complete, "p": p, "completed_conditions": len(rows),
                           "planned_conditions": 202 + p,
                           "completed_ids": [row["condition"]["id"] for row in rows],
                           "family_counts": dict(Counter(row["condition"]["family"] for row in rows)),
                           "sampling_mode": "quota_bins", "bin_edges_percent": list(BINS),
                           "bin_counts": {f: {str(b): sum(c["family"] == f and c.get("bin") == b for c in conditions)
                                                for b in range(5)} for f in ("anchor", "private")},
                           "clean_linkage": read(directory / "clean_linkage.json"),
                           "clean_linkage_sha256": digest(directory / "clean_linkage.json"),
                           "local_pooled": local_pooled_summary(rows, p),
                           "records": rows, "completed_utc": utc()}
                summary = save_summary(directory / "summary.json", summary, execution_migration)
                summaries.append({k: v for k, v in summary.items() if k != "records"})
                if hasattr(data_pipeline, "release_context"):
                    data_pipeline.release_context(ctx)
                del ctx, rows
                gc.collect()
            if any(not s["calibration_complete"] for s in summaries):
                incomplete = {"state": "incomplete_bins", "planned_fits": PLANNED_FITS,
                    "completed_fits": sum(s["completed_conditions"] for s in summaries),
                    "study_signature": study_signature, "participant_summaries": summaries,
                    "completed_utc": utc(), "reason": "One or more fixed linkage-bin quotas unfilled"}
                write(output / "incomplete.json", incomplete)
                report(state="incomplete_bins", stage="bounded_sweep_finished", completed_fits=incomplete["completed_fits"])
                return
            if sum(s["completed_conditions"] for s in summaries) != PLANNED_FITS:
                raise RuntimeError("Full sweep condition count differs")
            completion = {"state": "complete", "planned_fits": PLANNED_FITS, "completed_fits": PLANNED_FITS,
                  "study_signature": study_signature, "participant_summaries": summaries,
                  "completed_utc": utc()}
            if execution_migration is not None:
                completion["execution_migration"] = execution_migration
            write(output / "completion.json", completion)
            report(state="complete", stage="complete", completed_fits=PLANNED_FITS)
        except BaseException as exc:
            report(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                   error=f"{type(exc).__name__}: {exc}")
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--module-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--pilot-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
