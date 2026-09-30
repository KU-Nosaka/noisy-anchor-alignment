"""Prepare one dataset and supervise its isolated all-p T4 study.

Only a verified full 1097-fit receipt is completion. Exhausted revised bin
quotas produce a separate incomplete receipt, after fitting accepted conditions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import gc
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import uuid

from run_study import (BINS, HERE, PARTICIPANTS, PLANNED_FITS, VERSION, digest, immutable,
                       local_pooled_summary, object_hash, read, source_hashes, utc,
                       validate_schedule, validate_spec, verify_record, write)


def effective_protocol(spec, root, module_dir):
    root = Path(root).resolve()
    return {**spec, "study_root": str(root), "module_dir": str(Path(module_dir).resolve()),
            "model_cache_dir": str(root / "model_assets"), "design_root": str(root / "shared_design"),
            "array_cache_root": spec.get("array_cache_root", str(root / "working_arrays"))}


def _validate_participant(study, p, identity, study_signature):
    directory = study / f"p{p:03d}"
    summary = read(directory / "summary.json")
    full = summary.get("calibration_complete") is True
    if (summary.get("state") != ("complete" if full else "incomplete_bins")
            or summary.get("p") != p or summary.get("planned_conditions") != 202 + p
            or summary.get("sampling_mode") != "quota_bins" or summary.get("bin_edges_percent") != list(BINS)):
        raise RuntimeError(f"Invalid p={p} summary state or scientific recipe")
    records = summary.get("records", [])
    ids = [row["condition"]["id"] for row in records]
    if (len(ids) != len(set(ids)) or ids != summary.get("completed_ids")
            or len(ids) != summary.get("completed_conditions")):
        raise RuntimeError("Participant receipt count/IDs differ")
    provenance = read(directory / "input_provenance.json")
    if not isinstance(provenance.get("input_hash"), str) or len(provenance["input_hash"]) != 64:
        raise RuntimeError("Participant scientific input hash is missing")
    p_signature = object_hash({"study_signature": study_signature, "p": p,
                              "input_provenance": provenance})
    schedule = read(directory / "schedule.json")
    if (schedule.get("p") != p or schedule.get("signature") != p_signature
            or schedule.get("sampling_mode") != "quota_bins" or schedule.get("calibration_complete") is not full):
        raise RuntimeError("Schedule input identity/state differs")
    conditions = schedule["conditions"]
    validate_schedule(conditions, p, complete=full)
    if any(c.get("input_hash") != provenance["input_hash"] for c in conditions if c["family"] in ("anchor", "private")):
        raise RuntimeError("Noisy condition scientific input hash differs")
    if [row["condition"] for row in records] != conditions:
        raise RuntimeError("Summary differs from committed schedule")
    if summary.get("family_counts") != dict(Counter(c["family"] for c in conditions)):
        raise RuntimeError("Summary family counts differ")
    bins = {f: {str(b): sum(c["family"] == f and c.get("bin") == b for c in conditions)
                for b in range(5)} for f in ("anchor", "private")}
    if summary.get("bin_counts") != bins:
        raise RuntimeError("Summary bin counts differ")
    if full:
        accepted = read(directory / "calibration" / "accepted_conditions.json")
        if (accepted.get("input_hash") != provenance["input_hash"]
                or accepted.get("conditions_sha256") != object_hash(accepted.get("conditions"))
                or accepted.get("conditions") != [c for c in conditions if c["family"] in ("anchor", "private")]):
            raise RuntimeError("Accepted calibration conditions differ")
    clean_path = directory / "clean_linkage.json"
    if summary.get("clean_linkage_sha256") != digest(clean_path) or summary.get("clean_linkage") != read(clean_path):
        raise RuntimeError("Clean CASIA baseline differs")
    for row in records:
        condition = row["condition"]
        expected_seed = identity["specification"]["seed"] + (int(condition["participant"]) if condition["family"] == "local" else 0)
        if row.get("training_seed") != expected_seed:
            raise RuntimeError("Condition training seed differs")
        signature = object_hash({"study_signature": p_signature, "condition": condition,
                                 "training_seed": expected_seed})
        condition_dir = directory / "conditions" / condition["id"]
        if verify_record(condition_dir, signature) != row:
            raise RuntimeError("Summary differs from verified condition receipt")
        result = read(condition_dir / "result.json")
        manifest = read(condition_dir / "completion_manifest.json")
        history = result.get("training", {}).get("history", [])
        if (result != row["result"] or result.get("status") != "complete"
                or result.get("training", {}).get("epochs") != 15 or len(history) != 15
                or [entry.get("epoch") for entry in history] != list(range(1, 16))
                or manifest.get("signature") != result.get("signature")):
            raise RuntimeError("Training result is incomplete or its identity differs")
        artifacts = manifest.get("artifacts", {})
        if set(artifacts) != {"model", "predictions", "result"}:
            raise RuntimeError("Trainer completion lacks mandatory artifacts")
        for artifact in artifacts.values():
            name = artifact["path"]
            if Path(name).name != name or "/" in name or "\\" in name or digest(condition_dir / name) != artifact["sha256"]:
                raise RuntimeError("Trainer artifact changed or escaped condition directory")
    if summary.get("local_pooled") != local_pooled_summary(records, p):
        raise RuntimeError("Pooled local confusion summary differs")
    return summary


def validate_completion(root):
    """Validate all conditions in either full or honestly incomplete terminal state."""
    root = Path(root)
    study = root / "study"
    complete_path, incomplete_path = study / "completion.json", study / "incomplete.json"
    if complete_path.exists() == incomplete_path.exists():
        raise RuntimeError("Study lacks a unique durable terminal receipt")
    complete = complete_path.exists()
    path = complete_path if complete else incomplete_path
    terminal = read(path)
    identity = read(study / "run_identity.json")
    validate_spec(identity["specification"])
    signature = object_hash(identity)
    if (terminal.get("study_signature") != signature or terminal.get("planned_fits") != PLANNED_FITS
            or terminal.get("state") != ("complete" if complete else "incomplete_bins")):
        raise RuntimeError("Terminal study identity/state differs")
    summaries = [_validate_participant(study, p, identity, signature) for p in PARTICIPANTS]
    count = sum(s["completed_conditions"] for s in summaries)
    all_complete = all(s["calibration_complete"] for s in summaries)
    if (terminal.get("completed_fits") != count or (complete and (count != PLANNED_FITS or not all_complete))
            or (not complete and (all_complete or count >= PLANNED_FITS))):
        raise RuntimeError("Terminal receipt misstates completed fits/bin coverage")
    if terminal.get("participant_summaries") != [{k: v for k, v in s.items() if k != "records"} for s in summaries]:
        raise RuntimeError("Terminal summaries differ from per-p receipts")
    return {"state": terminal["state"], "planned_fits": PLANNED_FITS, "completed_fits": count,
            "study_receipt_sha256": digest(path), "study_signature": signature,
            "summary_sha256": {str(p): digest(study / f"p{p:03d}" / "summary.json") for p in PARTICIPANTS},
            "reason": terminal.get("reason")}


@contextmanager
def pipeline_worker(root):
    path = Path(root) / ".pipeline.lock"
    owner = {"pid": os.getpid(), "host": socket.gethostname(), "token": uuid.uuid4().hex, "started_utc": utc()}
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(owner, handle)
    except FileExistsError as exc:
        raise RuntimeError("Pipeline lock exists; inspect owner, never launch a duplicate or auto-clear a foreign lock") from exc
    try:
        yield owner
    finally:
        if path.exists() and read(path).get("token") == owner["token"]:
            path.unlink()


class Progress:
    def __init__(self, root, initial):
        self.root, self.state = Path(root), dict(initial)
        self.mutex, self.stop = threading.Lock(), threading.Event()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)

    def __call__(self, event=None, **fields):
        with self.mutex:
            self.state.update(event or {}, **fields, updated_utc=utc())
            write(self.root / "pipeline_status.json", self.state)
            print("CASIA_PIPELINE", json.dumps(self.state, allow_nan=False), flush=True)

    def heartbeat(self):
        while not self.stop.wait(30):
            self(heartbeat_utc=utc())

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    child.send_signal(signal.SIGINT)
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def run(root, config, device="cuda", prepare_only=False):
    root, config = Path(root).resolve(), Path(config).resolve()
    spec = validate_spec(read(config))
    source = Path(spec["source_root"]).resolve()
    protected = [source]
    if spec.get("cache_root"):
        protected.append(Path(spec["cache_root"]).resolve())
    if spec.get("prepared_cache_root"):
        protected.append(Path(spec["prepared_cache_root"]).resolve())
    if any(root == path or root.is_relative_to(path) for path in protected):
        raise RuntimeError("New pipeline root must be outside original inputs/caches")
    root.mkdir(parents=True, exist_ok=True)
    module_dir = Path(spec.get("module_dir", HERE)).resolve()
    if str(config.parent) not in sys.path:
        sys.path.insert(0, str(config.parent))
    effective = effective_protocol(spec, root, module_dir)
    cache_root = Path(spec.get("cache_root") or root / "prepared").resolve()
    child = None
    with pipeline_worker(root) as owner, Progress(root, {"version": VERSION, "dataset": spec["dataset"],
                           "planned_fits": PLANNED_FITS, **owner}) as progress:
        try:
            import torch
            import training
            torch.set_num_threads(spec["cpu_threads"])
            runtime = training.runtime_probe(device)
            if device == "cuda" and "T4" not in runtime.get("gpu", ""):
                raise RuntimeError("Select the requested T4 GPU")
            identity = {"version": VERSION, "configuration": spec, "effective_protocol": effective,
                        "source_sha256": source_hashes(config, module_dir), "root": str(root), "runtime": runtime}
            immutable(root / "pipeline_identity.json", identity)
            progress(state="running", stage="preparing_master", runtime=runtime)
            import data_pipeline
            data_pipeline.configure(effective)
            prepared_source = data_pipeline.prepare_inputs(effective, root, progress=progress)
            if prepared_source is not None:
                if Path(prepared_source).resolve() != source:
                    raise RuntimeError("Preparation changed the declared source root")
            progress(stage="master_prepared")
            if prepare_only:
                progress(state="prepared", stage="prepared_not_trained")
                return
            # Public-SVD preparation ran in this supervisor. Release its Python
            # objects and cached CUDA workspace before the child allocates T4 RAM.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            command = [sys.executable, "-u", str(HERE / "run_study.py"), "--config", str(config),
                       "--source-root", str(source), "--cache-root", str(cache_root),
                       "--module-dir", str(module_dir), "--output", str(root / "study"), "--device", device]
            env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
            env["PYTHONPATH"] = os.pathsep.join([str(HERE), str(config.parent), str(module_dir), env.get("PYTHONPATH", "")])
            child = subprocess.Popen(command, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, env=env)
            write(root / "study_process.json", {"pid": child.pid, "host": socket.gethostname(), "command": command, "started_utc": utc()})
            progress(stage="study_running", study_pid=child.pid)
            with (root / "study.log").open("a", encoding="utf-8") as log:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    if line.startswith("CASIA_SPECTRAL "):
                        event = json.loads(line[len("CASIA_SPECTRAL "):])
                        progress(stage="study_" + str(event.get("stage", "running")),
                                 completed_fits=event.get("completed_fits", 0), p=event.get("p"),
                                 study_state=event.get("state"), study_updated_utc=event.get("updated_utc"))
            code = child.wait()
            if code:
                raise RuntimeError(f"Study exited {code}; checkpoints are preserved")
            terminal = validate_completion(root)
            terminal.update(completed_utc=utc(), pipeline_identity_sha256=digest(root / "pipeline_identity.json"))
            write(root / ("pipeline_completion.json" if terminal["state"] == "complete" else "pipeline_incomplete.json"), terminal)
            progress(terminal, stage=terminal["state"])
        except BaseException as exc:
            stop_child(child)
            progress(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(args.root, args.config, args.device, args.prepare_only)
