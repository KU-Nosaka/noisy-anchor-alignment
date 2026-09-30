"""One immutable, nested pmax50 VGGFace2 master for the full CASIA study."""
from __future__ import annotations
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import vgg_io as io

VERSION = "vggface2-full-uncapped-pmax50-20260928-v1"
SEED = 20260928
P_VALUES = (2, 5, 10, 20, 50)
PUBLIC_CANDIDATE_IDENTITIES = 100


def frozen_modules(module_dir):
    directory = str(Path(module_dir).resolve())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    base = importlib.import_module("celeba_public_svd_identity100_runner")
    core = importlib.import_module("core_protocols")
    if Path(base.__file__).resolve().parent != Path(directory):
        raise RuntimeError("Frozen protocol module came from a different directory")
    return base, core


def eligible_identity_ids(records):
    """Released train IDs with 51 defined labels and both Smiling classes."""
    return {row["numeric_identity"] for row in records if row["official_split"] == "train"
            and row["defined"] >= 51 and row["positive"] and row["negative"]}


def metadata_candidates(source_root, records, base, seed, reporter):
    """Decode all eligible-ID reserves plus a bounded pool of public IDs."""
    import pandas as pd
    train = {r["numeric_identity"]: r for r in records if r["official_split"] == "train"}
    eligible = eligible_identity_ids(records)
    public_order = base._hash_order(seed, "public-identities", train)[:PUBLIC_CANDIDATE_IDENTITIES]
    public_set = set(public_order)
    rows = {}
    total = 0
    for frame in pd.read_csv(source_root / "maad/MAAD_Face.csv", usecols=["Filename", "Smiling"],
                             dtype={"Filename": str, "Smiling": np.int8}, chunksize=200000):
        numeric = frame.Filename.str.slice(1, 7).astype(np.int32)
        subset = frame[numeric.isin(eligible) & frame.Smiling.ne(0)]
        for row in subset.itertuples(index=False):
            name = io.normalize_path(row.Filename)
            if name in rows:
                raise RuntimeError("Duplicate candidate MAAD filename")
            rows[name] = (int(name[1:7]), int(row.Smiling == 1))
        total += len(frame)
        reporter.emit("full_metadata_candidates", rows=total, candidate_images=len(rows))
    count = 0
    for count, name in enumerate(io.iter_train_names(source_root / "meta/train_list.txt"), 1):
        identity = int(name[1:7])
        if identity not in train:
            raise RuntimeError("Non-training identity in official training image list")
        if identity in public_set:
            rows.setdefault(name, (identity, -1))  # Public pixels need no utility label.
        if count % 10000 == 0:
            reporter.emit("public_identity_candidates", rows=count, candidate_images=len(rows))
    if count != 3141890 or len(eligible) != 6246:
        raise RuntimeError("Pinned metadata capacity differs from verified audit")
    return rows, eligible, public_order


def _unique_valid(names, results, used_rgb, used_source):
    kept, rejected = [], []
    local_rgb, local_source = set(), set()
    for name in names:
        receipt = results[name]
        reason = None
        if not receipt["valid"]:
            reason = receipt["reason"]
        else:
            rgb, source = receipt["rgb64_sha256"], receipt["source_image_sha256"]
            if rgb in used_rgb or rgb in local_rgb:
                reason = "exact_rgb64_duplicate"
            elif source in used_source or source in local_source:
                reason = "exact_source_bytes_duplicate"
        if reason is not None:
            rejected.append({"filename": name, "reason": reason})
        else:
            kept.append(name)
            local_rgb.add(rgb); local_source.add(source)
    return kept, rejected


def freeze_cohort(rows, eligible, public_order, results, candidate_index, base, seed):
    """Public-ID prefix first, then valid private identities, preserving original allocation."""
    by_id = {}
    for name, (identity, label) in rows.items():
        by_id.setdefault(identity, []).append(name)
    public_ids, public_pool, public_rejections = [], [], []
    used_rgb, used_source = set(), set()
    for identity in public_order:
        ordered = base._hash_order(seed, "public-images", by_id.get(identity, []))
        kept, rejected = _unique_valid(ordered, results, used_rgb, used_source)
        public_ids.append(identity); public_pool.extend(kept); public_rejections.extend(rejected)
        used_rgb.update(results[name]["rgb64_sha256"] for name in kept)
        used_source.update(results[name]["source_image_sha256"] for name in kept)
        if len(public_pool) >= 10000:
            break
    if len(public_pool) < 10000:
        raise RuntimeError("Fixed100 public candidate identities cannot supply10000 valid distinct images")
    public_names = base._hash_order(seed, "public-images", public_pool)[:10000]
    # Only selected public images need content exclusion; all public IDs remain private-excluded.
    used_rgb = {results[name]["rgb64_sha256"] for name in public_names}
    used_source = {results[name]["source_image_sha256"] for name in public_names}
    selected, rejected_ids, image_rejections = [], [], list(public_rejections)
    order = base._hash_order(seed, "participant-identities", eligible - set(public_ids))
    for identity in order:
        filenames = [name for name in by_id[identity] if rows[name][1] in (0, 1)]
        ordered = base._hash_order(seed, f"identity-{identity}-images", filenames)
        kept, rejected = _unique_valid(ordered, results, used_rgb, used_source)
        positives = sum(rows[name][1] for name in kept)
        if len(kept) < 51 or positives == 0 or positives == len(kept):
            rejected_ids.append({"identity": identity, "valid_defined_images": len(kept),
                                 "positive": positives, "reason": "post_qc_min51_or_both_labels_failed"})
            image_rejections.extend(rejected)
            continue
        reserved, protocol = kept[0], tuple(kept[1:])
        selected.append(base.IdentityStat(identity=identity, reserved_filename=reserved,
                         protocol_filenames=protocol,
                         protocol_labels=tuple(rows[name][1] for name in protocol),
                         protocol_sources=tuple(candidate_index[name] for name in protocol)))
        used_rgb.update(results[name]["rgb64_sha256"] for name in kept)
        used_source.update(results[name]["source_image_sha256"] for name in kept)
        image_rejections.extend(rejected)
        if len(selected) == 5000:
            break
    if len(selected) != 5000:
        raise RuntimeError(f"Only{len(selected)} valid participant identities; no QC relaxation")
    cfg = SimpleNamespace(seed=seed, max_participants=50, identities_per_participant=100)
    buckets = base._balanced_identity_allocation(selected, cfg)
    assignments = []
    for (participant, split), items in sorted(buckets.items()):
        for item in sorted(items, key=lambda x: x.identity):
            assignments.append({"participant": participant, "split": split, "identity": item.identity,
                "reserved_filename": item.reserved_filename,
                "reserved_source": candidate_index[item.reserved_filename],
                "protocol_filenames": list(item.protocol_filenames),
                "protocol_labels": list(item.protocol_labels), "protocol_sources": list(item.protocol_sources),
                "protocol_image_count": item.image_count, "protocol_positive_count": item.positive_count})
    for participant in range(50):
        for split, count in {"train": 70, "val": 15, "test": 15}.items():
            items = [r for r in assignments if r["participant"] == participant and r["split"] == split]
            labels = {label for r in items for label in r["protocol_labels"]}
            if len(items) != count or labels != {0, 1}:
                raise RuntimeError(f"Participant{participant}/{split} quota or utility class coverage failure")
    return {"public_ids": public_ids, "public_filenames": public_names, "assignments": assignments,
            "rejected_identities": rejected_ids, "image_rejections": image_rejections,
            "private_identity_selection_order": [item.identity for item in selected]}


def project_selected(raw, candidate_index, master, F, out, reporter, device):
    import torch
    path, receipt_path = out / "reduced_f400.npy", out / "projection_receipt.json"
    selected = sorted({int(i) for row in master["assignments"] for i in row["protocol_sources"]}
                      | {int(row["reserved_source"]) for row in master["assignments"]})
    signature = {"cohort_hash": io.canonical_hash(master), "basis_file_sha256": io.file_sha256(out / "basis/vgg_public_basis.pt"),
                 "raw_shape": list(raw.shape), "projection_device": str(device), "dtype": "float32", "batch_size": 512}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["signature"] != signature or io.file_sha256(path) != receipt["sha256"]:
            raise RuntimeError("Existing full projection differs")
        return receipt
    projected = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(raw), 400))
    basis = F.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(selected), 512):
            ids = np.asarray(selected[start:start + 512], dtype=np.int64)
            images = torch.from_numpy(np.asarray(raw[ids]).copy()).to(device=device, dtype=torch.float32)
            projected[ids] = (images.flatten(1).div_(255.) @ basis).cpu().numpy()
            if start % (512 * 20) == 0:
                projected.flush()
                reporter.emit("project_full_cohort", rows=min(start + 512, len(selected)), total=len(selected))
    projected.flush()
    receipt = {"signature": signature, "file": path.name, "bytes": path.stat().st_size,
               "sha256": io.file_sha256(path), "projected_rows": len(selected)}
    io.atomic_json(receipt_path, receipt)
    return receipt


def verify_master_artifacts(out, manifest, reporter, *, freshly_written=False):
    """Run once in the preparation/launcher process, not once per p child."""
    records = [manifest["raw_data"], manifest["projection"],
               {"file": "basis/vgg_public_basis.pt", "sha256": manifest["basis"]["basis_file_sha256"]}]
    verified = {}
    for record in records:
        path = out / record["file"]
        if "bytes" in record and path.stat().st_size != record["bytes"]:
            raise RuntimeError("Prepared master artifact size differs")
        reporter.emit("verify_completed_master", force=True, artifact=record["file"], bytes=path.stat().st_size)
        actual = record["sha256"] if freshly_written else io.file_sha256(path)
        if actual != record["sha256"]:
            raise RuntimeError(f"Completed master checksum mismatch: {record['file']}")
        verified[record["file"]] = {"sha256": actual, "bytes": path.stat().st_size}
    receipt = {"master_manifest_hash": manifest["manifest_hash"], "manifest_file_sha256": io.file_sha256(out / "master_manifest.json"),
               "artifacts": verified,
               "verification": "full sequential byte hashes at creation or launcher/resume; per-p children verify receipt and sizes"}
    io.atomic_json(out / "master_verified.json", receipt)
    return receipt


def prepare_master(source_root, cache_root, module_dir, seed=SEED, device="cuda", progress=None):
    source, out = Path(source_root).resolve(), Path(cache_root).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Full cache must be isolated from read-only VGG source data")
    out.mkdir(parents=True, exist_ok=True)
    reporter = io.Progress(out, progress)
    sources = io.checked_sources(source, Path(__file__).with_name("vgg_asset_pins.json"))
    base, _ = frozen_modules(module_dir)
    code = {Path(__file__).name: io.file_sha256(__file__), "vgg_io.py": io.file_sha256(io.__file__),
            "frozen_base": io.file_sha256(base.__file__)}
    config = {"version": VERSION, "seed": int(seed), "p_values": list(P_VALUES), "pmax": 50,
              "identities_per_client": 100, "identity_split_counts": {"train": 70, "val": 15, "test": 15},
              "eligibility": "officialtrain; >=51 defined Smiling images and both labels, after QC",
              "utility_images": "ALL remaining valid defined images after one reserved reference; no cap",
              "public_images": 10000, "public_candidate_id_bound": PUBLIC_CANDIDATE_IDENTITIES,
              "allocation": "unmodified frozen _balanced_identity_allocation",
              "preprocessing": "stored bbox1.3total expansion/clamp/center-square/bilinearRGB64",
              "h": 400, "projection_device": str(device), "auditor_score_selection": False}
    signature = io.canonical_hash({"config": config, "sources": sources, "code": code})
    manifest_path = out / "master_manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["preparation_signature"] != signature or saved["manifest_hash"] != io.canonical_hash({k: v for k, v in saved.items() if k != "manifest_hash"}):
            raise RuntimeError("Full master input/code changed; existing master is immutable")
        verify_master_artifacts(out, saved, reporter)
        reporter.emit("full_master_reused", force=True, manifest_hash=saved["manifest_hash"])
        return saved
    lock_path = out / "preparation_identity.json"
    lock = {"signature": signature, "config": config, "code": code, "sources": sources}
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise RuntimeError("An interrupted preparation has different inputs; use separate cache")
    io.atomic_json(lock_path, lock)
    records, metadata_audit = io.read_metadata(source, sources, reporter)
    if io.file_sha256(source / "meta/train_list.txt") != sources["meta/train_list.txt"]["sha256"]:
        raise RuntimeError("Official training list checksum mismatch")
    rows, eligible, public_order = metadata_candidates(source, records, base, seed, reporter)
    wanted = set(rows)
    io.atomic_json(out / "candidate_plan.json", {"signature": signature, "candidate_images": len(rows),
                   "candidate_filename_hash": io.canonical_hash(sorted(rows)), "eligible_ids": sorted(eligible),
                   "public_identity_candidate_order": public_order})
    boxes = io.read_boxes(source / "meta/bb_landmark.tar.gz", wanted,
                         sources["meta/bb_landmark.tar.gz"]["sha256"], reporter)
    raw, candidate_index, results = io.decode_candidates(source, out, wanted, boxes,
                                            sources["data/vggface2_train.tar.gz"], reporter)
    del boxes, wanted
    cohort = freeze_cohort(rows, eligible, public_order, results, candidate_index, base, seed)
    io.atomic_json(out / "cohort_before_projection.json", cohort)
    public = np.stack([raw[candidate_index[name]] for name in cohort["public_filenames"]])
    public_provenance = {"public_identity_ids": cohort["public_ids"], "filenames": cohort["public_filenames"],
                         "private_identity_ids": sorted(r["identity"] for r in cohort["assignments"]),
                         "identity_disjoint_from_all5000": True}
    F, basis_meta = io.fit_public_basis(public, out / "basis", seed, device, progress, public_provenance)
    del public, results, rows
    projection = project_selected(raw, candidate_index, cohort, F, out, reporter, device)
    candidate_receipt_path = out / "candidate_integrity.json"
    # The decoder computed the full file hash after flushing the memmap. Read
    # its compact integrity receipt without reparsing a million image receipts.
    candidate_receipt = json.loads(candidate_receipt_path.read_text())
    raw_record = {"file": "vgg_candidate_rgb64.npy", "bytes": (out / "vgg_candidate_rgb64.npy").stat().st_size,
                  "sha256": candidate_receipt["sha256"], "shape": list(raw.shape), "dtype": "uint8"}
    del candidate_receipt, raw
    diagnostics = base._balance_diagnostics({"assignments": cohort["assignments"]}, SimpleNamespace(n_participants=50))
    manifest = {"version": VERSION, "preparation_signature": signature, "config": config, "sources": sources,
                "code_sha256": code, "metadata_audit": metadata_audit, "assignments": cohort["assignments"],
                "public_svd": {"identity_ids": cohort["public_ids"], "filenames": cohort["public_filenames"],
                    "image_count": 10000, "uncentered": True, "all5000_identities_disjoint": True},
                "raw_data": raw_record, "projection": projection, "basis": basis_meta,
                "participant_design": {"master_capacity_participants": 50, "identities_per_participant": 100,
                    "split_identity_counts": {"train": 70, "val": 15, "test": 15}, "nested_by_client_prefix": True},
                "qc": {"post_qc_minimum_defined": 51, "both_labels_before_reference": True,
                    "exact_rgb64_and_source_bytes_deduplicated": True, "rejected_identities": cohort["rejected_identities"],
                    "image_rejections": cohort["image_rejections"], "near_duplicate_or_human_identity_audit": False,
                    "selected_identity_order": cohort["private_identity_selection_order"], "auditor_scores_used": False},
                "allocation_diagnostics": diagnostics,
                "total_protocol_images": sum(r["protocol_image_count"] for r in cohort["assignments"]),
                "cache_validation": "full compressed archive and raw/projection hashes at creation; full raw/projection hashes once per launcher/resume; child contexts verify completion receipt and byte sizes",
                "interrupted_projection_policy": "an uncommitted projection with no receipt may be regenerated in this isolated cache; completed checksummed projections never overwritten"}
    manifest["manifest_hash"] = io.canonical_hash(manifest)
    io.atomic_json(manifest_path, manifest)
    verify_master_artifacts(out, manifest, reporter, freshly_written=True)
    reporter.emit("full_master_complete", force=True, protocol_images=manifest["total_protocol_images"],
                  manifest_hash=manifest["manifest_hash"])
    return manifest
