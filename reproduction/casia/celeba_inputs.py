"""Compute the matched CelebA cohort, public SVD and embedded rows from raw files.

No downloaded data, prepared pixels, projection or experiment results are shipped
with this module. Committed caches are immutable and checked before every launch.
"""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import torch

import celeba_public_svd_identity100_runner as base
import vgg_io as io
import vgg_master
import vgg_context

SEED = 20260713
VERSION = "celeba-raw-uncentered-pmax50-public-reproduction-v1"


def find_metadata(root, names):
    for name in names:
        paths = sorted(Path(root).rglob(name))
        if paths:
            if len(paths) != 1:
                raise RuntimeError(f"Ambiguous metadata file {name}: {paths}")
            return paths[0]
    raise FileNotFoundError(f"Missing downloaded metadata: one of {names}")


def load_metadata(root, *, expected_images=202599, expected_identities=10177):
    """Accept official CelebA TXT attributes and the common CSV conversion."""
    import pandas as pd
    attr_path = find_metadata(root, ("list_attr_celeba.txt", "list_attr_celeba.csv"))
    identity_path = find_metadata(root, ("identity_CelebA.txt", "identity_CelebA.csv",
                                        "list_identity_celeba.txt", "list_identity_celeba.csv"))
    if attr_path.suffix.lower() == ".txt":
        with attr_path.open(encoding="utf-8") as stream:
            declared = int(stream.readline().strip())
            columns = stream.readline().split()
        attr = pd.read_csv(attr_path, sep=r"\s+", skiprows=2,
                           names=["image_id", *columns])
        if len(attr) != declared:
            raise RuntimeError("Official attribute row count disagrees with its header")
    else:
        attr = pd.read_csv(attr_path)
    if identity_path.suffix.lower() == ".txt":
        identity = pd.read_csv(identity_path, sep=r"\s+", header=None,
                               names=["image_id", "identity"])
    else:
        identity = pd.read_csv(identity_path)
    if "image_id" not in attr.columns:
        attr = attr.rename(columns={attr.columns[0]: "image_id"})
    if "image_id" not in identity.columns:
        identity = identity.rename(columns={identity.columns[0]: "image_id"})
    identity_column = next((name for name in identity.columns if name != "image_id"), None)
    if identity_column is None or "Smiling" not in attr.columns:
        raise RuntimeError("Downloaded metadata lacks identity or Smiling")
    if (not attr.image_id.is_unique or not identity.image_id.is_unique
            or set(attr.image_id) != set(identity.image_id)):
        raise RuntimeError("Attribute and identity image IDs must match one-to-one")
    frame = attr[["image_id", "Smiling"]].merge(
        identity[["image_id", identity_column]], on="image_id", validate="one_to_one",
        sort=False).rename(columns={identity_column: "identity"})
    if not frame.Smiling.isin([-1, 1]).all():
        raise RuntimeError("Smiling must contain only the official -1/+1 labels")
    frame["identity"] = frame.identity.astype(np.int64)
    frame["label"] = (frame.Smiling.astype(int) > 0).astype(np.int64)
    frame["source_index"] = np.arange(len(frame), dtype=np.int64)
    if ((expected_images is not None and len(frame) != expected_images)
            or (expected_identities is not None and frame.identity.nunique() != expected_identities)):
        raise RuntimeError("Expected complete CelebA metadata: 202599 images and 10177 identities")
    frame.attrs["identity_metadata_sha256"] = io.file_sha256(identity_path)
    frame.attrs["attribute_metadata_sha256"] = io.file_sha256(attr_path)
    return frame, attr_path, identity_path


def image_directory(root, first_name):
    candidates = sorted(path for path in Path(root).rglob("img_align_celeba") if path.is_dir())
    matches = [path for path in candidates if (path / first_name).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError("Expected one img_align_celeba directory with aligned JPEGs")
    return matches[0]


def verify_raw_images(directory, names, reporter):
    """Hash the supplied JPEG bytes, including on resume; filenames alone are insufficient."""
    import hashlib
    digest, total = hashlib.sha256(), 0
    for index, name in enumerate(names):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing CelebA JPEG: {path}")
        size, value = path.stat().st_size, io.file_sha256(path)
        digest.update(json.dumps([name, size, value], separators=(",", ":")).encode())
        total += size
        if index % 1000 == 0:
            reporter.emit("verify_raw_jpeg_bytes", checked=index + 1, total=len(names))
    return {"images": len(names), "bytes": total, "ordered_files_sha256": digest.hexdigest()}


def decode_pixels(image_dir, names, out, signature, reporter):
    from PIL import Image
    path, receipt_path = out / "celeba_rgb64.npy", out / "pixels_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt["signature"] != signature or not path.exists()
                or io.file_sha256(path) != receipt["sha256"]):
            raise RuntimeError("Committed CelebA pixels differ; use a separate cache")
        return np.load(path, mmap_mode="r", allow_pickle=False), receipt
    raw = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8,
                                    shape=(len(names), 3, 64, 64))
    reporter.emit("decode_rgb64", force=True, total=len(names))
    for index, name in enumerate(names):
        with Image.open(image_dir / name) as image:
            image = image.convert("RGB")
            width, height = image.size
            side = min(width, height)
            left, top = (width - side) // 2, (height - side) // 2
            image = image.crop((left, top, left + side, top + side)).resize(
                (64, 64), Image.Resampling.BILINEAR)
            raw[index] = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
        if index % 5120 == 0:
            raw.flush()
            reporter.emit("decode_rgb64", images=index + 1, total=len(names))
    raw.flush()
    receipt = {"signature": signature, "file": path.name, "bytes": path.stat().st_size,
               "sha256": io.file_sha256(path), "shape": list(raw.shape), "dtype": "uint8"}
    io.atomic_json(receipt_path, receipt)
    return raw, receipt


def prepare_master(source_root, cache_root, module_dir, seed=SEED, device="cuda", progress=None):
    source, out = Path(source_root).resolve(), Path(cache_root).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Preparation cache must be separate from the downloaded raw data")
    out.mkdir(parents=True, exist_ok=True)
    reporter = io.Progress(out, progress)
    reporter.emit("read_metadata_and_allocate", force=True)
    frame, attr_path, identity_path = load_metadata(source)
    image_dir = image_directory(source, str(frame.iloc[0].image_id))
    source_pixels = verify_raw_images(image_dir, frame.image_id.astype(str).tolist(), reporter)
    sources = {"attributes": {"path": str(attr_path), "sha256": io.file_sha256(attr_path)},
               "identities": {"path": str(identity_path), "sha256": io.file_sha256(identity_path)},
               "aligned_jpegs": source_pixels}
    config = {"version": VERSION, "seed": int(seed), "p_values": list(vgg_master.P_VALUES),
              "pmax": 50, "identities_per_client": 100, "public_images": 10000,
              "identity_split_counts": {"train": 70, "val": 15, "test": 15}, "h": 400,
              "eligibility": "at least two images; reserve one; all remaining images used",
              "allocation": "production hash order and balanced identity allocation",
              "preprocessing": "center-square aligned RGB, bilinear64, flatten NCHW/255",
              "projection_device": str(device), "auditor_score_selection": False}
    code = {"celeba_inputs.py": io.file_sha256(__file__),
            "frozen_base": io.file_sha256(base.__file__), "vgg_io.py": io.file_sha256(io.__file__),
            "vgg_master.py": io.file_sha256(vgg_master.__file__)}
    signature = io.canonical_hash({"config": config, "sources": sources, "code": code})
    manifest_path = out / "master_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["preparation_signature"] != signature or manifest["manifest_hash"] !=
                io.canonical_hash({k: v for k, v in manifest.items() if k != "manifest_hash"})):
            raise RuntimeError("Raw data, preparation code or configuration changed; cache is immutable")
        vgg_master.verify_master_artifacts(out, manifest, reporter)
        reporter.emit("full_master_reused", force=True, manifest_hash=manifest["manifest_hash"])
        return manifest
    lock = {"signature": signature, "config": config, "sources": sources, "code": code}
    base._write_or_validate_json(out / "preparation_identity.json", lock, "raw preparation")
    cfg = base.ExperimentConfig(seed=int(seed), device=device, data_root=str(source), cache_dir=str(out))
    original = base._build_master_manifest(frame, cfg, sources)
    assignments = original["assignments"]
    names = sorted(set(original["public_svd"]["filenames"]) | {
        name for row in assignments for name in [row["reserved_filename"], *row["protocol_filenames"]]})
    candidate_index = {name: index for index, name in enumerate(names)}
    for row in assignments:
        row["reserved_source"] = candidate_index[row["reserved_filename"]]
        row["protocol_sources"] = [candidate_index[name] for name in row["protocol_filenames"]]
    raw, raw_record = decode_pixels(image_dir, names, out, signature, reporter)
    public = np.stack([raw[candidate_index[name]] for name in original["public_svd"]["filenames"]])
    provenance = {"public_identity_ids": original["public_svd"]["identity_ids"],
                  "filenames": original["public_svd"]["filenames"],
                  "private_identity_ids": sorted(row["identity"] for row in assignments),
                  "identity_disjoint_from_all5000": True}
    F, basis = io.fit_public_basis(public, out / "basis", int(seed), device, progress, provenance)
    del public
    cohort = {"assignments": assignments}
    projection = vgg_master.project_selected(raw, candidate_index, cohort, F, out, reporter, device)
    manifest = {"version": VERSION, "preparation_signature": signature, "config": config,
                "sources": sources, "code_sha256": code, "assignments": assignments,
                "raw_data": raw_record, "projection": projection, "basis": basis,
                "public_svd": original["public_svd"], "participant_design": original["participant_design"],
                "deterministic_identity_exclusion": original["legacy_test_quarantine"],
                "allocation_diagnostics": base._balance_diagnostics(original, cfg),
                "total_protocol_images": sum(row["protocol_image_count"] for row in assignments),
                "cache_validation": "actual raw JPEG byte hashes plus full generated memmap/basis hashes at every launch"}
    manifest["manifest_hash"] = io.canonical_hash(manifest)
    io.atomic_json(manifest_path, manifest)
    vgg_master.verify_master_artifacts(out, manifest, reporter, freshly_written=True)
    reporter.emit("full_master_complete", force=True, protocol_images=manifest["total_protocol_images"],
                  manifest_hash=manifest["manifest_hash"])
    return manifest


def prepare_inputs(source_root, prepared_root, module_dir, protocol_config=None, progress=None):
    spec = protocol_config or {}
    return prepare_master(source_root, prepared_root, module_dir,
                          seed=int(spec.get("dataset_seed", SEED)),
                          device=spec.get("preparation_device", "cuda"), progress=progress)


def load_frozen_data(source_root, p, module_dir, out_dir, cache_root=None, progress=None):
    return vgg_context.load_frozen_data(source_root, p, module_dir, out_dir,
                                       cache_root=cache_root, progress=progress)
