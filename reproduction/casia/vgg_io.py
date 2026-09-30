"""Pinned VGGFace2 metadata, pixel IO and public-SVD primitives.

The full-study cohort/allocation lives in vgg_master.py. No model training or
auditor scores are used by these metadata and image-processing helpers.
"""

from __future__ import annotations

import collections

import csv

import hashlib

import io

import json

import math

import os

import random

import re

import tarfile

import time

from pathlib import Path

import numpy as np

from PIL import Image

VERSION = "vggface2-casia-clean-preparation-20260928-v1"

DEFAULT_SEED = 20260928

IMAGE_RE = re.compile(r"^(n\d{6})/(\d+_\d+\.jpg)$")

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()

def named_seed(seed, *parts):
    # Deliberately matches the original CelebA helper, including JSON spacing.
    return int.from_bytes(hashlib.sha256(json.dumps([int(seed), *parts]).encode()).digest()[:8],
                          "little") & ((1 << 63) - 1)

def order_key(seed, namespace, value):
    return canonical_hash([int(seed), namespace, value])

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Stream the million-row decode receipt instead of building a second giant
    # JSON string in RAM. Bytes match the ordinary indented JSON serializer.
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)

def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()

class Progress:
    def __init__(self, out_dir, callback=None):
        self.out_dir = Path(out_dir)
        self.callback = callback
        self.last = 0.0

    def emit(self, stage, *, force=False, **details):
        now = time.monotonic()
        if not force and now - self.last < 30:
            return
        event = {"version": VERSION, "stage": stage, "unix_time": time.time(), **details}
        try:
            import psutil
            event["process_rss_bytes"] = psutil.Process().memory_info().rss
        except ImportError:
            pass
        atomic_json(self.out_dir / "vgg_preparation_status.json", event)
        if self.callback:
            self.callback(**event)
        self.last = now

def normalize_path(name):
    parts = str(name).replace("\\", "/").split("/")
    if ".." in parts:
        raise ValueError("Parent traversal in image path")
    path = "/".join(parts[-2:])
    if not IMAGE_RE.fullmatch(path):
        raise ValueError(f"Unexpected VGGFace2 image path: {name}")
    return path

def checked_sources(source_root, pins_path):
    """Verify actual downloaded bytes; no external download receipts are required."""
    root = Path(source_root)
    pins = json.loads(Path(pins_path).read_text(encoding="utf-8"))
    records = {}
    for pin in pins["assets"]:
        path = root / pin["path"]
        if not path.is_file() or path.stat().st_size != pin["size"]:
            raise RuntimeError(f"Missing or wrong-size raw asset: {path}")
        if file_sha256(path) != pin["sha256"]:
            raise RuntimeError(f"Downloaded raw asset SHA256 differs: {path}")
        records[pin["path"]] = dict(pin)
    return records

def validate_maad_rows(block):
    """Check released MAAD identity tokens and ternary attribute labels."""
    parsed = block.Filename.str.extract(r"^(n\d{6})/(\d+_\d+\.jpg)$")
    if parsed.isna().any().any() or not block.Smiling.isin([-1, 0, 1]).all():
        raise RuntimeError("Malformed MAAD filename or Smiling encoding")
    expected = parsed[0].str.slice(1).astype(np.int32).astype(str)
    exceptions = []
    for index in block.index[expected.to_numpy() != block.Identity.to_numpy()]:
        row = block.loc[index]
        exception = {"csv_line": int(index) + 2, "Filename": row.Filename, "Identity": row.Identity}
        if exception != {"csv_line": 250636, "Filename": "n000713/0437_101.jpg", "Identity": "713\\"}:
            raise RuntimeError(f"Unverified numeric identity mismatch: {exception}")
        exceptions.append(exception)
    return parsed[0].to_numpy(), exceptions


def read_metadata(root, sources, progress):
    import pandas as pd
    identity_path = root / "meta/identity_meta.csv"
    maad_path = root / "maad/MAAD_Face.csv"
    for name in ["meta/identity_meta.csv", "maad/MAAD_Face.csv"]:
        progress.emit("verify_metadata", force=True, asset=name)
        if file_sha256(root / name) != sources[name]["sha256"]:
            raise RuntimeError(f"Metadata checksum mismatch: {name}")
    meta = pd.read_csv(identity_path, skipinitialspace=True).set_index("Class_ID")
    if len(meta) != 9131 or not meta.index.is_unique or meta.Flag.value_counts().to_dict() != {1: 8631, 0: 500}:
        raise RuntimeError("Unexpected VGGFace2 identity metadata")
    header = list(pd.read_csv(maad_path, nrows=0).columns)
    if len(header) != 49 or header[:2] != ["Filename", "Identity"] or "Smiling" not in header:
        raise RuntimeError("Unexpected MAAD-Face column schema")
    counts = collections.defaultdict(lambda: [0, 0, 0])
    exceptions = []
    total = 0
    for block in pd.read_csv(maad_path, usecols=["Filename", "Identity", "Smiling"],
                             dtype={"Filename": str, "Identity": str, "Smiling": np.int8}, chunksize=200000):
        identities, block_exceptions = validate_maad_rows(block)
        exceptions.extend(block_exceptions)
        block = block.assign(identity=identities)
        grouped = block.groupby(["identity", "Smiling"]).size()
        for (identity, label), n in grouped.items():
            counts[identity][int(label) + 1] += int(n)
        total += len(block)
        progress.emit("count_smiling_metadata", rows=total)
    if total != 3308040 or len(exceptions) != 1 or set(counts) != set(meta.index):
        raise RuntimeError("MAAD row/identity/exception count differs from verified source")
    records = []
    for identity, row in meta.iterrows():
        neg, undefined, pos = counts[identity]
        if neg + undefined + pos > int(row.Sample_Num):
            raise RuntimeError("More MAAD rows than published VGG images")
        records.append({"identity": identity, "numeric_identity": int(identity[1:]),
                        "official_split": "train" if int(row.Flag) else "test",
                        "images": int(row.Sample_Num), "defined": pos + neg,
                        "positive": pos, "negative": neg, "undefined": undefined})
    return records, {"rows": total, "columns": header, "source_identity_token_exceptions": exceptions,
                     "identity_authority": "Filename folder matched to VGGFace2 Class_ID",
                     "missing_annotation_rows": int(meta.Sample_Num.sum()) - total,
                     "label_origin": "MAAD confidence-filtered attribute-transfer predictions; Smiling mainly from CelebA"}

def iter_train_names(path):
    """The official list can contain empty lines; count actual image records."""
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            name = line.strip()
            if name:
                yield normalize_path(name)

def read_boxes(path, wanted, expected_hash, progress):
    progress.emit("verify_bbox_metadata", force=True)
    if file_sha256(path) != expected_hash:
        raise RuntimeError("Bounding-box metadata checksum mismatch")
    boxes = {}
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if member.name.lstrip("./") != "bb_landmark/loose_bb_train.csv":
                continue
            if not member.isfile():
                raise RuntimeError("Bounding-box CSV is not a regular file")
            stream = archive.extractfile(member)
            # TextIOWrapper probes seekability; tarfile's streaming _Stream has
            # no seekable() in Python3.13. Decode complete binary lines without
            # wrapping the nonseekable archive member in another buffered layer.
            reader = csv.DictReader(line.decode("utf-8") for line in stream)
            if reader.fieldnames != ["NAME_ID", "X", "Y", "W", "H"]:
                raise RuntimeError(f"Unexpected bounding-box columns: {reader.fieldnames}")
            for i, row in enumerate(reader, 1):
                name = normalize_path(row["NAME_ID"] + ".jpg")
                if name in wanted:
                    if name in boxes:
                        raise RuntimeError(f"Duplicate bounding-box path: {name}")
                    boxes[name] = [float(row[key]) for key in ["X", "Y", "W", "H"]]
                if i % 10000 == 0:
                    progress.emit("read_bbox_metadata", rows=i, found=len(boxes), needed=len(wanted))
            break
    return boxes

def crop_rgb64(encoded, box):
    x, y, w, h = box
    if not all(math.isfinite(v) for v in box) or w <= 0 or h <= 0:
        raise ValueError("nonfinite_or_nonpositive_bbox")
    with Image.open(io.BytesIO(encoded)) as image:
        image.load()  # Truncated JPEGs remain errors; LOAD_TRUNCATED_IMAGES is never enabled.
        image = image.convert("RGB")
        width, height = image.size
        if x >= width or y >= height or x + w <= 0 or y + h <= 0:
            raise ValueError("bbox_has_no_image_intersection")
        cx, cy = x + w / 2, y + h / 2
        bounds = [max(0, math.floor(cx - 1.3 * w / 2)), max(0, math.floor(cy - 1.3 * h / 2)),
                  min(width, math.ceil(cx + 1.3 * w / 2)), min(height, math.ceil(cy + 1.3 * h / 2))]
        if bounds[2] - bounds[0] < 2 or bounds[3] - bounds[1] < 2:
            raise ValueError("clipped_bbox_smaller_than_2_pixels")
        image = image.crop(tuple(bounds))
        cw, ch = image.size
        side = min(cw, ch)
        left, top = (cw - side) // 2, (ch - side) // 2
        image = image.crop((left, top, left + side, top + side)).resize((64, 64), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()
    if np.max(array) == np.min(array):
        raise ValueError("constant_rgb64_image")
    return array, {"original_size": [width, height], "bbox_xywh": list(box),
                   "expanded_clipped_ltrb": bounds, "center_square_ltrb": [left, top, left + side, top + side]}

class HashingReader:
    def __init__(self, stream, progress, expected):
        self.stream, self.progress, self.expected = stream, progress, expected
        self.digest, self.bytes = hashlib.sha256(), 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.bytes += len(data)
        self.progress.emit("stream_image_archive", compressed_bytes=self.bytes, expected_bytes=self.expected)
        return data

def decode_candidates(root, out, wanted, boxes, source, progress):
    names = sorted(wanted)
    index = {name: i for i, name in enumerate(names)}
    raw_path = out / "vgg_candidate_rgb64.npy"
    receipt_path = out / "vgg_candidate_decode_receipts.json"
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (saved["names"] != names or saved["archive_sha256"] != source["sha256"]
                or saved["archive_bytes"] != source["size"]
                or saved.get("preprocessing_code_sha256") != file_sha256(__file__)
                or file_sha256(raw_path) != saved["candidate_array_sha256"]):
            raise RuntimeError("Completed candidate cache differs; never overwritten")
        data = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        if data.dtype != np.uint8 or data.shape != (len(names), 3, 64, 64):
            raise RuntimeError("Invalid completed candidate cache shape")
        atomic_json(out / "candidate_integrity.json", {"sha256": saved["candidate_array_sha256"],
                    "bytes": raw_path.stat().st_size, "shape": list(data.shape),
                    "archive_sha256": saved["archive_sha256"]})
        progress.emit("candidate_decoding_reused", force=True, candidates=len(names))
        return data, index, saved["results"]
    data = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.uint8, shape=(len(names), 3, 64, 64))
    results, seen = {}, set()
    progress.emit("stream_image_archive", force=True, needed=len(names), compressed_bytes=0,
                  expected_bytes=source["size"])
    with (root / "data/vggface2_train.tar.gz").open("rb") as file:
        reader = HashingReader(file, progress, source["size"])
        with tarfile.open(fileobj=reader, mode="r|gz", bufsize=1 << 20) as archive:
            for member in archive:
                archive.members.clear()  # A streaming scan must not retain 3M TarInfo records.
                if not member.isfile():
                    continue
                try:
                    name = normalize_path(member.name)
                except ValueError:
                    continue  # Non-image archive metadata is never extracted.
                if name not in index:
                    continue
                if name in seen:
                    raise RuntimeError(f"Duplicate selected path in image archive: {name}")
                seen.add(name)
                if name not in boxes:
                    results[name] = {"valid": False, "reason": "missing_official_bbox"}
                    continue
                if member.size <= 0 or member.size > 64 << 20:
                    results[name] = {"valid": False, "reason": "invalid_or_over_64MiB_image_size"}
                    continue
                encoded = archive.extractfile(member).read()
                try:
                    array, geometry = crop_rgb64(encoded, boxes[name])
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    results[name] = {"valid": False, "reason": f"decode_or_geometry:{type(exc).__name__}:{exc}"}
                    continue
                data[index[name]] = array
                results[name] = {"valid": True, "rgb64_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                                 "source_image_sha256": hashlib.sha256(encoded).hexdigest(), **geometry}
                if len(seen) % 2048 == 0:
                    data.flush()
                    progress.emit("decode_selected_candidates", found=len(seen), needed=len(names),
                                  compressed_bytes=reader.bytes, expected_bytes=source["size"])
        while reader.read(8 << 20):
            pass
        actual = reader.digest.hexdigest()
    data.flush()
    if reader.bytes != source["size"] or actual != source["sha256"]:
        raise RuntimeError("Image archive SHA256 differs after complete compressed-byte stream")
    for name in wanted - seen:
        results[name] = {"valid": False, "reason": "missing_archive_member"}
    array_hash = file_sha256(raw_path)
    atomic_json(receipt_path, {"names": names, "results": results,
                "archive_sha256": actual, "archive_bytes": reader.bytes,
                "preprocessing_code_sha256": file_sha256(__file__), "candidate_array_sha256": array_hash})
    atomic_json(out / "candidate_integrity.json", {"sha256": array_hash, "bytes": raw_path.stat().st_size,
                "shape": list(data.shape), "archive_sha256": actual})
    progress.emit("candidate_decoding_complete", force=True, candidates=len(names),
                  valid=sum(r["valid"] for r in results.values()), archive_sha256=actual)
    return data, index, results

def fit_public_basis(public_uint8, out_dir, seed=DEFAULT_SEED, device="cuda", progress=None,
                     public_manifest=None):
    """Same public raw-pixel algorithm as CelebA; no centering/FaceNet/SVD fitting on private data."""
    import torch
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reporter = Progress(out, progress)
    images = torch.as_tensor(public_uint8)
    if images.dtype != torch.uint8 or tuple(images.shape) != (10000, 3, 64, 64):
        raise ValueError("Public SVD requires exactly10000 RGB64 uint8 images")
    image_hash = hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest()
    spec = {"seed": int(seed), "public_pixel_sha256": image_hash, "q": 420, "niter": 4,
            "input_dim": 12288, "output_dim": 400, "uncentered": True, "device": str(device)}
    path, manifest_path = out / "vgg_public_basis.pt", out / "vgg_public_basis_manifest.json"
    if path.exists() or manifest_path.exists():
        if not (path.exists() and manifest_path.exists()):
            raise RuntimeError("Incomplete existing SVD output preserved")
        manifest = json.loads(manifest_path.read_text())
        if manifest["spec"] != spec or file_sha256(path) != manifest["basis_file_sha256"]:
            raise RuntimeError("Existing SVD output differs; never overwritten")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return payload["F"], manifest
    reporter.emit("fit_public_svd", force=True, device=str(device), images=10000, q=420, niter=4)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    svd_seed = named_seed(seed, "public-svd")
    random.seed(svd_seed)
    np.random.seed(svd_seed % (2**32 - 1))
    torch.manual_seed(svd_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(svd_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    matrix = images.float().flatten(1).div_(255.0).to(device)
    _, singular, candidate = torch.pca_lowrank(matrix, q=420, center=False, niter=4)
    basis_q, _ = torch.linalg.qr(candidate[:, :400].double().cpu(), mode="reduced")
    F, retained = basis_q.float().contiguous(), singular[:400].double().cpu()
    residual = float(torch.linalg.matrix_norm(F.double().T @ F.double() - torch.eye(400)))
    if residual > 1e-3:
        raise RuntimeError(f"SVD basis orthogonality failure: {residual}")
    temporary = path.with_suffix(".tmp")
    torch.save({"F": F, "singular_values": retained}, temporary)
    os.replace(temporary, path)
    manifest = {"version": VERSION, "spec": spec, "uncentered": True, "center_argument": False,
                "input_dim": 12288, "output_dim": 400, "fit_image_count": 10000,
                "q": 420, "niter": 4, "public_manifest": public_manifest,
                "algorithm": "torch.pca_lowrank float32(center=False); top400; CPUfloat64QR;float32F",
                "svd_seed": svd_seed, "torch_version": torch.__version__, "device": str(device),
                "orthogonality_residual": residual, "retained_singular_values": retained.tolist(),
                "basis_matrix_sha256": hashlib.sha256(F.numpy().tobytes()).hexdigest(),
                "basis_file": path.name, "basis_file_sha256": file_sha256(path)}
    manifest["manifest_hash"] = canonical_hash(manifest)
    atomic_json(manifest_path, manifest)
    reporter.emit("public_svd_complete", force=True, basis_sha256=manifest["basis_matrix_sha256"])
    return F, manifest
