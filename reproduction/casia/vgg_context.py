"""Memory-bounded adapter from the VGG pmax50 master to the shared protocol."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import torch

import vgg_io as io
import vgg_master as master_builder


class IndexedRawMemmap:
    """Expose requested RGB64 rows while keeping the complete cache disk-backed."""
    def __init__(self, pixels, source_indices):
        self.pixels = pixels
        self.source_indices = np.asarray(source_indices, dtype=np.int64)
        self.shape = (len(self.source_indices), 3, 64, 64)
        self.dtype = torch.uint8

    def __len__(self):
        return len(self.source_indices)

    def __getitem__(self, key):
        if isinstance(key, torch.Tensor):
            key = key.detach().cpu().numpy()
        positions = self.source_indices[key]
        return torch.from_numpy(np.asarray(self.pixels[positions]).copy())


def prepare_inputs(source_root, prepared_root, module_dir, protocol_config=None, progress=None):
    spec = protocol_config or {}
    return master_builder.prepare_master(source_root, prepared_root, module_dir,
            seed=int(spec.get("dataset_seed", master_builder.SEED)),
            device=spec.get("preparation_device", "cuda"), progress=progress)


def load_frozen_data(source_root, p, module_dir, out_dir, cache_root=None, progress=None):
    if p not in master_builder.P_VALUES:
        raise ValueError("Unsupported VGG participant count")
    out = Path(out_dir).resolve()
    receipt_directory = Path(cache_root).resolve() if cache_root is not None else out.parent / "prepared"
    cache, verification_path = receipt_directory, receipt_directory / "master_verified.json"
    manifest_path = cache / "master_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Run prepare_dataset on the downloaded raw files before loading a context")
    manifest = json.loads(manifest_path.read_text())
    if manifest["manifest_hash"] != io.canonical_hash({k: v for k, v in manifest.items() if k != "manifest_hash"}):
        raise RuntimeError("VGG master manifest self-hash differs")
    if manifest["config"]["pmax"] != 50 or manifest["config"]["identities_per_client"] != 100:
        raise RuntimeError("VGG master identity design changed")
    verification = json.loads(verification_path.read_text())
    if (verification["master_manifest_hash"] != manifest["manifest_hash"]
            or verification["manifest_file_sha256"] != io.file_sha256(manifest_path)):
        raise RuntimeError("Full master verification receipt differs")
    base, core = master_builder.frozen_modules(module_dir)
    code_hashes = {path.name: io.file_sha256(path) for path in sorted(Path(module_dir).glob("*.py"))}
    if io.file_sha256(base.__file__) != manifest["code_sha256"]["frozen_base"]:
        raise RuntimeError("VGG master allocation protocol source changed")
    for artifact in [manifest["raw_data"], manifest["projection"]]:
        path = cache / artifact["file"]
        if (path.stat().st_size != artifact["bytes"]
                or verification["artifacts"][artifact["file"]] != {"sha256": artifact["sha256"], "bytes": artifact["bytes"]}):
            raise RuntimeError("Prepared VGG artifact size differs")
    raw = np.load(cache / manifest["raw_data"]["file"], mmap_mode="r", allow_pickle=False)
    reduced = np.load(cache / manifest["projection"]["file"], mmap_mode="r", allow_pickle=False)
    if raw.dtype != np.uint8 or raw.shape[1:] != (3, 64, 64) or reduced.dtype != np.float32 or reduced.shape != (len(raw), 400):
        raise RuntimeError("VGG raw/projection cache shape or dtype differs")
    basis_path = cache / "basis/vgg_public_basis.pt"
    if io.file_sha256(basis_path) != manifest["basis"]["basis_file_sha256"]:
        raise RuntimeError("VGG public basis checksum differs")
    payload = torch.load(basis_path, map_location="cpu", weights_only=True)
    F = payload["F"].float()
    projection = {**manifest["basis"], "public_private_identity_disjoint": True,
                  "master_manifest_hash": manifest["manifest_hash"]}
    reducer = base.PublicSVDReducer(F, payload["singular_values"].double(), projection)
    cfg = SimpleNamespace(mode="full", seed=int(manifest["config"]["seed"]), n_participants=p,
          max_participants=50, identities_per_participant=100, public_svd_images=10000,
          projection_side=20, reduced_dim=400, anchor_rows=1600, anchor_ratio=4.,
          svd_q=420, svd_niter=4, deterministic=True, device="cpu", face_batch_size=128,
          face_model="casia-webface", linkage_candidates=10, linkage_episodes=10,
          bootstrap_resamples=10000, reference_participant=0, target_participant=1,
          batch_size=256, eval_batch_size=512, downstream_epochs=15,
          learning_rate=1e-3, weight_decay=0., output_dir=str(out), cache_dir=str(cache))
    assignments = [row for row in manifest["assignments"] if row["participant"] < p]
    if len(assignments) != 100 * p or len({r["identity"] for r in assignments}) != 100 * p:
        raise RuntimeError("Active nested VGG participant identities differ")
    split_data, blocks, derived = {}, {}, {}
    for split in ("train", "val", "test"):
        rows = []
        for identity in assignments:
            if identity["split"] != split:
                continue
            for name, label, source in zip(identity["protocol_filenames"], identity["protocol_labels"], identity["protocol_sources"]):
                rows.append((identity["participant"], name, label, source, identity["identity"]))
        rows.sort(key=lambda row: (row[0], base.stable_hash([cfg.seed, "protocol-row", split, row[1]], 64)))
        sources = np.asarray([row[3] for row in rows], dtype=np.int64)
        values = {"raw_uint8": IndexedRawMemmap(raw, sources),
                  "reduced": torch.from_numpy(np.asarray(reduced[sources])),
                  "labels": torch.as_tensor([row[2] for row in rows], dtype=torch.long),
                  "identities": torch.as_tensor([row[4] for row in rows], dtype=torch.long),
                  "source_indices": torch.from_numpy(sources),
                  "participant_ids": torch.as_tensor([row[0] for row in rows], dtype=torch.long),
                  "filenames": tuple(row[1] for row in rows)}
        data = base.SplitData(**values)
        split_data[split] = data
        xs, ys, indices = [], [], []
        for participant in range(p):
            positions = torch.nonzero(data.participant_ids == participant, as_tuple=False).flatten()
            if not len(positions) or not torch.equal(positions, torch.arange(int(positions[0]), int(positions[-1]) + 1)):
                raise RuntimeError("Participant rows are not contiguous")
            first, last = int(positions[0]), int(positions[-1]) + 1
            xs.append(data.reduced[first:last])  # float32 view; shared pipeline converts batches to float64.
            ys.append(data.labels[first:last]); indices.append(positions)
            if set(ys[-1].tolist()) != {0, 1}:
                raise RuntimeError("Participant utility split lacks a class")
        blocks[split] = (xs, ys, indices)
        derived[split] = {"rows": len(rows), "identity_count": len(set(row[4] for row in rows)),
                          "source_indices_hash": io.canonical_hash(sources.tolist()),
                          "filename_hash": io.canonical_hash(list(data.filenames)),
                          "projection_source_sha256": manifest["projection"]["sha256"],
                          "raw_source_sha256": manifest["raw_data"]["sha256"],
                          "raw_access": "lazy indexed memmap", "blocks_dtype": "float32"}
        del rows
    reserved = sorted(assignments, key=lambda row: row["identity"])
    ref_indices = np.asarray([row["reserved_source"] for row in reserved], dtype=np.int64)
    refs = base.ReferenceData(torch.from_numpy(np.asarray(raw[ref_indices])),
                             torch.as_tensor([row["identity"] for row in reserved], dtype=torch.long),
                             tuple(row["reserved_filename"] for row in reserved))
    dataset = "CelebA" if manifest["version"].startswith("celeba-") else "VGGFace2_MAAD"
    data_manifest = {"dataset": dataset, "task": "Smiling", "projection": projection,
                     "master_manifest_hash": manifest["manifest_hash"],
                     "participant_design": {"active_participants": p}, "derived_splits": derived}
    data = base.CelebADataBundle(**split_data, references=refs, reducer=reducer, manifest=data_manifest)
    linkage = base.make_face_linkage_manifest(data, blocks, cfg, include_test=True)
    if len(linkage.query_identities) != 100:
        raise RuntimeError("The linkage estimator requires100 protected identities")
    source_files = {str(manifest_path): io.file_sha256(manifest_path),
                    str(verification_path): io.file_sha256(verification_path),
                    str(cache / manifest["raw_data"]["file"]): manifest["raw_data"]["sha256"],
                    str(cache / manifest["projection"]["file"]): manifest["projection"]["sha256"],
                    str(basis_path): manifest["basis"]["basis_file_sha256"],
                    str(Path(__file__)): io.file_sha256(__file__)}
    source_identity = {"dataset": dataset, "master_manifest_hash": manifest["manifest_hash"],
                       "source_module_sha256": code_hashes, "active_participants": p,
                       "active_identity_hash": io.canonical_hash(sorted(r["identity"] for r in assignments)),
                       "raw_sources_sha256": io.canonical_hash(manifest["sources"]),
                       "cache_integrity_policy": manifest["cache_validation"]}
    return SimpleNamespace(cfg=cfg, base=base, core=core, data=data, blocks=blocks,
                           labels={split: split_data[split].labels for split in split_data}, linkage=linkage,
                           query_reduced=base._query_tensor_from_blocks(linkage, blocks, 1).double(), decoder=F.T,
                           source_files=source_files, source_config={**manifest["config"], "module_sha256": code_hashes}, projection=projection,
                           derived_splits=derived, source_identity=source_identity)


def load_context(source_root, out_dir, p, module_dir, audit_device="cuda", protocol_config=None,
                 prepared_root=None, progress=None):
    from context_common import finalize_context
    inputs = load_frozen_data(source_root, p, module_dir, out_dir, cache_root=prepared_root, progress=progress)
    return finalize_context(inputs, out_dir, audit_device, protocol_config, progress)
