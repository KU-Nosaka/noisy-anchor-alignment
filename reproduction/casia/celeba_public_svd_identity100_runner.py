"""Frozen numerical cohort, GDP-input and linkage helpers for the public raw-data reproduction.

Extracted from the evaluated production implementation; no prior data or results
are imported. The deterministic CelebA exclusion set is recomputed from metadata.
"""

from __future__ import annotations

import dataclasses

import hashlib

import json

import math

import os

import random

import tempfile

from dataclasses import asdict, dataclass

from pathlib import Path

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

import torch

from torch import Tensor, nn

from torch.nn import functional as torch_f


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = "full"
    seed: int = 20260713
    data_root: Optional[str] = None
    cache_dir: str = "cache/current_casia/celeba"
    output_dir: str = "build/current_casia/celeba"
    local_cache_dir: str = "cache/current_casia/celeba"
    device: str = "cuda"
    deterministic: bool = True
    n_participants: int = 50
    max_participants: int = 50
    identities_per_participant: int = 100
    public_svd_images: int = 10000
    projection_side: int = 20
    svd_q: int = 420
    svd_niter: int = 4
    face_batch_size: int = 128
    projection_batch_size: int = 512
    smoke_protocol_images_per_identity: Optional[int] = None
    reference_participant: int = 0
    target_participant: int = 1
    anchor_ratio: float = 4.0
    linkage_candidates: int = 10
    linkage_episodes: int = 10
    bootstrap_resamples: int = 10000
    batch_size: int = 256
    eval_batch_size: int = 512
    downstream_epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    face_model: str = "casia-webface"

    @property
    def reduced_dim(self):
        return self.projection_side ** 2

    @property
    def anchor_rows(self):
        return int(math.ceil(self.anchor_ratio * self.reduced_dim))

    def to_dict(self):
        return asdict(self)


DATASET_NAME = "CelebA"

RAW_SHAPE = (3, 64, 64)

LINKAGE_CHANCE = 0.10

DATA_CACHE_SCHEMA_VERSION = 1

PUBLIC_SVD_IMAGES = 10_000

SPLIT_IDENTITY_COUNTS = {"train": 70, "val": 15, "test": 15}

def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")

def stable_hash(value: Any, length: int = 32) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]

def named_seed(base_seed: int, *parts: Any) -> int:
    digest = hashlib.sha256(
        json.dumps([int(base_seed), *parts], default=_json_default).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)

def _hash_order(seed: int, label: str, values: Iterable[Any]) -> list[Any]:
    return sorted(values, key=lambda value: stable_hash([seed, label, value], 64))

def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)

def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)

def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)

def _write_or_validate_json(path: Path, payload: Mapping[str, Any], label: str) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if stable_hash(existing) != stable_hash(payload):
            raise RuntimeError(f"Refusing incompatible resume: {label} differs at {path}")
        return
    _atomic_json(path, payload)

@dataclass(frozen=True)
class IdentityStat:
    identity: int
    reserved_filename: str
    protocol_filenames: tuple[str, ...]
    protocol_labels: tuple[int, ...]
    protocol_sources: tuple[int, ...]

    @property
    def image_count(self) -> int:
        return len(self.protocol_filenames)

    @property
    def positive_count(self) -> int:
        return int(sum(self.protocol_labels))

def _balanced_identity_allocation(
    stats: Sequence[IdentityStat], cfg: ExperimentConfig
) -> dict[tuple[int, str], list[IdentityStat]]:
    """Fill exact identity-count slots while balancing images and prevalence."""

    required = cfg.max_participants * cfg.identities_per_participant
    if len(stats) != required:
        raise ValueError(f"Expected {required} participant identities, received {len(stats)}")
    total_images = sum(item.image_count for item in stats)
    total_positive = sum(item.positive_count for item in stats)
    prevalence = total_positive / max(1, total_images)
    mean_images = total_images / required
    capacities = {
        (participant, split): SPLIT_IDENTITY_COUNTS[split]
        for participant in range(cfg.max_participants)
        for split in ("train", "val", "test")
    }
    buckets = {key: [] for key in capacities}
    loads = {key: 0 for key in capacities}
    positives = {key: 0 for key in capacities}
    order = sorted(
        stats,
        key=lambda item: (
            -item.image_count,
            -abs(item.positive_count / max(1, item.image_count) - prevalence),
            stable_hash([cfg.seed, "allocation", item.identity], 64),
        ),
    )
    for item in order:
        candidates = [key for key, cap in capacities.items() if len(buckets[key]) < cap]
        if not candidates:
            raise AssertionError("Identity allocation exhausted all capacities early")

        def score(key: tuple[int, str]) -> tuple[float, str]:
            _, split = key
            target_load = mean_images * SPLIT_IDENTITY_COUNTS[split]
            new_load = loads[key] + item.image_count
            new_positive = positives[key] + item.positive_count
            load_error = ((new_load - target_load) / max(1.0, target_load)) ** 2
            prevalence_error = (new_positive / max(1, new_load) - prevalence) ** 2
            fill = (len(buckets[key]) + 1) / capacities[key]
            return (
                load_error + 2.0 * prevalence_error + 1.0e-3 * fill,
                stable_hash([cfg.seed, "allocation-tie", item.identity, key], 64),
            )

        chosen = min(candidates, key=score)
        buckets[chosen].append(item)
        loads[chosen] += item.image_count
        positives[chosen] += item.positive_count
    for (participant, split), items in buckets.items():
        expected = SPLIT_IDENTITY_COUNTS[split]
        if len(items) != expected:
            raise AssertionError(
                f"Participant {participant} {split} has {len(items)} identities, expected {expected}"
            )
    return buckets

def _complete_identity_indices_legacy(
    identities: np.ndarray, size: int, seed: int
) -> np.ndarray:
    """Reproduce the complete-identity cap used by the prior CrossSVD smoke."""

    identities = np.asarray(identities, dtype=np.int64)
    rng = np.random.default_rng(seed)
    row_order = np.argsort(identities, kind="stable")
    sorted_identities = identities[row_order]
    boundaries = np.flatnonzero(np.diff(sorted_identities)) + 1
    groups = list(np.split(row_order, boundaries))
    rng.shuffle(groups)
    chosen: list[np.ndarray] = []
    used = 0
    for indices in groups:
        if used + len(indices) <= size:
            chosen.append(indices)
            used += len(indices)
    return np.concatenate(chosen).astype(np.int64) if chosen else np.empty(0, dtype=np.int64)

def _legacy_cross_svd_test_identities(frame) -> list[int]:
    """Quarantine identities whose test pixels the earlier smoke opened."""

    legacy_seed = 20260712
    identities = np.asarray(sorted(frame["identity"].unique()), dtype=np.int64)
    rng = np.random.default_rng(named_seed(legacy_seed, "identity_disjoint_split"))
    rng.shuffle(identities)
    n_train = int(round(0.70 * len(identities)))
    n_val = int(round(0.15 * len(identities)))
    test_ids = set(identities[n_train + n_val :].tolist())
    subset = frame[frame["identity"].isin(test_ids)].reset_index(drop=True)
    indices = _complete_identity_indices_legacy(
        subset["identity"].to_numpy(),
        2_500,
        named_seed(legacy_seed, "limit", "test"),
    )
    return sorted(map(int, subset.iloc[indices]["identity"].unique().tolist()))

def _build_master_manifest(frame, cfg: ExperimentConfig, archive_manifest: Mapping[str, Any]) -> dict[str, Any]:
    grouped = {int(identity): group.copy() for identity, group in frame.groupby("identity", sort=True)}
    eligible = [identity for identity, group in grouped.items() if len(group) >= 2]
    public_identities: list[int] = []
    public_count = 0
    for identity in _hash_order(cfg.seed, "public-identities", eligible):
        public_identities.append(int(identity))
        public_count += len(grouped[int(identity)])
        if public_count >= cfg.public_svd_images:
            break
    public_set = set(public_identities)
    public_rows = frame[frame["identity"].isin(public_set)]
    public_filenames = _hash_order(
        cfg.seed, "public-images", [str(value) for value in public_rows["image_id"].tolist()]
    )[: cfg.public_svd_images]
    if len(public_filenames) != PUBLIC_SVD_IMAGES:
        raise RuntimeError("Could not select exactly 10,000 public SVD images")

    legacy_test_identities = _legacy_cross_svd_test_identities(frame)
    legacy_quarantine = set(legacy_test_identities)
    remaining = [
        identity
        for identity in eligible
        if identity not in public_set and identity not in legacy_quarantine
    ]
    required = cfg.max_participants * cfg.identities_per_participant
    chosen_identities = _hash_order(cfg.seed, "participant-identities", remaining)[:required]
    if len(chosen_identities) != required:
        raise RuntimeError("Too few eligible non-public CelebA identities")
    stats: list[IdentityStat] = []
    for identity in chosen_identities:
        group = grouped[int(identity)]
        rows = {
            str(row.image_id): (int(row.label), int(row.source_index))
            for row in group.itertuples(index=False)
        }
        ordered = _hash_order(cfg.seed, f"identity-{identity}-images", rows)
        reserved = str(ordered[0])
        protocol = tuple(str(value) for value in ordered[1:])
        if not protocol:
            raise AssertionError("Every selected identity must retain a protocol image")
        stats.append(
            IdentityStat(
                identity=int(identity),
                reserved_filename=reserved,
                protocol_filenames=protocol,
                protocol_labels=tuple(rows[name][0] for name in protocol),
                protocol_sources=tuple(rows[name][1] for name in protocol),
            )
        )
    buckets = _balanced_identity_allocation(stats, cfg)
    assignments = []
    for (participant, split), items in sorted(buckets.items()):
        for item in sorted(items, key=lambda value: value.identity):
            assignments.append(
                {
                    "participant": participant,
                    "split": split,
                    "identity": item.identity,
                    "reserved_filename": item.reserved_filename,
                    "protocol_filenames": list(item.protocol_filenames),
                    "protocol_labels": list(item.protocol_labels),
                    "protocol_sources": list(item.protocol_sources),
                    "protocol_image_count": item.image_count,
                    "protocol_positive_count": item.positive_count,
                }
            )
    participant_set = {int(item["identity"]) for item in assignments}
    if public_set & participant_set:
        raise AssertionError("Public and participant identities overlap")
    if legacy_quarantine & participant_set:
        raise AssertionError("A legacy-opened test identity entered the new participant design")
    reserved = {str(item["reserved_filename"]) for item in assignments}
    protocol = {
        name for item in assignments for name in map(str, item["protocol_filenames"])
    }
    if reserved & protocol:
        raise AssertionError("Reserved references entered protocol data")
    manifest = {
        "schema_version": DATA_CACHE_SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "seed": cfg.seed,
        "archive": dict(archive_manifest),
        "metadata_hashes": {
            "identity": frame.attrs["identity_metadata_sha256"],
            "attributes": frame.attrs["attribute_metadata_sha256"],
        },
        "eligibility": "at least two available images before reserving one reference",
        "public_svd": {
            "uncentered": True,
            "image_count": len(public_filenames),
            "filenames": public_filenames,
            "identity_ids": sorted(public_set),
            "identity_count": len(public_set),
        },
        "legacy_test_quarantine": {
            "source_notebook": "I_GDP_PA_CelebA_CrossSVD_Colab.ipynb",
            "legacy_seed": 20260712,
            "legacy_test_cap": 2_500,
            "identity_ids": legacy_test_identities,
            "identity_hash": stable_hash(legacy_test_identities),
            "policy": "excluded from all new participant data",
        },
        "participant_design": {
            "master_capacity_participants": cfg.max_participants,
            "identities_per_participant": cfg.identities_per_participant,
            "split_identity_counts": SPLIT_IDENTITY_COUNTS,
            "globally_participant_disjoint": True,
            "reserved_references_excluded_from_protocol": True,
        },
        "assignments": assignments,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    return manifest

@dataclass
class PublicSVDReducer:
    F: Tensor
    singular_values: Tensor
    basis_manifest: dict[str, Any]

    @property
    def F_pinv(self) -> Tensor:
        return self.F.mT

    def manifest(self) -> dict[str, Any]:
        return dict(self.basis_manifest)

@dataclass
class SplitData:
    raw_uint8: Tensor
    labels: Tensor
    identities: Tensor
    source_indices: Tensor
    participant_ids: Tensor
    reduced: Tensor
    filenames: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.labels.numel())

@dataclass
class ReferenceData:
    raw_uint8: Tensor
    identities: Tensor
    filenames: tuple[str, ...]

    def by_identity(self) -> dict[int, int]:
        return {int(identity): index for index, identity in enumerate(self.identities.tolist())}

@dataclass
class CelebADataBundle:
    train: SplitData
    val: SplitData
    test: SplitData
    references: ReferenceData
    reducer: PublicSVDReducer
    manifest: dict[str, Any]

    def split(self, name: str) -> SplitData:
        if name not in {"train", "val", "test"}:
            raise KeyError(name)
        return getattr(self, name)

    def participant_blocks(
        self, split: str, device: torch.device | str = "cpu"
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        data = self.split(split)
        xs: list[Tensor] = []
        ys: list[Tensor] = []
        indices: list[Tensor] = []
        n_participants = int(self.manifest["participant_design"]["active_participants"])
        for participant in range(n_participants):
            idx = torch.nonzero(data.participant_ids == participant, as_tuple=False).flatten()
            indices.append(idx)
            xs.append(data.reduced[idx].to(device=device, dtype=torch.float64))
            ys.append(data.labels[idx])
        return xs, ys, indices

def _assignment_frame(master: Mapping[str, Any], split: str, cfg: ExperimentConfig):
    import pandas as pd

    rows = []
    cap = cfg.smoke_protocol_images_per_identity if cfg.mode == "smoke" else None
    for assignment in master["assignments"]:
        if assignment["split"] != split or int(assignment["participant"]) >= cfg.n_participants:
            continue
        names = list(assignment["protocol_filenames"])
        labels = list(assignment["protocol_labels"])
        sources = list(assignment["protocol_sources"])
        if cap is not None:
            names, labels, sources = names[:cap], labels[:cap], sources[:cap]
        for name, label, source in zip(names, labels, sources):
            rows.append(
                {
                    "image_id": str(name),
                    "label": int(label),
                    "identity": int(assignment["identity"]),
                    "source_index": int(source),
                    "participant": int(assignment["participant"]),
                }
            )
    rows.sort(
        key=lambda row: (
            row["participant"],
            stable_hash([cfg.seed, "protocol-row", split, row["image_id"]], 64),
        )
    )
    return pd.DataFrame(rows)

def _balance_diagnostics(master: Mapping[str, Any], cfg: ExperimentConfig) -> dict[str, Any]:
    rows = []
    for assignment in master["assignments"]:
        if int(assignment["participant"]) >= cfg.n_participants:
            continue
        rows.append(
            {
                "participant": int(assignment["participant"]),
                "split": str(assignment["split"]),
                "images": int(assignment["protocol_image_count"]),
                "positive": int(assignment["protocol_positive_count"]),
            }
        )
    diagnostics: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        values = []
        for participant in sorted({row["participant"] for row in rows}):
            selected = [row for row in rows if row["participant"] == participant and row["split"] == split]
            images = sum(row["images"] for row in selected)
            positive = sum(row["positive"] for row in selected)
            values.append(
                {
                    "participant": participant,
                    "identity_count": len(selected),
                    "image_count": images,
                    "smiling_prevalence": positive / max(1, images),
                }
            )
        counts = [item["image_count"] for item in values]
        prevalences = [item["smiling_prevalence"] for item in values]
        diagnostics[split] = {
            "participants": values,
            "image_count_range": [min(counts), max(counts)],
            "prevalence_range": [min(prevalences), max(prevalences)],
            "image_count_relative_range": (max(counts) - min(counts)) / max(1.0, float(np.mean(counts))),
            "prevalence_absolute_range": max(prevalences) - min(prevalences),
        }
    return diagnostics

@torch.no_grad()
def _face_embeddings(model: nn.Module, images: Tensor, cfg: ExperimentConfig) -> Tensor:
    device = resolve_device(cfg.device)
    outputs = []
    divide_by_255 = images.dtype == torch.uint8
    for start in range(0, len(images), cfg.face_batch_size):
        batch = images[start : start + cfg.face_batch_size].to(device=device, dtype=torch.float32)
        if divide_by_255:
            batch = batch / 255.0
        batch = torch_f.interpolate(batch, size=(160, 160), mode="bilinear", align_corners=False)
        batch = (batch * 255.0 - 127.5) / 128.0
        outputs.append(model(batch).cpu())
    return torch_f.normalize(torch.cat(outputs), dim=1) if outputs else torch.empty((0, 512))

CandidateToken = tuple[str, int]

@dataclass(frozen=True)
class FaceLinkageManifest:
    query_splits: tuple[str, ...]
    query_split_indices: tuple[int, ...]
    query_target_positions: tuple[int, ...]
    query_identities: tuple[int, ...]
    source_galleries: tuple[tuple[tuple[CandidateToken, ...], ...], ...]
    cross_galleries: tuple[tuple[tuple[CandidateToken, ...], ...], ...]
    seed: int
    n_candidates: int = 10
    episodes: int = 10

    @property
    def manifest_hash(self) -> str:
        return stable_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "manifest_hash": self.manifest_hash}

def make_face_linkage_manifest(
    data: CelebADataBundle,
    blocks: Mapping[str, tuple[list[Tensor], list[Tensor], list[Tensor]]],
    cfg: ExperimentConfig,
    include_test: bool,
) -> FaceLinkageManifest:
    """One protected query per identity; the bootstrap unit is identity."""

    query_splits = ("train", "val", "test") if include_test else ("train", "val")
    candidates = []
    for split_name in query_splits:
        split = data.split(split_name)
        target_global = blocks[split_name][2][cfg.target_participant]
        by_identity: dict[int, list[tuple[int, int]]] = {}
        for target_position, global_index in enumerate(target_global.tolist()):
            identity = int(split.identities[global_index])
            by_identity.setdefault(identity, []).append((target_position, global_index))
        for identity, positions in by_identity.items():
            ordered = _hash_order(
                cfg.seed,
                f"linkage-query-{split_name}-{identity}",
                positions,
            )
            target_position, global_index = ordered[0]
            candidates.append((split_name, int(global_index), int(target_position), identity))
    candidates.sort(key=lambda value: (value[0], value[3]))
    expected = 100 if include_test else 85
    if len(candidates) != expected:
        raise RuntimeError(
            f"Protected participant {cfg.target_participant} supplied {len(candidates)} linkage "
            f"identities; expected {expected}"
        )
    reference_identities = sorted(map(int, data.references.identities.tolist()))
    reference_set = set(reference_identities)
    if any(identity not in reference_set for *_, identity in candidates):
        raise RuntimeError("A protected linkage identity lacks its reserved reference")
    rng = np.random.default_rng(named_seed(cfg.seed, "reserved-reference-linkage", include_test))
    source_all = []
    cross_all = []
    for split_name, global_index, _, identity in candidates:
        impostor_pool = np.asarray(
            [value for value in reference_identities if value != identity], dtype=np.int64
        )
        source_episodes = []
        cross_episodes = []
        for _ in range(cfg.linkage_episodes):
            impostor_ids = rng.choice(
                impostor_pool, cfg.linkage_candidates - 1, replace=False
            ).tolist()
            source_gallery: list[CandidateToken] = [
                (f"source_{split_name}", int(global_index)),
                *(("reference", int(value)) for value in impostor_ids),
            ]
            cross_gallery: list[CandidateToken] = [
                ("reference", int(identity)),
                *(("reference", int(value)) for value in impostor_ids),
            ]
            rng.shuffle(source_gallery)
            rng.shuffle(cross_gallery)
            source_episodes.append(tuple(source_gallery))
            cross_episodes.append(tuple(cross_gallery))
        source_all.append(tuple(source_episodes))
        cross_all.append(tuple(cross_episodes))
    return FaceLinkageManifest(
        query_splits=tuple(value[0] for value in candidates),
        query_split_indices=tuple(value[1] for value in candidates),
        query_target_positions=tuple(value[2] for value in candidates),
        query_identities=tuple(value[3] for value in candidates),
        source_galleries=tuple(source_all),
        cross_galleries=tuple(cross_all),
        seed=named_seed(cfg.seed, "reserved-reference-linkage", include_test),
        n_candidates=cfg.linkage_candidates,
        episodes=cfg.linkage_episodes,
    )

def _identity_bootstrap_ci(
    per_identity: Sequence[float], cfg: ExperimentConfig, seed: int
) -> list[float]:
    values = np.asarray(per_identity, dtype=np.float64)
    if not len(values):
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    means = []
    remaining = max(200, cfg.bootstrap_resamples)
    while remaining:
        batch = min(500, remaining)
        indices = rng.integers(0, len(values), size=(batch, len(values)))
        means.append(values[indices].mean(axis=1))
        remaining -= batch
    return [float(value) for value in np.quantile(np.concatenate(means), [0.025, 0.975])]

def _candidate_image(data: CelebADataBundle, token: CandidateToken) -> Tensor:
    kind, key = token
    if kind == "reference":
        position = data.references.by_identity()[int(key)]
        return data.references.raw_uint8[position]
    if kind.startswith("source_"):
        split = kind.removeprefix("source_")
        return data.split(split).raw_uint8[int(key)]
    raise KeyError(token)

def _evaluate_face_linkage(
    recovered_queries: Tensor,
    manifest: FaceLinkageManifest,
    embedding_cache: Mapping[CandidateToken, Tensor],
    face_model: nn.Module,
    cfg: ExperimentConfig,
) -> dict[str, Any]:
    recovered_embeddings = _face_embeddings(face_model, recovered_queries.clamp(0, 1), cfg)
    result: dict[str, Any] = {
        "chance_top1": LINKAGE_CHANCE,
        "n_identities": len(manifest.query_identities),
        "episodes_per_identity": manifest.episodes,
        "manifest_hash": manifest.manifest_hash,
        "bootstrap_unit": "identity",
    }
    for task, episode_groups in (
        ("source_image", manifest.source_galleries),
        ("cross_image_identity", manifest.cross_galleries),
    ):
        per_identity = []
        ranks = []
        for position, galleries in enumerate(episode_groups):
            identity = int(manifest.query_identities[position])
            split = manifest.query_splits[position]
            source_index = int(manifest.query_split_indices[position])
            genuine: CandidateToken = (
                (f"source_{split}", source_index)
                if task == "source_image"
                else ("reference", identity)
            )
            successes = []
            for gallery in galleries:
                candidates = torch.stack([embedding_cache[token] for token in gallery])
                similarities = candidates @ recovered_embeddings[position]
                true_position = gallery.index(genuine)
                order = torch.argsort(similarities, descending=True)
                rank = int(torch.nonzero(order == true_position, as_tuple=False)[0]) + 1
                ranks.append(rank)
                successes.append(int(rank == 1))
            per_identity.append(float(np.mean(successes)))
        rank_array = np.asarray(ranks, dtype=np.int64)
        result[task] = {
            "top1": float(np.mean(rank_array == 1)),
            "top3": float(np.mean(rank_array <= 3)),
            "mrr": float(np.mean(1.0 / rank_array)),
            "mean_rank": float(rank_array.mean()),
            "identity_top1": per_identity,
            "ci95_identity_cluster": _identity_bootstrap_ci(
                per_identity, cfg, named_seed(cfg.seed, "linkage-ci", task)
            ),
            "n_episodes": int(len(rank_array)),
        }
    return result

def _query_tensor_from_blocks(
    manifest: FaceLinkageManifest,
    blocks: Mapping[str, tuple[list[Tensor], list[Tensor], list[Tensor]]],
    participant: int,
) -> Tensor:
    rows = [
        blocks[split][0][participant][position]
        for split, position in zip(manifest.query_splits, manifest.query_target_positions)
    ]
    return torch.stack(rows) if rows else torch.empty((0, 0), dtype=torch.float64)
