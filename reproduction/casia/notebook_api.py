"""Readable public entry points for the current matched CASIA experiments.

All preparation starts with downloaded dataset bytes. Feature and experiment
caches are generated locally and validated against the current source hashes;
there is no dependence on the authors' Drive folders or historical results.
"""
from pathlib import Path
import gc
import importlib
import json

HERE = Path(__file__).resolve().parent
DATASET_SEEDS = {"celeba": 20260713, "vggface2": 20260928}
VERSION = "casia-full-h400-cnn-spectral-rebinned-20260928-v2"


def _dataset(value):
    name = str(value).lower().strip()
    if name not in DATASET_SEEDS:
        raise ValueError("Dataset must be 'celeba' or 'vggface2'")
    return name


def _separate(*roots):
    paths = [Path(root).resolve() for root in roots]
    for index, path in enumerate(paths):
        for other in paths[index + 1:]:
            if path == other or path in other.parents or other in path.parents:
                raise ValueError(f"Raw inputs, caches and outputs must be isolated: {path}, {other}")
    return paths


def _report(**event):
    print("CASIA_PREPARATION", json.dumps(event, sort_keys=True, allow_nan=False), flush=True)


def validate_raw_layout(dataset, data_root):
    """Check paths cheaply before expensive actual-byte verification in preparation."""
    name, root = _dataset(dataset), Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Download the raw {name} files into {root} first")
    if name == "celeba":
        from celeba_inputs import find_metadata
        attributes = find_metadata(root, ("list_attr_celeba.txt", "list_attr_celeba.csv"))
        identities = find_metadata(root, ("identity_CelebA.txt", "identity_CelebA.csv",
                                          "list_identity_celeba.txt", "list_identity_celeba.csv"))
        folders = [path for path in root.rglob("img_align_celeba") if path.is_dir()]
        if not folders:
            raise FileNotFoundError("Missing extracted img_align_celeba JPEG directory")
        return {"dataset": name, "data_root": str(root), "attributes": str(attributes),
                "identities": str(identities), "image_directories": [str(path) for path in folders]}
    pins = json.loads((HERE / "vgg_asset_pins.json").read_text())["assets"]
    for pin in pins:
        path = root / pin["path"]
        if not path.is_file() or path.stat().st_size != pin["size"]:
            raise FileNotFoundError(f"Missing or wrong-size downloaded file: {path}")
    return {"dataset": name, "data_root": str(root),
            "raw_files": [pin["path"] for pin in pins], "raw_bytes": sum(pin["size"] for pin in pins)}


def prepare_dataset(dataset, data_root, cache_root, device="cuda", progress=None):
    """Validate metadata/bytes, decode RGB64, fit the public SVD and project private rows.

    This stage freezes 50 participants with 100 identities each, identity-disjoint
    70/15/15 splits and separate reference images. It never trains utility models
    or selects cohorts using auditor outcomes. A completed cache is fully checked
    and reused; a different raw input/code identity requires a separate cache.
    """
    name = _dataset(dataset)
    raw, cache = _separate(data_root, cache_root)
    validate_raw_layout(name, raw)
    callback = progress or _report
    callback(stage="metadata_rgb64_public_svd_projection", dataset=name,
             data_root=str(raw), cache_root=str(cache), device=device)
    import torch
    torch.set_num_threads(4)
    adapter = importlib.import_module("celeba_inputs" if name == "celeba" else "vgg_context")
    spec = {"dataset_seed": DATASET_SEEDS[name], "preparation_device": device}
    manifest = adapter.prepare_inputs(raw, cache, HERE, spec, callback)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"dataset": name, "cache_root": str(cache), "state": "prepared",
            "master_manifest_hash": manifest["manifest_hash"],
            "participant_count": 50, "identity_count": 5000,
            "protocol_images": manifest["total_protocol_images"],
            "public_svd_images": 10000, "h": 400,
            "manifest_path": str(cache / "master_manifest.json"),
            "verification_path": str(cache / "master_verified.json")}


def study_spec(dataset, data_root, cache_root, output_root, device="cuda"):
    """Fixed manuscript recipe; quota calibration is separate at each p."""
    name = _dataset(dataset)
    raw, cache, output = _separate(data_root, cache_root, output_root)
    return {
        "version": VERSION, "dataset": name, "dataset_label": "CelebA" if name == "celeba" else "VGGFace2",
        "dataset_seed": DATASET_SEEDS[name], "seed": 20260928,
        "source_root": str(raw), "cache_root": str(cache), "output": str(output),
        "module_dir": str(HERE), "input_adapter": "celeba_inputs" if name == "celeba" else "vgg_context",
        "preparation_device": device, "participants": [2, 5, 10, 20, 50],
        "master_participants": 50, "identities_per_participant": 100,
        "identity_split_counts": {"train": 70, "val": 15, "test": 15},
        "public_svd_images": 10000, "h": 400, "raw_shape": [3, 64, 64],
        "cnn_input_shape": [1, 20, 20], "utility_model": "original 9381-parameter CNN",
        "feature_extractor": "public uncentered linear SVD400 projection",
        "alignment": "spectral-only; leading right singular subspace then participant blockwise polar",
        "gpm_iterations": 0, "paired_gpm_comparison": False,
        "anchor_rows": 1600, "anchor_normalization": "centered unit isometric; A.T @ A = I",
        "linkage_auditor": "frozen InceptionResnetV1 pretrained CASIA-WebFace 20180408-102900",
        "auditor_checkpoint_sha256": "7a67afdbbc995fce5e10128675e318799a70698c2f433ba75dd7eb9a2f096e7d",
        "pretraining_identity_overlap_screened": False, "reference_treatment": "projected_decoded",
        "anchor_attack": "PA-OP", "private_attack": "known-secret",
        "audit_query_identities": 100, "audit_episodes_per_identity": 10, "audit_candidates_per_episode": 10,
        "sampling_mode": "quota_bins", "linkage_bin_edges_percent": [0, 15, 25, 35, 45, 100],
        "linkage_bin_rule": "[lo,hi), except [45,100]", "conditions_per_bin_per_family": 20,
        "max_proposals_per_family": 1500, "noise_families": ["anchor", "private"],
        "noise_scale_upper_bounds": {"anchor": 0.08, "private": 3.0},
        "noise_scale_units": "absolute Gaussian standard deviation",
        "local_models": "fresh per participant at each p; pooled test-confusion summary",
        "control_models_per_p": ["i_gdp", "c_gdp"], "planned_training_fits": 1097,
        "epochs": 15, "batch_size": 256, "learning_rate": 0.001, "weight_decay": 0,
        "loss": "unweighted binary cross entropy with logits",
        "checkpoint_selection": "highest validation balanced accuracy; earliest tie", "test_threshold": 0.5,
        "training_device_requested": "NVIDIA T4 CUDA", "audit_device": "CUDA float32",
        "cpu_threads": 4, "alignment_device": "CPU float64",
        "training_precision": "float32; no autocast or TF32; deterministic CUDA",
        "resume_policy": "one worker; immutable current inputs/configuration; completed records and epoch checkpoints",
        "sampling_note": "independent linkage-conditioned calibration for every p; not paired accepted draws",
    }


def build_study_config(dataset, data_root, cache_root, output_root, device="cuda"):
    """Write an immutable experiment.json, containing no historical paths or tensor pins."""
    from data_pipeline import lock_json
    from run_study import validate_spec
    spec = validate_spec(study_spec(dataset, data_root, cache_root, output_root, device))
    path = Path(spec["output"]) / "specs" / "experiment.json"
    lock_json(path, spec)
    return path


def load_context(dataset, data_root, cache_root, context_root, p=10, device="cuda", progress=None):
    """Load fresh prepared inputs and generate a shared GDP bank/CASIA galleries.

    context_root owns experiment context receipts, shared_design and model_assets;
    p-specific receipts go under pNNN. It must be outside the raw/prepared roots.
    Call prepare_dataset first. This entry point does not run utility training.
    """
    import data_pipeline as dp
    name = _dataset(dataset)
    raw, cache, context = _separate(data_root, cache_root, context_root)
    if not (cache / "master_verified.json").is_file():
        raise RuntimeError("Call prepare_dataset first; no raw-to-feature stage may be skipped")
    spec = study_spec(name, raw, cache, context, device)
    spec.update(study_root=str(context), design_root=str(context / "shared_design"),
                model_cache_dir=str(context / "model_assets"))
    return dp.load_context(raw, context / f"p{p:03d}", p, HERE, audit_device=device,
                           protocol_config=spec, cache_root=cache, progress=progress or _report)


def run_privacy_utility(config_path, device="cuda"):
    """Prepare/check current raw inputs and execute the bounded 1097-fit sweep.

    Exhausting a calibration budget produces incomplete_bins with exact accepted
    counts; it never substitutes or silently relaxes quotas. CUDA execution requires
    the requested T4. CPU is intended for tests, not the published full experiment.
    """
    import run_pipeline
    spec = json.loads(Path(config_path).read_text())
    run_pipeline.run(spec["output"], config_path, device=device)
    return run_pipeline.validate_completion(spec["output"])


def inspect_results(output_root):
    """Read durable progress and terminal receipts without executing experiment work."""
    from run_study import read, digest
    import run_pipeline
    root = Path(output_root)
    if (root / "study/completion.json").exists() or (root / "study/incomplete.json").exists():
        return run_pipeline.validate_completion(root)
    status = read(root / "pipeline_status.json") if (root / "pipeline_status.json").exists() else {"state": "not_started"}
    records = sorted((root / "study").glob("p*/conditions/*/record.json"))
    ids = set()
    for path in records:
        row = read(path)
        key = (row["condition"]["p"], row["condition"]["id"])
        if key in ids:
            raise RuntimeError("Duplicate committed condition identity")
        for relative, expected in row["artifact_hashes"].items():
            artifact = (path.parent / relative).resolve()
            if path.parent.resolve() not in artifact.parents or digest(artifact) != expected:
                raise RuntimeError(f"Changed committed artifact: {artifact}")
        ids.add(key)
    return {"status": status, "verified_completed_fits": len(ids), "planned_fits": 1097}


def export_privacy_utility(output_root):
    """Validate terminal receipts, then export summaries and figures locally."""
    import run_pipeline
    import render_results
    verified = run_pipeline.validate_completion(output_root)
    render_results.summarize(Path(output_root))
    target = Path(output_root) / "figures"
    figure_paths = [str(path.resolve()) for path in sorted(target.glob("*"))
                    if path.suffix.lower() in (".png", ".pdf", ".svg")]
    if not figure_paths:
        raise RuntimeError("Renderer did not produce any figure artifacts")
    return {"verification": verified, "figure_paths": figure_paths,
            "artifacts": str((target / "figure_provenance.json").resolve())}
