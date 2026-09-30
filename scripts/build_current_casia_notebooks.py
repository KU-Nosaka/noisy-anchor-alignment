"""Build and validate the four readable Colab notebooks for the current study.

No notebook embeds source archives or previous experiment state. The readable
cells import the versioned reproduction/casia modules in the same checkout.
Run this script with --check before committing generated notebooks.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "current"
REPO_URL = "https://github.com/KU-Nosaka/noisy-anchor-alignment.git"
SPECS = (
    ("10_celeba_casia_privacy_utility.ipynb", "celeba", "full"),
    ("11_vggface2_casia_privacy_utility.ipynb", "vggface2", "full"),
    ("12_celeba_casia_attack_comparison.ipynb", "celeba", "attacks"),
    ("13_vggface2_casia_attack_comparison.ipynb", "vggface2", "attacks"),
)


def cell(kind: str, name: str, text: str) -> dict:
    source = textwrap.dedent(text).strip() + "\n"
    value = {
        "cell_type": kind,
        "id": name,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def markdown(name: str, text: str) -> dict:
    return cell("markdown", name, text)


def code(name: str, text: str) -> dict:
    return cell("code", name, text)


def make_notebook(filename: str, dataset: str, mode: str) -> dict:
    title = "CelebA" if dataset == "celeba" else "VGGFace2"
    cohort_seed = 20260713 if dataset == "celeba" else 20260928
    full = mode == "full"
    heading = "privacy–utility sweep" if full else "MP / AM / OP attack comparison"
    output_name = "privacy_utility" if full else "attack_comparison500"
    raw_layout = (
        "`img_align_celeba/*.jpg`, `list_attr_celeba.txt` (or the documented CSV "
        "variant), and `identity_CelebA.txt` (or its CSV variant)."
        if dataset == "celeba" else
        "`data/vggface2_train.tar.gz`, `meta/bb_landmark.tar.gz`, "
        "`meta/identity_meta.csv`, `meta/train_list.txt`, and `maad/MAAD_Face.csv`."
    )
    resources = (
        "Use a high-RAM runtime when possible (24–32 GB host RAM is a useful "
        "planning range), with about 20 GiB free disk as a planning budget for "
        "preparation, feature rows, checkpoints, and results. The decoded and "
        "projected cache is under about 2 GB, in addition to the raw images. "
        "These are planning bounds, not guaranteed minima."
        if dataset == "celeba" else
        "Plan for at least 80 GB free local disk before preparation and about "
        "100 GB for the full sweep/checkpoints, with 24–32 GB host RAM. The raw "
        "files occupy about 38.6 GB; the decoded candidate cache is about "
        "15.1 GB and projected rows about 2.0 GB. The raw RGB cache is memory "
        "mapped rather than copied completely into RAM. These are planning "
        "bounds, not guaranteed minima; no paid runtime settings are changed."
    )
    experiment = (
        "The bounded sweep evaluates p = 2, 5, 10, 20, 50. Each participant has "
        "100 identities, split 70 / 15 / 15 by identity. Each noise family targets "
        "20 accepted observations in each absolute linkage interval [0,15), "
        "[15,25), [25,35), [35,45), [45,100] percent. Calibration stops after at "
        "most 1,500 proposals per family and participant count; an unfilled quota "
        "is reported as incomplete_bins. With all quotas filled, the expected "
        "unique fit counts are 204, 207, 212, 222, 252, totaling 1,097. "
        "These counts include collaborative controls and Local models without "
        "counting a reused prefitted C-GDP model twice."
        if full else
        "The comparison fixes p = 10 and independently samples 500 anchor-noise "
        "scales from Uniform(0, 0.08). MP, AM, and OP use exactly the same noisy "
        "uploads for each draw. There is no bin calibration, acceptance filter, "
        "or utility training. A separate zero-noise diagnostic is excluded from "
        "the 500 observations, mean summaries, and fitted curves."
    )
    cells = [
        markdown("overview", f"""
        # {title}: current CASIA {heading}

        Run the cells in order on a **Tesla T4 GPU**. This notebook reruns the
        experiment from raw images and metadata in the repository. It prepares
        its own cohort, RGB64 image cache, public SVD feature mapping, and frozen
        protocol context. It requires no previous study, feature cache, checkpoint,
        result bundle, or private Google Drive folder.

        {experiment}

        Both datasets use h = 400, r = 1,600 centered orthonormal anchors,
        participant-specific GDP transformations, spectral-only alignment, and
        the frozen CASIA-WebFace InceptionResnetV1 identity auditor. The full
        utility study uses the original CNN, 15 epochs, Adam at 1e-3, batch size
        256, and selection by validation balanced accuracy.

        The raw datasets are assumed to have been downloaded legally already.
        Place their archives and metadata in `data/raw/{dataset}/` **inside the
        repository**, following the repository's documented raw-data layout.
        A fresh Colab clone copies code, not ignored local datasets: upload/copy
        those raw inputs to the clone before the preparation cell. The notebook
        does not connect Google Drive or change a running experiment.

        Plan for a long run and substantial CPU RAM and local disk in addition
        to the T4's 16 GB device memory. {resources}
        Local Colab storage is ephemeral. Keep
        or download the output receipts and completed records before ending a
        runtime. To resume, restore the same cache/output folders and rerun in
        order; the code verifies provenance before reusing completed work.
        """),
        markdown("repo_heading", """
        ## 1. Repository setup

        Clone the published `main` branch into `/content/noisy-anchor-alignment`.
        An existing checkout is reused without deleting raw data or outputs.
        For local Jupyter, change `REPO_ROOT` to the existing repository directory.
        """),
        code("repository_setup", f"""
        from pathlib import Path
        import subprocess
        import sys

        REPO_ROOT = Path("/content/noisy-anchor-alignment")
        REPO_URL = {REPO_URL!r}
        REPO_REF = "main"

        if not REPO_ROOT.exists():
            subprocess.check_call([
                "git", "clone", "--depth", "1", "--branch", REPO_REF,
                REPO_URL, str(REPO_ROOT),
            ])
        else:
            print("Reusing existing repository directory:", REPO_ROOT)

        API_ROOT = REPO_ROOT / "reproduction" / "casia"
        if not (API_ROOT / "notebook_api.py").is_file():
            raise FileNotFoundError(
                "This checkout does not contain the current CASIA reproduction "
                "API. Use the updated repository main branch before continuing."
            )
        sys.path.insert(0, str(API_ROOT))
        print("Reproduction modules:", API_ROOT)
        """),
        markdown("install_heading", """
        ## 2. Install packages and check the runtime

        Retain Colab's CUDA-enabled PyTorch and torchvision. Install the frozen
        auditor package with `--no-deps` so its older dependency constraints do
        not downgrade those GPU packages. No model is fine-tuned by this notebook.
        """),
        code("install_packages", """
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "numpy", "scipy", "pandas", "pillow", "matplotlib",
            "scikit-learn", "tqdm", "requests", "gdown", "psutil",
        ])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--no-deps",
            "facenet-pytorch==2.6.0",
        ])

        import torch
        import torchvision

        if not torch.cuda.is_available():
            raise RuntimeError("Select a GPU runtime with a Tesla T4 before running.")
        GPU_NAME = torch.cuda.get_device_name(0)
        if "T4" not in GPU_NAME:
            raise RuntimeError(f"Expected a Tesla T4; this runtime provides {GPU_NAME}.")
        DEVICE = "cuda"
        print("GPU:", GPU_NAME)
        print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__)
        """),
        markdown("paths_heading", f"""
        ## 3. Paths and fixed study settings

        Only the raw input location needs to be supplied. All caches and new
        outputs remain in this checkout. The constants below document the fixed
        recipe implemented by the versioned reproduction API; the generated
        configuration and receipts record the effective settings. To implement
        a different experiment, change the versioned runner configuration rather
        than only editing these descriptive constants.

        Required entries relative to the raw dataset root: {raw_layout}
        """),
        code("configure_paths", f"""
        import json
        import shutil
        import psutil
        import notebook_api as api

        DATASET = {dataset!r}
        DATA_ROOT = REPO_ROOT / "data" / "raw" / DATASET
        CACHE_ROOT = REPO_ROOT / "cache" / "current_casia" / DATASET
        OUTPUT_ROOT = REPO_ROOT / "build" / "current_casia" / DATASET / {output_name!r}
        CONTEXT_ROOT = OUTPUT_ROOT / "context"

        STUDY_SEED = 20260928
        COHORT_SEED = {cohort_seed}
        ATTACK_SEED = 20260930
        FEATURE_DIMENSION = 400
        ANCHOR_ROWS = 1600
        PARTICIPANTS = [2, 5, 10, 20, 50]
        IDENTITIES_PER_PARTICIPANT = 100
        LINKAGE_BIN_EDGES = [0, 15, 25, 35, 45, 100]
        OBSERVATIONS_PER_BIN = 20
        CALIBRATION_PROPOSAL_CAP = 1500
        EXPECTED_FITS_BY_P = {{2: 204, 5: 207, 10: 212, 20: 222, 50: 252}}
        EXPECTED_FULL_FITS = 1097
        ATTACK_DRAWS = 500
        ANCHOR_NOISE_RANGE = (0.0, 0.08)
        PRIVATE_NOISE_RANGE = (0.0, 3.0)

        if not DATA_ROOT.is_dir():
            raise FileNotFoundError(
                f"Put the raw {{DATASET}} images/archives and metadata inside "
                f"the repository at {{DATA_ROOT}} before continuing."
            )
        for path in (CACHE_ROOT, OUTPUT_ROOT):
            path.mkdir(parents=True, exist_ok=True)
        print("Raw data:", DATA_ROOT)
        print("Preparation cache:", CACHE_ROOT)
        print("New output:", OUTPUT_ROOT)
        print("Free local disk (GiB):", round(shutil.disk_usage(REPO_ROOT).free / 2**30, 1))
        print("Host RAM total / available (GiB):", round(psutil.virtual_memory().total / 2**30, 1),
              "/", round(psutil.virtual_memory().available / 2**30, 1))
        """),
        markdown("prepare_heading", f"""
        ## 4. Validate raw inputs, select identities, and preprocess images

        The preparation stage validates raw metadata and image sources, excludes
        reserved reference images from utility uploads, fixes the 50-participant
        identity allocation, and builds RGB64 images in channel-first order.
        CelebA uses centered square crops. VGGFace2 uses the supplied face boxes
        expanded by 1.3, clipped to the image, and centered to a square.
        VGGFace2 utility uses the released MAAD-Face Smiling labels; images with
        unknown labels are excluded.

        **This cell is always required.** A missing raw preparation cannot be
        skipped merely because a later receipt or old output folder exists.
        Verified caches may be reused, but a new run computes everything from
        raw inputs. Progress messages identify the preparation and SVD stages.
        """),
        code("prepare_dataset", """
        prepared_receipt = api.prepare_dataset(
            DATASET, data_root=DATA_ROOT, cache_root=CACHE_ROOT, device=DEVICE,
        )
        print(json.dumps(prepared_receipt, indent=2, default=str))
        """),
        markdown("svd_heading", """
        ## 5. Public SVD feature mapping

        The preceding preparation cell fits an **uncentered truncated SVD on
        10,000 public images**, whose identities are disjoint from participant
        identities. It produces a d = 12,288 by h = 400 orthonormal feature
        basis, projects the participant/query/reference rows, and writes source
        and feature receipts. An existing basis is reused only after the raw
        source and configuration checks pass. No pretrained/private feature
        file is supplied as an input.

        The receipt printed above identifies the completed feature preparation.
        The next stage verifies that prepared data before constructing the
        frozen protocol context; a failed SVD stage prevents the experiment.
        """),
        code("verify_feature_stage", """
        expected_preparation = {
            "state": "prepared", "participant_count": 50, "identity_count": 5000,
            "public_svd_images": 10000, "h": 400,
        }
        for key, expected in expected_preparation.items():
            if prepared_receipt.get(key) != expected:
                raise RuntimeError(f"Preparation receipt mismatch for {key}: {prepared_receipt.get(key)}")
        for key in ("manifest_path", "verification_path"):
            if not Path(prepared_receipt[key]).is_file():
                raise FileNotFoundError(f"Missing completed feature receipt: {prepared_receipt[key]}")
        print("Public SVD and participant projections verified:", prepared_receipt["master_manifest_hash"])
        """),
        markdown("context_heading", """
        ## 6. Frozen CASIA auditor and protocol context

        Load and freeze CASIA-WebFace InceptionResnetV1. The code verifies the
        pretrained asset and runs the same RGB64 projection/decoding, 160-pixel
        bilinear resize, standardization, and normalized cosine matching used
        in the study. It does not train the auditor on either evaluation dataset.
        """),
        code("load_context", f"""
        context = api.load_context(
            DATASET, data_root=DATA_ROOT, cache_root=CACHE_ROOT,
            context_root=CONTEXT_ROOT, p={2 if full else 10}, device=DEVICE,
        )
        print("Frozen context is ready for", DATASET, "at p = {2 if full else 10}.")
        """),
        markdown("protocol_heading", """
        ## 7. GDP, shared anchors, and identity galleries

        The verified context fixes Haar orthogonal GDP matrices and Gaussian
        translations for 50 participants, r = 1,600 centered orthonormal anchor
        rows, and the same protected participant and genuine references. The
        reference/colluding participant is participant 0; the protected one is
        participant 1. C-GDP uses the reference participant's transformation. For
        each collaboration size, 100 protected queries use ten ten-way galleries
        each: 1,000 lineups with 10% random-guessing accuracy. Distractors come
        from that active collaboration; lineups stay fixed across methods and
        noise draws at the same p. Source/configuration hashes are retained in
        the experiment records.
        """),
    ]

    if full:
        cells.extend([
            code("build_study_config", """
            CONFIG_PATH = api.build_study_config(
                DATASET, data_root=DATA_ROOT, cache_root=CACHE_ROOT,
                output_root=OUTPUT_ROOT,
            )
            study_config = json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
            print("Configuration:", CONFIG_PATH)
            print(json.dumps(study_config, indent=2))
            """),
            markdown("run_heading", """
            ## 8. Run the bounded privacy–utility sweep

            Run both noise placements at all five participant counts. Calibration
            measures linkage first and accepts up to 20 observations per bin,
            using fresh noise for proposals. Utility then trains the original
            CNN for accepted observations and controls. This can take more than
            one Colab session; the output folder stores resumable per-condition
            records. The finite cap can produce `incomplete_bins`, which is
            reported honestly instead of relabeling observations or silently
            filling bins. Full success requires all 1,097 unique fits and quotas.
            """),
            code("run_experiment", """
            terminal_summary = api.run_privacy_utility(CONFIG_PATH, device=DEVICE)
            print(json.dumps(terminal_summary, indent=2, default=str))
            """),
            markdown("inspect_heading", """
            ## 9. Inspect committed results and export figures

            Inspect durable record counts, per-p schedules, calibration quotas,
            and terminal status before using the results. Plotting reads those
            saved observations; it does not train another model. Save the output
            directory and exported PDFs/PNGs before ending the runtime.
            """),
            code("inspect_results", """
            verified_summary = api.inspect_results(OUTPUT_ROOT)
            print(json.dumps(verified_summary, indent=2, default=str))
            """),
            code("export_figures", """
            export_receipt = api.export_privacy_utility(OUTPUT_ROOT)
            print(json.dumps(export_receipt["verification"], indent=2, default=str))
            figure_paths = export_receipt["figure_paths"]
            from IPython.display import FileLink, Image, display

            for figure_path in figure_paths:
                path = Path(figure_path)
                if not path.is_file():
                    raise FileNotFoundError(f"Exported figure is missing: {path}")
                print(path)
                if path.suffix.lower() == ".png":
                    display(Image(filename=str(path)))
                else:
                    display(FileLink(str(path)))
            """),
        ])
    else:
        cells.extend([
            markdown("run_heading", """
            ## 8. Run 500 uniform draws and all three attacks

            Each draw generates one fresh noisy anchor upload for every active
            participant. All three estimators attack that same upload, using
            the same protected queries and galleries. No utility classifier is
            trained. The seed and unconditional noise schedule are committed
            before draw evaluation. A zero-noise diagnostic is saved separately.
            """),
            code("run_experiment", """
            import attack_comparison as attack

            terminal_summary = attack.run_attack_comparison(
                DATASET, data_root=DATA_ROOT, cache_root=CACHE_ROOT,
                output_root=OUTPUT_ROOT, device=DEVICE, seed=ATTACK_SEED,
            )
            print(json.dumps(terminal_summary, indent=2, default=str))
            """),
            markdown("inspect_heading", """
            ## 9. Inspect paired records and export the comparison

            Full success requires exactly 500 unique random draws and all three
            attack outcomes for every draw. Inspect the separate zero-noise
            diagnostic, immutable draw receipts, and completed schedule before
            interpreting the curves. The exported spline curves describe these
            observations; their shaded 95% bootstrap bands resample draw indices
            jointly across attacks, with the diagnostic excluded.
            """),
            code("inspect_results", """
            verified_summary = attack.inspect_results(OUTPUT_ROOT)
            print(json.dumps(verified_summary, indent=2, default=str))
            """),
            code("export_figures", """
            exported_files = attack.export_figures(OUTPUT_ROOT)
            print(json.dumps(exported_files, indent=2, default=str))
            from IPython.display import FileLink, Image, display

            for figure_path in exported_files.values():
                path = Path(figure_path)
                if not path.is_file():
                    raise FileNotFoundError(f"Exported artifact is missing: {path}")
                print(path)
                if path.suffix.lower() == ".png":
                    display(Image(filename=str(path)))
                else:
                    display(FileLink(str(path)))
            """),
        ])

    cells.append(markdown("completion", """
    ## Retain the run

    Keep the raw input hashes, preparation and configuration receipts, condition
    records, calibration/schedule receipts, and terminal summary together. A
    live progress message alone is not proof of completion. Resume only from
    these verified folders; do not mix outputs from different seeds or settings.
    """))
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": filename, "provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "current_casia": {
                "dataset": dataset, "mode": mode,
                "cohort_seed": cohort_seed, "study_seed": 20260928,
                "attack_seed": 20260930,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def canonical_bytes(notebook: dict) -> bytes:
    return (json.dumps(notebook, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def validate_notebook(notebook: dict, filename: str) -> None:
    if (notebook.get("nbformat"), notebook.get("nbformat_minor")) != (4, 5):
        raise ValueError(f"{filename}: expected notebook format 4.5")
    cells = notebook.get("cells", [])
    ids = [value.get("id") for value in cells]
    if len(ids) != len(set(ids)) or any(not name for name in ids):
        raise ValueError(f"{filename}: missing or duplicate cell IDs")
    required = ["repository_setup", "install_packages", "configure_paths", "prepare_dataset", "verify_feature_stage",
                "load_context", "run_experiment", "inspect_results", "export_figures"]
    if any(name not in ids for name in required):
        raise ValueError(f"{filename}: a required execution stage is missing")
    if [ids.index(name) for name in required] != sorted(ids.index(name) for name in required):
        raise ValueError(f"{filename}: execution stages are out of order")
    for value in cells:
        if value.get("cell_type") == "code":
            if value.get("execution_count") is not None or value.get("outputs") != []:
                raise ValueError(f"{filename}: saved execution state is not cleared")
            ast.parse("".join(value["source"]), filename=f"{filename}:{value['id']}")
    preparation = next(value for value in cells if value["id"] == "prepare_dataset")
    tree = ast.parse("".join(preparation["source"]))
    unconditional = any(
        isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "prepare_dataset"
        for statement in tree.body
    )
    if not unconditional:
        raise ValueError(f"{filename}: raw preparation must be an unconditional stage")
    all_source = "\n".join("".join(value["source"]) for value in cells)
    for forbidden in ("drive.mount", "/content/drive/", "base64", "gzip.decompress"):
        if forbidden in all_source:
            raise ValueError(f"{filename}: forbidden prior-state/opaque source marker: {forbidden}")
    if "--no-deps" not in all_source or "facenet-pytorch==2.6.0" not in all_source:
        raise ValueError(f"{filename}: frozen-auditor installation must avoid torch downgrades")
    current = notebook["metadata"]["current_casia"]
    expected_seed = 20260713 if current["dataset"] == "celeba" else 20260928
    if current["cohort_seed"] != expected_seed:
        raise ValueError(f"{filename}: incorrect cohort seed")
    if current["mode"] == "full":
        if "run_privacy_utility" not in all_source or "EXPECTED_FULL_FITS = 1097" not in all_source:
            raise ValueError(f"{filename}: full-study specification is incomplete")
    else:
        if "run_privacy_utility(" in all_source or "run_attack_comparison" not in all_source:
            raise ValueError(f"{filename}: attack notebooks must not train utility models")
        if "ATTACK_DRAWS = 500" not in all_source or "ANCHOR_NOISE_RANGE = (0.0, 0.08)" not in all_source:
            raise ValueError(f"{filename}: uniform attack schedule is incomplete")


def self_test() -> None:
    """Ensure the validator catches the mistakes that break staged reruns."""
    import copy

    reference = make_notebook(*SPECS[0])
    validate_notebook(reference, "valid.ipynb")
    mutations = []
    saved_output = copy.deepcopy(reference)
    next(value for value in saved_output["cells"] if value["cell_type"] == "code")["outputs"] = [{"output_type": "stream", "text": "stale"}]
    mutations.append(saved_output)
    missing_preparation = copy.deepcopy(reference)
    missing_preparation["cells"] = [value for value in missing_preparation["cells"] if value["id"] != "prepare_dataset"]
    mutations.append(missing_preparation)
    guarded_preparation = copy.deepcopy(reference)
    next(value for value in guarded_preparation["cells"] if value["id"] == "prepare_dataset")["source"] = [
        "if CACHE_ROOT.exists():\n", "    prepared_receipt = api.prepare_dataset(DATASET, DATA_ROOT, CACHE_ROOT)\n",
    ]
    mutations.append(guarded_preparation)
    for mutant in mutations:
        try:
            validate_notebook(mutant, "invalid.ipynb")
        except ValueError:
            continue
        raise AssertionError("Notebook validator accepted an invalid staged rerun")
    print("Validator self-tests passed: saved output, missing preparation, guarded preparation.")
    export_cell_smoke_test()


def export_cell_smoke_test() -> None:
    """Execute final cells with the two real exporter return shapes, without GPU work."""
    import contextlib
    import io
    import tempfile
    import types
    from unittest.mock import patch

    with tempfile.TemporaryDirectory(prefix="casia-notebook-export-test-") as directory:
        output = Path(directory)
        files = {extension: output / f"comparison.{extension}" for extension in ("png", "pdf", "svg")}
        for path in files.values():
            path.write_bytes(b"test export placeholder")
        outputs = []
        display_module = types.ModuleType("IPython.display")
        display_module.display = outputs.append
        display_module.Image = lambda filename: ("image", filename)
        display_module.FileLink = lambda filename: ("link", filename)
        ipython_module = types.ModuleType("IPython")
        ipython_module.display = display_module
        full_return = {"verification": {"state": "complete", "verified_completed_fits": 1097},
                       "figure_paths": [str(path) for path in files.values()]}
        attack_return = {key: str(path) for key, path in files.items()}
        for key, filename in (("observations", "observations.csv"),
                              ("spline", "spline_coordinates.csv"),
                              ("provenance", "figure_provenance.json")):
            path = output / filename
            path.write_text("test export placeholder", encoding="utf-8")
            attack_return[key] = str(path)
        for spec in (SPECS[0], SPECS[2]):
            namespace = {
                "Path": Path, "json": json, "OUTPUT_ROOT": output,
                "api": types.SimpleNamespace(export_privacy_utility=lambda root: full_return),
                "attack": types.SimpleNamespace(export_figures=lambda root: attack_return),
            }
            final_cell = next(value for value in make_notebook(*spec)["cells"] if value["id"] == "export_figures")
            before = len(outputs)
            with patch.dict("sys.modules", {"IPython": ipython_module, "IPython.display": display_module}):
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile("".join(final_cell["source"]), spec[0], "exec"), namespace)
            expected_outputs = 3 if spec[2] == "full" else 6
            if len(outputs) - before != expected_outputs:
                raise AssertionError(f"{spec[0]}: exported artifacts were not displayed")
    print("Export-cell smoke passed for full verification/path-list and attack path-mapping results.")


def validate_api_signatures() -> None:
    """Check notebook calls against the checked-in APIs without importing torch."""
    modules = {}
    for alias, module_name in (("api", "notebook_api"), ("attack", "attack_comparison")):
        module_path = ROOT / "reproduction" / "casia" / f"{module_name}.py"
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        modules[alias] = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    checked = set()
    for spec in SPECS:
        notebook = make_notebook(*spec)
        for value in notebook["cells"]:
            if value["cell_type"] != "code":
                continue
            tree = ast.parse("".join(value["source"]))
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                    continue
                owner = call.func.value
                if not isinstance(owner, ast.Name) or owner.id not in modules:
                    continue
                functions = modules[owner.id]
                if call.func.attr not in functions:
                    raise ValueError(f"{spec[0]}: missing {owner.id}.{call.func.attr} API")
                signature = functions[call.func.attr].args
                positional = signature.posonlyargs + signature.args
                names = {arg.arg for arg in positional + signature.kwonlyargs}
                if signature.vararg is None and len(call.args) > len(positional):
                    raise ValueError(f"{spec[0]}: too many positional arguments for {call.func.attr}")
                for keyword in call.keywords:
                    if keyword.arg is not None and keyword.arg not in names and signature.kwarg is None:
                        raise ValueError(f"{spec[0]}: unsupported {call.func.attr} keyword {keyword.arg}")
                supplied = {arg.arg for arg in positional[:len(call.args)]}
                supplied.update(keyword.arg for keyword in call.keywords if keyword.arg)
                required = {arg.arg for arg in positional[:len(positional)-len(signature.defaults)]}
                required.update(arg.arg for arg, default in zip(signature.kwonlyargs, signature.kw_defaults) if default is None)
                if not required.issubset(supplied):
                    raise ValueError(f"{spec[0]}: missing {call.func.attr} arguments {sorted(required-supplied)}")
                checked.add(f"{owner.id}.{call.func.attr}")
    print("Checked reproduction API signatures:", ", ".join(sorted(checked)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate checked-in notebooks without modifying them")
    parser.add_argument("--self-test", action="store_true", help="Check that validation rejects broken staged notebooks")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if not args.check:
        OUT.mkdir(parents=True, exist_ok=True)
    for filename, dataset, mode in SPECS:
        notebook = make_notebook(filename, dataset, mode)
        validate_notebook(notebook, filename)
        expected = canonical_bytes(notebook)
        destination = OUT / filename
        if args.check:
            observed = destination.read_bytes()
            validate_notebook(json.loads(observed), filename)
            if observed != expected:
                raise SystemExit(f"{destination}: regenerate using this script before committing")
        else:
            destination.write_bytes(expected)
        print(f"{'Verified' if args.check else 'Built'} {filename}: {len(notebook['cells'])} cells, sha256 {hashlib.sha256(expected).hexdigest()}")
    actual = set(OUT.glob("*.ipynb"))
    expected_paths = {OUT / name for name, _, _ in SPECS}
    if actual != expected_paths:
        raise SystemExit("notebooks/current must contain exactly the four current CASIA notebooks")
    if args.check:
        validate_api_signatures()


if __name__ == "__main__":
    main()
