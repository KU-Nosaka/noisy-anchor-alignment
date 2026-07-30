"""Regenerate every main-manuscript and supplementary PDF.

The scientific renderers retain their original validation logic. This driver
only prepares the compact qualitative inputs, executes the renderers, and
collects the manuscript-facing PDFs into one predictable directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "build"
INPUT_BUILD = BUILD / "inputs"
RENDER_BUILD = BUILD / "renderers"
PAPER_BUILD = BUILD / "paper_figures"


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member in {archive}: {member.filename}")
        handle.extractall(destination)


def run_renderer(filename: str, env_updates: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(env_updates)
    command = [sys.executable, str(REPO / "figures" / filename)]
    print(f"[render] {filename}", flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Renderer did not create {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Do not run the repository integrity checks before rendering.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect already rendered PDFs without rerunning the renderers.",
    )
    args = parser.parse_args()

    if not args.skip_validation and not args.collect_only:
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "validate_repository.py")],
            cwd=REPO,
            check=True,
        )

    mnist_quant = RENDER_BUILD / "mnist_quantitative"
    mnist_utility = RENDER_BUILD / "mnist_utility"
    mnist_recon = RENDER_BUILD / "mnist_reconstructions"
    celeba_quant = RENDER_BUILD / "celeba_p010_quantitative"
    celeba_recon = RENDER_BUILD / "celeba_reconstructions"
    celeba_count = RENDER_BUILD / "celeba_participant_count"

    if not args.collect_only:
        mnist_input = INPUT_BUILD / "mnist_reconstructions"
        celeba_input = INPUT_BUILD / "celeba_reconstructions"
        safe_extract(
            REPO / "figure_inputs" / "mnist_reconstruction_cells.zip",
            mnist_input,
        )
        safe_extract(
            REPO / "figure_inputs" / "celeba_mixed_reconstructions.zip",
            celeba_input,
        )
        safe_extract(
            REPO / "figure_inputs" / "celeba_identity_reconstructions.zip",
            celeba_input,
        )
        run_renderer(
            "render_mnist_quantitative.py",
            {"REPRO_MNIST_QUANT_OUT": str(mnist_quant)},
        )
        run_renderer(
            "render_mnist_utility.py",
            {"REPRO_MNIST_UTILITY_OUT": str(mnist_utility)},
        )
        run_renderer(
            "render_mnist_reconstructions.py",
            {
                "REPRO_MNIST_RECON_INPUT": str(mnist_input),
                "REPRO_MNIST_RECON_OUT": str(mnist_recon),
            },
        )
        run_renderer(
            "render_celeba_p010_quantitative.py",
            {"REPRO_CELEBA_P010_OUT": str(celeba_quant)},
        )
        run_renderer(
            "render_celeba_reconstructions.py",
            {
                "REPRO_CELEBA_RECON_INPUT": str(celeba_input),
                "REPRO_CELEBA_RECON_OUT": str(celeba_recon),
            },
        )
        run_renderer(
            "render_celeba_participant_count.py",
            {"REPRO_CELEBA_COUNT_OUT": str(celeba_count)},
        )

    main_files = {
        "mnist_exact_source_linkage_four_panel.pdf":
            mnist_quant / "mnist_exact_source_linkage_four_panel.pdf",
        "mnist_reconstructions_mixed_exact_source.pdf":
            mnist_recon / "mnist_reconstructions_mixed_exact_source.pdf",
        "mnist_utility_balanced_accuracy_two_panel.pdf":
            mnist_utility / "mnist_utility_balanced_accuracy_two_panel.pdf",
        "mnist_balanced_accuracy_versus_exact_source_linkage_op.pdf":
            mnist_quant / "mnist_balanced_accuracy_versus_exact_source_linkage_op.pdf",
        "celeba_cross_identity_leakage_private_and_anchor_methods_two_panel.pdf":
            celeba_quant / "figures" / "leakage_noise"
            / "celeba_cross_identity_leakage_private_and_anchor_methods_two_panel.pdf",
        "celeba_smiling_balanced_accuracy_two_panel.pdf":
            celeba_quant / "figures" / "utility_noise"
            / "celeba_smiling_balanced_accuracy_two_panel.pdf",
        "celeba_smiling_balanced_accuracy_vs_cross_identity_linkage_op.pdf":
            celeba_quant / "figures" / "privacy_utility"
            / "celeba_smiling_balanced_accuracy_vs_cross_identity_leakage_op.pdf",
        "celeba_gpm_convergence_diagnostics.pdf":
            celeba_quant / "figures" / "gpm"
            / "celeba_gpm_convergence_diagnostics.pdf",
        "celeba_p010_reconstruction_leakage_ordered.pdf":
            celeba_recon / "figures" / "reconstruction"
            / "celeba_p010_reconstruction_leakage_ordered.pdf",
        "celeba_privacy_utility_by_participant_count.pdf":
            celeba_count / "figures"
            / "celeba_privacy_utility_by_participant_count.pdf",
    }
    supplement_files = {
        **{
            f"mnist_reconstructions_digit_{digit}_exact_source.pdf":
                mnist_recon / f"mnist_reconstructions_digit_{digit}_exact_source.pdf"
            for digit in range(10)
        },
        **{
            f"celeba_reconstructions_identity_{slot:02d}.pdf":
                celeba_recon / "figures" / "reconstruction" / "identity_specific"
                / f"celeba_reconstructions_identity_{slot:02d}.pdf"
            for slot in range(1, 11)
        },
        "celeba_participant_gpm_stopping.pdf":
            celeba_count / "figures" / "celeba_participant_gpm_stopping.pdf",
    }

    for name, source in main_files.items():
        copy_required(source, PAPER_BUILD / "main" / name)
    for name, source in supplement_files.items():
        copy_required(source, PAPER_BUILD / "supplement" / name)

    generated = [
        *(PAPER_BUILD / "main" / name for name in main_files),
        *(PAPER_BUILD / "supplement" / name for name in supplement_files),
    ]
    manifest = {
        "main_figure_count": len(main_files),
        "supplement_figure_count": len(supplement_files),
        "files": {
            path.relative_to(PAPER_BUILD).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in generated
        },
    }
    (PAPER_BUILD / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Created {len(main_files)} main and {len(supplement_files)} "
        f"supplementary PDFs under {PAPER_BUILD}"
    )


if __name__ == "__main__":
    main()
