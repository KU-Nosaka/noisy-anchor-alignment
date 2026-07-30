"""Re-render the MNIST reconstruction grids from the persisted replay tiles.

This lightweight renderer deliberately does not import the experiment runtime.
It applies the current main-manuscript Figure 2 layout to the mixed grid and
to all ten fixed-digit supplementary grids.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = Path(
    os.environ.get(
        "REPRO_MNIST_RECON_INPUT",
        str(ROOT / "build" / "inputs" / "mnist_reconstructions"),
    )
).expanduser().resolve()
OUTPUT_ROOT = Path(
    os.environ.get(
        "REPRO_MNIST_RECON_OUT",
        str(ROOT / "build" / "renderers" / "mnist_reconstructions"),
    )
).expanduser().resolve()
CELL_ROOT = INPUT_ROOT / "cells"
SCHEDULE_PATH = INPUT_ROOT / "selected_exact_source_schedules.json"

DIGITS = tuple(range(10))
N_COLUMNS = 10
BASELINE = 0.367
METRIC = "linkage_embedding_top1"
ROW_ORDER = (
    "original",
    "c_gdp_known_secret",
    "aa_i_gdp_known_anchor",
    "c_gdp_private_data_noise",
    "aa_i_gdp_private_data_noise",
    "aa_i_gdp_anchor_noise_mp",
    "aa_i_gdp_anchor_noise_op",
    "aa_i_gdp_anchor_noise_alignment_map",
)

ROW_STYLE: dict[str, dict[str, str]] = {
    "original": {
        "title": "Original",
        "math1": r"$\boldsymbol{X}_j$",
        "math2": "",
        "face": "#edf1f5",
    },
    "c_gdp_known_secret": {
        "title": "C-GDP | known-secret inversion",
        "math1": (
            r"$\widehat{\widetilde{\boldsymbol{X}}}_j="
            r"(\boldsymbol{Y}_j-\mathbf{1}_{n_j}\boldsymbol{\Psi}_s^{\mathsf{T}})"
            r"\boldsymbol{O}_s^{\mathsf{T}}=\widetilde{\boldsymbol{X}}_j,$"
        ),
        "math2": (
            r"$\widehat{\boldsymbol{X}}_j="
            r"\widehat{\widetilde{\boldsymbol{X}}}_j\boldsymbol{F}^\dagger="
            r"\boldsymbol{X}_j\boldsymbol{F}\boldsymbol{F}^\dagger$"
        ),
        "face": "#e8f2f8",
    },
    "aa_i_gdp_known_anchor": {
        "title": "AA-I-GDP | known-anchor inversion",
        "math1": (
            r"$\widehat{\widetilde{\boldsymbol{X}}}_j="
            r"(\boldsymbol{Y}_j-\mathbf{1}_{n_j}\boldsymbol{b}_j^{\mathsf{T}})"
            r"(\boldsymbol{A}^{\mathsf{T}}\overline{\boldsymbol{B}}_j)^{\mathsf{T}}="
            r"\widetilde{\boldsymbol{X}}_j,$"
        ),
        "math2": (
            r"$\widehat{\boldsymbol{X}}_j="
            r"\widehat{\widetilde{\boldsymbol{X}}}_j\boldsymbol{F}^\dagger="
            r"\boldsymbol{X}_j\boldsymbol{F}\boldsymbol{F}^\dagger$"
        ),
        "face": "#e8f2f8",
    },
    "c_gdp_private_data_noise": {
        "title": "C-GDP (private-data noise) | known-secret inversion",
        "math1": (
            r"$\widehat{\widetilde{\boldsymbol{X}}}_j="
            r"(\boldsymbol{Y}_j-\mathbf{1}_{n_j}\boldsymbol{\Psi}_s^{\mathsf{T}})"
            r"\boldsymbol{O}_s^{\mathsf{T}}="
            r"\widetilde{\boldsymbol{X}}_j+\sigma\boldsymbol{W}_j\boldsymbol{O}_s^{\mathsf{T}},$"
        ),
        "math2": (
            r"$\widehat{\boldsymbol{X}}_j="
            r"\boldsymbol{X}_j\boldsymbol{F}\boldsymbol{F}^\dagger+"
            r"\sigma\boldsymbol{W}_j\boldsymbol{O}_s^{\mathsf{T}}\boldsymbol{F}^\dagger$"
        ),
        "face": "#e8f2f8",
    },
    "aa_i_gdp_private_data_noise": {
        "title": "AA-I-GDP (private-data noise) | known-anchor inversion",
        "math1": (
            r"$\widehat{\widetilde{\boldsymbol{X}}}_j="
            r"(\boldsymbol{Y}_j-\mathbf{1}_{n_j}\boldsymbol{b}_j^{\mathsf{T}})"
            r"(\boldsymbol{A}^{\mathsf{T}}\overline{\boldsymbol{B}}_j)^{\mathsf{T}}="
            r"\widetilde{\boldsymbol{X}}_j+\sigma\boldsymbol{W}_j\boldsymbol{O}_j^{\mathsf{T}},$"
        ),
        "math2": (
            r"$\widehat{\boldsymbol{X}}_j="
            r"\boldsymbol{X}_j\boldsymbol{F}\boldsymbol{F}^\dagger+"
            r"\sigma\boldsymbol{W}_j\boldsymbol{O}_j^{\mathsf{T}}\boldsymbol{F}^\dagger$"
        ),
        "face": "#e8f2f8",
    },
    "aa_i_gdp_anchor_noise_mp": {
        "title": "AA-I-GDP (anchor noise) | MP attack",
        "math1": (
            r"$\widehat{\widetilde{\boldsymbol{X}}}_j^{\mathrm{MP}}="
            r"(\boldsymbol{Y}_j-\mathbf{1}_{n_j}\boldsymbol{b}_j^{\mathsf{T}})"
            r"(\boldsymbol{A}^{\mathsf{T}}\overline{\boldsymbol{B}}_j)^\dagger,$"
        ),
        "math2": (
            r"$\widehat{\boldsymbol{X}}_j^{\mathrm{MP}}="
            r"\widehat{\widetilde{\boldsymbol{X}}}_j^{\mathrm{MP}}\boldsymbol{F}^\dagger"
            r"\ \longrightarrow_{v\to0}\ \boldsymbol{X}_j\boldsymbol{F}\boldsymbol{F}^\dagger$"
        ),
        "face": "#fbf4e7",
    },
    "aa_i_gdp_anchor_noise_op": {
        "title": "AA-I-GDP (anchor noise) | OP attack",
        "math1": (
            r"$\widehat{\widetilde{\boldsymbol{X}}}_j^{\mathrm{OP}}="
            r"(\boldsymbol{Y}_j-\mathbf{1}_{n_j}\boldsymbol{b}_j^{\mathsf{T}})"
            r"[\Pi(\boldsymbol{A}^{\mathsf{T}}\overline{\boldsymbol{B}}_j)]^{\mathsf{T}},$"
        ),
        "math2": (
            r"$\widehat{\boldsymbol{X}}_j^{\mathrm{OP}}="
            r"\widehat{\widetilde{\boldsymbol{X}}}_j^{\mathrm{OP}}\boldsymbol{F}^\dagger"
            r"\ \longrightarrow_{v\to0}\ \boldsymbol{X}_j\boldsymbol{F}\boldsymbol{F}^\dagger$"
        ),
        "face": "#fbf4e7",
    },
    "aa_i_gdp_anchor_noise_alignment_map": {
        "title": "AA-I-GDP (anchor noise) | AM attack",
        "math1": (
            r"$\widehat{\widetilde{\boldsymbol{X}}}_j^{\mathrm{AM}}="
            r"(\boldsymbol{Y}_j-\mathbf{1}_{n_j}\boldsymbol{b}_j^{\mathsf{T}})"
            r"[\Pi(\boldsymbol{O}_c\widehat{\boldsymbol{R}}_c)"
            r"\widehat{\boldsymbol{R}}_j^{\mathsf{T}}]^{\mathsf{T}},$"
        ),
        "math2": (
            r"$\widehat{\boldsymbol{X}}_j^{\mathrm{AM}}="
            r"\widehat{\widetilde{\boldsymbol{X}}}_j^{\mathrm{AM}}\boldsymbol{F}^\dagger"
            r"\ \longrightarrow_{v\to0}\ \boldsymbol{X}_j\boldsymbol{F}\boldsymbol{F}^\dagger$"
        ),
        "face": "#fbf4e7",
    },
}


def cell_path(digit: int, row: str, column: int) -> Path:
    return CELL_ROOT / f"digit_{digit}" / f"c{column + 1:02d}_{row}.png"


def row_scores(
    schedules: dict[str, list[dict[str, object]]], row: str
) -> list[float | None]:
    if row == "original":
        return [None] * N_COLUMNS
    if row in ("c_gdp_known_secret", "aa_i_gdp_known_anchor"):
        return [BASELINE] * N_COLUMNS
    if row in ("c_gdp_private_data_noise", "aa_i_gdp_private_data_noise"):
        return [float(item[METRIC]) for item in schedules["private"]]
    key = {
        "aa_i_gdp_anchor_noise_mp": "PA-MP",
        "aa_i_gdp_anchor_noise_op": "PA-OP",
        "aa_i_gdp_anchor_noise_alignment_map": "PA-AM",
    }[row]
    return [float(item[METRIC]) for item in schedules[key]]


def render_grid(
    *,
    image_paths: dict[str, list[Path]],
    schedules: dict[str, list[dict[str, object]]],
    output_stem: Path,
    column_labels: list[str] | None,
) -> None:
    fig = plt.figure(figsize=(19.2, 14.7), facecolor="white")
    grid = fig.add_gridspec(
        len(ROW_ORDER),
        N_COLUMNS + 1,
        width_ratios=[5.55] + [1.0] * N_COLUMNS,
        hspace=0.19,
        wspace=0.035,
        left=0.006,
        right=0.997,
        bottom=0.025,
        top=0.975 if column_labels else 0.995,
    )
    for row_index, row in enumerate(ROW_ORDER):
        style = ROW_STYLE[row]
        label_ax = fig.add_subplot(grid[row_index, 0])
        label_ax.set_facecolor(style["face"])
        label_ax.set_xticks([])
        label_ax.set_yticks([])
        for spine in label_ax.spines.values():
            spine.set_visible(False)
        title_parts = [part.strip() for part in style["title"].split("|")]
        label_ax.text(
            0.025,
            0.94,
            title_parts[0],
            ha="left",
            va="top",
            fontsize=23.0,
            fontweight="bold",
            color="#22374d",
            transform=label_ax.transAxes,
        )
        if len(title_parts) > 1:
            label_ax.text(
                0.025,
                0.75,
                title_parts[1],
                ha="left",
                va="top",
                fontsize=22.0,
                fontweight="bold",
                color="#22374d",
                transform=label_ax.transAxes,
            )
        if row == "original":
            label_ax.text(
                0.025,
                0.38,
                style["math1"],
                ha="left",
                va="center",
                fontsize=25.0,
                color="#22374d",
                transform=label_ax.transAxes,
            )
        else:
            for y, field in ((0.39, "math1"), (0.12, "math2")):
                label_ax.text(
                    0.025,
                    y,
                    style[field],
                    ha="left",
                    va="center",
                    fontsize=21.5,
                    color="#314b65",
                    transform=label_ax.transAxes,
                )
        scores = row_scores(schedules, row)
        for column in range(N_COLUMNS):
            ax = fig.add_subplot(grid[row_index, column + 1])
            with Image.open(image_paths[row][column]) as image:
                ax.imshow(
                    np.asarray(image),
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                    interpolation="nearest",
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index == 0 and column_labels:
                ax.set_title(
                    column_labels[column],
                    fontsize=21.5,
                    fontweight="bold",
                    color="#53677c",
                    pad=5,
                )
            score = scores[column]
            if score is not None:
                ax.text(
                    0.5,
                    -0.075,
                    f"{100.0 * score:.1f}%",
                    ha="center",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=21.5,
                    color="#43566b",
                    clip_on=False,
                )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=300, facecolor="white")
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    schedules = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    mixed_paths = {
        row: [cell_path(digit, row, digit) for digit in DIGITS]
        for row in ROW_ORDER
    }
    render_grid(
        image_paths=mixed_paths,
        schedules=schedules,
        output_stem=OUTPUT_ROOT / "mnist_reconstructions_mixed_exact_source",
        column_labels=[str(digit) for digit in DIGITS],
    )
    for digit in DIGITS:
        paths = {
            row: [cell_path(digit, row, column) for column in range(N_COLUMNS)]
            for row in ROW_ORDER
        }
        render_grid(
            image_paths=paths,
            schedules=schedules,
            output_stem=OUTPUT_ROOT
            / f"mnist_reconstructions_digit_{digit}_exact_source",
            column_labels=None,
        )
    print("Re-rendered the mixed grid and ten supplementary digit grids.")


if __name__ == "__main__":
    main()
