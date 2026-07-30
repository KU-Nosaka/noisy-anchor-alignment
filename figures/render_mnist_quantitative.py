"""Render the revised MNIST exact-source-linkage figures.

The authoritative 609-record aggregate contains two hundred fitted noise
observations per noisy protocol.  Only ``linkage_embedding_top1`` is used as a
privacy metric.  Deterministic endpoint controls are retained in the source
aggregate but excluded from every plot, spline, bootstrap, and summary value.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO = SCRIPT_PATH.parents[1]
HERE = Path(
    os.environ.get(
        "REPRO_MNIST_QUANT_OUT",
        str(REPO / "build" / "renderers" / "mnist_quantitative"),
    )
).expanduser().resolve()
SOURCE = REPO / "results" / "mnist" / "mnist_combined_609_results.json.gz"
SOURCE_CSV = HERE / "mnist_exact_source_source_records.csv"
PLOT_CSV = HERE / "mnist_exact_source_plot_data.csv"
METADATA_JSON = HERE / "mnist_exact_source_figure_metadata.json"
SUMMARY_JSON = HERE / "mnist_exact_source_numerical_summary.json"
SUMMARY_MD = HERE / "mnist_exact_source_numerical_summary.md"

PRIVATE_ROLES = {"random", "supplemental_private_sigma_35_50_v1"}
ANCHOR_ROLES = {"random", "supplemental_anchor_v_0_0p2_v1"}
PRIVATE_UPPER = 50.0
ANCHOR_UPPER = 0.5
CHANCE = 0.10
BOOTSTRAP_RESAMPLES = 10_000
GRID_SIZE = 201

COLORS = {
    "c_private": "#0072B2",
    "aa_private": "#D55E00",
    "mp": "#0072B2",
    "op": "#E69F00",
    "am": "#009E73",
    "anchor_utility": "#CC79A7",
    "i_gdp": "#009E73",
    "clean": "#333333",
    "chance": "#666666",
    "band": "#777777",
}

LABELS = {
    "c_private": "C-GDP (private-data noise)",
    "aa_private": "AA-I-GDP (private-data noise)",
    "mp": "AA-I-GDP (anchor noise): MP attack",
    "op": "AA-I-GDP (anchor noise): OP attack",
    "am": "AA-I-GDP (anchor noise): alignment-map attack",
}

ATTACKS = {
    "c_private": "known-secret",
    "aa_private": "AA-known-anchor",
    "mp": "PA-MP",
    "op": "PA-OP",
    "am": "PA-AM",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records() -> list[dict[str, Any]]:
    with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = list(payload["records"])
    if len(records) != 609:
        raise ValueError(f"Expected 609 authoritative records, found {len(records)}")
    expected = {
        ("c_gdp", "random"): 1,
        ("aa_i_gdp", "random"): 1,
        ("i_gdp", "random"): 1,
        ("c_gdp_an", "random"): 100,
        ("aa_i_gdp_an", "random"): 100,
        ("pa_i_gdp", "random"): 100,
        ("c_gdp_an", "supplemental_private_sigma_35_50_v1"): 100,
        ("aa_i_gdp_an", "supplemental_private_sigma_35_50_v1"): 100,
        ("pa_i_gdp", "supplemental_anchor_v_0_0p2_v1"): 100,
        ("c_gdp_an", "endpoint_zero"): 1,
        ("aa_i_gdp_an", "endpoint_zero"): 1,
        ("pa_i_gdp", "endpoint_zero"): 1,
        ("c_gdp_an", "endpoint_upper"): 1,
        ("aa_i_gdp_an", "endpoint_upper"): 1,
        ("pa_i_gdp", "endpoint_upper"): 1,
    }
    actual: dict[tuple[str, str], int] = {}
    for row in records:
        key = (str(row["protocol"]), str(row["run_role"]))
        actual[key] = actual.get(key, 0) + 1
    if actual != expected:
        raise ValueError(f"Unexpected schedule composition: {actual}")
    return records


def record_one(records: Iterable[dict[str, Any]], protocol: str) -> dict[str, Any]:
    matches = [row for row in records if row["protocol"] == protocol and row["run_role"] == "random"]
    if len(matches) != 1:
        raise ValueError(f"Expected one {protocol} baseline, found {len(matches)}")
    return matches[0]


def attack(record: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in record.get("attacks", []) if item["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected attack {name!r} once in {record['run_id']}")
    return matches[0]


def source_linkage(record: dict[str, Any], attack_name: str) -> float:
    """Return ten-way exact-source linkage accuracy from the frozen embedding."""
    return float(attack(record, attack_name)["metrics"]["linkage_embedding_top1"])


def balanced_accuracy(record: dict[str, Any]) -> float:
    return float(record["utility"]["test_balanced_accuracy"])


def select_fit_records(records: list[dict[str, Any]], protocol: str) -> list[dict[str, Any]]:
    roles = ANCHOR_ROLES if protocol == "pa_i_gdp" else PRIVATE_ROLES
    selected = [row for row in records if row["protocol"] == protocol and row["run_role"] in roles]
    selected.sort(key=lambda row: (str(row["run_role"]), int(row["draw"])))
    if len(selected) != 200:
        raise ValueError(f"Expected 200 fit observations for {protocol}, found {len(selected)}")
    return selected


def validate_private_equivalence(
    c_rows: list[dict[str, Any]], aa_rows: list[dict[str, Any]]
) -> dict[str, float | int]:
    if len(c_rows) != len(aa_rows):
        raise AssertionError("Private C-GDP/AA-I-GDP counts differ")
    for c_row, aa_row in zip(c_rows, aa_rows):
        c_key = (c_row["run_role"], int(c_row["draw"]))
        aa_key = (aa_row["run_role"], int(aa_row["draw"]))
        linkage_fields_equal = (
            c_key == aa_key
            and float(c_row["noise_sigma"]) == float(aa_row["noise_sigma"])
            and source_linkage(c_row, ATTACKS["c_private"])
            == source_linkage(aa_row, ATTACKS["aa_private"])
        )
        if not linkage_fields_equal:
            raise AssertionError(f"Private sigma/exact-source-linkage equivalence failed at {c_key}")
    utility_differences = np.asarray(
        [abs(balanced_accuracy(c) - balanced_accuracy(a)) for c, a in zip(c_rows, aa_rows)],
        dtype=np.float64,
    )
    return {
        "exact_sigma_linkage_pairs": len(c_rows),
        "exact_utility_pairs": int(np.count_nonzero(utility_differences == 0.0)),
        "max_abs_utility_difference": float(np.max(utility_differences)),
    }


def validate_linkage_manifest(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Verify that every stored linkage score uses one common 10-way manifest."""
    hashes: set[str] = set()
    candidate_counts: set[int] = set()
    query_counts: set[int] = set()
    chance_levels: set[float] = set()
    attack_count = 0
    for record in records:
        for item in record.get("attacks", []):
            metrics = item.get("metrics", {})
            if "linkage_embedding_top1" not in metrics:
                continue
            hashes.add(str(metrics["linkage_manifest_hash"]))
            candidate_counts.add(int(metrics["linkage_n_candidates"]))
            query_counts.add(int(metrics["linkage_n_queries"]))
            chance_levels.add(float(metrics["linkage_chance_top1"]))
            attack_count += 1
    if len(hashes) != 1:
        raise AssertionError(f"Expected one linkage manifest, found {len(hashes)}")
    if candidate_counts != {10} or query_counts != {1000} or chance_levels != {CHANCE}:
        raise AssertionError(
            f"Unexpected linkage protocol: candidates={candidate_counts}, "
            f"queries={query_counts}, chance={chance_levels}"
        )
    return {
        "manifest_hash": next(iter(hashes)),
        "candidate_count": 10,
        "query_count": 1000,
        "chance_top1": CHANCE,
        "validated_attack_records": attack_count,
    }


def spline_design(normalized_x: np.ndarray) -> np.ndarray:
    x = np.asarray(normalized_x, dtype=np.float64).reshape(-1)
    knots = np.linspace(0.0, 1.0, 7, dtype=np.float64)[1:-1]
    columns = [np.ones_like(x), x, x * x, x**3]
    columns.extend(np.maximum(x - knot, 0.0) ** 3 for knot in knots)
    return np.column_stack(columns)


def conditional_spline(x: np.ndarray, y: np.ndarray, upper: float, seed: int) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = spline_design(np.clip(x / upper, 0.0, 1.0))
    grid = np.linspace(0.0, upper, GRID_SIZE, dtype=np.float64)
    grid_design = spline_design(grid / upper)
    gram = design.T @ design
    ridge = 1.0e-8 * max(float(np.trace(gram) / design.shape[1]), 1.0)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(gram + penalty, design.T @ y)
    estimate = grid_design @ beta
    rng = np.random.default_rng(seed)
    probability = np.full(len(x), 1.0 / len(x), dtype=np.float64)
    predictions = np.empty((BOOTSTRAP_RESAMPLES, GRID_SIZE), dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 256):
        stop = min(start + 256, BOOTSTRAP_RESAMPLES)
        counts = rng.multinomial(len(x), probability, size=stop - start).astype(np.float64, copy=False)
        boot_gram = np.einsum("bn,np,nq->bpq", counts, design, design, optimize=True)
        boot_rhs = np.einsum("bn,np,n->bp", counts, design, y, optimize=True)
        boot_beta = np.linalg.solve(boot_gram + penalty[None, :, :], boot_rhs[..., None])[..., 0]
        predictions[start:stop] = boot_beta @ grid_design.T
    lower, upper_band = np.quantile(predictions, [0.025, 0.975], axis=0)
    return {
        "x": grid,
        "estimate": np.clip(estimate, 0.0, 1.0),
        "lower": np.clip(lower, 0.0, 1.0),
        "upper": np.clip(upper_band, 0.0, 1.0),
        "ridge": ridge,
        "seed": seed,
    }


def direct_spline(leakage: np.ndarray, utility: np.ndarray, seed: int) -> dict[str, Any]:
    x = 100.0 * np.asarray(leakage, dtype=np.float64)
    y = 100.0 * np.asarray(utility, dtype=np.float64)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if not x_max > x_min:
        raise ValueError("Exact-source-linkage support has zero width")
    design = spline_design((x - x_min) / (x_max - x_min))
    grid = np.linspace(x_min, x_max, GRID_SIZE, dtype=np.float64)
    grid_design = spline_design((grid - x_min) / (x_max - x_min))
    gram = design.T @ design
    ridge = 1.0e-8 * max(float(np.trace(gram) / design.shape[1]), 1.0)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    estimate = grid_design @ np.linalg.solve(gram + penalty, design.T @ y)
    rng = np.random.default_rng(seed)
    probability = np.full(len(x), 1.0 / len(x), dtype=np.float64)
    predictions = np.empty((BOOTSTRAP_RESAMPLES, GRID_SIZE), dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 256):
        stop = min(start + 256, BOOTSTRAP_RESAMPLES)
        counts = rng.multinomial(len(x), probability, size=stop - start).astype(np.float64, copy=False)
        boot_gram = np.einsum("bn,np,nq->bpq", counts, design, design, optimize=True)
        boot_rhs = np.einsum("bn,np,n->bp", counts, design, y, optimize=True)
        boot_beta = np.linalg.solve(boot_gram + penalty[None, :, :], boot_rhs[..., None])[..., 0]
        predictions[start:stop] = boot_beta @ grid_design.T
    lower, upper_band = np.quantile(predictions, [0.025, 0.975], axis=0)
    return {
        "x": grid,
        "estimate": np.clip(estimate, 0.0, 100.0),
        "lower": np.clip(lower, 0.0, 100.0),
        "upper": np.clip(upper_band, 0.0, 100.0),
        "support": [x_min, x_max],
        "ridge": ridge,
        "seed": seed,
    }


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10.5,
        "axes.titlesize": 12.0, "axes.labelsize": 11.0, "legend.fontsize": 9.6,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.linewidth": 0.8,
        "lines.linewidth": 2.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def configure_wide_matplotlib() -> None:
    """Typography for wide multi-panel PDFs placed at manuscript text width."""

    configure_matplotlib()
    mpl.rcParams.update({
        "font.size": 17.0,
        "axes.titlesize": 17.0,
        "axes.labelsize": 18.0,
        "legend.fontsize": 14.5,
        "xtick.labelsize": 15.5,
        "ytick.labelsize": 15.5,
    })


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.2, width=0.7)


def dedupe_legend(ax: plt.Axes, **kwargs: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(unique.values(), unique.keys(), **kwargs)


def save_figure(
    fig: plt.Figure,
    stem: str,
    *,
    tight: bool = True,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for suffix in ("png", "pdf", "svg"):
        path = HERE / f"{stem}.{suffix}"
        kwargs: dict[str, Any] = {}
        if tight:
            kwargs["bbox_inches"] = "tight"
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        paths[suffix] = path.name
    return paths


def x_for(records: list[dict[str, Any]], anchor: bool) -> np.ndarray:
    field = "anchor_noise_sigma" if anchor else "noise_sigma"
    return np.asarray([float(row[field]) for row in records], dtype=np.float64)


def y_linkage(records: list[dict[str, Any]], attack_name: str) -> np.ndarray:
    return np.asarray([source_linkage(row, attack_name) for row in records], dtype=np.float64)


def y_utility(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([balanced_accuracy(row) for row in records], dtype=np.float64)


def draw_linkage_panel(
    ax: plt.Axes,
    key: str,
    records: list[dict[str, Any]],
    fit: dict[str, Any],
    baselines: dict[str, float],
    title: str,
    *,
    aa_records: list[dict[str, Any]] | None = None,
) -> None:
    private = key == "c_private"
    upper = PRIVATE_UPPER if private else ANCHOR_UPPER
    style_axis(ax)
    ax.set_xlim(0.0, upper)
    ax.set_ylim(5.0, 40.0)
    ax.set_yticks(np.arange(5.0, 41.0, 5.0))
    ax.set_xticks(np.arange(0.0, 51.0, 10.0) if private else np.arange(0.0, 0.51, 0.1))
    ax.set_xlabel(r"Noise Scale $\sigma$" if private else r"Noise Scale $v$")
    ax.set_ylabel("Exact-Source Linkage Accuracy (%)")
    ax.set_title(title, loc="left", pad=9)
    x = x_for(records, not private)
    y = y_linkage(records, ATTACKS[key])
    band_color = COLORS["band"] if private else COLORS[key]
    ax.fill_between(fit["x"], 100 * fit["lower"], 100 * fit["upper"], color=band_color,
                    alpha=0.14, linewidth=0, label="95% conditional band" + (" (shared)" if private else ""), zorder=1)
    ax.scatter(x, 100 * y, s=18, marker="o", color=COLORS[key], alpha=0.23, linewidths=0, zorder=3)
    ax.plot(fit["x"], 100 * fit["estimate"], color=COLORS[key], linewidth=2.6,
            label=LABELS[key], zorder=5)
    if private:
        assert aa_records is not None
        aa_x = x_for(aa_records, False)
        aa_y = y_linkage(aa_records, ATTACKS["aa_private"])
        ax.scatter(aa_x, 100 * aa_y, s=25, marker="s", facecolors="none",
                   edgecolors=COLORS["aa_private"], alpha=0.40, linewidths=0.75, zorder=4)
        ax.plot(fit["x"], 100 * fit["estimate"], color=COLORS["aa_private"],
                linestyle=(0, (5.2, 2.2)), linewidth=2.1, label=LABELS["aa_private"], zorder=6)
        ax.axhline(100 * baselines["c_source"], color=COLORS["c_private"],
                   linestyle=(0, (1.0, 1.8)), linewidth=2.3, label="C-GDP (no noise)", zorder=2)
    ax.axhline(100 * CHANCE, color=COLORS["chance"], linestyle=(0, (1.2, 2.2)),
               linewidth=1.2, label="Random guess (10%)", zorder=1)
    ax.axhline(100 * baselines["aa_source"], color=COLORS["aa_private"],
               linestyle=(0, (4.2, 1.6, 1.0, 1.6)), linewidth=1.9,
               label="AA-I-GDP (no noise)", zorder=3)
    if private:
        ax.text(0.03, 0.045, r"C-GDP and AA-I-GDP coincide at every sampled $\sigma$",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=16.0, color="#333333")
    dedupe_legend(ax, loc="upper right", frameon=True, fancybox=False, framealpha=0.93,
                  edgecolor="#B0B0B0", borderpad=0.45, handlelength=2.8, labelspacing=0.35)


def render_figure1(
    c_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    fits: dict[str, dict[str, Any]],
    baselines: dict[str, float],
) -> dict[str, dict[str, str]]:
    """Render MNIST Figure 1 with the final two-panel CelebA layout."""

    configure_matplotlib()
    mpl.rcParams.update({
        "font.size": 17.0,
        "axes.titlesize": 17.0,
        "axes.labelsize": 18.0,
        "legend.fontsize": 14.5,
        "xtick.labelsize": 15.5,
        "ytick.labelsize": 15.5,
    })

    figure_colors = {
        "private": "#CC79A7",
        "mp": "#0072B2",
        "op": "#E69F00",
        "am": "#009E73",
    }
    figure_labels = {
        "private": "C-GDP / AA-I-GDP (private noise)",
        "mp": "AA-I-GDP (anchor noise): MP",
        "op": "AA-I-GDP (anchor noise): OP",
        "am": "AA-I-GDP (anchor noise): AM",
    }

    def prepare_axis(ax: plt.Axes, *, private: bool) -> None:
        style_axis(ax)
        ax.set_xlim(0.0, PRIVATE_UPPER if private else ANCHOR_UPPER)
        ax.set_ylim(5.0, 40.0)
        ax.set_yticks(np.arange(5.0, 41.0, 5.0))
        ax.set_xticks(
            np.arange(0.0, 51.0, 10.0)
            if private
            else np.arange(0.0, 0.51, 0.1)
        )
        ax.set_xlabel(r"Noise Scale $\sigma$" if private else r"Noise Scale $v$")
        ax.set_ylabel("")

    def draw_series(
        ax: plt.Axes,
        key: str,
        rows: list[dict[str, Any]],
        fit: dict[str, Any],
        *,
        show_guides: bool,
    ) -> None:
        private = key == "private"
        attack_key = "c_private" if private else key
        color = figure_colors[key]
        x = x_for(rows, not private)
        y = y_linkage(rows, ATTACKS[attack_key])
        ax.fill_between(
            fit["x"],
            100.0 * fit["lower"],
            100.0 * fit["upper"],
            color=color,
            alpha=0.15 if private else 0.09,
            linewidth=0,
            zorder=1,
        )
        if private:
            ax.scatter(
                x,
                100.0 * y,
                s=18,
                color=color,
                alpha=0.25,
                linewidths=0,
                zorder=3,
            )
        else:
            converged = np.asarray(
                [bool(row["gpm"]["converged"]) for row in rows]
            )
            ax.scatter(
                x[converged],
                100.0 * y[converged],
                s=13,
                color=color,
                alpha=0.20,
                linewidths=0,
                zorder=3,
            )
            ax.scatter(
                x[~converged],
                100.0 * y[~converged],
                s=15,
                facecolors="none",
                edgecolors=color,
                alpha=0.24,
                linewidths=0.7,
                zorder=3,
            )
        ax.plot(
            fit["x"],
            100.0 * fit["estimate"],
            color=color,
            linewidth=2.55,
            zorder=5,
        )
        if show_guides:
            ax.axhline(
                100.0 * CHANCE,
                color="#666666",
                linestyle=(0, (1.3, 2.2)),
                linewidth=1.15,
                zorder=2,
            )
            ax.axhline(
                100.0 * (
                    baselines["c_source"] if private else baselines["aa_source"]
                ),
                color="#333333",
                linestyle=(0, (5.0, 2.0, 1.0, 2.0)),
                linewidth=1.7,
                zorder=2,
            )

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.8), sharey=True)
    prepare_axis(axes[0], private=True)
    draw_series(
        axes[0],
        "private",
        c_rows,
        fits["source_private"],
        show_guides=True,
    )
    prepare_axis(axes[1], private=False)
    for index, key in enumerate(("mp", "op", "am")):
        draw_series(
            axes[1],
            key,
            anchor_rows,
            fits[f"source_{key}"],
            show_guides=index == 0,
        )

    fig.supylabel(
        "Exact-Source Linkage Accuracy (%)",
        x=0.012,
        y=0.575,
        fontsize=mpl.rcParams["axes.labelsize"],
    )
    legend_handles = [
        Line2D([], [], color=figure_colors["private"], linewidth=2.55),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=5.5,
            markerfacecolor="#777777",
            markeredgewidth=0,
        ),
        Line2D([], [], color=figure_colors["mp"], linewidth=2.55),
        Line2D([], [], color=figure_colors["op"], linewidth=2.55),
        Line2D([], [], color=figure_colors["am"], linewidth=2.55),
        Patch(facecolor="#777777", edgecolor="none", alpha=0.18),
        Line2D(
            [],
            [],
            color="#666666",
            linestyle=(0, (1.3, 2.2)),
            linewidth=1.15,
        ),
        Line2D(
            [],
            [],
            color="#333333",
            linestyle=(0, (5.0, 2.0, 1.0, 2.0)),
            linewidth=1.7,
        ),
    ]
    legend_labels = [
        figure_labels["private"],
        "200 random noise draws",
        figure_labels["mp"],
        figure_labels["op"],
        figure_labels["am"],
        "95% conditional band",
        "Random guess (10%)",
        "C-GDP / AA-I-GDP",
    ]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.0,
        labelspacing=0.35,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.31,
        top=0.84,
        wspace=0.14,
    )
    for panel_index, (ax, title) in enumerate(
        zip(
            axes,
            (
                "C-GDP / AA-I-GDP (private noise)",
                "AA-I-GDP (anchor noise)",
            ),
        )
    ):
        fig.text(
            ax.get_position().x0,
            0.94,
            f"({chr(ord('a') + panel_index)}) {title}",
            ha="left",
            va="top",
            fontsize=mpl.rcParams["axes.titlesize"],
        )

    outputs = {
        "mnist_exact_source_linkage_private_and_anchor_methods_two_panel":
            save_figure(
                fig,
                "mnist_exact_source_linkage_private_and_anchor_methods_two_panel",
                tight=False,
            ),
        # Preserve the manuscript's current include path while replacing its
        # former four-panel contents with the updated two-panel figure.
        "mnist_exact_source_linkage_four_panel":
            save_figure(
                fig,
                "mnist_exact_source_linkage_four_panel",
                tight=False,
            ),
    }
    plt.close(fig)
    return outputs


def draw_utility_panel(
    ax: plt.Axes,
    records: list[dict[str, Any]],
    fit: dict[str, Any],
    baselines: dict[str, float],
    title: str,
    *,
    aa_records: list[dict[str, Any]] | None = None,
    aa_fit: dict[str, Any] | None = None,
) -> None:
    private = aa_records is not None
    upper = PRIVATE_UPPER if private else ANCHOR_UPPER
    style_axis(ax)
    ax.set_xlim(0.0, upper)
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks(np.arange(0.0, 101.0, 10.0))
    ax.set_xticks(np.arange(0.0, 51.0, 10.0) if private else np.arange(0.0, 0.51, 0.1))
    ax.set_xlabel(r"Private-Data Noise Scale $\sigma$" if private else r"Anchor-Noise Scale $v$")
    ax.set_ylabel("MNIST Test Balanced Accuracy (%)")
    ax.set_title(title, loc="left", pad=9)
    x = x_for(records, not private)
    y = y_utility(records)
    color = COLORS["c_private"] if private else COLORS["anchor_utility"]
    ax.fill_between(fit["x"], 100 * fit["lower"], 100 * fit["upper"], color=color,
                    alpha=0.15, linewidth=0, label="95% conditional bands" if private else "95% conditional band", zorder=1)
    if private:
        assert aa_fit is not None
        ax.fill_between(aa_fit["x"], 100 * aa_fit["lower"], 100 * aa_fit["upper"],
                        color=COLORS["aa_private"], alpha=0.10, linewidth=0, zorder=1)
        ax.scatter(x, 100 * y, s=18, marker="o", color=COLORS["c_private"], alpha=0.25, linewidths=0, zorder=3)
        ax.scatter(x_for(aa_records, False), 100 * y_utility(aa_records), s=25, marker="s",
                   facecolors="none", edgecolors=COLORS["aa_private"], alpha=0.42, linewidths=0.75, zorder=4)
        ax.plot(fit["x"], 100 * fit["estimate"], color=COLORS["c_private"], linewidth=2.8,
                label="C-GDP (private-data noise)", zorder=5)
        ax.plot(aa_fit["x"], 100 * aa_fit["estimate"], color=COLORS["aa_private"], linestyle=(0, (5.2, 2.2)),
                linewidth=2.0, label="AA-I-GDP (private-data noise)", zorder=6)
    else:
        converged = np.asarray([bool(row["gpm"]["converged"]) for row in records])
        ax.scatter(x[converged], 100 * y[converged], s=20, color=color, alpha=0.42, linewidths=0,
                   label="Fit observation: GPM converged", zorder=3)
        ax.scatter(x[~converged], 100 * y[~converged], s=25, facecolors="none", edgecolors=color,
                   alpha=0.48, linewidths=0.85, label="Fit observation: 500-iteration cap", zorder=4)
        ax.plot(fit["x"], 100 * fit["estimate"], color=color, linewidth=2.6,
                label="AA-I-GDP (anchor noise)", zorder=5)
    ax.axhline(10.0, color="#777777", linestyle=(0, (1.2, 2.2)), linewidth=1.2,
               label="Random guess (10%)", zorder=1)
    ax.axhline(100 * baselines["c_utility"], color=COLORS["clean"], linestyle=(0, (1.0, 1.8)),
               linewidth=1.8, label="C-GDP / AA-I-GDP (no noise)", zorder=2)
    ax.axhline(100 * baselines["i_utility"], color=COLORS["i_gdp"], linestyle=(0, (5.0, 2.0, 1.0, 2.0)),
               linewidth=2.0, label="I-GDP", zorder=3)
    dedupe_legend(ax, loc="upper right", frameon=True, fancybox=False, framealpha=0.93,
                  edgecolor="#B0B0B0", borderpad=0.45, handlelength=2.8, labelspacing=0.35)


def draw_direct_tradeoff(
    ax: plt.Axes,
    c_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    private_fit: dict[str, Any],
    anchor_fit: dict[str, Any],
    baselines: dict[str, float],
) -> None:
    style_axis(ax)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ax.set_xlim(5.0, 40.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_xticks(range(5, 41, 5))
    ax.set_yticks(range(0, 101, 10))
    ax.set_xlabel("Exact-Source Linkage Accuracy (%)")
    ax.set_ylabel("MNIST Test Balanced Accuracy (%)")
    ax.axvline(10.0, color="#6A6A6A", linestyle=(0, (1.5, 2.2)), linewidth=1.15,
               label="Random guess (10% on either axis)", zorder=1)
    ax.axhline(10.0, color="#6A6A6A", linestyle=(0, (1.5, 2.2)), linewidth=1.15, zorder=1)
    ax.text(0.025, 0.965, "preferred: upper left", transform=ax.transAxes,
            ha="left", va="top", fontsize=8.8, color="#444444")
    for rows, fit, attack_name, color, marker, linestyle, label, anchor in (
        (c_rows, private_fit, ATTACKS["c_private"], COLORS["c_private"], "o", "-",
         "C-GDP (private-data noise)", False),
        (anchor_rows, anchor_fit, ATTACKS["op"], COLORS["op"], "^", (0, (5.0, 2.0)),
         "AA-I-GDP (anchor noise), OP attack", True),
    ):
        x = 100 * y_linkage(rows, attack_name)
        y = 100 * y_utility(rows)
        if anchor:
            converged = np.asarray([bool(row["gpm"]["converged"]) for row in rows])
            ax.scatter(x[converged], y[converged], s=19, marker=marker, color=color,
                       alpha=0.27, linewidths=0, zorder=3)
            ax.scatter(x[~converged], y[~converged], s=21, marker=marker, facecolors="none",
                       edgecolors=color, alpha=0.30, linewidths=0.70, zorder=3)
        else:
            ax.scatter(x, y, s=19, marker=marker, color=color, alpha=0.22, linewidths=0, zorder=3)
        ax.fill_between(fit["x"], fit["lower"], fit["upper"], color=color, alpha=0.14, linewidth=0, zorder=2)
        ax.plot(fit["x"], fit["estimate"], color=color, linestyle=linestyle,
                linewidth=2.55, label=label, zorder=5)
    ax.scatter([100 * baselines["c_source"]], [100 * baselines["c_utility"]], s=132,
               marker="*", color="#222222", edgecolors="white", linewidths=0.55,
               label="C-GDP / AA-I-GDP (no noise)", zorder=9)
    ax.axhline(100 * baselines["i_utility"], color="#D55E00", linestyle=(0, (5.0, 2.0, 1.0, 2.0)),
               linewidth=2.0, label="I-GDP balanced accuracy", zorder=4)
    dedupe_legend(ax, loc="upper left", bbox_to_anchor=(1.015, 1.0), frameon=False,
                  handlelength=2.9, labelspacing=0.50, borderaxespad=0.0)


def write_source_csv(
    c_rows: list[dict[str, Any]], aa_rows: list[dict[str, Any]], anchor_rows: list[dict[str, Any]],
    baselines_records: dict[str, dict[str, Any]],
) -> int:
    fields = ["record_kind", "source_run_id", "run_role", "draw", "protocol", "series_key",
              "method_label", "attack", "noise_family", "noise_scale", "exact_source_linkage",
              "test_balanced_accuracy", "gpm_converged", "linkage_manifest_hash",
              "linkage_n_queries", "linkage_n_candidates"]
    count = 0
    with SOURCE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, rows in (("c_private", c_rows), ("aa_private", aa_rows), ("mp", anchor_rows),
                          ("op", anchor_rows), ("am", anchor_rows)):
            anchor = key in {"mp", "op", "am"}
            for row in rows:
                metrics = attack(row, ATTACKS[key])["metrics"]
                writer.writerow({
                    "record_kind": "fit_observation", "source_run_id": row["run_id"],
                    "run_role": row["run_role"], "draw": row["draw"], "protocol": row["protocol"],
                    "series_key": key, "method_label": LABELS[key], "attack": ATTACKS[key],
                    "noise_family": "anchor" if anchor else "private_data",
                    "noise_scale": f"{float(row['anchor_noise_sigma'] if anchor else row['noise_sigma']):.17g}",
                    "exact_source_linkage": f"{source_linkage(row, ATTACKS[key]):.17g}",
                    "test_balanced_accuracy": f"{balanced_accuracy(row):.17g}",
                    "gpm_converged": str(bool(row["gpm"]["converged"])) if anchor else "",
                    "linkage_manifest_hash": metrics["linkage_manifest_hash"],
                    "linkage_n_queries": metrics["linkage_n_queries"],
                    "linkage_n_candidates": metrics["linkage_n_candidates"],
                })
                count += 1
        for key, protocol, attack_name, label in (
            ("clean_c", "c_gdp", "C-exact", "C-GDP (no noise)"),
            ("clean_aa", "aa_i_gdp", "AA-known-anchor", "AA-I-GDP (no noise)"),
            ("i_gdp", "i_gdp", "", "I-GDP utility reference"),
        ):
            row = baselines_records[protocol]
            metrics = attack(row, attack_name)["metrics"] if attack_name else {}
            writer.writerow({
                "record_kind": "horizontal_reference", "source_run_id": row["run_id"],
                "run_role": row["run_role"], "draw": row["draw"], "protocol": protocol,
                "series_key": key, "method_label": label, "attack": attack_name,
                "noise_family": "none", "noise_scale": "",
                "exact_source_linkage": "" if not attack_name else f"{source_linkage(row, attack_name):.17g}",
                "test_balanced_accuracy": f"{balanced_accuracy(row):.17g}", "gpm_converged": "",
                "linkage_manifest_hash": metrics.get("linkage_manifest_hash", ""),
                "linkage_n_queries": metrics.get("linkage_n_queries", ""),
                "linkage_n_candidates": metrics.get("linkage_n_candidates", ""),
            })
            count += 1
    return count


def write_plot_csv(
    series: dict[str, dict[str, Any]], fits: dict[str, dict[str, Any]], baselines: dict[str, float]
) -> int:
    fields = ["figure_family", "record_kind", "series_key", "method_label", "source_run_id",
              "run_role", "draw", "noise_scale", "exact_source_linkage", "balanced_accuracy",
              "lower_95", "upper_95"]
    rows_out: list[dict[str, Any]] = []

    def observation(family: str, key: str, row: dict[str, Any], attack_name: str, anchor: bool) -> None:
        rows_out.append({
            "figure_family": family, "record_kind": "observation", "series_key": key,
            "method_label": LABELS.get(key, key), "source_run_id": row["run_id"],
            "run_role": row["run_role"], "draw": row["draw"],
            "noise_scale": f"{float(row['anchor_noise_sigma'] if anchor else row['noise_sigma']):.17g}",
            "exact_source_linkage": f"{source_linkage(row, attack_name):.17g}",
            "balanced_accuracy": f"{balanced_accuracy(row):.17g}", "lower_95": "", "upper_95": "",
        })

    for key in ("c_private", "aa_private", "mp", "op", "am"):
        for row in series[key]["records"]:
            observation("exact_source_linkage_vs_noise", key, row, ATTACKS[key], key in {"mp", "op", "am"})
        fit = fits[f"source_{'private' if key in {'c_private','aa_private'} else key}"]
        for x, estimate, low, high in zip(fit["x"], fit["estimate"], fit["lower"], fit["upper"]):
            rows_out.append({"figure_family": "exact_source_linkage_vs_noise", "record_kind": "spline_estimate",
                             "series_key": key, "method_label": LABELS[key], "source_run_id": "", "run_role": "fit",
                             "draw": "", "noise_scale": f"{x:.17g}", "exact_source_linkage": f"{estimate:.17g}",
                             "balanced_accuracy": "", "lower_95": f"{low:.17g}", "upper_95": f"{high:.17g}"})
    for key, rows, fit_key, attack_name, anchor in (
        ("private", series["c_private"]["records"], "direct_private", ATTACKS["c_private"], False),
        ("anchor_op", series["op"]["records"], "direct_anchor", ATTACKS["op"], True),
    ):
        for row in rows:
            observation("balanced_accuracy_vs_exact_source_linkage_op", key, row, attack_name, anchor)
        fit = fits[fit_key]
        for x, estimate, low, high in zip(fit["x"], fit["estimate"], fit["lower"], fit["upper"]):
            rows_out.append({"figure_family": "balanced_accuracy_vs_exact_source_linkage_op", "record_kind": "direct_spline",
                             "series_key": key, "method_label": key, "source_run_id": "", "run_role": "fit", "draw": "",
                             "noise_scale": "", "exact_source_linkage": f"{x / 100.0:.17g}",
                             "balanced_accuracy": f"{estimate / 100.0:.17g}", "lower_95": f"{low / 100.0:.17g}",
                             "upper_95": f"{high / 100.0:.17g}"})
    for key, linkage, utility in (("clean", baselines["c_source"], baselines["c_utility"]),
                                ("i_gdp", "", baselines["i_utility"])):
        rows_out.append({"figure_family": "balanced_accuracy_vs_exact_source_linkage_op", "record_kind": "benchmark",
                         "series_key": key, "method_label": key, "source_run_id": "", "run_role": "reference", "draw": "",
                         "noise_scale": "", "exact_source_linkage": linkage, "balanced_accuracy": utility,
                         "lower_95": "", "upper_95": ""})
    with PLOT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)
    return len(rows_out)


def interpolate_fit(fit: dict[str, Any], x_value: float, field: str = "estimate") -> float:
    """Linearly interpolate one fitted quantity at an in-support x value."""
    grid = np.asarray(fit["x"], dtype=np.float64)
    values = np.asarray(fit[field], dtype=np.float64)
    if x_value < float(grid[0]) or x_value > float(grid[-1]):
        raise ValueError(f"Requested {x_value} outside fit support [{grid[0]}, {grid[-1]}]")
    return float(np.interp(x_value, grid, values))


def chance_crossings(fit: dict[str, Any], chance: float = CHANCE) -> list[float]:
    """Return all linearly interpolated crossings of a fitted leakage curve."""
    x = np.asarray(fit["x"], dtype=np.float64)
    delta = np.asarray(fit["estimate"], dtype=np.float64) - chance
    crossings: list[float] = []
    for index in range(len(x) - 1):
        left, right = float(delta[index]), float(delta[index + 1])
        if left == 0.0:
            crossings.append(float(x[index]))
        elif left * right < 0.0:
            fraction = -left / (right - left)
            crossings.append(float(x[index] + fraction * (x[index + 1] - x[index])))
    if float(delta[-1]) == 0.0:
        crossings.append(float(x[-1]))
    deduplicated: list[float] = []
    for value in crossings:
        if not deduplicated or abs(value - deduplicated[-1]) > 1.0e-10:
            deduplicated.append(value)
    return deduplicated


def validate_op_selection(anchor_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Confirm that OP is the strongest stored anchor-noise linkage attack."""
    attack_names = {"mp": ATTACKS["mp"], "op": ATTACKS["op"], "am": ATTACKS["am"]}
    scores = {
        key: y_linkage(anchor_rows, name)
        for key, name in attack_names.items()
    }
    means = {key: float(np.mean(values)) for key, values in scores.items()}
    medians = {key: float(np.median(values)) for key, values in scores.items()}
    maxima = np.column_stack([scores[key] for key in ("mp", "op", "am")]).max(axis=1)
    strongest_counts = {
        key: int(np.count_nonzero(np.isclose(values, maxima, rtol=0.0, atol=1.0e-15)))
        for key, values in scores.items()
    }
    strongest_mean = max(means, key=means.get)
    if strongest_mean != "op":
        raise AssertionError(f"OP is not strongest by mean exact-source linkage: {means}")
    return {
        "selection_rule": "largest mean exact-source linkage across the 200 paired anchor-noise observations",
        "selected_attack": "PA-OP",
        "mean_linkage": means,
        "median_linkage": medians,
        "pointwise_maximum_count_including_ties": strongest_counts,
    }


def build_numerical_summary(
    fits: dict[str, dict[str, Any]],
    baselines: dict[str, float],
    c_rows: list[dict[str, Any]],
    aa_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    private_equivalence: dict[str, float | int],
    op_selection: dict[str, Any],
) -> dict[str, Any]:
    private_points = [0.0, 5.0, 10.0, 20.0, 35.0, 50.0]
    anchor_points = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    noise_fits = {
        "c_gdp_and_aa_i_gdp_private_noise": fits["source_private"],
        "anchor_noise_mp": fits["source_mp"],
        "anchor_noise_op": fits["source_op"],
        "anchor_noise_alignment_map": fits["source_am"],
    }
    crossings = {
        key: chance_crossings(fit)
        for key, fit in noise_fits.items()
    }
    representative = {
        "private_noise": {
            f"sigma_{value:g}": {
                "fitted_exact_source_linkage": interpolate_fit(fits["source_private"], value),
                "fitted_c_gdp_balanced_accuracy": interpolate_fit(fits["utility_private_c"], value),
                "fitted_aa_i_gdp_balanced_accuracy": interpolate_fit(fits["utility_private_aa"], value),
            }
            for value in private_points
        },
        "anchor_noise": {
            f"v_{value:g}": {
                "fitted_mp_exact_source_linkage": interpolate_fit(fits["source_mp"], value),
                "fitted_op_exact_source_linkage": interpolate_fit(fits["source_op"], value),
                "fitted_alignment_map_exact_source_linkage": interpolate_fit(fits["source_am"], value),
                "fitted_balanced_accuracy": interpolate_fit(fits["utility_anchor"], value),
            }
            for value in anchor_points
        },
    }
    return {
        "metric": "ten-way exact-source linkage accuracy from frozen-auditor embeddings",
        "metric_field": "attacks[*].metrics.linkage_embedding_top1",
        "chance_accuracy": CHANCE,
        "linkage_protocol": manifest,
        "fit_observations": {
            "c_gdp_private_noise": len(c_rows),
            "aa_i_gdp_private_noise": len(aa_rows),
            "aa_i_gdp_anchor_noise": len(anchor_rows),
        },
        "noise_ranges": {"private_sigma": [0.0, PRIVATE_UPPER], "anchor_v": [0.0, ANCHOR_UPPER]},
        "noiseless_controls": baselines,
        "private_method_equivalence": private_equivalence,
        "op_attack_selection": op_selection,
        "fitted_chance_crossings": {
            key: {
                "all_crossings": values,
                "first_crossing": values[0] if values else None,
                "within_calibrated_range": bool(values),
            }
            for key, values in crossings.items()
        },
        "representative_fitted_values": representative,
        "direct_spline_support_percent": {
            "private_noise": fits["direct_private"]["support"],
            "anchor_noise_op": fits["direct_anchor"]["support"],
        },
        "endpoint_controls_used": False,
    }


def write_summary(summary: dict[str, Any]) -> None:
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    crossing_lines = []
    for key, value in summary["fitted_chance_crossings"].items():
        first = value["first_crossing"]
        crossing_lines.append(
            f"- {key}: " + (f"first 10% crossing at {first:.6g}" if first is not None else "no fitted 10% crossing")
        )
    op = summary["op_attack_selection"]
    markdown = "\n".join([
        "# MNIST exact-source-linkage numerical summary",
        "",
        "All privacy values use `attacks[*].metrics.linkage_embedding_top1` from the authoritative 609-record aggregate.",
        "The ten-way gallery contains 1,000 fixed queries and has a 10% chance level.",
        "",
        "## Validations",
        "",
        f"- 200 fit observations per noisy protocol.",
        f"- One common linkage manifest: `{summary['linkage_protocol']['manifest_hash']}`.",
        f"- C-GDP and AA-I-GDP private-noise linkage agrees in all {summary['private_method_equivalence']['exact_sigma_linkage_pairs']} paired observations.",
        f"- OP has the largest mean exact-source linkage ({100 * op['mean_linkage']['op']:.4f}%) and is pointwise strongest (ties included) in {op['pointwise_maximum_count_including_ties']['op']} of 200 paired draws.",
        "",
        "## Fitted chance crossings",
        "",
        *crossing_lines,
        "",
        "## Noiseless controls",
        "",
        f"- C-GDP / AA-I-GDP exact-source linkage: {100 * summary['noiseless_controls']['c_source']:.4f}%.",
        f"- C-GDP / AA-I-GDP balanced accuracy: {100 * summary['noiseless_controls']['c_utility']:.4f}%.",
        f"- I-GDP balanced-accuracy reference: {100 * summary['noiseless_controls']['i_utility']:.4f}%.",
        "",
        "The full representative-value table is stored in the companion JSON file.",
        "",
    ])
    SUMMARY_MD.write_text(markdown, encoding="utf-8")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    records = load_records()
    c_rows = select_fit_records(records, "c_gdp_an")
    aa_rows = select_fit_records(records, "aa_i_gdp_an")
    anchor_rows = select_fit_records(records, "pa_i_gdp")
    private_equivalence = validate_private_equivalence(c_rows, aa_rows)
    manifest = validate_linkage_manifest(records)
    op_selection = validate_op_selection(anchor_rows)
    baseline_records = {key: record_one(records, key) for key in ("c_gdp", "aa_i_gdp", "i_gdp")}
    baselines = {
        "c_source": source_linkage(baseline_records["c_gdp"], "C-exact"),
        "aa_source": source_linkage(baseline_records["aa_i_gdp"], "AA-known-anchor"),
        "c_utility": balanced_accuracy(baseline_records["c_gdp"]),
        "aa_utility": balanced_accuracy(baseline_records["aa_i_gdp"]),
        "i_utility": balanced_accuracy(baseline_records["i_gdp"]),
    }
    if baselines["c_source"] != baselines["aa_source"] or baselines["c_utility"] != baselines["aa_utility"]:
        raise AssertionError("Clean C-GDP/AA-I-GDP baselines differ")

    private_x = x_for(c_rows, False)
    anchor_x = x_for(anchor_rows, True)
    fits = {
        "source_private": conditional_spline(private_x, y_linkage(c_rows, ATTACKS["c_private"]), PRIVATE_UPPER, 24071911),
        "source_mp": conditional_spline(anchor_x, y_linkage(anchor_rows, ATTACKS["mp"]), ANCHOR_UPPER, 24071912),
        "source_op": conditional_spline(anchor_x, y_linkage(anchor_rows, ATTACKS["op"]), ANCHOR_UPPER, 24071913),
        "source_am": conditional_spline(anchor_x, y_linkage(anchor_rows, ATTACKS["am"]), ANCHOR_UPPER, 24071914),
        "utility_private_c": conditional_spline(private_x, y_utility(c_rows), PRIVATE_UPPER, 24072011),
        "utility_private_aa": conditional_spline(private_x, y_utility(aa_rows), PRIVATE_UPPER, 24072011),
        "utility_anchor": conditional_spline(anchor_x, y_utility(anchor_rows), ANCHOR_UPPER, 24072012),
        "direct_private": direct_spline(y_linkage(c_rows, ATTACKS["c_private"]), y_utility(c_rows), 24072031),
        "direct_anchor": direct_spline(y_linkage(anchor_rows, ATTACKS["op"]), y_utility(anchor_rows), 24072032),
    }
    series = {
        "c_private": {"records": c_rows}, "aa_private": {"records": aa_rows},
        "mp": {"records": anchor_rows}, "op": {"records": anchor_rows}, "am": {"records": anchor_rows},
    }
    source_count = write_source_csv(c_rows, aa_rows, anchor_rows, baseline_records)
    plot_count = write_plot_csv(series, fits, baselines)
    summary = build_numerical_summary(
        fits, baselines, c_rows, aa_rows, anchor_rows, manifest, private_equivalence, op_selection
    )
    write_summary(summary)
    outputs: dict[str, dict[str, str]] = {}

    outputs.update(render_figure1(c_rows, anchor_rows, fits, baselines))

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.7, 5.4))
    draw_direct_tradeoff(ax, c_rows, anchor_rows, fits["direct_private"], fits["direct_anchor"], baselines)
    fig.subplots_adjust(left=0.10, right=0.70, bottom=0.12, top=0.96)
    outputs["mnist_balanced_accuracy_versus_exact_source_linkage_op"] = save_figure(
        fig, "mnist_balanced_accuracy_versus_exact_source_linkage_op"
    )
    plt.close(fig)

    hash_paths = [Path(__file__).resolve(), SOURCE_CSV, PLOT_CSV, SUMMARY_JSON, SUMMARY_MD]
    for group in outputs.values():
        hash_paths.extend(HERE / filename for filename in group.values())
    metadata = {
        "authoritative_source": str(SOURCE.relative_to(REPO)).replace("\\", "/"),
        "authoritative_source_sha256": sha256_file(SOURCE),
        "authoritative_record_count": len(records),
        "fit_roles": {"private": sorted(PRIVATE_ROLES), "anchor": sorted(ANCHOR_ROLES)},
        "fit_observations_per_noisy_protocol": 200,
        "validation_counts": {
            "c_gdp_an": len(c_rows), "aa_i_gdp_an": len(aa_rows), "pa_i_gdp": len(anchor_rows),
            "source_csv_rows": source_count, "plot_csv_rows": plot_count,
            "private_exact_sigma_linkage_pairs": private_equivalence["exact_sigma_linkage_pairs"],
            "private_exact_utility_pairs": private_equivalence["exact_utility_pairs"],
            "private_max_abs_utility_difference": private_equivalence["max_abs_utility_difference"],
            "anchor_gpm_converged": sum(bool(row["gpm"]["converged"]) for row in anchor_rows),
            "anchor_gpm_iteration_cap": sum(not bool(row["gpm"]["converged"]) for row in anchor_rows),
        },
        "endpoint_controls_in_fit": False,
        "endpoint_controls_in_plots": False,
        "linkage_protocol": manifest,
        "op_attack_selection": op_selection,
        "metric": "exact_source_linkage",
        "metric_field": "attacks[*].metrics.linkage_embedding_top1",
        "metric_ylabel": "Exact-Source Linkage Accuracy (%)",
        "private_x_upper": PRIVATE_UPPER,
        "anchor_x_upper": ANCHOR_UPPER,
        "baselines": baselines,
        "spline": {"degree": 3, "interior_knots": 5, "grid_size": GRID_SIZE,
                   "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap": "multinomial percentile",
                   "seeds": {key: value["seed"] for key, value in fits.items()}},
        "direct_spline_support_percent": {
            "private": fits["direct_private"]["support"], "anchor_op": fits["direct_anchor"]["support"]
        },
        "summary_files": {"json": SUMMARY_JSON.name, "markdown": SUMMARY_MD.name},
        "outputs": outputs,
        "sha256": {path.name: sha256_file(path) for path in hash_paths},
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
