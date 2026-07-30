"""Build the self-contained CelebA p=10 publication-figure bundle.

The authoritative quantitative input is the 207-record JSON aggregate from the
current participant-count sweep.  Random draws alone are used for conditional
spline fits and bootstrap bands; deterministic endpoint controls are excluded
from the quantitative plots and fitting.  The qualitative reconstruction grid
uses a separate deterministic replay artifact whose columns are ordered by
decreasing saved cross-image identity-linkage accuracy.  The v=40 equivalence control is
retained in the tidy
source table and metadata, but is outside the calibrated v in [0, 0.25] range.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


try:
    SCRIPT_PATH = Path(__file__).resolve()
except NameError:  # Jupyter/Colab code cells do not define ``__file__``.
    SCRIPT_PATH = Path.cwd() / "render_celeba_p010_figures.py"

ROOT = SCRIPT_PATH.parents[1]
BUNDLE = Path(
    os.environ.get(
        "REPRO_CELEBA_RECON_OUT",
        str(ROOT / "build" / "renderers" / "celeba_reconstructions"),
    )
).expanduser().resolve()
INPUT_ROOT = Path(
    os.environ.get(
        "REPRO_CELEBA_RECON_INPUT",
        str(ROOT / "build" / "inputs" / "celeba_reconstructions"),
    )
).expanduser().resolve()
DATA_DIR = BUNDLE / "data"
FIGURE_DIR = BUNDLE / "figures"
RAW_JSON = ROOT / "results" / "celeba" / "p010_combined_407_results.json.gz"
RECONSTRUCTION_SWEEP_DIR = INPUT_ROOT / "reconstruction_leakage_ordered_v2"
IDENTITY_RECONSTRUCTION_SWEEP_DIR = (
    INPUT_ROOT / "reconstruction_leakage_ordered_by_identity_v1"
)

BOOTSTRAP_RESAMPLES = 10_000
GRID_SIZE = 201
LINKAGE_CHANCE = 0.10
BALANCED_CHANCE = 0.50
PRIVATE_DISPLAY_UPPER = 1.5

METRICS: dict[str, dict[str, Any]] = {
    "cross_image_identity": {
        "label": "Identity Linkage Accuracy",
        "ylabel": "Identity Linkage Accuracy (%)",
        "short": "cross_identity",
        "definition": (
            "Face-embedding top-1 matching to a different reserved image of "
            "the same identity among ten identity candidates"
        ),
    },
}

ATTACK_SERIES: dict[str, dict[str, str]] = {
    "private": {
        "protocol": "c_gdp_an",
        "attack": "known-secret",
        "label": "C-GDP (private-data noise): known-secret attack",
        "noise_field": "noise_sigma",
        "noise_symbol": r"$\sigma$",
        "color": "#0072B2",
    },
    "mp": {
        "protocol": "pa_i_gdp",
        "attack": "PA-MP",
        "label": "AA-I-GDP (anchor noise): MP attack",
        "noise_field": "anchor_noise_sigma",
        "noise_symbol": r"$v$",
        "color": "#0072B2",
    },
    "op": {
        "protocol": "pa_i_gdp",
        "attack": "PA-OP",
        "label": "AA-I-GDP (anchor noise): OP attack",
        "noise_field": "anchor_noise_sigma",
        "noise_symbol": r"$v$",
        "color": "#E69F00",
    },
    "am": {
        "protocol": "pa_i_gdp",
        "attack": "PA-AM",
        "label": "AA-I-GDP (anchor noise): alignment-map attack",
        "noise_field": "anchor_noise_sigma",
        "noise_symbol": r"$v$",
        "color": "#009E73",
    },
}

UTILITY_SERIES: dict[str, dict[str, str]] = {
    "private": {
        "protocol": "c_gdp_an",
        "label": "C-GDP (private-data noise)",
        "noise_field": "noise_sigma",
        "noise_symbol": r"$\sigma$",
        "color": "#0072B2",
    },
    "anchor": {
        "protocol": "pa_i_gdp",
        "label": "AA-I-GDP (anchor noise)",
        "noise_field": "anchor_noise_sigma",
        "noise_symbol": r"$v$",
        "color": "#CC79A7",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records() -> list[dict[str, Any]]:
    with gzip.open(RAW_JSON, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    all_records = list(payload["records"])
    if payload.get("record_count") != 407 or len(all_records) != 407:
        raise ValueError("Expected exactly 407 authoritative p=10 records")
    records = [
        row
        for row in all_records
        if not str(row["run_role"]).startswith("supplemental_")
    ]
    if len(records) != 207:
        raise ValueError("Expected 207 core p=10 records after filtering")
    if {int(row["n_participants"]) for row in records} != {10}:
        raise ValueError("The aggregate is not exclusively the p=10 condition")
    expected = {
        ("c_gdp", "clean"): 1,
        ("i_gdp", "clean"): 1,
        ("c_gdp_an", "random"): 100,
        ("pa_i_gdp", "random"): 100,
        ("c_gdp_an", "endpoint_zero"): 1,
        ("pa_i_gdp", "endpoint_zero"): 1,
        ("c_gdp_an", "endpoint_upper"): 1,
        ("pa_i_gdp", "endpoint_upper"): 1,
        ("pa_i_gdp", "i_gdp_equivalence_control"): 1,
    }
    actual: dict[tuple[str, str], int] = {}
    for row in records:
        key = (str(row["protocol"]), str(row["run_role"]))
        actual[key] = actual.get(key, 0) + 1
    if actual != expected:
        raise ValueError(f"Unexpected schedule composition: {actual}")
    return records


def attack(record: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in record.get("attacks", []) if item["name"] == name]
    if len(matches) != 1:
        raise KeyError(f"Expected attack {name!r} once in {record['run_id']}")
    return matches[0]


def attack_linkage(record: dict[str, Any], name: str, metric: str) -> float:
    return float(attack(record, name)["metrics"]["face_linkage"][metric]["top1"])


def record_one(
    records: Iterable[dict[str, Any]], protocol: str, run_role: str
) -> dict[str, Any]:
    matches = [
        row
        for row in records
        if row["protocol"] == protocol and row["run_role"] == run_role
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {protocol}/{run_role}, found {len(matches)}")
    return matches[0]


def spline_design(normalized_x: np.ndarray) -> np.ndarray:
    x = np.asarray(normalized_x, dtype=np.float64).reshape(-1)
    knots = np.linspace(0.0, 1.0, 7, dtype=np.float64)[1:-1]
    columns = [np.ones_like(x), x, x * x, x**3]
    columns.extend(np.maximum(x - knot, 0.0) ** 3 for knot in knots)
    return np.column_stack(columns)


def conditional_spline(
    x: np.ndarray,
    y: np.ndarray,
    upper: float,
    seed: int,
    *,
    bootstrap: bool = True,
) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xn = np.clip(x / upper, 0.0, 1.0)
    grid = np.linspace(0.0, upper, GRID_SIZE, dtype=np.float64)
    design = spline_design(xn)
    grid_design = spline_design(grid / upper)
    gram = design.T @ design
    ridge = 1.0e-8 * max(float(np.trace(gram) / design.shape[1]), 1.0)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(gram + penalty, design.T @ y)
    estimate = grid_design @ beta
    if not bootstrap:
        return {
            "x": grid,
            "estimate": estimate,
            "lower": estimate,
            "upper": estimate,
            "ridge": ridge,
            "seed": seed,
        }
    rng = np.random.default_rng(seed)
    probability = np.full(len(x), 1.0 / len(x), dtype=np.float64)
    predictions = np.empty((BOOTSTRAP_RESAMPLES, GRID_SIZE), dtype=np.float64)
    chunk = 256
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        stop = min(start + chunk, BOOTSTRAP_RESAMPLES)
        counts = rng.multinomial(len(x), probability, size=stop - start).astype(
            np.float64, copy=False
        )
        boot_gram = np.einsum(
            "bn,np,nq->bpq", counts, design, design, optimize=True
        )
        boot_rhs = np.einsum("bn,np,n->bp", counts, design, y, optimize=True)
        boot_beta = np.linalg.solve(
            boot_gram + penalty[None, :, :], boot_rhs[..., None]
        )[..., 0]
        predictions[start:stop] = boot_beta @ grid_design.T
    lower, upper_band = np.quantile(predictions, [0.025, 0.975], axis=0)
    return {
        "x": grid,
        "estimate": estimate,
        "lower": lower,
        "upper": upper_band,
        "ridge": ridge,
        "seed": seed,
    }


def noise_upper(records: list[dict[str, Any]], series_key: str) -> float:
    if series_key == "private":
        return float(record_one(records, "c_gdp_an", "endpoint_upper")["noise_sigma"])
    return float(
        record_one(records, "pa_i_gdp", "endpoint_upper")["anchor_noise_sigma"]
    )


def attack_series(
    records: list[dict[str, Any]], series_key: str, metric: str
) -> dict[str, Any]:
    definition = ATTACK_SERIES[series_key]
    selected = [
        row
        for row in records
        if row["protocol"] == definition["protocol"]
        and row["run_role"] in {"random", "endpoint_zero", "endpoint_upper"}
    ]
    by_role = {
        role: [row for row in selected if row["run_role"] == role]
        for role in ("random", "endpoint_zero", "endpoint_upper")
    }
    if len(by_role["random"]) != 100:
        raise ValueError(f"Expected 100 random rows for {series_key}")
    x = np.asarray(
        [float(row[definition["noise_field"]]) for row in by_role["random"]]
    )
    y = np.asarray(
        [attack_linkage(row, definition["attack"], metric) for row in by_role["random"]]
    )
    upper = noise_upper(records, series_key)
    fit = conditional_spline(
        x,
        y,
        upper,
        seed=26072000 + 100 * list(METRICS).index(metric) + list(ATTACK_SERIES).index(series_key),
    )
    return {"definition": definition, "by_role": by_role, "x": x, "y": y, "upper": upper, "fit": fit}


def utility_series(records: list[dict[str, Any]], series_key: str) -> dict[str, Any]:
    definition = UTILITY_SERIES[series_key]
    selected = [
        row
        for row in records
        if row["protocol"] == definition["protocol"]
        and row["run_role"] in {"random", "endpoint_zero", "endpoint_upper"}
    ]
    by_role = {
        role: [row for row in selected if row["run_role"] == role]
        for role in ("random", "endpoint_zero", "endpoint_upper")
    }
    if len(by_role["random"]) != 100:
        raise ValueError(f"Expected 100 random utility rows for {series_key}")
    x = np.asarray(
        [float(row[definition["noise_field"]]) for row in by_role["random"]]
    )
    y = np.asarray(
        [float(row["utility"]["test_balanced_accuracy"]) for row in by_role["random"]]
    )
    upper = noise_upper(records, "private" if series_key == "private" else "anchor")
    fit = conditional_spline(x, y, upper, seed=26072100 + list(UTILITY_SERIES).index(series_key))
    return {"definition": definition, "by_role": by_role, "x": x, "y": y, "upper": upper, "fit": fit}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.0,
            "legend.fontsize": 9.2,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def wide_figure_rc() -> dict[str, float]:
    """Font sizes for 13--14 inch multi-panel PDFs scaled to text width."""

    return {
        "font.size": 18.0,
        "axes.titlesize": 20.0,
        "axes.labelsize": 19.0,
        "legend.fontsize": 17.0,
        "xtick.labelsize": 17.0,
        "ytick.labelsize": 17.0,
    }


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


def combined_legend_entries(axes: Iterable[plt.Axes]) -> tuple[list[Any], list[str]]:
    """Collect one handle for every distinct legend label across several axes."""
    unique: dict[str, Any] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
    return list(unique.values()), list(unique.keys())


def save_figure(fig: plt.Figure, subdir: str, stem: str) -> dict[str, str]:
    target = FIGURE_DIR / subdir
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for suffix in ("png", "pdf", "svg"):
        path = target / f"{stem}.{suffix}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        paths[suffix] = path.relative_to(BUNDLE).as_posix()
    return paths


def endpoint_value(row: dict[str, Any], definition: dict[str, str], metric: str) -> float:
    return attack_linkage(row, definition["attack"], metric)


def draw_linkage_panel(
    ax: plt.Axes,
    prepared: dict[str, Any],
    metric: str,
    title: str,
    *,
    compact: bool = False,
) -> None:
    definition = prepared["definition"]
    by_role = prepared["by_role"]
    fit = prepared["fit"]
    upper = prepared["upper"]
    display_upper = (
        PRIVATE_DISPLAY_UPPER
        if definition["protocol"] == "c_gdp_an"
        else upper
    )
    style_axis(ax)
    ax.set_xlim(0.0, display_upper)
    if definition["protocol"] == "c_gdp_an":
        ax.set_xticks(np.arange(0.0, PRIVATE_DISPLAY_UPPER + 0.001, 0.25))
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks(np.arange(0.0, 101.0, 10.0))
    ax.set_xlabel(f"Noise Scale {definition['noise_symbol']}")
    ax.set_ylabel(METRICS[metric]["ylabel"])
    ax.set_title(title, loc="left", pad=8)
    color = definition["color"]
    random = by_role["random"]
    xs = np.asarray([float(row[definition["noise_field"]]) for row in random])
    ys = np.asarray([attack_linkage(row, definition["attack"], metric) for row in random])
    ax.fill_between(
        fit["x"], 100.0 * np.clip(fit["lower"], 0, 1), 100.0 * np.clip(fit["upper"], 0, 1),
        color=color, alpha=0.15, linewidth=0, label="95% conditional band", zorder=1,
    )
    if definition["protocol"] == "pa_i_gdp":
        converged = np.asarray([bool(row["gpm"]["converged"]) for row in random])
        ax.scatter(xs[converged], 100.0 * ys[converged], s=18, color=color, alpha=0.28,
                   linewidths=0, label="Random draw: GPM converged", zorder=3)
        ax.scatter(xs[~converged], 100.0 * ys[~converged], s=21, facecolors="none",
                   edgecolors=color, alpha=0.33, linewidths=0.7,
                   label="Random draw: 500-iteration cap", zorder=3)
    else:
        ax.scatter(xs, 100.0 * ys, s=18, color=color, alpha=0.25, linewidths=0,
                   label="100 random draws", zorder=3)
    ax.plot(fit["x"], 100.0 * np.clip(fit["estimate"], 0, 1), color=color,
            linewidth=2.55, label=definition["label"], zorder=5)
    zero = by_role["endpoint_zero"][0]
    ax.axhline(100.0 * LINKAGE_CHANCE, color="#666666", linestyle=(0, (1.3, 2.2)),
               linewidth=1.15, label="Random guess (10%)", zorder=2)
    baseline = endpoint_value(zero, definition, metric)
    ax.axhline(100.0 * baseline, color="#D55E00", linestyle=(0, (5.0, 2.0, 1.0, 2.0)),
               linewidth=1.7, label=("C-GDP (no noise)" if definition["protocol"] == "c_gdp_an"
                                     else "AA-I-GDP (no anchor noise)"), zorder=2)
    if not compact:
        dedupe_legend(ax, loc="upper right", frameon=True, fancybox=False,
                      framealpha=0.93, edgecolor="#B0B0B0", borderpad=0.45,
                      handlelength=2.7, labelspacing=0.34)


def render_linkage_figures(
    records: list[dict[str, Any]], outputs: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    prepared = {
        metric: {key: attack_series(records, key, metric) for key in ATTACK_SERIES}
        for metric in METRICS
    }
    panel_names = {
        "private": "Private-data noise",
        "mp": "Anchor noise: MP attack",
        "op": "Anchor noise: OP attack",
        "am": "Anchor noise: alignment-map attack",
    }
    for metric, metric_spec in METRICS.items():
        for key in ATTACK_SERIES:
            fig, ax = plt.subplots(figsize=(7.25, 5.25))
            draw_linkage_panel(
                ax,
                prepared[metric][key],
                metric,
                f"{metric_spec['label']}: {panel_names[key].lower()}",
            )
            fig.subplots_adjust(left=0.13, right=0.98, bottom=0.13, top=0.90)
            stem = f"celeba_{metric_spec['short']}_leakage_{'private_data_noise' if key == 'private' else 'anchor_noise_' + key}"
            outputs[stem] = save_figure(fig, "leakage_noise", stem)
            plt.close(fig)

        with mpl.rc_context(wide_figure_rc()):
            fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.4), sharey=True)
            for panel_index, (ax, (key, panel)) in enumerate(
                zip(axes.flat, panel_names.items())
            ):
                prefix = f"({chr(ord('a') + panel_index)}) "
                draw_linkage_panel(
                    ax,
                    prepared[metric][key],
                    metric,
                    prefix + panel,
                    compact=True,
                )
            for ax in axes.flat:
                ax.set_ylabel("")
            fig.supylabel(METRICS[metric]["ylabel"], x=0.012)
            handles, labels = combined_legend_entries(axes.flat)
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02),
                       ncol=2, frameon=False, handlelength=2.6, columnspacing=1.2)
            fig.subplots_adjust(left=0.085, right=0.985, bottom=0.37, top=0.97, hspace=0.34, wspace=0.22)
            stem = f"celeba_{metric_spec['short']}_leakage_four_panel"
            outputs[stem] = save_figure(fig, "leakage_noise", stem)
            plt.close(fig)

    return prepared


def draw_utility_panel(
    ax: plt.Axes,
    prepared: dict[str, Any],
    controls: dict[str, float],
    title: str,
    *,
    compact: bool = False,
) -> None:
    definition = prepared["definition"]
    by_role = prepared["by_role"]
    fit = prepared["fit"]
    upper = prepared["upper"]
    display_upper = (
        PRIVATE_DISPLAY_UPPER
        if definition["protocol"] == "c_gdp_an"
        else upper
    )
    color = definition["color"]
    style_axis(ax)
    ax.set_xlim(0.0, display_upper)
    if definition["protocol"] == "c_gdp_an":
        ax.set_xticks(np.arange(0.0, PRIVATE_DISPLAY_UPPER + 0.001, 0.25))
    ax.set_ylim(45.0, 90.0)
    ax.set_yticks(np.arange(45.0, 91.0, 5.0))
    ax.set_xlabel(f"Noise Scale {definition['noise_symbol']}")
    ax.set_ylabel("Smiling Test Balanced Accuracy (%)")
    ax.set_title(title, loc="left", pad=8)
    random = by_role["random"]
    xs = np.asarray([float(row[definition["noise_field"]]) for row in random])
    ys = np.asarray([float(row["utility"]["test_balanced_accuracy"]) for row in random])
    ax.fill_between(fit["x"], 100.0 * fit["lower"], 100.0 * fit["upper"],
                    color=color, alpha=0.15, linewidth=0, label="95% conditional band", zorder=1)
    if definition["protocol"] == "pa_i_gdp":
        converged = np.asarray([bool(row["gpm"]["converged"]) for row in random])
        ax.scatter(xs[converged], 100.0 * ys[converged], s=18, color=color, alpha=0.28,
                   linewidths=0, label="Random draw: GPM converged", zorder=3)
        ax.scatter(xs[~converged], 100.0 * ys[~converged], s=21, facecolors="none",
                   edgecolors=color, alpha=0.34, linewidths=0.7,
                   label="Random draw: 500-iteration cap", zorder=3)
    else:
        ax.scatter(xs, 100.0 * ys, s=18, color=color, alpha=0.25, linewidths=0,
                   label="100 random draws", zorder=3)
    ax.plot(fit["x"], 100.0 * fit["estimate"], color=color, linewidth=2.6,
            label=definition["label"], zorder=5)
    ax.axhline(100.0 * BALANCED_CHANCE, color="#888888", linestyle=(0, (1.3, 2.2)),
               linewidth=1.1, label="Balanced random guess (50%)", zorder=2)
    ax.axhline(100.0 * controls["c_gdp"], color="#333333", linestyle=(0, (1.2, 1.8)),
               linewidth=1.9, label="C-GDP (no noise)", zorder=2)
    if definition["protocol"] == "pa_i_gdp":
        ax.axhline(100.0 * controls["aa_zero"], color="#D55E00",
                   linestyle=(0, (5.0, 2.0, 1.0, 2.0)), linewidth=1.7,
                   label="AA-I-GDP (no anchor noise)", zorder=2)
    ax.axhline(100.0 * controls["i_gdp"], color="#009E73", linestyle=(0, (5.5, 2.2, 1.2, 2.2)),
               linewidth=2.0, label="I-GDP", zorder=2)
    if not compact:
        dedupe_legend(ax, loc="upper right", frameon=True, fancybox=False,
                      framealpha=0.93, edgecolor="#B0B0B0", borderpad=0.45,
                      handlelength=2.7, labelspacing=0.34)


def render_utility_figures(
    records: list[dict[str, Any]], outputs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, float]]:
    prepared = {key: utility_series(records, key) for key in UTILITY_SERIES}
    controls = {
        "c_gdp": float(record_one(records, "c_gdp", "clean")["utility"]["test_balanced_accuracy"]),
        "i_gdp": float(record_one(records, "i_gdp", "clean")["utility"]["test_balanced_accuracy"]),
        "aa_zero": float(record_one(records, "pa_i_gdp", "endpoint_zero")["utility"]["test_balanced_accuracy"]),
    }
    titles = {
        "private": "C-GDP (private noise)",
        "anchor": "AA-I-GDP (anchor noise)",
    }
    for key in UTILITY_SERIES:
        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        draw_utility_panel(ax, prepared[key], controls, titles[key])
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.13, top=0.90)
        stem = f"celeba_smiling_balanced_accuracy_{'private_data_noise' if key == 'private' else 'anchor_noise'}"
        outputs[stem] = save_figure(fig, "utility_noise", stem)
        plt.close(fig)
    with mpl.rc_context(wide_figure_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.35), sharey=True)
        for ax, key in zip(axes, UTILITY_SERIES):
            draw_utility_panel(ax, prepared[key], controls, titles[key], compact=True)
        axes[1].set_ylabel("")
        handles, labels = combined_legend_entries(axes)
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015),
                   ncol=3, frameon=False, handlelength=2.6, columnspacing=1.2)
        fig.subplots_adjust(left=0.085, right=0.985, bottom=0.40, top=0.94, wspace=0.16)
        stem = "celeba_smiling_balanced_accuracy_two_panel"
        outputs[stem] = save_figure(fig, "utility_noise", stem)
        plt.close(fig)
    return prepared, controls


def draw_tradeoff_panel(
    ax: plt.Axes,
    metric: str,
    leakage_prepared: dict[str, dict[str, Any]],
    utility_prepared: dict[str, Any],
    controls: dict[str, float],
    records: list[dict[str, Any]],
    title: str,
    *,
    compact: bool = False,
) -> None:
    style_axis(ax)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(45.0, 90.0)
    ax.set_xticks(np.arange(0.0, 101.0, 10.0))
    ax.set_yticks(np.arange(45.0, 91.0, 5.0))
    ax.set_xlabel(f"{METRICS[metric]['label']} (%)")
    ax.set_ylabel("Smiling Test Balanced Accuracy (%)")
    # The manuscript caption explains the panel meaning.  Keep only an
    # explicit panel marker for multi-panel variants and omit plot titles.
    if title.startswith("("):
        ax.set_title(title.split(maxsplit=1)[0], loc="left", pad=8)
    ax.axvline(10.0, color="#777777", linestyle=(0, (1.3, 2.2)), linewidth=1.1,
               label="Random guess (10% leakage)", zorder=1)
    ax.axhline(50.0, color="#AAAAAA", linestyle=(0, (1.3, 2.2)), linewidth=1.0, zorder=1)
    ax.text(0.025, 0.955, "preferred: upper left", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.0, color="#444444")

    for attack_key, utility_key, color, marker, linestyle, label in (
        ("private", "private", "#0072B2", "o", "-", "C-GDP (private-data noise)"),
        ("op", "anchor", "#E69F00", "^", (0, (5.0, 2.0)), "AA-I-GDP (anchor noise): OP attack"),
    ):
        leak = leakage_prepared[metric][attack_key]
        util = utility_prepared[utility_key]
        random_leak = leak["by_role"]["random"]
        random_util = util["by_role"]["random"]
        by_id_util = {row["run_id"]: row for row in random_util}
        pairs = [(row, by_id_util[row["run_id"]]) for row in random_leak]
        lx = np.asarray([attack_linkage(a, leak["definition"]["attack"], metric) for a, _ in pairs])
        uy = np.asarray([float(u["utility"]["test_balanced_accuracy"]) for _, u in pairs])
        if attack_key == "op":
            converged = np.asarray([bool(a["gpm"]["converged"]) for a, _ in pairs])
            ax.scatter(100.0 * lx[converged], 100.0 * uy[converged], s=20, marker=marker,
                       color=color, alpha=0.25, linewidths=0, zorder=3)
            ax.scatter(100.0 * lx[~converged], 100.0 * uy[~converged], s=24, marker=marker,
                       facecolors="none", edgecolors=color, alpha=0.33, linewidths=0.75, zorder=3)
        else:
            ax.scatter(100.0 * lx, 100.0 * uy, s=18, marker=marker, color=color,
                       alpha=0.22, linewidths=0, zorder=3)
        lf, uf = leak["fit"], util["fit"]
        ax.fill(
            np.concatenate([100.0 * np.clip(lf["lower"], 0, 1), 100.0 * np.clip(lf["upper"], 0, 1)[::-1]]),
            np.concatenate([100.0 * np.clip(uf["lower"], 0, 1), 100.0 * np.clip(uf["upper"], 0, 1)[::-1]]),
            color=color, alpha=0.10, linewidth=0, zorder=1,
        )
        ax.plot(100.0 * np.clip(lf["estimate"], 0, 1),
                100.0 * np.clip(uf["estimate"], 0, 1), color=color,
                linestyle=linestyle, linewidth=2.6, label=label, zorder=5)
    clean = record_one(records, "c_gdp", "clean")
    clean_leakage = attack_linkage(clean, "C-exact", metric)
    ax.scatter([100.0 * clean_leakage], [100.0 * controls["c_gdp"]], s=128,
               marker="*", color="#222222", edgecolors="white", linewidths=0.55,
               label="C-GDP (no noise)", zorder=9)
    ax.axhline(100.0 * controls["i_gdp"], color="#D55E00",
               linestyle=(0, (5.0, 2.0, 1.0, 2.0)), linewidth=2.0,
               label="I-GDP balanced accuracy", zorder=2)
    if not compact:
        dedupe_legend(ax, loc="lower right", frameon=True, fancybox=False,
                      framealpha=0.93, edgecolor="#B0B0B0", borderpad=0.45,
                      handlelength=2.7, labelspacing=0.36)


def render_tradeoff_figures(
    records: list[dict[str, Any]],
    leakage_prepared: dict[str, dict[str, Any]],
    utility_prepared: dict[str, Any],
    controls: dict[str, float],
    outputs: dict[str, Any],
) -> None:
    for metric, metric_spec in METRICS.items():
        fig, ax = plt.subplots(figsize=(8.2, 5.45))
        draw_tradeoff_panel(ax, metric, leakage_prepared, utility_prepared, controls,
                            records, metric_spec["label"])
        fig.subplots_adjust(left=0.105, right=0.98, bottom=0.14, top=0.90)
        stem = f"celeba_smiling_balanced_accuracy_vs_{metric_spec['short']}_leakage_op"
        outputs[stem] = save_figure(fig, "privacy_utility", stem)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.45), sharey=True)
    for ax, metric, prefix in zip(axes, METRICS, ("(a)", "(b)")):
        draw_tradeoff_panel(ax, metric, leakage_prepared, utility_prepared, controls,
                            records, f"{prefix} {METRICS[metric]['label']}", compact=True)
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=3, frameon=False, handlelength=2.6)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.25, top=0.93, wspace=0.12)
    stem = "celeba_smiling_balanced_accuracy_vs_identity_linkage_two_panel_op"
    outputs[stem] = save_figure(fig, "privacy_utility", stem)
    plt.close(fig)


def render_gpm_diagnostics(records: list[dict[str, Any]], outputs: dict[str, Any]) -> None:
    random = [row for row in records if row["protocol"] == "pa_i_gdp" and row["run_role"] == "random"]
    v = np.asarray([float(row["anchor_v"]) for row in random])
    converged = np.asarray([bool(row["gpm"]["converged"]) for row in random])
    iterations = np.asarray([int(row["gpm"]["iterations"]) for row in random])
    residual = np.asarray([float(row["gpm"]["fixed_point_residual"]) for row in random])
    elapsed = np.asarray([float(row["gpm"]["elapsed_seconds"]) for row in random])
    # This three-panel PDF is scaled from roughly 14 inches to one manuscript
    # text width.  Use source fonts that remain at least 7 pt after that
    # reduction; the default 8--11 pt source fonts become only 3--5 pt.
    with mpl.rc_context(wide_figure_rc()):
        fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))
        for ax, y, ylabel, log in (
            (axes[0], iterations, "GPM Iterations", False),
            (axes[1], residual, "Fixed-Point Residual", True),
            (axes[2], elapsed, "GPM Optimization Time (s)", False),
        ):
            style_axis(ax)
            ax.scatter(v[converged], y[converged], s=34, color="#0072B2", alpha=0.55,
                       linewidths=0, label="Converged")
            ax.scatter(v[~converged], y[~converged], s=38, facecolors="none",
                       edgecolors="#D55E00", alpha=0.55, linewidths=1.2,
                       label="500-iteration cap")
            ax.set_xlabel(r"Anchor-Noise Scale $v$")
            ax.set_ylabel(ylabel)
            if log:
                ax.set_yscale("log")
        dedupe_legend(axes[0], loc="lower left", bbox_to_anchor=(0.0, 1.01),
                      ncol=2, frameon=False, borderaxespad=0.0)
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.84, wspace=0.36)
        stem = "celeba_gpm_convergence_diagnostics"
        outputs[stem] = save_figure(fig, "gpm", stem)
        plt.close(fig)


def reconstruction_row_templates() -> list[dict[str, Any]]:
    """Return the shared labels, equations, and colors for reconstruction grids."""

    return [
        {
            "key": "original",
            "label": "Original",
            "math": [r"$\boldsymbol{X}_j$"],
            "color": "#F1F4F7",
        },
        {
            "key": "c_gdp_clean",
            "label": "C-GDP  |  known-secret inversion",
            "math": [
                r"$\widehat{\widetilde{\boldsymbol{X}}}_j=(\boldsymbol{Y}_j-\mathbf{1}_{n_j}\Psi_s^\top)\boldsymbol{O}_s^\top=\widetilde{\boldsymbol{X}}_j$",
                r"$\widehat{\boldsymbol{X}}_j=\widetilde{\boldsymbol{X}}_j\boldsymbol{F}^\dagger=\boldsymbol{X}_j\boldsymbol{F}\boldsymbol{F}^\dagger$",
            ],
            "color": "#EAF2F7",
        },
        {
            "key": "c_gdp_private",
            "label": "C-GDP (private-data noise)  |  known-secret inversion",
            "math": [
                r"$\widehat{\widetilde{\boldsymbol{X}}}_j=(\boldsymbol{Y}_j-\mathbf{1}_{n_j}\Psi_s^\top)\boldsymbol{O}_s^\top=\widetilde{\boldsymbol{X}}_j+\sigma\boldsymbol{W}_j\boldsymbol{O}_s^\top$",
                r"$\widehat{\boldsymbol{X}}_j=\boldsymbol{X}_j\boldsymbol{F}\boldsymbol{F}^\dagger+\sigma\boldsymbol{W}_j\boldsymbol{O}_s^\top\boldsymbol{F}^\dagger$",
            ],
            "color": "#E8F4FA",
        },
        {
            "key": "aa_anchor_mp",
            "label": "AA-I-GDP (anchor noise)  |  MP attack",
            "math": [
                r"$\widehat{\widetilde{\boldsymbol{X}}}_j^{\mathrm{MP}}=(\boldsymbol{Y}_j-\mathbf{1}_{n_j}\boldsymbol{b}_j^\top)(\boldsymbol{A}^\top\overline{\boldsymbol{B}}_j)^\dagger$",
                r"$\widehat{\boldsymbol{X}}_j^{\mathrm{MP}}=\widehat{\widetilde{\boldsymbol{X}}}_j^{\mathrm{MP}}\boldsymbol{F}^\dagger,\quad v\to0:\ \widehat{\boldsymbol{X}}_j^{\mathrm{MP}}\to\boldsymbol{X}_j\boldsymbol{F}\boldsymbol{F}^\dagger$",
            ],
            "color": "#FBF4E7",
        },
        {
            "key": "aa_anchor_op",
            "label": "AA-I-GDP (anchor noise)  |  OP attack",
            "math": [
                r"$\widehat{\widetilde{\boldsymbol{X}}}_j^{\mathrm{OP}}=(\boldsymbol{Y}_j-\mathbf{1}_{n_j}\boldsymbol{b}_j^\top)[\Pi(\boldsymbol{A}^\top\overline{\boldsymbol{B}}_j)]^\top$",
                r"$\widehat{\boldsymbol{X}}_j^{\mathrm{OP}}=\widehat{\widetilde{\boldsymbol{X}}}_j^{\mathrm{OP}}\boldsymbol{F}^\dagger,\quad v\to0:\ \widehat{\boldsymbol{X}}_j^{\mathrm{OP}}\to\boldsymbol{X}_j\boldsymbol{F}\boldsymbol{F}^\dagger$",
            ],
            "color": "#FBF4E7",
        },
        {
            "key": "aa_anchor_am",
            "label": "AA-I-GDP (anchor noise)  |  AM attack",
            "math": [
                r"$\widehat{\widetilde{\boldsymbol{X}}}_j^{\mathrm{AM}}=(\boldsymbol{Y}_j-\mathbf{1}_{n_j}\boldsymbol{b}_j^\top)[\Pi(\boldsymbol{O}_c\widehat{\boldsymbol{R}}_c)\widehat{\boldsymbol{R}}_j^\top]^\top$",
                r"$\widehat{\boldsymbol{X}}_j^{\mathrm{AM}}=\widehat{\widetilde{\boldsymbol{X}}}_j^{\mathrm{AM}}\boldsymbol{F}^\dagger,\quad v\to0:\ \widehat{\boldsymbol{X}}_j^{\mathrm{AM}}\to\boldsymbol{X}_j\boldsymbol{F}\boldsymbol{F}^\dagger$",
            ],
            "color": "#FBF4E7",
        },
    ]


def render_reconstruction_grid(
    rows: list[dict[str, Any]],
    outputs: dict[str, Any],
    *,
    stem: str,
    subdir: str = "reconstruction",
) -> None:
    """Render one six-row, ten-column reconstruction grid."""

    expected_keys = [item["key"] for item in reconstruction_row_templates()]
    if [item["key"] for item in rows] != expected_keys:
        raise ValueError("A reconstruction grid has a different row order")
    if any(len(item["images"]) != 10 for item in rows):
        raise ValueError("Every reconstruction row must contain ten images")
    for row in rows:
        if row["scales"] is not None and len(row["scales"]) != 10:
            raise ValueError("Every reconstruction annotation row needs ten values")

    # Reserve a dedicated inter-row gutter for the leakage percentages printed
    # below noisy reconstructions.  This keeps each value visually attached to
    # its own row instead of crowding the following row.
    # The native width is intentionally large to preserve ten face columns.
    # Source typography is sized for the 0.331 scale factor used when the PDF
    # is inserted at the manuscript's 458-point text width.
    fig = plt.figure(figsize=(19.2, 11.0))
    grid = fig.add_gridspec(
        len(rows),
        11,
        width_ratios=[5.55] + [1.0] * 10,
        height_ratios=[1.0] * len(rows),
        left=0.006,
        right=0.997,
        bottom=0.025,
        top=0.995,
        wspace=0.035,
        hspace=0.19,
    )
    for row_index, row in enumerate(rows):
        label_axis = fig.add_subplot(grid[row_index, 0])
        label_axis.set_facecolor(row["color"])
        label_axis.set_xticks([])
        label_axis.set_yticks([])
        for spine in label_axis.spines.values():
            spine.set_visible(False)
        label_parts = [part.strip() for part in row["label"].split("|")]
        label_axis.text(
            0.025,
            0.94,
            label_parts[0],
            transform=label_axis.transAxes,
            ha="left",
            va="top",
            fontsize=23.0,
            fontweight="bold",
        )
        if len(label_parts) > 1:
            label_axis.text(
                0.025,
                0.75,
                label_parts[1],
                transform=label_axis.transAxes,
                ha="left",
                va="top",
                fontsize=22.0,
                fontweight="bold",
            )
        equations = row["math"]
        if len(equations) == 1:
            label_axis.text(
                0.025,
                0.34,
                equations[0],
                transform=label_axis.transAxes,
                ha="left",
                va="center",
                fontsize=25.0,
            )
        else:
            label_axis.text(
                0.025,
                0.39,
                equations[0],
                transform=label_axis.transAxes,
                ha="left",
                va="center",
                fontsize=21.5,
            )
            label_axis.text(
                0.025,
                0.12,
                equations[1],
                transform=label_axis.transAxes,
                ha="left",
                va="center",
                fontsize=21.5,
            )
        for column, image in enumerate(row["images"]):
            axis = fig.add_subplot(grid[row_index, column + 1])
            axis.imshow(image, interpolation="bicubic")
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row["scales"] is not None:
                axis.text(
                    0.5,
                    -0.075,
                    row["scales"][column],
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=21.5,
                    clip_on=False,
                )
    outputs[stem] = save_figure(fig, subdir, stem)
    plt.close(fig)


def render_reconstruction_noise_sweep(
    records: list[dict[str, Any]], outputs: dict[str, Any]
) -> None:
    """Render ten replayed faces ordered by the saved leakage estimates."""

    metadata_path = RECONSTRUCTION_SWEEP_DIR / "metadata.json"
    if not metadata_path.is_file():
        print(
            "Skipping leakage-ordered reconstructions; missing replay metadata: "
            f"{metadata_path}"
        )
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_rows = [
        "original",
        "c_gdp_clean",
        "c_gdp_private",
        "aa_anchor_mp",
        "aa_anchor_op",
        "aa_anchor_am",
    ]
    if metadata.get("row_order") != expected_rows:
        raise ValueError("The reconstruction replay row order differs")
    private_schedule = list(metadata.get("private_schedule", []))
    anchor_schedule = dict(metadata.get("anchor_schedule", {}))
    if len(private_schedule) != 10 or set(anchor_schedule) != {"mp", "op", "am"}:
        raise ValueError("The reconstruction replay must contain ten columns")
    if any(len(anchor_schedule[key]) != 10 for key in ("mp", "op", "am")):
        raise ValueError("Every anchor attack must contain ten replayed columns")
    private_leakage = [
        float(item["cross_image_identity_top1"]) for item in private_schedule
    ]
    anchor_leakage = {
        key: [
            float(item["cross_image_identity_top1"])
            for item in anchor_schedule[key]
        ]
        for key in ("mp", "op", "am")
    }
    for key, values in {"private": private_leakage, **anchor_leakage}.items():
        if not all(a > b for a, b in zip(values, values[1:])):
            raise ValueError(f"{key} linkage is not strictly decreasing")

    def load_row(key: str) -> list[np.ndarray]:
        paths = [
            RECONSTRUCTION_SWEEP_DIR / "tiles" / key / f"col_{column:02d}.png"
            for column in range(10)
        ]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing reconstruction tiles: {missing}")
        images = [plt.imread(path)[..., :3] for path in paths]
        shapes = {image.shape for image in images}
        if len(shapes) != 1:
            raise ValueError(f"Reconstruction row {key} has inconsistent image sizes")
        return images

    private_labels = [f"{100.0 * value:.1f}%" for value in private_leakage]
    anchor_labels = {
        key: [f"{100.0 * value:.1f}%" for value in anchor_leakage[key]]
        for key in ("mp", "op", "am")
    }
    clean_c_gdp_linkage = attack_linkage(
        record_one(records, "c_gdp", "clean"),
        "C-exact",
        "cross_image_identity",
    )
    clean_c_gdp_labels = [f"{100.0 * clean_c_gdp_linkage:.1f}%"] * 10
    clean_aa_linkages = [anchor_leakage[key][0] for key in ("mp", "op", "am")]
    if not all(
        math.isclose(value, clean_c_gdp_linkage, rel_tol=0.0, abs_tol=1e-12)
        for value in clean_aa_linkages
    ):
        raise ValueError(
            "The zero-anchor-noise AA-I-GDP leakage does not match clean C-GDP"
        )
    scales = {
        "original": None,
        "c_gdp_clean": clean_c_gdp_labels,
        "c_gdp_private": private_labels,
        "aa_anchor_mp": anchor_labels["mp"],
        "aa_anchor_op": anchor_labels["op"],
        "aa_anchor_am": anchor_labels["am"],
    }
    rows = [
        {
            **template,
            "images": load_row(template["key"]),
            "scales": scales[template["key"]],
        }
        for template in reconstruction_row_templates()
    ]
    render_reconstruction_grid(
        rows,
        outputs,
        stem="celeba_p010_reconstruction_leakage_ordered",
    )


def render_identity_reconstruction_sweeps(
    records: list[dict[str, Any]], outputs: dict[str, Any]
) -> None:
    """Render ten fixed-identity supplementary reconstruction grids."""

    metadata_path = IDENTITY_RECONSTRUCTION_SWEEP_DIR / "metadata.json"
    if not metadata_path.is_file():
        print(
            "Skipping fixed-identity reconstructions; missing replay metadata: "
            f"{metadata_path}"
        )
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_rows = [item["key"] for item in reconstruction_row_templates()]
    if metadata.get("row_order") != expected_rows:
        raise ValueError("The fixed-identity replay row order differs")
    identity_sweeps = list(metadata.get("identity_sweeps", []))
    if len(identity_sweeps) != 10:
        raise ValueError("The fixed-identity replay must contain ten identities")
    diagonal_audit = list(metadata.get("diagonal_audit", []))
    if (
        len(diagonal_audit) != 10
        or not metadata.get("all_diagonal_rows_bitwise_identical", False)
        or not all(
            item.get("all_rows_bitwise_identical", False)
            for item in diagonal_audit
        )
    ):
        raise ValueError("The mixed-grid diagonal audit did not pass")

    private_leakage = [
        float(value)
        for value in metadata["private_cross_image_identity_top1"]
    ]
    anchor_leakage = {
        key: [float(value) for value in metadata["anchor_cross_image_identity_top1"][key]]
        for key in ("mp", "op", "am")
    }
    if len(private_leakage) != 10 or any(
        len(anchor_leakage[key]) != 10 for key in ("mp", "op", "am")
    ):
        raise ValueError("The fixed-identity leakage schedules need ten columns")
    for key, values in {"private": private_leakage, **anchor_leakage}.items():
        if not all(left > right for left, right in zip(values, values[1:])):
            raise ValueError(f"{key} linkage is not strictly decreasing")

    clean_c_gdp_linkage = attack_linkage(
        record_one(records, "c_gdp", "clean"),
        "C-exact",
        "cross_image_identity",
    )
    clean_aa_linkages = [anchor_leakage[key][0] for key in ("mp", "op", "am")]
    if not all(
        math.isclose(value, clean_c_gdp_linkage, rel_tol=0.0, abs_tol=1e-12)
        for value in clean_aa_linkages
    ):
        raise ValueError(
            "The zero-anchor-noise AA-I-GDP leakage does not match clean C-GDP"
        )
    scales = {
        "original": None,
        "c_gdp_clean": [f"{100.0 * clean_c_gdp_linkage:.1f}%"] * 10,
        "c_gdp_private": [f"{100.0 * value:.1f}%" for value in private_leakage],
        "aa_anchor_mp": [
            f"{100.0 * value:.1f}%" for value in anchor_leakage["mp"]
        ],
        "aa_anchor_op": [
            f"{100.0 * value:.1f}%" for value in anchor_leakage["op"]
        ],
        "aa_anchor_am": [
            f"{100.0 * value:.1f}%" for value in anchor_leakage["am"]
        ],
    }

    for identity_slot, identity in enumerate(identity_sweeps):
        if int(identity.get("identity_slot", -1)) != identity_slot:
            raise ValueError("The fixed-identity sweep order differs")
        query = dict(identity.get("query", {}))
        if int(query.get("identity", -1)) < 0:
            raise ValueError(f"Identity slot {identity_slot} lacks query provenance")
        columns = list(identity.get("columns", []))
        if len(columns) != 10 or any(
            int(item.get("column", -1)) != column
            or int(item.get("identity_slot", -1)) != identity_slot
            for column, item in enumerate(columns)
        ):
            raise ValueError(
                f"Identity slot {identity_slot} does not contain ten ordered columns"
            )

        def load_row(key: str) -> list[np.ndarray]:
            paths = [
                IDENTITY_RECONSTRUCTION_SWEEP_DIR
                / "identities"
                / f"identity_{identity_slot:02d}"
                / "tiles"
                / key
                / f"col_{column:02d}.png"
                for column in range(10)
            ]
            missing = [path for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Missing identity {identity_slot} reconstruction tiles: {missing}"
                )
            images = [plt.imread(path)[..., :3] for path in paths]
            shapes = {image.shape for image in images}
            if len(shapes) != 1:
                raise ValueError(
                    f"Identity {identity_slot} row {key} has inconsistent image sizes"
                )
            return images

        rows = [
            {
                **template,
                "images": load_row(template["key"]),
                "scales": scales[template["key"]],
            }
            for template in reconstruction_row_templates()
        ]
        render_reconstruction_grid(
            rows,
            outputs,
            stem=f"celeba_reconstructions_identity_{identity_slot + 1:02d}",
            subdir="reconstruction/identity_specific",
        )


def write_tidy_source(records: list[dict[str, Any]]) -> Path:
    path = DATA_DIR / "celeba_p010_attack_level_source.csv"
    fields = [
        "run_id", "n_participants", "protocol", "run_role", "draw", "lambda",
        "private_lambda", "noise_sigma", "anchor_v", "anchor_noise_sigma",
        "test_balanced_accuracy", "test_accuracy", "test_macro_f1", "test_auroc",
        "runtime_seconds", "gpm_converged", "gpm_iterations", "gpm_termination_reason",
        "gpm_elapsed_seconds", "attack", "attack_role", "best_attack",
        "cross_image_identity_top1", "cross_image_identity_ci95_low",
        "cross_image_identity_ci95_high", "source_image_top1",
        "source_image_ci95_low", "source_image_ci95_high", "pixel_nrmse",
        "pixel_ssim", "pixel_psnr", "reduced_nrmse", "rotation_relative_error",
        "translation_relative_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            gpm = row.get("gpm") or {}
            best = (row.get("best_attack") or {}).get("name")
            for item in row.get("attacks", []):
                metrics = item["metrics"]
                face = metrics["face_linkage"]
                cross = face["cross_image_identity"]
                source = face["source_image"]
                pixel = metrics["pixel"]
                reduced = metrics["reduced"]
                writer.writerow(
                    {
                        "run_id": row["run_id"],
                        "n_participants": row["n_participants"],
                        "protocol": row["protocol"],
                        "run_role": row["run_role"],
                        "draw": row["draw"],
                        "lambda": row["lambda"],
                        "private_lambda": row["private_lambda"],
                        "noise_sigma": row["noise_sigma"],
                        "anchor_v": row["anchor_v"],
                        "anchor_noise_sigma": row["anchor_noise_sigma"],
                        "test_balanced_accuracy": row["utility"]["test_balanced_accuracy"],
                        "test_accuracy": row["utility"]["test_accuracy"],
                        "test_macro_f1": row["utility"]["test_macro_f1"],
                        "test_auroc": row["utility"]["test_auroc"],
                        "runtime_seconds": row["runtime_seconds"],
                        "gpm_converged": "" if not gpm else gpm["converged"],
                        "gpm_iterations": "" if not gpm else gpm["iterations"],
                        "gpm_termination_reason": "" if not gpm else gpm["termination_reason"],
                        "gpm_elapsed_seconds": "" if not gpm else gpm["elapsed_seconds"],
                        "attack": item["name"],
                        "attack_role": item["role"],
                        "best_attack": item["name"] == best,
                        "cross_image_identity_top1": cross["top1"],
                        "cross_image_identity_ci95_low": cross["ci95_identity_cluster"][0],
                        "cross_image_identity_ci95_high": cross["ci95_identity_cluster"][1],
                        "source_image_top1": source["top1"],
                        "source_image_ci95_low": source["ci95_identity_cluster"][0],
                        "source_image_ci95_high": source["ci95_identity_cluster"][1],
                        "pixel_nrmse": pixel["nrmse"],
                        "pixel_ssim": pixel["ssim"],
                        "pixel_psnr": pixel["psnr_db"],
                        "reduced_nrmse": reduced["nrmse"],
                        "rotation_relative_error": metrics["rotation_relative_error"],
                        "translation_relative_error": metrics["translation_relative_error"],
                    }
                )
    return path


def write_plot_data(
    leakage: dict[str, dict[str, Any]], utility: dict[str, Any]
) -> Path:
    path = DATA_DIR / "celeba_p010_plot_coordinates.csv"
    fields = ["figure_family", "metric", "series", "noise_scale", "estimate", "lower", "upper"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in METRICS:
            for key in ATTACK_SERIES:
                fit = leakage[metric][key]["fit"]
                for x, estimate, low, high in zip(fit["x"], fit["estimate"], fit["lower"], fit["upper"]):
                    writer.writerow({"figure_family": "linkage_vs_noise", "metric": metric,
                                     "series": key, "noise_scale": x, "estimate": estimate,
                                     "lower": low, "upper": high})
        for key in UTILITY_SERIES:
            fit = utility[key]["fit"]
            for x, estimate, low, high in zip(fit["x"], fit["estimate"], fit["lower"], fit["upper"]):
                writer.writerow({"figure_family": "utility_vs_noise", "metric": "test_balanced_accuracy",
                                 "series": key, "noise_scale": x, "estimate": estimate,
                                 "lower": low, "upper": high})
        for metric in METRICS:
            for attack_key, utility_key in (("private", "private"), ("op", "anchor")):
                lf = leakage[metric][attack_key]["fit"]
                uf = utility[utility_key]["fit"]
                for x, leak, acc in zip(lf["x"], lf["estimate"], uf["estimate"]):
                    writer.writerow({"figure_family": "privacy_utility", "metric": metric,
                                     "series": attack_key, "noise_scale": x,
                                     "estimate": f"leakage={leak:.17g};accuracy={acc:.17g}",
                                     "lower": "", "upper": ""})
    return path


def write_readme(metadata: dict[str, Any]) -> None:
    text = f"""# CelebA p=10 reproducible figure bundle

This folder reproduces the CelebA participant-count-sweep figures for **p=10**.
The authoritative aggregate contains **207 fits**: two clean controls, 100
C-GDP private-data-noise draws, 100 AA-I-GDP anchor-noise draws, four endpoint
controls, and one separate AA-I-GDP v=40 equivalence control.

## Metrics

- Utility: held-out CelebA `Smiling` test balanced accuracy; chance is 50%.
- Primary privacy metric: cross-image identity linkage, using a different
  reserved image of the same identity in each ten-way gallery; chance is 10%.
## Reproduction

Open `notebook/CelebA_p010_figures.ipynb` in Colab or Jupyter and run all cells.
The notebook reads the source JSON in `data/` and recreates the plots in
`figures/`.  It uses 100 random observations per noisy family, a fixed cubic
regression spline with five interior knots, and {BOOTSTRAP_RESAMPLES:,}
paired-draw bootstrap resamples for pointwise 95% conditional bands.
Deterministic endpoint controls are excluded from the quantitative plots and
the fits; they are used only in the qualitative endpoint comparison.

The plot x-axis uses the realized private-data standard deviation `sigma`,
not the stored multiplier `lambda`.  For presentation, its upper bound is
rounded to {metadata['private_sigma_display_upper']:.1f}; the exact retained
endpoint is {metadata['private_sigma_upper']:.12g}.  Anchor noise uses the
absolute scale `v` from 0 to {metadata['anchor_v_upper']:.12g}.  The v=40
control is retained in the source data but excluded from calibrated-range
curve fitting.

PNG files are intended for preview, PDFs for manuscript inclusion, and SVGs
remain editable.  `SHA256SUMS.txt` records all bundle files except itself.

The qualitative reconstruction collection uses a deterministic replay of ten
completed specifications in one large MNIST-style grid.  Each column contains
a distinct source face paired across rows.  Within every noisy row, the first
column is the zero-noise control and the remaining columns are ordered by
decreasing saved cross-image identity-linkage accuracy.  The rightmost image is
the least-protective completed draw no better than the 10% chance rate when one
exists; otherwise it is the closest attained result.  The row annotations show
the saved leakage accuracy, not the noise scale.  No reconstruction is
interpolated or synthesized.

When `data/reconstruction_leakage_ordered_by_identity_v1` is present, the same
renderer additionally produces ten supplementary grids under
`figures/reconstruction/identity_specific/`.  Each grid fixes one of the ten
mixed-grid identities across the same ten leakage-ordered columns.  Its
identity-slot/column diagonal is hash-locked to the corresponding mixed-grid
tile.
"""
    (BUNDLE / "README.md").write_text(text, encoding="utf-8")


def write_sha256_manifest() -> None:
    path = BUNDLE / "SHA256SUMS.txt"
    rows = []
    for item in sorted(BUNDLE.rglob("*")):
        if item.is_file() and item != path:
            rows.append(f"{sha256_file(item)}  {item.relative_to(BUNDLE).as_posix()}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    configure_matplotlib()
    outputs: dict[str, Any] = {}
    render_reconstruction_noise_sweep(records, outputs)
    render_identity_reconstruction_sweeps(records, outputs)
    metadata = {
        "dataset": "CelebA",
        "participant_count": 10,
        "source_record_count": len(records),
        "source_sha256": sha256_file(RAW_JSON),
        "output_files": outputs,
    }
    metadata_path = DATA_DIR / "celeba_reconstruction_figure_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_sha256_manifest()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
