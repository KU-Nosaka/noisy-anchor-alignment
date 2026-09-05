"""Build updated CelebA p=10 supplemental quantitative figures.

The authoritative quantitative input is the 407-record combined JSON aggregate.
Each noisy family contributes 100 original random draws and 100 matching
supplemental draws. Deterministic endpoints and the v=40 equivalence control are
retained in the source table but excluded from fits and displayed observations.
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
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


try:
    SCRIPT_PATH = Path(__file__).resolve()
except NameError:  # Jupyter/Colab code cells do not define ``__file__``.
    SCRIPT_PATH = Path.cwd() / "render_celeba_p010_figures.py"

ROOT = SCRIPT_PATH.parents[1]
BUNDLE = Path(
    os.environ.get(
        "REPRO_CELEBA_P010_OUT",
        str(ROOT / "build" / "renderers" / "celeba_p010_quantitative"),
    )
).expanduser().resolve()
DATA_DIR = BUNDLE / "data"
FIGURE_DIR = BUNDLE / "figures"
RAW_JSON = ROOT / "results" / "celeba" / "p010_combined_407_results.json.gz"
RECONSTRUCTION_SWEEP_DIR = DATA_DIR / "reconstruction_leakage_ordered_v2"
IDENTITY_RECONSTRUCTION_SWEEP_DIR = (
    DATA_DIR / "reconstruction_leakage_ordered_by_identity_v1"
)

BOOTSTRAP_RESAMPLES = 10_000
GRID_SIZE = 201
LINKAGE_CHANCE = 0.10
BALANCED_CHANCE = 0.50
PRIVATE_DISPLAY_UPPER = 3.0
PRIVATE_FIT_ROLES = {"random", "supplemental_private_sigma_1p5_3p0_v1"}
ANCHOR_FIT_ROLES = {"random", "supplemental_anchor_v_0_0p05_v1"}
FIT_OBSERVATIONS_PER_FAMILY = 200

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
        "label": "C-GDP (private-data noise)",
        "noise_field": "noise_sigma",
        "noise_symbol": r"$\sigma$",
        "color": "#CC79A7",
    },
    "mp": {
        "protocol": "pa_i_gdp",
        "attack": "PA-MP",
        "label": "NAA-GDP: MP attack",
        "noise_field": "anchor_noise_sigma",
        "noise_symbol": r"$v$",
        "color": "#0072B2",
    },
    "op": {
        "protocol": "pa_i_gdp",
        "attack": "PA-OP",
        "label": "NAA-GDP: OP attack",
        "noise_field": "anchor_noise_sigma",
        "noise_symbol": r"$v$",
        "color": "#E69F00",
    },
    "am": {
        "protocol": "pa_i_gdp",
        "attack": "PA-AM",
        "label": "NAA-GDP: AM attack",
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
        "label": "NAA-GDP",
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
    records = list(payload["records"])
    if payload.get("record_count") != 407 or len(records) != 407:
        raise ValueError("Expected exactly 407 authoritative p=10 records")
    if {int(row["n_participants"]) for row in records} != {10}:
        raise ValueError("The aggregate is not exclusively the p=10 condition")
    expected = {
        ("c_gdp", "clean"): 1,
        ("i_gdp", "clean"): 1,
        ("c_gdp_an", "random"): 100,
        ("pa_i_gdp", "random"): 100,
        ("c_gdp_an", "supplemental_private_sigma_1p5_3p0_v1"): 100,
        ("pa_i_gdp", "supplemental_anchor_v_0_0p05_v1"): 100,
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
        return 3.0
    return 0.25


def fit_roles(series_key: str) -> set[str]:
    return PRIVATE_FIT_ROLES if series_key == "private" else ANCHOR_FIT_ROLES


def attack_series(
    records: list[dict[str, Any]], series_key: str, metric: str
) -> dict[str, Any]:
    definition = ATTACK_SERIES[series_key]
    selected = [
        row
        for row in records
        if row["protocol"] == definition["protocol"]
        and row["run_role"] in fit_roles(series_key) | {"endpoint_zero", "endpoint_upper"}
    ]
    by_role = {
        role: [row for row in selected if row["run_role"] == role]
        for role in fit_roles(series_key) | {"endpoint_zero", "endpoint_upper"}
    }
    observations = [row for role in sorted(fit_roles(series_key)) for row in by_role[role]]
    if len(observations) != FIT_OBSERVATIONS_PER_FAMILY:
        raise ValueError(f"Expected 200 fit observations for {series_key}")
    x = np.asarray(
        [float(row[definition["noise_field"]]) for row in observations]
    )
    y = np.asarray(
        [attack_linkage(row, definition["attack"], metric) for row in observations]
    )
    upper = noise_upper(records, series_key)
    fit = conditional_spline(
        x,
        y,
        upper,
        seed=26072000 + 100 * list(METRICS).index(metric) + list(ATTACK_SERIES).index(series_key),
    )
    return {"definition": definition, "by_role": by_role, "observations": observations,
            "x": x, "y": y, "upper": upper, "fit": fit}


def utility_series(records: list[dict[str, Any]], series_key: str) -> dict[str, Any]:
    definition = UTILITY_SERIES[series_key]
    selected = [
        row
        for row in records
        if row["protocol"] == definition["protocol"]
        and row["run_role"] in fit_roles("private" if series_key == "private" else "anchor") | {"endpoint_zero", "endpoint_upper"}
    ]
    by_role = {
        role: [row for row in selected if row["run_role"] == role]
        for role in fit_roles("private" if series_key == "private" else "anchor") | {"endpoint_zero", "endpoint_upper"}
    }
    roles = fit_roles("private" if series_key == "private" else "anchor")
    observations = [row for role in sorted(roles) for row in by_role[role]]
    if len(observations) != FIT_OBSERVATIONS_PER_FAMILY:
        raise ValueError(f"Expected 200 fit utility observations for {series_key}")
    x = np.asarray(
        [float(row[definition["noise_field"]]) for row in observations]
    )
    y = np.asarray(
        [float(row["utility"]["test_balanced_accuracy"]) for row in observations]
    )
    upper = noise_upper(records, "private" if series_key == "private" else "anchor")
    fit = conditional_spline(x, y, upper, seed=26072100 + list(UTILITY_SERIES).index(series_key))
    return {"definition": definition, "by_role": by_role, "observations": observations,
            "x": x, "y": y, "upper": upper, "fit": fit}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.0,
            "legend.fontsize": 8.4,
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
    """Readable typography after a wide two-panel PDF is scaled to text width."""

    return {
        "font.size": 17.0,
        "axes.titlesize": 17.0,
        "axes.labelsize": 18.0,
        "legend.fontsize": 14.5,
        "xtick.labelsize": 15.5,
        "ytick.labelsize": 15.5,
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


def compact_linkage_legend_entries(
    axes: Iterable[plt.Axes],
) -> tuple[list[Any], list[str]]:
    """Return short, ordered legend entries for manuscript two-panel figures."""

    handles, labels = combined_legend_entries(axes)
    by_label = dict(zip(labels, handles))
    label_map = {
        "C-GDP (private-data noise)": "C-GDP (private-data noise)",
        "NAA-GDP: MP attack":
            "NAA-GDP: MP",
        "NAA-GDP: OP attack":
            "NAA-GDP: OP",
        "NAA-GDP: AM attack":
            "NAA-GDP: AM",
        "Individual observations": "Individual observations",
        "GPM converged":
            "Anchor noise: GPM converged",
        "500-iteration cap":
            "Anchor noise: 500-iteration cap",
        "C-GDP": "C-GDP",
        "AA-GDP": "AA-GDP",
    }
    order = (
        "C-GDP (private-data noise)",
        "Individual observations",
        "NAA-GDP: MP attack",
        "NAA-GDP: OP attack",
        "NAA-GDP: AM attack",
        "95% conditional band",
        "Random guess (10%)",
        "C-GDP",
        "AA-GDP",
    )
    selected_handles: list[Any] = []
    selected_labels: list[str] = []
    shared_proxies: dict[str, Any] = {
        "Individual observations": Line2D(
            [], [], linestyle="none", marker="o", markersize=5.5,
            markerfacecolor="#777777", markeredgewidth=0,
        ),
        "95% conditional band": Patch(
            facecolor="#777777", edgecolor="none", alpha=0.18
        ),
        "GPM converged": Line2D(
            [], [], linestyle="none", marker="o", markersize=5.5,
            markerfacecolor="#777777", markeredgewidth=0,
        ),
        "500-iteration cap": Line2D(
            [], [], linestyle="none", marker="o", markersize=5.5,
            markerfacecolor="none", markeredgecolor="#777777",
            markeredgewidth=0.8,
        ),
    }
    for label in order:
        if label in by_label:
            selected_handles.append(
                shared_proxies.get(label, by_label[label])
            )
            selected_labels.append(label_map.get(label, label))
    return selected_handles, selected_labels


def save_figure(
    fig: plt.Figure,
    subdir: str,
    stem: str,
    *,
    tight: bool = True,
) -> dict[str, str]:
    target = FIGURE_DIR / subdir
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for suffix in ("png", "pdf", "svg"):
        path = target / f"{stem}.{suffix}"
        kwargs: dict[str, Any] = {}
        if tight:
            kwargs["bbox_inches"] = "tight"
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
    show_guides: bool = True,
    overlay: bool = False,
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
        tick_step = 0.5 if compact else 0.25
        ax.set_xticks(np.arange(0.0, PRIVATE_DISPLAY_UPPER + 0.001, tick_step))
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks(np.arange(0.0, 101.0, 10.0))
    ax.set_xlabel(f"Noise Scale {definition['noise_symbol']}")
    ax.set_ylabel(METRICS[metric]["ylabel"])
    ax.set_title(title, loc="left", pad=8)
    color = definition["color"]
    observations = prepared["observations"]
    xs = np.asarray([float(row[definition["noise_field"]]) for row in observations])
    ys = np.asarray([attack_linkage(row, definition["attack"], metric) for row in observations])
    ax.fill_between(
        fit["x"], 100.0 * np.clip(fit["lower"], 0, 1), 100.0 * np.clip(fit["upper"], 0, 1),
        color=color, alpha=(0.09 if overlay else 0.15), linewidth=0,
        label="95% conditional band", zorder=1,
    )
    if definition["protocol"] == "pa_i_gdp":
        converged = np.asarray([bool(row["gpm"]["converged"]) for row in observations])
        ax.scatter(xs[converged], 100.0 * ys[converged],
                   s=(13 if overlay else 18), color=color,
                   alpha=(0.20 if overlay else 0.28),
                   linewidths=0, label="GPM converged", zorder=3)
        ax.scatter(xs[~converged], 100.0 * ys[~converged],
                   s=(15 if overlay else 21), facecolors="none",
                   edgecolors=color, alpha=(0.24 if overlay else 0.33),
                   linewidths=0.7,
                   label="500-iteration cap", zorder=3)
    else:
        ax.scatter(xs, 100.0 * ys, s=18, color=color, alpha=0.25, linewidths=0,
                   label="Individual observations", zorder=3)
    ax.plot(fit["x"], 100.0 * np.clip(fit["estimate"], 0, 1), color=color,
            linewidth=2.55, label=definition["label"], zorder=5)
    if show_guides:
        zero = by_role["endpoint_zero"][0]
        ax.axhline(100.0 * LINKAGE_CHANCE, color="#666666", linestyle=(0, (1.3, 2.2)),
                   linewidth=1.15, label="Random guess (10%)", zorder=2)
        baseline = endpoint_value(zero, definition, metric)
        ax.axhline(100.0 * baseline, color="#333333", linestyle=(0, (5.0, 2.0, 1.0, 2.0)),
                   linewidth=1.7, label=("C-GDP" if definition["protocol"] == "c_gdp_an"
                                         else "AA-GDP"), zorder=2)
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
        "private": "\nC-GDP (private-data noise)",
        "mp": "\nNAA-GDP under MP attack",
        "op": "\nNAA-GDP under OP attack",
        "am": "\nNAA-GDP under AM attack",
    }
    for metric, metric_spec in METRICS.items():
        for key in ATTACK_SERIES:
            fig, ax = plt.subplots(figsize=(7.25, 5.25))
            draw_linkage_panel(
                ax,
                prepared[metric][key],
                metric,
                panel_names[key],
            )
            fig.subplots_adjust(left=0.13, right=0.98, bottom=0.13, top=0.90)
            stem = f"celeba_{metric_spec['short']}_leakage_{'private_data_noise' if key == 'private' else 'anchor_noise_' + key}"
            outputs[stem] = save_figure(fig, "leakage_noise", stem)
            plt.close(fig)

        with mpl.rc_context(wide_figure_rc()):
            fig, axes = plt.subplots(
                1, 2, figsize=(13.6, 6.8), sharey=True
            )
            draw_linkage_panel(
                axes[0],
                prepared[metric]["private"],
                metric,
                "",
                compact=True,
            )
            for key in ("mp", "op", "am"):
                draw_linkage_panel(
                    axes[1],
                    prepared[metric][key],
                    metric,
                    "",
                    compact=True,
                    show_guides=(key == "mp"),
                    overlay=True,
                )
            for ax in axes:
                ax.set_ylabel("")
            fig.supylabel(
                METRICS[metric]["ylabel"],
                x=0.012,
                y=0.575,
                fontsize=mpl.rcParams["axes.labelsize"],
            )
            handles, labels = compact_linkage_legend_entries(axes)
            fig.legend(
                handles,
                labels,
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
                        "C-GDP (private-data noise)",
                        "NAA-GDP",
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
            stem = (
                f"celeba_{metric_spec['short']}_leakage_"
                "private_and_anchor_methods_two_panel"
            )
            outputs[stem] = save_figure(
                fig, "leakage_noise", stem, tight=False
            )
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
    observations = prepared["observations"]
    xs = np.asarray([float(row[definition["noise_field"]]) for row in observations])
    ys = np.asarray([float(row["utility"]["test_balanced_accuracy"]) for row in observations])
    ax.fill_between(fit["x"], 100.0 * fit["lower"], 100.0 * fit["upper"],
                    color=color, alpha=0.15, linewidth=0, label="95% conditional band", zorder=1)
    if definition["protocol"] == "pa_i_gdp":
        converged = np.asarray([bool(row["gpm"]["converged"]) for row in observations])
        ax.scatter(xs[converged], 100.0 * ys[converged], s=18, color=color, alpha=0.28,
                   linewidths=0, zorder=3)
        ax.scatter(xs[~converged], 100.0 * ys[~converged], s=21, facecolors="none",
                   edgecolors=color, alpha=0.34, linewidths=0.7, zorder=3)
    else:
        ax.scatter(xs, 100.0 * ys, s=18, color=color, alpha=0.25, linewidths=0,
                   label="Individual observations", zorder=3)
    ax.plot(fit["x"], 100.0 * fit["estimate"], color=color, linewidth=2.6,
            label=definition["label"], zorder=5)
    ax.axhline(100.0 * BALANCED_CHANCE, color="#888888", linestyle=(0, (1.3, 2.2)),
               linewidth=1.1, label="Balanced random guess (50%)", zorder=2)
    ax.axhline(100.0 * controls["c_gdp"], color="#333333", linestyle=(0, (1.2, 1.8)),
               linewidth=1.9, label="C-GDP", zorder=2)
    if definition["protocol"] == "pa_i_gdp":
        ax.axhline(100.0 * controls["aa_zero"], color="#D55E00",
                   linestyle=(0, (5.0, 2.0, 1.0, 2.0)), linewidth=1.7,
                   label="AA-GDP", zorder=2)
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
        "private": "C-GDP (private-data noise)",
        "anchor": "NAA-GDP",
    }
    for key in UTILITY_SERIES:
        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        draw_utility_panel(ax, prepared[key], controls, titles[key])
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.13, top=0.90)
        stem = f"celeba_smiling_balanced_accuracy_{'private_data_noise' if key == 'private' else 'anchor_noise'}"
        outputs[stem] = save_figure(fig, "utility_noise", stem)
        plt.close(fig)
    with mpl.rc_context(wide_figure_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.8), sharey=True)
        for ax, key in zip(axes, UTILITY_SERIES):
            draw_utility_panel(ax, prepared[key], controls, "", compact=True)
        axes[1].set_ylabel("")
        handles, labels = combined_legend_entries(axes)
        by_label = dict(zip(labels, handles))
        by_label["95% conditional band"] = Patch(
            facecolor="#777777", edgecolor="none", alpha=0.18
        )
        order = (
            "C-GDP (private-data noise)",
            "Individual observations",
            "NAA-GDP",
            "95% conditional band",
            "Balanced random guess (50%)",
            "C-GDP",
            "AA-GDP",
            "I-GDP",
        )
        handles = [by_label[label] for label in order if label in by_label]
        labels = [label for label in order if label in by_label]
        fig.legend(
            handles,
            labels,
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
            zip(axes, (titles["private"], titles["anchor"]))
        ):
            fig.text(
                ax.get_position().x0,
                0.94,
                f"({chr(ord('a') + panel_index)}) {title}",
                ha="left",
                va="top",
                fontsize=mpl.rcParams["axes.titlesize"],
            )
        stem = "celeba_smiling_balanced_accuracy_two_panel"
        outputs[stem] = save_figure(
            fig, "utility_noise", stem, tight=False
        )
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

    for attack_key, utility_key, color, marker, linestyle, label in (
        ("private", "private", "#0072B2", "o", "-", "C-GDP (private-data noise)"),
        ("op", "anchor", "#E69F00", "^", (0, (5.0, 2.0)), "NAA-GDP"),
    ):
        leak = leakage_prepared[metric][attack_key]
        util = utility_prepared[utility_key]
        random_leak = leak["observations"]
        random_util = util["observations"]
        by_id_util = {row["run_id"]: row for row in random_util}
        pairs = [(row, by_id_util[row["run_id"]]) for row in random_leak]
        lx = np.asarray([attack_linkage(a, leak["definition"]["attack"], metric) for a, _ in pairs])
        uy = np.asarray([float(u["utility"]["test_balanced_accuracy"]) for _, u in pairs])
        if attack_key == "op":
            converged = np.asarray([bool(a["gpm"]["converged"]) for a, _ in pairs])
            ax.scatter(100.0 * lx[converged], 100.0 * uy[converged], s=30, marker=marker,
                       color=color, alpha=0.48, linewidths=0, zorder=4)
            ax.scatter(100.0 * lx[~converged], 100.0 * uy[~converged], s=34, marker=marker,
                       facecolors="none", edgecolors=color, alpha=0.62, linewidths=0.9, zorder=4)
        else:
            ax.scatter(100.0 * lx, 100.0 * uy, s=30, marker=marker, color=color,
                       alpha=0.48, linewidths=0, zorder=4)
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
    ax.axhline(100.0 * controls["i_gdp"], color="#009E73",
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
    random = [row for row in records if row["protocol"] == "pa_i_gdp" and row["run_role"] in ANCHOR_FIT_ROLES]
    if len(random) != FIT_OBSERVATIONS_PER_FAMILY:
        raise ValueError(f"Expected 200 GPM diagnostic observations, found {len(random)}")
    v = np.asarray([float(row["anchor_v"]) for row in random])
    converged = np.asarray([bool(row["gpm"]["converged"]) for row in random])
    iterations = np.asarray([int(row["gpm"]["iterations"]) for row in random])
    residual = np.asarray([float(row["gpm"]["fixed_point_residual"]) for row in random])
    elapsed = np.asarray([float(row["gpm"]["elapsed_seconds"]) for row in random])
    with mpl.rc_context(wide_figure_rc()):
        fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.2))
        for ax, y, ylabel, log in (
            (axes[0], iterations, "GPM Iterations", False),
            (axes[1], residual, "Fixed-Point Residual", True),
            (axes[2], elapsed, "GPM Optimization Time (s)", False),
        ):
            style_axis(ax)
            ax.scatter(v[converged], y[converged], s=22, color="#0072B2", alpha=0.55,
                       linewidths=0, label="Converged")
            ax.scatter(v[~converged], y[~converged], s=24, facecolors="none",
                       edgecolors="#D55E00", alpha=0.55, linewidths=0.8,
                       label="500-iteration cap")
            ax.set_xlabel(r"Anchor-Noise Scale $v$")
            ax.set_xlim(0.0, 0.25)
            ax.set_ylabel(ylabel)
            if log:
                ax.set_yscale("log")
        dedupe_legend(axes[0], loc="lower left", bbox_to_anchor=(0.0, 1.01),
                      ncol=2, frameon=False, borderaxespad=0.0)
        fig.subplots_adjust(left=0.065, right=0.99, bottom=0.19, top=0.78, wspace=0.30)
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
            "label": "C-GDP  |  known-parameter inversion",
            "math": [
                r"$\widehat{\widetilde{\boldsymbol{X}}}_j=(\boldsymbol{Y}_j-\mathbf{1}_{n_j}\Psi_s^\top)\boldsymbol{O}_s^\top=\widetilde{\boldsymbol{X}}_j$",
                r"$\widehat{\boldsymbol{X}}_j=\widetilde{\boldsymbol{X}}_j\boldsymbol{F}^\dagger=\boldsymbol{X}_j\boldsymbol{F}\boldsymbol{F}^\dagger$",
            ],
            "color": "#EAF2F7",
        },
        {
            "key": "c_gdp_private",
            "label": "C-GDP (private-data noise)  |  known-parameter inversion",
            "math": [
                r"$\widehat{\widetilde{\boldsymbol{X}}}_j=(\boldsymbol{Y}_j-\mathbf{1}_{n_j}\Psi_s^\top)\boldsymbol{O}_s^\top=\widetilde{\boldsymbol{X}}_j+\sigma\boldsymbol{W}_j\boldsymbol{O}_s^\top$",
                r"$\widehat{\boldsymbol{X}}_j=\boldsymbol{X}_j\boldsymbol{F}\boldsymbol{F}^\dagger+\sigma\boldsymbol{W}_j\boldsymbol{O}_s^\top\boldsymbol{F}^\dagger$",
            ],
            "color": "#E8F4FA",
        },
        {
            "key": "aa_anchor_mp",
            "label": "NAA-GDP  |  MP attack",
            "math": [
                r"$\widehat{\widetilde{\boldsymbol{X}}}_j^{\mathrm{MP}}=(\boldsymbol{Y}_j-\mathbf{1}_{n_j}\boldsymbol{b}_j^\top)(\boldsymbol{A}^\top\overline{\boldsymbol{B}}_j)^\dagger$",
                r"$\widehat{\boldsymbol{X}}_j^{\mathrm{MP}}=\widehat{\widetilde{\boldsymbol{X}}}_j^{\mathrm{MP}}\boldsymbol{F}^\dagger,\quad v\to0:\ \widehat{\boldsymbol{X}}_j^{\mathrm{MP}}\to\boldsymbol{X}_j\boldsymbol{F}\boldsymbol{F}^\dagger$",
            ],
            "color": "#FBF4E7",
        },
        {
            "key": "aa_anchor_op",
            "label": "NAA-GDP  |  OP attack",
            "math": [
                r"$\widehat{\widetilde{\boldsymbol{X}}}_j^{\mathrm{OP}}=(\boldsymbol{Y}_j-\mathbf{1}_{n_j}\boldsymbol{b}_j^\top)[\Pi(\boldsymbol{A}^\top\overline{\boldsymbol{B}}_j)]^\top$",
                r"$\widehat{\boldsymbol{X}}_j^{\mathrm{OP}}=\widehat{\widetilde{\boldsymbol{X}}}_j^{\mathrm{OP}}\boldsymbol{F}^\dagger,\quad v\to0:\ \widehat{\boldsymbol{X}}_j^{\mathrm{OP}}\to\boldsymbol{X}_j\boldsymbol{F}\boldsymbol{F}^\dagger$",
            ],
            "color": "#FBF4E7",
        },
        {
            "key": "aa_anchor_am",
            "label": "NAA-GDP  |  alignment-map attack",
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
    fig = plt.figure(figsize=(19.2, 9.0))
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
        label_axis.text(
            0.025,
            0.82,
            row["label"],
            transform=label_axis.transAxes,
            ha="left",
            va="top",
            fontsize=12.0,
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
                fontsize=11.5,
            )
        else:
            label_axis.text(
                0.025,
                0.49,
                equations[0],
                transform=label_axis.transAxes,
                ha="left",
                va="center",
                fontsize=9.5,
            )
            label_axis.text(
                0.025,
                0.21,
                equations[1],
                transform=label_axis.transAxes,
                ha="left",
                va="center",
                fontsize=9.5,
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
                    fontsize=12.0,
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
            "The zero-anchor-noise AA-GDP leakage does not match clean C-GDP"
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
            "The zero-anchor-noise AA-GDP leakage does not match clean C-GDP"
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
    text = f"""# Updated CelebA p=10 supplemental quantitative figures

This folder renders updated CelebA figures for **p=10** from the authoritative
407-record combined aggregate. Each noisy family contains **200 fit
observations**: 100 original random draws and 100 matching supplemental draws.
The four endpoint controls and NAA-GDP v=40 equivalence control are excluded
from fits and are not displayed as endpoint markers.

## Metrics

- Utility: held-out CelebA `Smiling` test balanced accuracy; chance is 50%.
- Primary privacy metric: cross-image identity linkage, using a different
  reserved image of the same identity in each ten-way gallery; chance is 10%.
## Reproduction

Run `CelebA_p010_combined_figures.ipynb`, or execute
`python render_celeba_updated_figures.py` directly. The renderer reads the
authoritative JSON from `outputs/extra_results_drive/folder_1/` and recreates
the plots in `figures/`. It uses 200 observations per noisy family, a fixed cubic
regression spline with five interior knots, and {BOOTSTRAP_RESAMPLES:,}
multinomial paired-draw bootstrap resamples for pointwise 95% conditional bands.
Deterministic endpoint controls are excluded from the quantitative plots and
the fits; they are used only in the qualitative endpoint comparison.

The private-data fit and x-axis use sigma in [0, 3.0]. Anchor noise uses the
absolute scale v in [0, 0.25]. The identity-linkage display is divided into
two manuscript figures: private-data noise with MP, and OP with alignment map.

PNG files are intended for preview, PDFs for manuscript inclusion, and SVGs
remain editable.  `SHA256SUMS.txt` records all bundle files except itself.

This update is quantitative only; no reconstruction artifacts are generated.
"""
    (BUNDLE / "README.md").write_text(text, encoding="utf-8")


def write_sha256_manifest() -> None:
    path = BUNDLE / "SHA256SUMS.txt"
    rows = []
    for item in sorted(BUNDLE.rglob("*")):
        if item.is_file() and item != path and "__pycache__" not in item.parts:
            rows.append(f"{sha256_file(item)}  {item.relative_to(BUNDLE).as_posix()}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    configure_matplotlib()
    outputs: dict[str, Any] = {}
    tidy_path = write_tidy_source(records)
    leakage = render_linkage_figures(records, outputs)
    utility, controls = render_utility_figures(records, outputs)
    render_tradeoff_figures(records, leakage, utility, controls, outputs)
    render_gpm_diagnostics(records, outputs)
    plot_data_path = write_plot_data(leakage, utility)
    private_endpoint = record_one(records, "c_gdp_an", "endpoint_upper")
    anchor_endpoint = record_one(records, "pa_i_gdp", "endpoint_upper")
    equivalence = record_one(records, "pa_i_gdp", "i_gdp_equivalence_control")
    private_fit = [row for row in records if row["protocol"] == "c_gdp_an" and row["run_role"] in PRIVATE_FIT_ROLES]
    anchor_random = [row for row in records if row["protocol"] == "pa_i_gdp" and row["run_role"] in ANCHOR_FIT_ROLES]
    metadata = {
        "dataset": "CelebA",
        "participant_count": 10,
        "task": "Smiling",
        "record_count": len(records),
        "fit_observations_per_noisy_family": FIT_OBSERVATIONS_PER_FAMILY,
        "fit_roles": {"private": sorted(PRIVATE_FIT_ROLES), "anchor": sorted(ANCHOR_FIT_ROLES)},
        "private_sigma_upper": 3.0,
        "private_sigma_display_upper": PRIVATE_DISPLAY_UPPER,
        "private_lambda_upper": float(private_endpoint["private_lambda"]),
        "private_reduced_train_rms": float(private_endpoint["noise_sigma"]) / float(private_endpoint["private_lambda"]),
        "anchor_v_upper": 0.25,
        "anchor_equivalence_control_v": float(equivalence["anchor_v"]),
        "balanced_accuracy_chance": BALANCED_CHANCE,
        "linkage_chance": LINKAGE_CHANCE,
        "gpm_random_converged": sum(bool(row["gpm"]["converged"]) for row in anchor_random),
        "gpm_random_iteration_cap": sum(not bool(row["gpm"]["converged"]) for row in anchor_random),
        "fit_observation_counts": {"private": len(private_fit), "anchor": len(anchor_random)},
        "controls": controls,
        "metrics": {key: value["definition"] for key, value in METRICS.items()},
        "spline": {"degree": 3, "interior_knots": 5, "grid_size": GRID_SIZE,
                   "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                   "endpoint_controls_in_fit": False},
        "source_sha256": sha256_file(RAW_JSON),
        "endpoint_controls_displayed": False,
        "tidy_source": tidy_path.relative_to(BUNDLE).as_posix(),
        "plot_coordinates": plot_data_path.relative_to(BUNDLE).as_posix(),
        "output_files": outputs,
    }
    metadata_path = DATA_DIR / "celeba_p010_figure_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_readme(metadata)
    write_sha256_manifest()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
