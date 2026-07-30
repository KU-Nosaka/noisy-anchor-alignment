"""Render the completed CelebA participant-count sensitivity analysis.

For p in {10, 30, 50}, the analysis combines the 100 broad anchor-noise
observations over 0 < v < 0.25 with the 100 scale-paired focused observations
over 0 < v < 0.05.  Each focused observation reuses the broad observation's
draw-level randomness and differs only in amplitude.  The regression splines
therefore use all 200 observations, while the bootstrap resamples the 100 base
draw IDs and carries each broad/focused pair together.

The main figure compares the AA-I-GDP anchor-noise privacy--utility trajectory
across participant counts.  The companion GPM figure and CSV quantify the
numerical stopping transition using the same 200 observations per condition.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
OUTPUT_DIR = Path(
    os.environ.get(
        "REPRO_CELEBA_COUNT_OUT",
        str(REPO_ROOT / "build" / "renderers" / "celeba_participant_count"),
    )
).expanduser().resolve()
FIGURE_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"
RESULT_ROOT = REPO_ROOT / "results" / "celeba"
CORE_PATHS = {
    10: RESULT_ROOT / "p010_combined_407_results.json.gz",
    30: RESULT_ROOT / "p030_core_207_results.json.gz",
    50: RESULT_ROOT / "p050_core_207_results.json.gz",
}
SUPPLEMENT_PATHS = {
    10: RESULT_ROOT / "p010_combined_407_results.json.gz",
    30: RESULT_ROOT / "p030_low_v_100_results.json.gz",
    50: RESULT_ROOT / "p050_low_v_100_results.json.gz",
}
COMPLETION_PATHS = {
    30: RESULT_ROOT / "manifests" / "p030_low_v" / "completion_summary.json",
    50: RESULT_ROOT / "manifests" / "p050_low_v" / "completion_summary.json",
}

PARTICIPANT_COUNTS = (10, 30, 50)
BASE_DRAWS = 100
OBSERVATIONS_PER_PARTICIPANT_COUNT = 200
BOOTSTRAP_RESAMPLES = 10_000
GRID_SIZE = 251
ANCHOR_UPPER = 0.25
FOCUSED_UPPER = 0.05
LINKAGE_CHANCE = 0.10
BALANCED_CHANCE = 0.50

COLORS = {
    10: "#0072B2",
    30: "#E69F00",
    50: "#009E73",
    "chance": "#888888",
    "grid": "#D9D9D9",
    "cap": "#D55E00",
}
PARTICIPANT_STYLE = {
    10: {"marker": "o"},
    30: {"marker": "s"},
    50: {"marker": "^"},
}


def load_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def attack_linkage(record: dict[str, Any], attack_name: str = "PA-OP") -> float:
    attacks = {
        str(attack["name"]): attack for attack in record.get("attacks", [])
    }
    if attack_name not in attacks:
        best = record.get("best_attack") or {}
        if best.get("name") == attack_name:
            attacks[attack_name] = best
    try:
        return float(
            attacks[attack_name]["metrics"]["face_linkage"]
            ["cross_image_identity"]["top1"]
        )
    except KeyError as exc:
        raise ValueError(
            f"Missing {attack_name} linkage in run {record.get('run_id')}"
        ) from exc


def utility(record: dict[str, Any]) -> float:
    return float(record["utility"]["test_balanced_accuracy"])


def one(records: Iterable[dict[str, Any]], protocol: str, role: str) -> dict[str, Any]:
    matches = [
        row
        for row in records
        if row.get("protocol") == protocol and row.get("run_role") == role
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {protocol}/{role} record, found {len(matches)}"
        )
    return matches[0]


def load_condition(p: int) -> dict[str, Any]:
    core_path = CORE_PATHS[p]
    core_payload = load_json(core_path)
    core_all = list(core_payload["records"])
    if p == 10:
        if int(core_payload.get("record_count", -1)) != 407 or len(core_all) != 407:
            raise ValueError("p=10: combined stage is not the completed 407-record stage")
        core = [
            row
            for row in core_all
            if not str(row.get("run_role", "")).startswith("supplemental_")
        ]
    else:
        core = core_all
    if len(core) != 207:
        raise ValueError(f"p={p}: core stage is not the completed 207-record stage")
    if {int(row["n_participants"]) for row in core} != {p}:
        raise ValueError(f"p={p}: core participant count mismatch")

    broad = sorted(
        [
            row
            for row in core
            if row.get("protocol") == "pa_i_gdp"
            and row.get("run_role") == "random"
        ],
        key=lambda row: int(row["draw"]),
    )
    if len(broad) != BASE_DRAWS or [int(row["draw"]) for row in broad] != list(
        range(BASE_DRAWS)
    ):
        raise ValueError(f"p={p}: broad anchor stratum is not draws 0..99")

    supplement_payload = load_json(SUPPLEMENT_PATHS[p])
    supplement_all = list(supplement_payload["records"])
    focused = sorted(
        [
            row
            for row in supplement_all
            if row.get("protocol") == "pa_i_gdp"
            and row.get("run_role") == "supplemental_anchor_v_0_0p05_v1"
        ],
        key=lambda row: int(row["draw"]),
    )
    if len(focused) != BASE_DRAWS or [int(row["draw"]) for row in focused] != list(
        range(BASE_DRAWS)
    ):
        raise ValueError(f"p={p}: focused anchor stratum is not draws 0..99")
    if {int(row["n_participants"]) for row in focused} != {p}:
        raise ValueError(f"p={p}: focused participant count mismatch")

    if p in COMPLETION_PATHS:
        completion = load_json(COMPLETION_PATHS[p])
        status = completion.get("status", {})
        if int(status.get("completed", -1)) != 100 or int(
            status.get("planned", -1)
        ) != 100:
            raise ValueError(f"p={p}: focused stage is not complete")
        if not bool(completion.get("stage_runtime", {}).get("completed")):
            raise ValueError(f"p={p}: focused runtime is not marked complete")

    for broad_row, focused_row in zip(broad, focused):
        draw = int(broad_row["draw"])
        if int(focused_row["draw"]) != draw:
            raise ValueError(f"p={p}: focused draw ordering mismatch")
        if str(focused_row.get("paired_source_run_id")) != str(
            broad_row.get("run_id")
        ):
            raise ValueError(f"p={p}, draw={draw}: source run ID mismatch")
        broad_v = float(broad_row["anchor_v"])
        focused_v = float(focused_row["anchor_v"])
        if not np.isclose(focused_v, 0.2 * broad_v, rtol=0.0, atol=1.0e-15):
            raise ValueError(f"p={p}, draw={draw}: focused amplitude is not paired")
        if not bool(focused_row.get("paired_scale_only_design")):
            raise ValueError(f"p={p}, draw={draw}: scale-only pairing is not locked")

    rows = broad + focused
    if len(rows) != OBSERVATIONS_PER_PARTICIPANT_COUNT:
        raise AssertionError("Each participant count must provide 200 anchor rows")
    cluster = np.asarray([int(row["draw"]) for row in rows], dtype=int)
    if not np.array_equal(np.bincount(cluster, minlength=BASE_DRAWS), np.full(BASE_DRAWS, 2)):
        raise ValueError(f"p={p}: each base draw must have exactly two amplitudes")

    controls = {
        "c_gdp": {
            "label": "C-GDP",
            "utility": utility(one(core, "c_gdp", "clean")),
            "linkage": attack_linkage(one(core, "c_gdp", "clean"), "C-exact"),
        },
        "aa_zero": {
            "label": r"AA-I-GDP ($v=0$)",
            "utility": utility(one(core, "pa_i_gdp", "endpoint_zero")),
            "linkage": attack_linkage(
                one(core, "pa_i_gdp", "endpoint_zero"), "PA-OP"
            ),
        },
        "i_gdp": {
            "label": "I-GDP",
            "utility": utility(one(core, "i_gdp", "clean")),
        },
    }
    return {
        "p": p,
        "core_path": core_path,
        "supplement_path": SUPPLEMENT_PATHS[p],
        "core": core,
        "broad": broad,
        "focused": focused,
        "rows": rows,
        "cluster": cluster,
        "v": np.asarray([float(row["anchor_v"]) for row in rows]),
        "linkage": np.asarray([attack_linkage(row) for row in rows]),
        "utility": np.asarray([utility(row) for row in rows]),
        "converged": np.asarray(
            [bool(row["gpm"]["converged"]) for row in rows], dtype=bool
        ),
        "iterations": np.asarray(
            [int(row["gpm"]["iterations"]) for row in rows], dtype=int
        ),
        "controls": controls,
        "v_pt": float(rows[0]["gpm"]["ling_empirical_v_pt"]),
    }


def validate_cross_condition_pairing(conditions: dict[int, dict[str, Any]]) -> None:
    reference_broad = np.asarray(
        [float(row["anchor_v"]) for row in conditions[10]["broad"]]
    )
    reference_focused = np.asarray(
        [float(row["anchor_v"]) for row in conditions[10]["focused"]]
    )
    for p in PARTICIPANT_COUNTS[1:]:
        broad = np.asarray([float(row["anchor_v"]) for row in conditions[p]["broad"]])
        focused = np.asarray(
            [float(row["anchor_v"]) for row in conditions[p]["focused"]]
        )
        if not np.array_equal(broad, reference_broad) or not np.array_equal(
            focused, reference_focused
        ):
            raise ValueError("Anchor-noise amplitudes are not exactly paired across p")


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


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="both", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.2, width=0.7)


def spline_design(normalized_x: np.ndarray) -> np.ndarray:
    x = np.asarray(normalized_x, dtype=np.float64).reshape(-1)
    knots = np.linspace(0.0, 1.0, 7, dtype=np.float64)[1:-1]
    columns = [np.ones_like(x), x, x * x, x**3]
    columns.extend(np.maximum(x - knot, 0.0) ** 3 for knot in knots)
    return np.column_stack(columns)


def fit_coefficients(design: np.ndarray, outcome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gram = design.T @ design
    ridge = 1.0e-8 * max(float(np.trace(gram) / design.shape[1]), 1.0)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(gram + penalty, design.T @ outcome)
    return beta, penalty


def paired_cluster_spline(
    condition: dict[str, Any],
    cluster_counts: np.ndarray,
) -> dict[str, np.ndarray | float]:
    v = np.asarray(condition["v"], dtype=np.float64)
    leakage = np.asarray(condition["linkage"], dtype=np.float64)
    utility_values = np.asarray(condition["utility"], dtype=np.float64)
    clusters = np.asarray(condition["cluster"], dtype=int)
    if len(v) != 200 or cluster_counts.shape != (BOOTSTRAP_RESAMPLES, BASE_DRAWS):
        raise ValueError("Unexpected paired-cluster design")
    if not (
        np.all(np.isfinite(v))
        and np.all(np.isfinite(leakage))
        and np.all(np.isfinite(utility_values))
    ):
        raise ValueError("Spline inputs contain non-finite values")

    design = spline_design(v / ANCHOR_UPPER)
    noise_grid = np.linspace(0.0, ANCHOR_UPPER, GRID_SIZE, dtype=np.float64)
    grid_design = spline_design(noise_grid / ANCHOR_UPPER)
    leakage_beta, penalty = fit_coefficients(design, leakage)
    utility_beta, _ = fit_coefficients(design, utility_values)
    leakage_estimate = grid_design @ leakage_beta
    utility_estimate = grid_design @ utility_beta

    leakage_predictions = np.empty(
        (BOOTSTRAP_RESAMPLES, GRID_SIZE), dtype=np.float64
    )
    utility_predictions = np.empty(
        (BOOTSTRAP_RESAMPLES, GRID_SIZE), dtype=np.float64
    )
    for start in range(0, BOOTSTRAP_RESAMPLES, 256):
        stop = min(start + 256, BOOTSTRAP_RESAMPLES)
        observation_weights = cluster_counts[start:stop, clusters].astype(
            np.float64, copy=False
        )
        boot_gram = np.einsum(
            "bn,np,nq->bpq", observation_weights, design, design, optimize=True
        )
        leakage_rhs = np.einsum(
            "bn,np,n->bp", observation_weights, design, leakage, optimize=True
        )
        utility_rhs = np.einsum(
            "bn,np,n->bp", observation_weights, design, utility_values, optimize=True
        )
        regularized = boot_gram + penalty[None, :, :]
        leakage_boot_beta = np.linalg.solve(
            regularized, leakage_rhs[..., None]
        )[..., 0]
        utility_boot_beta = np.linalg.solve(
            regularized, utility_rhs[..., None]
        )[..., 0]
        leakage_predictions[start:stop] = leakage_boot_beta @ grid_design.T
        utility_predictions[start:stop] = utility_boot_beta @ grid_design.T

    linkage_lower, linkage_upper = np.quantile(
        leakage_predictions, [0.025, 0.975], axis=0
    )
    utility_lower, utility_upper = np.quantile(
        utility_predictions, [0.025, 0.975], axis=0
    )
    return {
        "noise": noise_grid,
        "linkage_estimate": leakage_estimate,
        "linkage_lower": linkage_lower,
        "linkage_upper": linkage_upper,
        "utility_estimate": utility_estimate,
        "utility_lower": utility_lower,
        "utility_upper": utility_upper,
        "observed_v_min": float(v.min()),
        "observed_v_max": float(v.max()),
    }


def logistic_midpoint(v: np.ndarray, stopped: np.ndarray) -> float:
    x = np.column_stack([np.ones(len(v)), np.asarray(v, dtype=np.float64)])
    y = np.asarray(stopped, dtype=np.float64)
    beta = np.zeros(2, dtype=np.float64)
    ridge = 1.0e-8
    for _ in range(100):
        eta = np.clip(x @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        weight = np.clip(probability * (1.0 - probability), 1.0e-8, None)
        hessian = x.T @ (weight[:, None] * x) + ridge * np.eye(2)
        gradient = x.T @ (y - probability) - ridge * beta
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if float(np.max(np.abs(step))) < 1.0e-10:
            break
    if beta[1] >= 0.0:
        raise ValueError("GPM stopping probability did not decrease with v")
    return float(-beta[0] / beta[1])


def first_descending_crossing(
    noise: np.ndarray, linkage: np.ndarray, target: float, *, lower: float, upper: float
) -> float:
    mask = (noise >= lower) & (noise <= upper)
    grid = noise[mask]
    values = linkage[mask]
    difference = values - target
    indices = np.flatnonzero(
        (difference[:-1] >= 0.0) & (difference[1:] <= 0.0)
    )
    if not len(indices):
        raise ValueError(f"No descending crossing for linkage target {target}")
    left = int(indices[0])
    y0, y1 = difference[left], difference[left + 1]
    weight = 0.0 if y0 == y1 else y0 / (y0 - y1)
    return float(grid[left] + weight * (grid[left + 1] - grid[left]))


def interpolate_on_noise(noise: np.ndarray, values: np.ndarray, point: float) -> float:
    return float(np.interp(point, noise, values))


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for suffix in ("png", "pdf", "svg"):
        path = FIGURE_DIR / f"{stem}.{suffix}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        outputs[suffix] = path.relative_to(OUTPUT_DIR).as_posix()
    plt.close(fig)
    return outputs


def draw_privacy_utility(
    conditions: dict[int, dict[str, Any]], fits: dict[int, dict[str, Any]]
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(8.25, 5.15))
    style_axis(ax)
    ax.set_xlim(8.0, 90.0)
    ax.set_ylim(48.0, 90.0)
    ax.set_xticks(np.arange(10.0, 91.0, 10.0))
    ax.set_yticks(np.arange(50.0, 91.0, 5.0))
    ax.set_xlabel("Identity Linkage Accuracy (%)")
    ax.set_ylabel("Smiling Test Balanced Accuracy (%)")
    ax.axvline(
        100.0 * LINKAGE_CHANCE,
        color=COLORS["chance"],
        linestyle=(0, (1.3, 2.2)),
        linewidth=1.05,
        zorder=1,
    )
    ax.axhline(
        100.0 * BALANCED_CHANCE,
        color="#AAAAAA",
        linestyle=(0, (1.3, 2.2)),
        linewidth=1.0,
        zorder=1,
    )

    for p in PARTICIPANT_COUNTS:
        condition = conditions[p]
        fit = fits[p]
        color = str(COLORS[p])
        marker = str(PARTICIPANT_STYLE[p]["marker"])
        ax.axhline(
            100.0 * float(condition["controls"]["i_gdp"]["utility"]),
            color=color,
            linestyle=(0, (5.5, 2.2)),
            linewidth=1.45,
            alpha=0.90,
            zorder=2,
        )
        ax.scatter(
            100.0 * np.asarray(condition["linkage"]),
            100.0 * np.asarray(condition["utility"]),
            s=21,
            marker=marker,
            color=color,
            alpha=0.48,
            linewidths=0,
            zorder=5,
        )
        ax.scatter(
            100.0 * float(condition["controls"]["c_gdp"]["linkage"]),
            100.0 * float(condition["controls"]["c_gdp"]["utility"]),
            s=82,
            marker="*",
            facecolor=color,
            edgecolor="#222222",
            linewidth=0.55,
            zorder=7,
        )
        ax.scatter(
            100.0 * float(condition["controls"]["aa_zero"]["linkage"]),
            100.0 * float(condition["controls"]["aa_zero"]["utility"]),
            s=58,
            marker="D",
            facecolor="none",
            edgecolor=color,
            linewidth=1.35,
            zorder=8,
        )
        band_x = 100.0 * np.concatenate(
            [
                np.clip(np.asarray(fit["linkage_lower"]), 0.0, 1.0),
                np.clip(np.asarray(fit["linkage_upper"]), 0.0, 1.0)[::-1],
            ]
        )
        band_y = 100.0 * np.concatenate(
            [
                np.clip(np.asarray(fit["utility_lower"]), 0.0, 1.0),
                np.clip(np.asarray(fit["utility_upper"]), 0.0, 1.0)[::-1],
            ]
        )
        ax.fill(band_x, band_y, color=color, alpha=0.055, linewidth=0, zorder=1)
        ax.plot(
            100.0 * np.clip(np.asarray(fit["linkage_estimate"]), 0.0, 1.0),
            100.0 * np.clip(np.asarray(fit["utility_estimate"]), 0.0, 1.0),
            color=color,
            linewidth=1.65,
            alpha=0.42,
            zorder=4,
        )

    participant_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[p],
            marker=PARTICIPANT_STYLE[p]["marker"],
            linestyle="none",
            markersize=5.4,
            label=rf"$p={p}$",
        )
        for p in PARTICIPANT_COUNTS
    ]
    method_handles = [
        Line2D(
            [0], [0], color="#444444", linewidth=1.65, alpha=0.42,
            label="AA-I-GDP (anchor noise): OP attack"
        ),
        Line2D(
            [0], [0], color="#444444", marker="*", markeredgecolor="#222222",
            linestyle="none", markersize=8.5, label="C-GDP"
        ),
        Line2D(
            [0], [0], color="#444444", marker="D", markerfacecolor="none",
            linestyle="none", markersize=5.8, label=r"AA-I-GDP ($v=0$)"
        ),
        Line2D(
            [0], [0], color="#444444", linestyle=(0, (5.5, 2.2)),
            linewidth=1.6, label="I-GDP"
        ),
        Line2D(
            [0], [0], color=COLORS["chance"], linestyle=(0, (1.3, 2.2)),
            linewidth=1.05, label="Chance levels"
        ),
    ]
    fig.legend(
        handles=participant_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.115),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.8,
    )
    fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=5,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.05,
    )
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.31, top=0.97)
    return save_figure(fig, "celeba_privacy_utility_by_participant_count")


def draw_gpm(conditions: dict[int, dict[str, Any]], summaries: dict[int, dict[str, Any]]) -> dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(14.25, 4.55), sharex=True, sharey=True)
    for ax, p in zip(axes, PARTICIPANT_COUNTS):
        condition = conditions[p]
        summary = summaries[p]
        v = np.asarray(condition["v"])
        iterations = np.asarray(condition["iterations"])
        stopped = np.asarray(condition["converged"])
        focused = np.arange(len(v)) >= BASE_DRAWS
        style_axis(ax)
        for mask, marker in ((~focused, "o"), (focused, "s")):
            ax.scatter(
                v[mask & stopped], iterations[mask & stopped], s=23,
                marker=marker, color=COLORS[p], alpha=0.58, linewidths=0, zorder=3
            )
            ax.scatter(
                v[mask & ~stopped], iterations[mask & ~stopped], s=25,
                marker=marker, facecolors="none", edgecolors=COLORS["cap"],
                alpha=0.58, linewidths=0.8, zorder=3
            )
        ax.axvline(
            float(summary["v_50_logistic"]), color="#222222",
            linestyle=(0, (5.0, 2.0)), linewidth=1.35, zorder=2
        )
        ax.axvline(
            float(summary["v_pt"]), color="#666666",
            linestyle=(0, (1.4, 2.0)), linewidth=1.35, zorder=2
        )
        ax.set_xlim(0.0, ANCHOR_UPPER)
        ax.set_xticks(np.arange(0.0, ANCHOR_UPPER + 0.001, 0.05))
        ax.set_ylim(0.0, 525.0)
        ax.set_yticks(np.arange(0.0, 501.0, 100.0))
        ax.set_xlabel("Anchor-Noise Scale $v$")
        ax.set_title(f"({chr(97 + PARTICIPANT_COUNTS.index(p))}) $p={p}$", loc="left", pad=8)
    axes[0].set_ylabel("GPM Iterations")
    handles = [
        Line2D([], [], color="#555555", marker="o", linestyle="none", markersize=5.4,
               label=r"Broad draw ($0<v<0.25$)"),
        Line2D([], [], color="#555555", marker="s", linestyle="none", markersize=5.4,
               label=r"Focused draw ($0<v<0.05$)"),
        Line2D([], [], color="#555555", marker="o", linestyle="none", markersize=5.4,
               label="Stopped before cap"),
        Line2D([], [], color=COLORS["cap"], marker="o", markerfacecolor="none",
               linestyle="none", markersize=5.6, label="Reached 500-iteration cap"),
        Line2D([], [], color="#222222", linestyle=(0, (5.0, 2.0)), linewidth=1.35,
               label=r"Logistic midpoint $v_{50}$"),
        Line2D([], [], color="#666666", linestyle=(0, (1.4, 2.0)), linewidth=1.35,
               label=r"Empirical reference $v_{\mathrm{PT}}$"),
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3,
        frameon=False, handlelength=2.7, columnspacing=1.35
    )
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.30, top=0.92, wspace=0.10)
    return save_figure(fig, "celeba_participant_gpm_stopping")


def write_outputs(
    conditions: dict[int, dict[str, Any]], fits: dict[int, dict[str, Any]]
) -> tuple[dict[str, str], dict[int, dict[str, Any]]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    draws_path = DATA_DIR / "celeba_privacy_utility_draws_by_participant_count.csv"
    draw_fields = [
        "p", "base_draw", "sampling_stratum", "run_role", "run_id",
        "paired_source_run_id", "v", "identity_linkage_accuracy",
        "test_balanced_accuracy", "gpm_reached_numerical_stopping_criterion",
        "gpm_iterations",
    ]
    with draws_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=draw_fields)
        writer.writeheader()
        for p in PARTICIPANT_COUNTS:
            condition = conditions[p]
            for index, row in enumerate(condition["rows"]):
                writer.writerow(
                    {
                        "p": p,
                        "base_draw": int(row["draw"]),
                        "sampling_stratum": "broad" if index < BASE_DRAWS else "focused",
                        "run_role": row["run_role"],
                        "run_id": row["run_id"],
                        "paired_source_run_id": row.get("paired_source_run_id", ""),
                        "v": float(row["anchor_v"]),
                        "identity_linkage_accuracy": attack_linkage(row),
                        "test_balanced_accuracy": utility(row),
                        "gpm_reached_numerical_stopping_criterion": bool(row["gpm"]["converged"]),
                        "gpm_iterations": int(row["gpm"]["iterations"]),
                    }
                )
    paths["draws"] = draws_path.relative_to(OUTPUT_DIR).as_posix()

    spline_path = DATA_DIR / "celeba_privacy_utility_spline_by_participant_count.csv"
    spline_fields = [
        "p", "v", "identity_linkage_accuracy", "estimated_test_balanced_accuracy",
        "identity_linkage_lower_95", "identity_linkage_upper_95",
        "test_balanced_accuracy_lower_95", "test_balanced_accuracy_upper_95",
    ]
    with spline_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=spline_fields)
        writer.writeheader()
        for p in PARTICIPANT_COUNTS:
            fit = fits[p]
            for values in zip(
                fit["noise"], fit["linkage_estimate"], fit["utility_estimate"],
                fit["linkage_lower"], fit["linkage_upper"],
                fit["utility_lower"], fit["utility_upper"],
            ):
                writer.writerow(dict(zip(spline_fields[1:], [float(value) for value in values]), p=p))
    paths["spline"] = spline_path.relative_to(OUTPUT_DIR).as_posix()

    controls_path = DATA_DIR / "celeba_privacy_utility_controls_by_participant_count.csv"
    with controls_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "p",
            "method",
            "identity_linkage_accuracy",
            "test_balanced_accuracy",
            "figure_representation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p in PARTICIPANT_COUNTS:
            for method in ("c_gdp", "aa_zero", "i_gdp"):
                writer.writerow(
                    {
                        "p": p,
                        "method": method,
                        "identity_linkage_accuracy": conditions[p]["controls"][
                            method
                        ].get("linkage", ""),
                        "test_balanced_accuracy": conditions[p]["controls"][method]["utility"],
                        "figure_representation": (
                            "horizontal utility reference"
                            if method == "i_gdp"
                            else "measured privacy-utility point"
                        ),
                    }
                )
    paths["controls"] = controls_path.relative_to(OUTPUT_DIR).as_posix()

    targets = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
    common_rows: list[dict[str, Any]] = []
    by_target: dict[float, list[dict[str, Any]]] = {target: [] for target in targets}
    for p in PARTICIPANT_COUNTS:
        fit = fits[p]
        for target in targets:
            crossing = first_descending_crossing(
                np.asarray(fit["noise"]), np.asarray(fit["linkage_estimate"]), target,
                lower=float(fit["observed_v_min"]), upper=float(fit["observed_v_max"]),
            )
            fitted_utility = interpolate_on_noise(
                np.asarray(fit["noise"]), np.asarray(fit["utility_estimate"]), crossing
            )
            row = {
                "p": p,
                "target_identity_linkage": target,
                "first_crossing_v": crossing,
                "fitted_test_balanced_accuracy": fitted_utility,
                "minus_aa_i_gdp_v0": fitted_utility
                - float(conditions[p]["controls"]["aa_zero"]["utility"]),
                "minus_i_gdp": fitted_utility
                - float(conditions[p]["controls"]["i_gdp"]["utility"]),
                "aa_i_gdp_v0_accuracy": float(conditions[p]["controls"]["aa_zero"]["utility"]),
                "i_gdp_accuracy": float(conditions[p]["controls"]["i_gdp"]["utility"]),
                "fit_domain_v_min": float(fit["observed_v_min"]),
                "fit_domain_v_max": float(fit["observed_v_max"]),
                "attack": "PA-OP",
            }
            common_rows.append(row)
            by_target[target].append(row)
    common_path = OUTPUT_DIR / "celeba_anchor_utility_at_common_linkage.csv"
    with common_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(common_rows[0]))
        writer.writeheader()
        writer.writerows(common_rows)
    paths["common_linkage"] = common_path.relative_to(OUTPUT_DIR).as_posix()

    summary_path = DATA_DIR / "celeba_anchor_utility_common_linkage_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["target_identity_linkage", "p10_accuracy", "p30_accuracy", "p50_accuracy", "maximum_spread"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for target in (0.20, 0.30, 0.50, 0.70):
            values = {
                int(row["p"]): float(row["fitted_test_balanced_accuracy"])
                for row in by_target[target]
            }
            writer.writerow(
                {
                    "target_identity_linkage": target,
                    "p10_accuracy": values[10],
                    "p30_accuracy": values[30],
                    "p50_accuracy": values[50],
                    "maximum_spread": max(values.values()) - min(values.values()),
                }
            )
    paths["common_linkage_summary"] = summary_path.relative_to(OUTPUT_DIR).as_posix()

    gpm_summaries: dict[int, dict[str, Any]] = {}
    for p in PARTICIPANT_COUNTS:
        condition = conditions[p]
        v = np.asarray(condition["v"])
        stopped = np.asarray(condition["converged"])
        summary = {
            "p": p,
            "observations": len(v),
            "base_draw_clusters": BASE_DRAWS,
            "stopped_before_cap": int(np.sum(stopped)),
            "reached_iteration_cap": int(np.sum(~stopped)),
            "largest_stopped_v": float(np.max(v[stopped])),
            "smallest_capped_v": float(np.min(v[~stopped])),
            "v_50_logistic": logistic_midpoint(v, stopped),
            "v_pt": float(condition["v_pt"]),
        }
        summary["v_50_over_v_pt"] = summary["v_50_logistic"] / summary["v_pt"]
        gpm_summaries[p] = summary
    gpm_path = OUTPUT_DIR / "celeba_participant_gpm_summary.csv"
    with gpm_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gpm_summaries[10]))
        writer.writeheader()
        writer.writerows(gpm_summaries[p] for p in PARTICIPANT_COUNTS)
    paths["gpm_summary"] = gpm_path.relative_to(OUTPUT_DIR).as_posix()
    return paths, gpm_summaries


def main() -> None:
    configure_matplotlib()
    conditions = {p: load_condition(p) for p in PARTICIPANT_COUNTS}
    validate_cross_condition_pairing(conditions)
    rng = np.random.default_rng(26072301)
    cluster_counts = rng.multinomial(
        BASE_DRAWS,
        np.full(BASE_DRAWS, 1.0 / BASE_DRAWS),
        size=BOOTSTRAP_RESAMPLES,
    )
    fits = {
        p: paired_cluster_spline(conditions[p], cluster_counts)
        for p in PARTICIPANT_COUNTS
    }
    data_paths, gpm_summaries = write_outputs(conditions, fits)
    figure_paths = {
        "privacy_utility": draw_privacy_utility(conditions, fits),
        "gpm": draw_gpm(conditions, gpm_summaries),
    }
    manifest = {
        "participant_counts": list(PARTICIPANT_COUNTS),
        "observations_per_participant_count": OBSERVATIONS_PER_PARTICIPANT_COUNT,
        "base_draw_clusters_per_participant_count": BASE_DRAWS,
        "anchor_noise_range": [0.0, ANCHOR_UPPER],
        "focused_anchor_noise_range": [0.0, FOCUSED_UPPER],
        "x_metric": "cross-image 10-way identity-linkage top-1 accuracy",
        "y_metric": "smiling test balanced accuracy",
        "attack": "orthogonal Procrustes (PA-OP)",
        "fit": {
            "type": "separate fixed cubic truncated-power regression splines for linkage(v) and utility(v), displayed as a paired parametric trace",
            "interior_knots": 5,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_unit": "base draw ID; broad and focused amplitudes resampled together",
            "bands": "95% pointwise paired cluster-percentile-bootstrap parametric envelope",
            "endpoint_controls_included": False,
        },
        "data": data_paths,
        "figures": figure_paths,
    }
    (OUTPUT_DIR / "privacy_utility_participant_count_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
