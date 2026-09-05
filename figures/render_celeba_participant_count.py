"""Render Figure 10 from the standalone CelebA participant-count replay.

The figure uses exactly 200 paired anchor-noise draws for each
``p in {2, 5, 10, 20, 50}``: the 100 published draws with ``0 < v < 0.1``
(``run_role == "random"``) and the 100 supplemental draws of the low-noise
stratum (``run_role == "supplemental_anchor_v_0_0p05_v1"``), in which draw
``j`` reuses the quantile, noise tensors and seeds of published draw ``j`` at
exactly half its amplitude, so ``0 < v < 0.05``.  Draw indices are paired
across participant counts in both strata.  The figure plots held-out Smiling
balanced accuracy against OP cross-image identity-linkage accuracy.  The
deterministic I-GDP utility reference is displayed for every participant
count, whereas the C-GDP and noiseless AA-GDP controls are omitted from the
figure (they remain in the exported control table).  All controls are
excluded from the spline fits and bootstrap resampling.

The supplemental records are read from ``results/celeba/participant_replay/
supplemental/``: either the complete record tables
(``p0xx_supplemental_results.csv[.gz]``, same schema as the published tables)
or, until those are frozen, the compact per-draw tables
(``p0xx_supplemental_compact.csv``) exported by the supplemental notebook.

Besides the flat PNG/PDF/SVG, the renderer writes a layered PDF
(``*_layers.pdf``) whose pages share the flat figure's page box: page 1 holds
the axes, chance levels and legends, and each further page holds one
participant count's bootstrap band, observations (with its I-GDP reference)
or fitted spline (``LAYER_PAGES``).  Stacked as PDF optional-content layers
(``figures/latex/celeba_participant_count_layers.tex`` does this with the
ocgx2 package), the pages give readers check boxes that show or hide each
participant count and the spline fits; with every layer visible the stack is
the flat figure.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO = SCRIPT_PATH.parents[1]
OUTPUT_DIR = Path(
    os.environ.get(
        "REPRO_CELEBA_COUNT_OUT",
        REPO / "build" / "renderers" / "celeba_participant_count",
    )
).resolve()
DATA_DIR = Path(
    os.environ.get(
        "REPRO_CELEBA_REPLAY_DATA",
        REPO / "results" / "celeba" / "participant_replay",
    )
).resolve()
FIGURE_DIR = OUTPUT_DIR / "figures"

PARTICIPANT_COUNTS = (2, 5, 10, 20, 50)
RANDOM_DRAWS = 100
SUPPLEMENTAL_DRAWS = 100
TOTAL_DRAWS = RANDOM_DRAWS + SUPPLEMENTAL_DRAWS
ANCHOR_UPPER = 0.1
SUPPLEMENTAL_UPPER = 0.05
SUPPLEMENTAL_ROLE = "supplemental_anchor_v_0_0p05_v1"
SUPPLEMENTAL_DIR = DATA_DIR / "supplemental"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260801
GRID_SIZE = 251
LINKAGE_CHANCE = 0.10
BALANCED_CHANCE = 0.50

COLORS: dict[int | str, str] = {
    2: "#0072B2",
    5: "#E69F00",
    10: "#009E73",
    20: "#CC79A7",
    50: "#D55E00",
    "chance": "#888888",
    "grid": "#D9D9D9",
}

PARTICIPANT_STYLE = {
    2: {"marker": "o"},
    5: {"marker": "s"},
    10: {"marker": "^"},
    20: {"marker": "v"},
    50: {"marker": "P"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def attack_linkage(row: dict[str, str], attack_name: str) -> float:
    attacks = json.loads(row["attacks"])
    matches = [attack for attack in attacks if str(attack.get("name")) == attack_name]
    if len(matches) != 1:
        raise ValueError(
            f"Run {row.get('run_id')} contains {len(matches)} {attack_name} attacks"
        )
    return float(
        matches[0]["metrics"]["face_linkage"]["cross_image_identity"]["top1"]
    )


def utility(row: dict[str, str]) -> float:
    return float(row["utility.test_balanced_accuracy"])


def one(
    rows: Iterable[dict[str, str]], protocol: str, run_role: str
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["protocol"] == protocol and row["run_role"] == run_role
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {protocol}/{run_role} control, found {len(matches)}"
        )
    return matches[0]


def supplemental_path(p: int) -> tuple[Path, str]:
    """Locate the supplemental table for ``p``; complete records take precedence."""

    for name, fmt in (
        (f"p{p:03d}_supplemental_results.csv.gz", "complete"),
        (f"p{p:03d}_supplemental_results.csv", "complete"),
        (f"p{p:03d}_supplemental_compact.csv", "compact"),
    ):
        candidate = SUPPLEMENTAL_DIR / name
        if candidate.is_file():
            return candidate, fmt
    raise FileNotFoundError(
        f"p={p}: no supplemental table under {SUPPLEMENTAL_DIR} "
        "(expected p0xx_supplemental_results.csv[.gz] or p0xx_supplemental_compact.csv)"
    )


def load_supplemental(p: int, original_v: np.ndarray) -> dict[str, Any]:
    """Load the 100 paired low-noise draws and lock them to the published draws."""

    path, fmt = supplemental_path(p)
    rows = read_csv(path)
    if fmt == "complete":
        rows = [
            row
            for row in rows
            if row["protocol"] == "pa_i_gdp" and row["run_role"] == SUPPLEMENTAL_ROLE
        ]
        if {int(row["n_participants"]) for row in rows} != {p}:
            raise ValueError(f"p={p}: supplemental participant-count field mismatch")
    else:
        if {int(row["p"]) for row in rows} != {p}:
            raise ValueError(f"p={p}: compact supplemental table lists another p")
        if "run_role" in rows[0] and {row["run_role"] for row in rows} != {SUPPLEMENTAL_ROLE}:
            raise ValueError(f"p={p}: compact supplemental table has an unexpected run role")
    rows = sorted(rows, key=lambda row: int(row["draw"]))
    if len(rows) != SUPPLEMENTAL_DRAWS:
        raise ValueError(f"p={p}: expected 100 supplemental records, found {len(rows)}")
    draws = np.asarray([int(row["draw"]) for row in rows], dtype=int)
    if not np.array_equal(draws, np.arange(SUPPLEMENTAL_DRAWS)):
        raise ValueError(f"p={p}: supplemental records are not draws 0,...,99")
    v = np.asarray([float(row["anchor_v"]) for row in rows], dtype=float)
    if not np.all((v > 0.0) & (v < SUPPLEMENTAL_UPPER)):
        raise ValueError(f"p={p}: supplemental v values must lie strictly inside (0, 0.05)")
    if not np.array_equal(v, 0.5 * np.asarray(original_v, dtype=float)):
        raise ValueError(
            f"p={p}: supplemental draws are not exactly half of the published paired draws"
        )
    if fmt == "complete":
        linkage = np.asarray([attack_linkage(row, "PA-OP") for row in rows], dtype=float)
        utility_values = np.asarray([utility(row) for row in rows], dtype=float)
        converged = np.asarray([parse_bool(row["gpm.converged"]) for row in rows], dtype=bool)
        iterations = np.asarray([int(float(row["gpm.iterations"])) for row in rows], dtype=int)
        run_ids = [row["run_id"] for row in rows]
    else:
        linkage = np.asarray(
            [float(row["op_cross_image_identity_top1"]) for row in rows], dtype=float
        )
        utility_values = np.asarray(
            [float(row["test_balanced_accuracy"]) for row in rows], dtype=float
        )
        converged = np.asarray([parse_bool(row["gpm_converged"]) for row in rows], dtype=bool)
        iterations = np.asarray([int(float(row["gpm_iterations"])) for row in rows], dtype=int)
        run_ids = [row.get("run_id", "") for row in rows]
    if not (
        np.all(np.isfinite(linkage))
        and np.all(np.isfinite(utility_values))
        and np.all((linkage >= 0.0) & (linkage <= 1.0))
        and np.all((utility_values >= 0.0) & (utility_values <= 1.0))
    ):
        raise ValueError(f"p={p}: non-finite or out-of-range supplemental metric")
    return {
        "path": path,
        "path_sha256": sha256_file(path),
        "format": fmt,
        "draw": draws,
        "v": v,
        "linkage": linkage,
        "utility": utility_values,
        "converged": converged,
        "iterations": iterations,
        "run_ids": run_ids,
    }


def load_condition(p: int) -> dict[str, Any]:
    path = DATA_DIR / f"p{p:03d}_results.csv.gz"
    rows = read_csv(path)
    if len(rows) != RANDOM_DRAWS + 3:
        raise ValueError(f"p={p}: expected 103 records, found {len(rows)}")
    if {int(row["n_participants"]) for row in rows} != {p}:
        raise ValueError(f"p={p}: participant-count field mismatch")
    if {int(row["h"]) for row in rows} != {400}:
        raise ValueError(f"p={p}: expected h=400")
    if {int(row["anchor.rows"]) for row in rows} != {1600}:
        raise ValueError(f"p={p}: expected 1,600 anchor rows")
    if {row["anchor.design"] for row in rows} != {"centered_unit_isometric_v2"}:
        raise ValueError(f"p={p}: unexpected anchor design")
    if {row["task"] for row in rows} != {"Smiling"}:
        raise ValueError(f"p={p}: unexpected task")

    random_rows = sorted(
        [
            row
            for row in rows
            if row["protocol"] == "pa_i_gdp" and row["run_role"] == "random"
        ],
        key=lambda row: int(row["draw"]),
    )
    if len(random_rows) != RANDOM_DRAWS:
        raise ValueError(f"p={p}: expected 100 random anchor-noise records")
    draws = np.asarray([int(row["draw"]) for row in random_rows], dtype=int)
    if not np.array_equal(draws, np.arange(RANDOM_DRAWS)):
        raise ValueError(f"p={p}: random records are not draws 0,...,99")

    v = np.asarray([float(row["anchor_v"]) for row in random_rows], dtype=float)
    linkage = np.asarray(
        [attack_linkage(row, "PA-OP") for row in random_rows], dtype=float
    )
    utility_values = np.asarray([utility(row) for row in random_rows], dtype=float)
    converged = np.asarray(
        [parse_bool(row["gpm.converged"]) for row in random_rows], dtype=bool
    )
    iterations = np.asarray(
        [int(float(row["gpm.iterations"])) for row in random_rows], dtype=int
    )
    if not np.all((v > 0.0) & (v < ANCHOR_UPPER)):
        raise ValueError(f"p={p}: random v values must lie strictly inside (0, 0.1)")
    if not (
        np.all(np.isfinite(linkage))
        and np.all(np.isfinite(utility_values))
        and np.all((linkage >= 0.0) & (linkage <= 1.0))
        and np.all((utility_values >= 0.0) & (utility_values <= 1.0))
    ):
        raise ValueError(f"p={p}: non-finite or out-of-range metric")

    c_gdp = one(rows, "c_gdp", "clean")
    i_gdp = one(rows, "i_gdp", "clean")
    aa_zero = one(rows, "pa_i_gdp", "endpoint_zero")
    nonrandom = [row for row in rows if row["run_role"] != "random"]
    expected_controls = {
        ("c_gdp", "clean"),
        ("i_gdp", "clean"),
        ("pa_i_gdp", "endpoint_zero"),
    }
    if {(row["protocol"], row["run_role"]) for row in nonrandom} != expected_controls:
        raise ValueError(f"p={p}: unexpected control records")

    controls = {
        "c_gdp": {
            "label": "C-GDP",
            "utility": utility(c_gdp),
            "linkage": attack_linkage(c_gdp, "C-exact"),
            "run_id": c_gdp["run_id"],
        },
        "aa_zero": {
            "label": "AA-GDP",
            "utility": utility(aa_zero),
            "linkage": attack_linkage(aa_zero, "PA-OP"),
            "run_id": aa_zero["run_id"],
        },
        "i_gdp": {
            "label": "I-GDP",
            "utility": utility(i_gdp),
            "run_id": i_gdp["run_id"],
        },
    }

    supplemental = load_supplemental(p, v)
    return {
        "p": p,
        "path": path,
        "path_sha256": sha256_file(path),
        "rows": rows,
        "random_rows": random_rows,
        "supplemental": supplemental,
        "draw": np.concatenate([draws, supplemental["draw"]]),
        "stratum": np.asarray(
            ["random"] * RANDOM_DRAWS + [SUPPLEMENTAL_ROLE] * SUPPLEMENTAL_DRAWS
        ),
        "run_id": [row["run_id"] for row in random_rows] + list(supplemental["run_ids"]),
        "v": np.concatenate([v, supplemental["v"]]),
        "linkage": np.concatenate([linkage, supplemental["linkage"]]),
        "utility": np.concatenate([utility_values, supplemental["utility"]]),
        "converged": np.concatenate([converged, supplemental["converged"]]),
        "iterations": np.concatenate([iterations, supplemental["iterations"]]),
        "controls": controls,
        "anchor_sha256": rows[0]["anchor.tensor_sha256_float64_c_order"],
        "deployment_bank_sha256": rows[0]["deployment.tensor_file_sha256"],
        "noise_rng": rows[0]["noise_rng"],
        "face_linkage_manifest": rows[0][
            "best_attack.metrics.face_linkage.manifest_hash"
        ],
    }


def validate_pairing(conditions: dict[int, dict[str, Any]]) -> str:
    reference = np.asarray(conditions[PARTICIPANT_COUNTS[0]]["v"])
    for p in PARTICIPANT_COUNTS[1:]:
        if not np.array_equal(np.asarray(conditions[p]["v"]), reference):
            raise ValueError("Anchor-noise amplitudes are not exactly paired across p")
    if len({conditions[p]["anchor_sha256"] for p in PARTICIPANT_COUNTS}) != 1:
        raise ValueError("Anchor tensor differs across participant-count conditions")
    if len(
        {conditions[p]["deployment_bank_sha256"] for p in PARTICIPANT_COUNTS}
    ) != 1:
        raise ValueError("GDP parameter bank differs across participant-count conditions")
    return hashlib.sha256(reference.astype("<f8", copy=False).tobytes()).hexdigest()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 10.5,
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


def spline_design(normalized_v: np.ndarray) -> np.ndarray:
    x = np.asarray(normalized_v, dtype=np.float64).reshape(-1)
    knots = np.linspace(0.0, 1.0, 7, dtype=np.float64)[1:-1]
    columns = [np.ones_like(x), x, x * x, x**3]
    columns.extend(np.maximum(x - knot, 0.0) ** 3 for knot in knots)
    return np.column_stack(columns)


def fit_coefficients(
    design: np.ndarray, outcome: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    gram = design.T @ design
    ridge = 1.0e-8 * max(float(np.trace(gram) / design.shape[1]), 1.0)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(gram + penalty, design.T @ outcome)
    return beta, penalty


def fit_spline_with_bootstrap(
    condition: dict[str, Any], bootstrap_counts: np.ndarray
) -> dict[str, np.ndarray]:
    v = np.asarray(condition["v"], dtype=np.float64)
    linkage = np.asarray(condition["linkage"], dtype=np.float64)
    utility_values = np.asarray(condition["utility"], dtype=np.float64)
    design = spline_design(v / ANCHOR_UPPER)
    v_grid = np.linspace(0.0, ANCHOR_UPPER, GRID_SIZE, dtype=np.float64)
    grid_design = spline_design(v_grid / ANCHOR_UPPER)
    linkage_beta, penalty = fit_coefficients(design, linkage)
    utility_beta, _ = fit_coefficients(design, utility_values)

    linkage_predictions = np.empty(
        (BOOTSTRAP_RESAMPLES, GRID_SIZE), dtype=np.float64
    )
    utility_predictions = np.empty(
        (BOOTSTRAP_RESAMPLES, GRID_SIZE), dtype=np.float64
    )
    for start in range(0, BOOTSTRAP_RESAMPLES, 256):
        stop = min(start + 256, BOOTSTRAP_RESAMPLES)
        weights = bootstrap_counts[start:stop].astype(np.float64, copy=False)
        boot_gram = np.einsum(
            "bn,np,nq->bpq", weights, design, design, optimize=True
        )
        linkage_rhs = np.einsum(
            "bn,np,n->bp", weights, design, linkage, optimize=True
        )
        utility_rhs = np.einsum(
            "bn,np,n->bp", weights, design, utility_values, optimize=True
        )
        regularized = boot_gram + penalty[None, :, :]
        linkage_boot_beta = np.linalg.solve(
            regularized, linkage_rhs[..., None]
        )[..., 0]
        utility_boot_beta = np.linalg.solve(
            regularized, utility_rhs[..., None]
        )[..., 0]
        linkage_predictions[start:stop] = linkage_boot_beta @ grid_design.T
        utility_predictions[start:stop] = utility_boot_beta @ grid_design.T

    linkage_lower, linkage_upper = np.quantile(
        linkage_predictions, [0.025, 0.975], axis=0
    )
    utility_lower, utility_upper = np.quantile(
        utility_predictions, [0.025, 0.975], axis=0
    )
    return {
        "v": v_grid,
        "linkage_estimate": grid_design @ linkage_beta,
        "linkage_lower": linkage_lower,
        "linkage_upper": linkage_upper,
        "utility_estimate": grid_design @ utility_beta,
        "utility_lower": utility_lower,
        "utility_upper": utility_upper,
    }


FIGURE_STEM = "celeba_privacy_utility_by_participant_count"
LAYERS_STEM = f"{FIGURE_STEM}_layers"
# Page order of the layered PDF.  Every page has the same page box as the flat
# figure, so the pages can be stacked on top of each other: page 1 carries the
# axes, the chance levels and the legends; the remaining pages carry exactly one
# participant count's bootstrap band, observations (with its I-GDP reference)
# or fitted spline, in the drawing order of the flat figure.
LAYER_PAGES: tuple[tuple[str, int | None], ...] = (
    ("base", None),
    *(("band", p) for p in PARTICIPANT_COUNTS),
    *(("observations", p) for p in PARTICIPANT_COUNTS),
    *(("fit", p) for p in PARTICIPANT_COUNTS),
)


def save_figure(
    fig: plt.Figure, groups: dict[tuple[str, int | None], list[Any]]
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    """Write the flat figure (PNG/PDF/SVG) and the layered PDF.

    All files share one page box, computed from the complete figure, so that
    the layered pages overlay each other and the flat figure exactly.
    """

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    renderer = fig.canvas.get_renderer()
    page_box = fig.get_tightbbox(renderer).padded(
        float(mpl.rcParams["savefig.pad_inches"])
    )
    outputs: dict[str, Path] = {}
    for suffix in ("png", "pdf", "svg"):
        path = FIGURE_DIR / f"{FIGURE_STEM}.{suffix}"
        kwargs: dict[str, Any] = {"bbox_inches": page_box}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        outputs[suffix] = path

    from matplotlib.backends.backend_pdf import PdfPages

    ax = fig.axes[0]
    base_only = groups[("base", None)]
    layered = [artist for key, items in groups.items() if key[0] != "base" for artist in items]
    pages: list[dict[str, Any]] = []
    layers_path = FIGURE_DIR / f"{LAYERS_STEM}.pdf"
    with PdfPages(
        layers_path,
        metadata={
            "Title": "CelebA participant-count privacy-utility figure, one layer per page",
            "Subject": "; ".join(
                f"page {index}: {kind}" + (f" p={p}" if p is not None else "")
                for index, (kind, p) in enumerate(LAYER_PAGES, start=1)
            ),
        },
    ) as pdf:
        for index, (kind, p) in enumerate(LAYER_PAGES, start=1):
            visible = groups[(kind, p)]
            for artist in layered:
                artist.set_visible(artist in visible)
            if kind == "base":
                ax.set_axis_on()
                for artist in base_only:
                    artist.set_visible(True)
                pdf.savefig(fig, bbox_inches=page_box)
            else:
                ax.set_axis_off()
                for artist in base_only:
                    artist.set_visible(False)
                pdf.savefig(fig, bbox_inches=page_box, transparent=True)
            pages.append(
                {
                    "page": index,
                    "layer": kind,
                    "participant_count": p,
                    "artists": len(visible),
                }
            )
    ax.set_axis_on()
    for artist in [*base_only, *layered]:
        artist.set_visible(True)
    outputs["layers_pdf"] = layers_path
    plt.close(fig)
    return outputs, pages


def draw_figure(
    conditions: dict[int, dict[str, Any]], fits: dict[int, dict[str, np.ndarray]]
) -> tuple[plt.Figure, dict[tuple[str, int | None], list[Any]]]:
    groups: dict[tuple[str, int | None], list[Any]] = {key: [] for key in LAYER_PAGES}
    fig, ax = plt.subplots(figsize=(8.25, 5.25))
    style_axis(ax)
    ax.set_xlim(5.0, 82.0)
    ax.set_ylim(48.0, 90.5)
    ax.set_xticks(np.arange(10.0, 81.0, 10.0))
    ax.set_yticks(np.arange(50.0, 91.0, 5.0))
    ax.set_xlabel("Identity Linkage Accuracy (%)")
    ax.set_ylabel("Smiling Test Balanced Accuracy (%)")
    groups[("base", None)].append(
        ax.axvline(
            100.0 * LINKAGE_CHANCE,
            color=COLORS["chance"],
            linestyle=(0, (1.3, 2.2)),
            linewidth=1.05,
            zorder=1,
        )
    )
    groups[("base", None)].append(
        ax.axhline(
            100.0 * BALANCED_CHANCE,
            color="#AAAAAA",
            linestyle=(0, (1.3, 2.2)),
            linewidth=1.0,
            zorder=1,
        )
    )

    for p in PARTICIPANT_COUNTS:
        condition = conditions[p]
        fit = fits[p]
        color = str(COLORS[p])
        marker = str(PARTICIPANT_STYLE[p]["marker"])
        converged = np.asarray(condition["converged"], dtype=bool)
        x = 100.0 * np.asarray(condition["linkage"])
        y = 100.0 * np.asarray(condition["utility"])

        observations = groups[("observations", p)]
        observations.append(
            ax.axhline(
                100.0 * float(condition["controls"]["i_gdp"]["utility"]),
                color=color,
                linestyle=(0, (5.5, 2.2)),
                linewidth=1.25,
                alpha=0.78,
                zorder=2,
            )
        )

        observations.append(
            ax.scatter(
                x[converged],
                y[converged],
                s=23,
                marker=marker,
                facecolor=color,
                edgecolor="none",
                alpha=0.20,
                zorder=3,
            )
        )
        observations.append(
            ax.scatter(
                x[~converged],
                y[~converged],
                s=23,
                marker=marker,
                facecolor="none",
                edgecolor=color,
                linewidth=0.65,
                alpha=0.18,
                zorder=3,
            )
        )

        band_x = 100.0 * np.concatenate(
            [
                np.clip(fit["linkage_lower"], 0.0, 1.0),
                np.clip(fit["linkage_upper"], 0.0, 1.0)[::-1],
            ]
        )
        band_y = 100.0 * np.concatenate(
            [
                np.clip(fit["utility_lower"], 0.0, 1.0),
                np.clip(fit["utility_upper"], 0.0, 1.0)[::-1],
            ]
        )
        groups[("band", p)].extend(
            ax.fill(band_x, band_y, color=color, alpha=0.060, linewidth=0, zorder=1)
        )
        groups[("fit", p)].extend(
            ax.plot(
                100.0 * np.clip(fit["linkage_estimate"], 0.0, 1.0),
                100.0 * np.clip(fit["utility_estimate"], 0.0, 1.0),
                color=color,
                linewidth=1.85,
                alpha=0.88,
                zorder=5,
            )
        )
    participant_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[p],
            linestyle="-",
            linewidth=2.1,
            alpha=0.75,
            label=rf"$p={p}$",
        )
        for p in PARTICIPANT_COUNTS
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=1.85,
            alpha=0.88,
            label="NAA-GDP",
        ),
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=(0, (5.5, 2.2)),
            linewidth=1.4,
            label="I-GDP",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["chance"],
            linestyle=(0, (1.3, 2.2)),
            linewidth=1.05,
            label="Chance levels",
        ),
    ]
    groups[("base", None)].append(
        fig.legend(
            handles=participant_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.115),
            ncol=5,
            frameon=False,
            handlelength=2.3,
            columnspacing=1.7,
        )
    )
    groups[("base", None)].append(
        fig.legend(
            handles=method_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=3,
            frameon=False,
            fontsize=8.8,
            handlelength=2.35,
            columnspacing=1.45,
        )
    )
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.31, top=0.97)
    return fig, groups


def write_analysis_tables(
    conditions: dict[int, dict[str, Any]], fits: dict[int, dict[str, np.ndarray]]
) -> dict[str, Path]:
    observations_path = OUTPUT_DIR / "figure10_observations.csv"
    with observations_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "p",
            "stratum",
            "draw",
            "anchor_v",
            "identity_linkage_accuracy",
            "smiling_test_balanced_accuracy",
            "gpm_converged",
            "gpm_iterations",
            "run_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p in PARTICIPANT_COUNTS:
            condition = conditions[p]
            for index in range(TOTAL_DRAWS):
                writer.writerow(
                    {
                        "p": p,
                        "stratum": str(condition["stratum"][index]),
                        "draw": int(condition["draw"][index]),
                        "anchor_v": float(condition["v"][index]),
                        "identity_linkage_accuracy": float(
                            condition["linkage"][index]
                        ),
                        "smiling_test_balanced_accuracy": float(
                            condition["utility"][index]
                        ),
                        "gpm_converged": bool(condition["converged"][index]),
                        "gpm_iterations": int(condition["iterations"][index]),
                        "run_id": condition["run_id"][index],
                    }
                )

    controls_path = OUTPUT_DIR / "figure10_controls.csv"
    with controls_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["p", "method", "identity_linkage_accuracy", "utility", "run_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p in PARTICIPANT_COUNTS:
            for key in ("c_gdp", "aa_zero", "i_gdp"):
                control = conditions[p]["controls"][key]
                writer.writerow(
                    {
                        "p": p,
                        "method": control["label"],
                        "identity_linkage_accuracy": control.get("linkage", ""),
                        "utility": control["utility"],
                        "run_id": control["run_id"],
                    }
                )

    fits_path = OUTPUT_DIR / "figure10_fitted_curves.csv"
    with fits_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "p",
            "anchor_v",
            "identity_linkage_fit",
            "identity_linkage_lower",
            "identity_linkage_upper",
            "utility_fit",
            "utility_lower",
            "utility_upper",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p in PARTICIPANT_COUNTS:
            fit = fits[p]
            for index in range(GRID_SIZE):
                writer.writerow(
                    {
                        "p": p,
                        "anchor_v": float(fit["v"][index]),
                        "identity_linkage_fit": float(
                            fit["linkage_estimate"][index]
                        ),
                        "identity_linkage_lower": float(
                            fit["linkage_lower"][index]
                        ),
                        "identity_linkage_upper": float(
                            fit["linkage_upper"][index]
                        ),
                        "utility_fit": float(fit["utility_estimate"][index]),
                        "utility_lower": float(fit["utility_lower"][index]),
                        "utility_upper": float(fit["utility_upper"][index]),
                    }
                )
    return {
        "observations": observations_path,
        "controls": controls_path,
        "fits": fits_path,
    }


def interpolate_on_v(fit: dict[str, np.ndarray], field: str, v_value: float) -> float:
    return float(np.interp(v_value, fit["v"], fit[field]))


def utility_at_matched_linkage(fit: dict[str, np.ndarray], target: float) -> dict[str, float | None]:
    """Utility of the fitted curve where the fitted linkage first falls to ``target``."""

    v = np.asarray(fit["v"], dtype=float)
    linkage = np.asarray(fit["linkage_estimate"], dtype=float)
    utility_values = np.asarray(fit["utility_estimate"], dtype=float)
    for index in range(len(v) - 1):
        left, right = linkage[index] - target, linkage[index + 1] - target
        if left == 0.0 or left * right < 0.0:
            fraction = 0.0 if left == 0.0 else -left / (right - left)
            v_star = float(v[index] + fraction * (v[index + 1] - v[index]))
            return {"v": v_star, "utility": float(np.interp(v_star, v, utility_values))}
    return {"v": None, "utility": None}


def text_numbers(
    conditions: dict[int, dict[str, Any]], fits: dict[int, dict[str, np.ndarray]]
) -> dict[str, Any]:
    """The quantities quoted in the manuscript's participant-count paragraphs."""

    per_p: dict[str, Any] = {}
    for p in PARTICIPANT_COUNTS:
        fit = fits[p]
        i_gdp = float(conditions[p]["controls"]["i_gdp"]["utility"])
        per_p[str(p)] = {
            "fitted_utility_at_v_0p01": interpolate_on_v(fit, "utility_estimate", 0.01),
            "fitted_linkage_at_v_0p01": interpolate_on_v(fit, "linkage_estimate", 0.01),
            "utility_at_50pct_linkage": utility_at_matched_linkage(fit, 0.50),
            "utility_at_20pct_linkage": utility_at_matched_linkage(fit, 0.20),
            "fitted_utility_at_v_0p1": interpolate_on_v(fit, "utility_estimate", ANCHOR_UPPER),
            "i_gdp_utility": i_gdp,
            "gap_to_i_gdp_at_v_0p1_pp": 100.0 * (
                interpolate_on_v(fit, "utility_estimate", ANCHOR_UPPER) - i_gdp
            ),
            "c_gdp": {
                "linkage": float(conditions[p]["controls"]["c_gdp"]["linkage"]),
                "utility": float(conditions[p]["controls"]["c_gdp"]["utility"]),
            },
        }
    linkage_window = [per_p[str(p)]["fitted_linkage_at_v_0p01"] for p in PARTICIPANT_COUNTS]
    return {
        "per_p": per_p,
        "fitted_linkage_window_at_v_0p01": [min(linkage_window), max(linkage_window)],
        "max_abs_gap_to_i_gdp_at_v_0p1_pp": max(
            abs(per_p[str(p)]["gap_to_i_gdp_at_v_0p1_pp"]) for p in PARTICIPANT_COUNTS
        ),
        "utility_monotone_in_p_at_v_0p01": all(
            per_p[str(a)]["fitted_utility_at_v_0p01"] < per_p[str(b)]["fitted_utility_at_v_0p01"]
            for a, b in zip(PARTICIPANT_COUNTS, PARTICIPANT_COUNTS[1:])
        ),
    }


def main() -> dict[str, Any]:
    configure_matplotlib()
    conditions = {p: load_condition(p) for p in PARTICIPANT_COUNTS}
    paired_v_sha256 = validate_pairing(conditions)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_counts = rng.multinomial(
        TOTAL_DRAWS,
        np.full(TOTAL_DRAWS, 1.0 / TOTAL_DRAWS),
        size=BOOTSTRAP_RESAMPLES,
    )
    fits = {
        p: fit_spline_with_bootstrap(conditions[p], bootstrap_counts)
        for p in PARTICIPANT_COUNTS
    }
    fig, groups = draw_figure(conditions, fits)
    figures, layer_pages = save_figure(fig, groups)
    tables = write_analysis_tables(conditions, fits)

    numbers = text_numbers(conditions, fits)
    summary = {
        str(p): {
            "records": len(conditions[p]["rows"]),
            "random_draws": TOTAL_DRAWS,
            "published_draws": len(conditions[p]["random_rows"]),
            "supplemental_draws": int(len(conditions[p]["supplemental"]["draw"])),
            "supplemental_source": conditions[p]["supplemental"]["path"].relative_to(REPO).as_posix(),
            "supplemental_source_sha256": conditions[p]["supplemental"]["path_sha256"],
            "supplemental_source_format": conditions[p]["supplemental"]["format"],
            "gpm_converged_published": int(np.sum(conditions[p]["converged"][:RANDOM_DRAWS])),
            "gpm_converged_supplemental": int(np.sum(conditions[p]["converged"][RANDOM_DRAWS:])),
            "v_min": float(np.min(conditions[p]["v"])),
            "v_max": float(np.max(conditions[p]["v"])),
            "linkage_min": float(np.min(conditions[p]["linkage"])),
            "linkage_max": float(np.max(conditions[p]["linkage"])),
            "utility_min": float(np.min(conditions[p]["utility"])),
            "utility_max": float(np.max(conditions[p]["utility"])),
            "gpm_converged": int(np.sum(conditions[p]["converged"])),
            "gpm_capped": int(np.sum(~conditions[p]["converged"])),
            "source_csv": conditions[p]["path"].relative_to(REPO).as_posix(),
            "source_csv_sha256": conditions[p]["path_sha256"],
            "face_linkage_manifest": conditions[p]["face_linkage_manifest"],
        }
        for p in PARTICIPANT_COUNTS
    }
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "figure": "Figure 10: CelebA participant-count privacy--utility sensitivity",
        "participant_counts": list(PARTICIPANT_COUNTS),
        "random_draws_per_participant_count": TOTAL_DRAWS,
        "anchor_noise_distribution": (
            "paired draws over (0, 0.1): 100 published draws v_j ~ Uniform(0, 0.1) plus "
            "100 supplemental draws at exactly v_j / 2 (same quantile, noise tensors and seeds)"
        ),
        "supplemental_run_role": SUPPLEMENTAL_ROLE,
        "manuscript_numbers": numbers,
        "attack": "PA-OP cross-image identity linkage",
        "utility": "held-out Smiling balanced accuracy",
        "spline": "fixed cubic truncated-power basis with five equally spaced interior knots",
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "unit": "paired draw index, shared across participant counts",
            "interval": "95% pointwise percentile envelope",
        },
        "controls_excluded_from_fit": ["C-GDP", "I-GDP", "AA-GDP"],
        "layered_pdf": {
            "path": figures["layers_pdf"].relative_to(OUTPUT_DIR).as_posix(),
            "page_box": "identical to the flat figure on every page",
            "pages": layer_pages,
            "intended_use": (
                "stack the pages as PDF optional-content layers (e.g. with the "
                "ocgx2 LaTeX package; see figures/latex/) so that readers can show "
                "or hide each participant count and the spline fits"
            ),
        },
        "paired_v_sha256_float64_c_order": paired_v_sha256,
        "anchor_sha256_float64_c_order": conditions[2]["anchor_sha256"],
        "deployment_bank_file_sha256": conditions[2]["deployment_bank_sha256"],
        "conditions": summary,
        "outputs": {
            **{
                suffix: {
                    "path": path.relative_to(OUTPUT_DIR).as_posix(),
                    "sha256": sha256_file(path),
                }
                for suffix, path in figures.items()
            },
            **{
                name: {
                    "path": path.relative_to(OUTPUT_DIR).as_posix(),
                    "sha256": sha256_file(path),
                }
                for name, path in tables.items()
            },
        },
    }
    manifest_path = OUTPUT_DIR / "figure10_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("Validated and rendered Figure 10")
    for p in PARTICIPANT_COUNTS:
        item = summary[str(p)]
        print(
            f"p={p:>2}: {item['random_draws']} draws ({item['published_draws']} published + "
            f"{item['supplemental_draws']} supplemental, {item['supplemental_source_format']}); "
            f"GPM converged/capped={item['gpm_converged']}/{item['gpm_capped']}; "
            f"linkage={100*item['linkage_min']:.1f}--{100*item['linkage_max']:.1f}%; "
            f"utility={100*item['utility_min']:.1f}--{100*item['utility_max']:.1f}%"
        )
    for p in PARTICIPANT_COUNTS:
        item = numbers["per_p"][str(p)]
        at50, at20 = item["utility_at_50pct_linkage"], item["utility_at_20pct_linkage"]
        print(
            f"p={p:>2}: fit at v=0.01: utility {100*item['fitted_utility_at_v_0p01']:.2f}%, "
            f"linkage {100*item['fitted_linkage_at_v_0p01']:.2f}%; "
            f"utility at 50% linkage {100*at50['utility']:.2f}% (v={at50['v']:.4f}); "
            f"at 20% linkage {100*at20['utility']:.2f}% (v={at20['v']:.4f}); "
            f"I-GDP {100*item['i_gdp_utility']:.2f}%, gap at v=0.1 {item['gap_to_i_gdp_at_v_0p1_pp']:+.2f} pp"
        )
    print(
        f"linkage window at v=0.01: {100*numbers['fitted_linkage_window_at_v_0p01'][0]:.2f}--"
        f"{100*numbers['fitted_linkage_window_at_v_0p01'][1]:.2f}%; "
        f"max |gap to I-GDP| at v=0.1: {numbers['max_abs_gap_to_i_gdp_at_v_0p1_pp']:.2f} pp; "
        f"monotone in p at v=0.01: {numbers['utility_monotone_in_p_at_v_0p01']}"
    )
    print(f"PNG: {figures['png']}")
    print(f"Layered PDF ({len(layer_pages)} pages): {figures['layers_pdf']}")
    print(f"Manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    FIGURE10_MANIFEST = main()
