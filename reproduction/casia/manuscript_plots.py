"""Current manuscript figure geometry, applied only to fresh result summaries.

The bar colors, serif typography, participant markers, compact legends, and
reference-line styles reuse the September 2026 manuscript renderers. This
module consumes actual per-bin means/SDs; it never fills an absent bin.
"""
from pathlib import Path
import math

import numpy as np

BINS = (0, 15, 25, 35, 45, 100)
PS = (2, 5, 10, 20, 50)
LABELS = ("0–15", "15–25", "25–35", "35–45", "45–100")
COLORS = ("#506F8E", "#5E8A86", "#9B829F", "#B29A61", "#C88A77")
MARKERS = ("o", "s", "^", "D", "v")
CONTROLS = (
    ("c_gdp", "C-GDP", "#3F4550", (0, (1.2, 2))),
    ("i_gdp", "I-GDP", "#66796D", (0, (5, 2, 1, 2))),
    ("local_pooled", "Local", "#89779A", (0, (3.5, 2.2))),
)


def _save(fig, target, stem):
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = Path(target) / f"{stem}.{suffix}"
        fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=.04)
        paths.append(str(path))
    return paths


def validate(statistics_rows, controls):
    """Require one explicit row for every bin, including unfilled n=0 bins."""
    expected = {(p, family, b) for p in PS for family in ("anchor", "private") for b in range(5)}
    keys = [(r["p"], r["family"], r["bin"]) for r in statistics_rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Expected unique statistics rows for both families, all p and all five bins")
    for row in statistics_rows:
        n, mean, sd = row["n"], row["mean_utility_percent"], row["utility_sample_sd_percent"]
        if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= 20:
            raise ValueError("Invalid contributing observation count")
        if n == 0:
            if mean is not None or sd is not None:
                raise ValueError("Empty bin must not have invented means or error bars")
        elif mean is None or not math.isfinite(mean) or not 0 <= mean <= 100:
            raise ValueError("Utility mean must be finite and between 0 and 100")
        if n > 1 and (sd is None or not math.isfinite(sd) or sd < 0):
            raise ValueError("Sample SD missing or invalid")
        if n == 1 and sd is not None:
            raise ValueError("Sample SD is undefined for one observation")
    control_keys = [(r["p"], r["method"]) for r in controls]
    expected_controls = {(p, method) for p in PS for method, *_ in CONTROLS}
    if len(control_keys) != len(set(control_keys)) or set(control_keys) != expected_controls:
        raise ValueError("Expected one C-GDP, I-GDP and pooled Local control per p")
    if any(not math.isfinite(r["test_balanced_accuracy_percent"])
           or not 0 <= r["test_balanced_accuracy_percent"] <= 100 for r in controls):
        raise ValueError("Invalid utility control")


def _limits(rows, reference_values):
    lower = [r["mean_utility_percent"] - (r["utility_sample_sd_percent"] or 0)
             for r in rows if r["n"]] + list(reference_values)
    upper = [r["mean_utility_percent"] + (r["utility_sample_sd_percent"] or 0)
             for r in rows if r["n"]] + list(reference_values)
    return (min(50., np.floor(min(lower) / 5.) * 5.),
            max(100., np.ceil(max(upper) / 5.) * 5.))


def render(target, dataset, statistics_rows, controls):
    validate(statistics_rows, controls)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 18,
        "axes.labelsize": 23, "xtick.labelsize": 20, "ytick.labelsize": 19,
        "legend.fontsize": 17, "axes.linewidth": .8, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    })
    artifacts = []
    by_key = {(r["p"], r["family"], r["bin"]): r for r in statistics_rows}
    control_map = {(r["p"], r["method"]): r["test_balanced_accuracy_percent"] for r in controls}
    # Preserve the requested crop when all values fit. A fresh run with
    # below-chance outcomes expands the axis so those outcomes are not hidden.
    bar_limits = _limits([r for r in statistics_rows if r["p"] == 10],
                         [control_map[(10, method)] for method, *_ in CONTROLS])
    participant_limits = _limits([r for r in statistics_rows if r["family"] == "anchor"],
                                 control_map.values())

    fig, ax = plt.subplots(figsize=(8.7, 6.6))
    fig.subplots_adjust(left=.145, right=.97, bottom=.17, top=.74)
    x = np.arange(5)
    width = .34
    family_style = (("private", "C-GDP (private-data noise)", "#506F8E", -width/2),
                    ("anchor", "NAA-GDP", "#C88A77", width/2))
    handles = []
    for family, label, color, offset in family_style:
        rows = [by_key[(10, family, b)] for b in range(5)]
        present = [b for b, row in enumerate(rows) if row["n"]]
        ax.bar(x[present] + offset,
               [rows[b]["mean_utility_percent"] for b in present], width=width,
               yerr=[rows[b]["utility_sample_sd_percent"]
                     if rows[b]["utility_sample_sd_percent"] is not None else np.nan for b in present],
               color=color, edgecolor="white", linewidth=.6,
               error_kw={"ecolor": "#303030", "elinewidth": 1.2, "capsize": 3})
        handles.append(Patch(facecolor=color, label=label))
    for method, label, color, linestyle in CONTROLS:
        handles.append(ax.axhline(control_map[(10, method)], color=color,
                                  linestyle=linestyle, linewidth=1.9, label=label))
    ax.set(xlim=(-.55, 4.55), ylim=bar_limits, xlabel="Identity Linkage Accuracy (%)",
           ylabel="Test Balanced Accuracy (%)")
    ax.set_xticks(x, LABELS)
    ax.grid(axis="y", color="#E2E5E8", linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .995),
               ncol=2, frameon=False, columnspacing=1.4, handlelength=2.6)
    artifacts.extend(_save(fig, target, f"{dataset}_p10_h400_casia_privacy_utility"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.7, 6.6))
    fig.subplots_adjust(left=.145, right=.97, bottom=.15, top=.72)
    handles, refs = [], []
    for b, (label, color, marker) in enumerate(zip(LABELS, COLORS, MARKERS)):
        rows = [by_key[(p, "anchor", b)] for p in PS]
        means = [r["mean_utility_percent"] if r["n"] else np.nan for r in rows]
        sd = [r["utility_sample_sd_percent"]
              if r["utility_sample_sd_percent"] is not None else np.nan for r in rows]
        line = ax.errorbar(PS, means, yerr=sd, color=color, marker=marker,
            markersize=6.5, linewidth=1.65, elinewidth=1.05, capsize=3,
            capthick=1.05, markeredgecolor="white", markeredgewidth=.55, label=label)
        handles.append(line)
    for method, label, color, linestyle in CONTROLS:
        line, = ax.plot(PS, [control_map[(p, method)] for p in PS],
                        color=color, linestyle=linestyle, linewidth=1.9, label=label)
        refs.append(line)
    ax.set_xscale("log")
    ax.set(xlim=(1.75, 57), ylim=participant_limits, xlabel="Number of Participants",
           ylabel="Test Balanced Accuracy (%)")
    ax.set_xticks(PS, [str(p) for p in PS])
    ax.minorticks_off()
    ax.grid(axis="y", color="#E2E5E8", linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles, LABELS, title="NAA-GDP: Identity Linkage Accuracy (%)",
        loc="upper center", bbox_to_anchor=(.55, .995), ncol=3, frameon=False,
        handlelength=2, columnspacing=1.3, labelspacing=.35, title_fontsize=18)
    fig.legend(refs, [c[1] for c in CONTROLS], loc="upper center",
        bbox_to_anchor=(.55, .82), ncol=3, frameon=False, handlelength=2.7,
        columnspacing=1.6)
    artifacts.extend(_save(fig, target, f"{dataset}_anchor_participant_count"))
    plt.close(fig)
    return artifacts
