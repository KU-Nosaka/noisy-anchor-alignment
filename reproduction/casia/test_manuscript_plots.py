"""Small CPU tests of scientific summaries and the current manuscript plots."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.container import BarContainer, ErrorbarContainer

import manuscript_plots as plots
import notebook_api as api


def statistics_fixture():
    rows = [{"p": p, "family": family, "bin": b, "n": 20,
             "mean_utility_percent": 70.0 + b + p / 10,
             "utility_sample_sd_percent": 2.5}
            for p in plots.PS for family in ("anchor", "private") for b in range(5)]
    controls = [{"p": p, "method": method, "test_balanced_accuracy_percent": value}
                for p in plots.PS
                for method, value in (("c_gdp", 82.0), ("i_gdp", 67.0), ("local_pooled", 62.0))]
    return rows, controls


def rendered_figures(rows, controls):
    captured = []

    def capture(fig, target, stem):
        fig.canvas.draw()
        captured.append(fig)
        return []

    with tempfile.TemporaryDirectory(prefix="casia-plot-test-") as directory:
        with patch.object(plots, "_save", side_effect=capture):
            plots.render(directory, "celeba", rows, controls)
    return captured


class ManuscriptPlotTests(unittest.TestCase):
    def test_required_rows_controls_and_finite_counts(self):
        rows, controls = statistics_fixture()
        plots.validate(rows, controls)
        for broken_rows, broken_controls in (
            (rows + [copy.deepcopy(rows[0])], controls),
            (rows[:-1], controls),
            (rows, controls + [copy.deepcopy(controls[0])]),
            (rows, controls[:-1]),
        ):
            with self.subTest(rows=len(broken_rows), controls=len(broken_controls)):
                with self.assertRaises(ValueError):
                    plots.validate(broken_rows, broken_controls)
        for key, value in (("n", 21), ("n", True),
                           ("mean_utility_percent", float("nan")),
                           ("utility_sample_sd_percent", -1.0)):
            broken = copy.deepcopy(rows)
            broken[0][key] = value
            with self.subTest(field=key, value=value):
                with self.assertRaises(ValueError):
                    plots.validate(broken, controls)

    def test_missing_bin_is_omitted_and_breaks_participant_curve(self):
        rows, controls = statistics_fixture()
        missing = next(row for row in rows if (row["p"], row["family"], row["bin"]) == (10, "anchor", 0))
        missing.update(n=0, mean_utility_percent=None, utility_sample_sd_percent=None)
        bar_figure, participant_figure = rendered_figures(rows, controls)
        bars = [item for item in bar_figure.axes[0].containers if isinstance(item, BarContainer)]
        self.assertEqual([len(item.patches) for item in bars], [5, 4])
        curves = [item for item in participant_figure.axes[0].containers if isinstance(item, ErrorbarContainer)]
        y = np.asarray(curves[0].lines[0].get_ydata(), dtype=float)
        self.assertTrue(np.isnan(y[2]))
        self.assertTrue(np.isfinite(y[[0, 1, 3, 4]]).all())
        invented = copy.deepcopy(rows)
        next(row for row in invented if row["n"] == 0)["mean_utility_percent"] = 50.0
        with self.assertRaises(ValueError):
            plots.validate(invented, controls)

    def test_singleton_has_no_invented_zero_sd_error_bar(self):
        rows, controls = statistics_fixture()
        singleton = next(row for row in rows if (row["p"], row["family"], row["bin"]) == (10, "private", 0))
        singleton.update(n=1, utility_sample_sd_percent=None)
        bar_figure, _ = rendered_figures(rows, controls)
        bars = [item for item in bar_figure.axes[0].containers if isinstance(item, BarContainer)]
        segments = bars[0].errorbar.lines[2][0].get_segments()
        self.assertEqual(segments[0].size, 0)
        self.assertEqual(segments[1].shape, (2, 2))
        singleton["utility_sample_sd_percent"] = 0.0
        with self.assertRaises(ValueError):
            plots.validate(rows, controls)

    def test_error_bars_use_the_supplied_sample_sd(self):
        rows, controls = statistics_fixture()
        bar_figure, participant_figure = rendered_figures(rows, controls)
        for container in bar_figure.axes[0].containers:
            if isinstance(container, BarContainer):
                for segment in container.errorbar.lines[2][0].get_segments():
                    self.assertAlmostEqual(segment[1, 1] - segment[0, 1], 5.0)
        for container in participant_figure.axes[0].containers:
            if isinstance(container, ErrorbarContainer):
                for segment in container.lines[2][0].get_segments():
                    self.assertAlmostEqual(segment[1, 1] - segment[0, 1], 5.0)

    def test_crop_depends_only_on_displayed_data_and_preserves_error_extents(self):
        rows, controls = statistics_fixture()
        invisible = next(row for row in rows if (row["p"], row["family"], row["bin"]) == (2, "private", 0))
        invisible.update(mean_utility_percent=40.0, utility_sample_sd_percent=5.0)
        figures = rendered_figures(rows, controls)
        self.assertEqual([figure.axes[0].get_ylim() for figure in figures], [(50.0, 100.0), (50.0, 100.0)])
        visible = next(row for row in rows if (row["p"], row["family"], row["bin"]) == (10, "anchor", 0))
        visible.update(mean_utility_percent=51.0, utility_sample_sd_percent=8.0)
        figures = rendered_figures(rows, controls)
        self.assertEqual([figure.axes[0].get_ylim()[0] for figure in figures], [40.0, 40.0])
        visible.update(mean_utility_percent=96.0, utility_sample_sd_percent=8.0)
        figures = rendered_figures(rows, controls)
        self.assertEqual([figure.axes[0].get_ylim()[1] for figure in figures], [105.0, 105.0])

    def test_controls_legends_no_titles_and_no_chance_utility(self):
        rows, controls = statistics_fixture()
        figures = rendered_figures(rows, controls)
        for figure in figures:
            self.assertEqual(figure.axes[0].get_title(), "")
            self.assertTrue(figure._suptitle is None or not figure._suptitle.get_text())
            labels = [text.get_text() for legend in figure.legends for text in legend.get_texts()]
            self.assertIn("Local", labels)
            self.assertNotIn("Local-only", labels)
            self.assertFalse(any("chance" in label.lower() for label in labels))
        for line in figures[0].axes[0].lines:
            if line.get_label() in {"C-GDP", "I-GDP", "Local"}:
                expected = {"C-GDP": 82.0, "I-GDP": 67.0, "Local": 62.0}[line.get_label()]
                np.testing.assert_allclose(line.get_ydata(), expected)
        self.assertEqual(figures[1].axes[0].get_xscale(), "log")
        np.testing.assert_allclose(figures[1].axes[0].get_xticks(), plots.PS)


class ResultExportTests(unittest.TestCase):
    def test_actual_summary_schema_sample_sd_and_full_api_artifacts(self):
        """Use ten accepted pairs per p to distinguish sample SD from SEM/population SD."""
        with tempfile.TemporaryDirectory(prefix="casia-summary-test-") as directory:
            root = Path(directory)
            (root / "pipeline_identity.json").write_text(json.dumps({"configuration": {
                "dataset": "celeba", "dataset_label": "CelebA",
                "linkage_bin_edges_percent": list(plots.BINS),
            }}), encoding="utf-8")
            for p in plots.PS:
                records = []
                for family in ("anchor", "private"):
                    for b in range(5):
                        for j, utility in enumerate((0.7, 0.9)):
                            records.append({"condition": {
                                "id": f"{family}-{b}-{j}", "family": family, "bin": b,
                                "linkage_percent": (plots.BINS[b] + plots.BINS[b + 1]) / 2,
                                "scale": 0.02 if family == "anchor" else 0.5,
                            }, "result": {"test": {"balanced_accuracy": utility, "n": 40}}})
                for family, utility in (("c_gdp", 0.83), ("i_gdp", 0.68)):
                    records.append({"condition": {"id": family, "family": family},
                                    "result": {"test": {"balanced_accuracy": utility, "n": 40}}})
                summary = {"p": p, "bin_edges_percent": list(plots.BINS), "records": records,
                           "local_pooled": {"test_balanced_accuracy": 0.63, "n": 40}}
                folder = root / "study" / f"p{p:03d}"
                folder.mkdir(parents=True)
                (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            pipeline = types.ModuleType("run_pipeline")
            pipeline.validate_completion = lambda value: {"state": "incomplete_bins", "verified_completed_fits": 110}
            with patch.dict("sys.modules", {"run_pipeline": pipeline}):
                exported = api.export_privacy_utility(root)
            self.assertEqual(exported["verification"]["state"], "incomplete_bins")
            self.assertEqual(len(exported["figure_paths"]), 9)
            self.assertTrue(all(Path(path).is_file() for path in exported["figure_paths"]))
            self.assertTrue(Path(exported["artifacts"]).is_file())
            with (root / "figures" / "binned_statistics.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 50)
            for row in rows:
                self.assertEqual(int(row["n"]), 2)
                self.assertAlmostEqual(float(row["mean_utility_percent"]), 80.0)
                self.assertAlmostEqual(float(row["utility_sample_sd_percent"]), np.sqrt(200.0))
            with (root / "figures" / "control_utility.csv").open(encoding="utf-8", newline="") as stream:
                controls = list(csv.DictReader(stream))
            self.assertEqual(len(controls), 15)
            local = next(row for row in controls if row["p"] == "10" and row["method"] == "local_pooled")
            self.assertEqual(float(local["test_balanced_accuracy_percent"]), 63.0)
            provenance = json.loads(Path(exported["artifacts"]).read_text())
            self.assertFalse(provenance["all_bin_quotas_met"])


if __name__ == "__main__":
    unittest.main()
