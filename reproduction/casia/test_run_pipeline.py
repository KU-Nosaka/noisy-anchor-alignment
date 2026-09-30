"""Durable all-p completion, quota and worker ownership checks; no GPU needed."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_study as study
import run_pipeline as pipeline
import notebook_api


def conditions(p, full=True):
    result = [{"id": f"p{p:03d}_{family}", "p": p, "family": family} for family in ("c_gdp", "i_gdp")]
    result.extend({"id": f"p{p:03d}_local_{i}", "p": p, "family": "local", "participant": i} for i in range(p))
    if full:
        for family in ("anchor", "private"):
            for b in range(5):
                for i in range(20):
                    result.append({"id": f"p{p:03d}_{family}_{b}_{i}", "p": p, "family": family,
                                   "bin": b, "linkage_percent": (study.BINS[b] + study.BINS[b + 1]) / 2,
                                   "scale": .01 if family == "anchor" else .5})
    return result


def make_participant(root, p, identity, full):
    directory = root / "study" / f"p{p:03d}"
    provenance = {"fixed_input": p, "input_hash": study.object_hash({"fixed_input": p})}
    study.write(directory / "input_provenance.json", provenance)
    signature = study.object_hash({"study_signature": study.object_hash(identity), "p": p,
                                   "input_provenance": provenance})
    schedule = conditions(p, full)
    for condition in schedule:
        if condition["family"] in ("anchor", "private"):
            condition["input_hash"] = provenance["input_hash"]
    study.write(directory / "schedule.json", {"p": p, "signature": signature, "conditions": schedule,
                                               "sampling_mode": "quota_bins", "calibration_complete": full})
    if full:
        accepted = [c for c in schedule if c["family"] in ("anchor", "private")]
        study.write(directory / "calibration/accepted_conditions.json", {
            "input_hash": provenance["input_hash"], "conditions": accepted, "conditions_sha256": study.object_hash(accepted)})
    clean = {"linkage_percent": 54.5, "n_identities": 100}
    study.write(directory / "clean_linkage.json", clean)
    rows = []
    for condition in schedule:
        folder = directory / "conditions" / condition["id"]
        seed = identity["specification"]["seed"] + condition.get("participant", 0)
        condition_sig = study.object_hash({"study_signature": signature, "condition": condition, "training_seed": seed})
        result = {"status": "complete", "signature": condition_sig,
                  "training": {"epochs": 15, "history": [{"epoch": i} for i in range(1, 16)]},
                  "test": {"confusion_matrix": [[2, 1], [1, 2]], "n": 6}}
        study.write(folder / "result.json", result)
        (folder / "model.pt").write_bytes(b"fixture-model")
        (folder / "predictions.npz").write_bytes(b"fixture-predictions")
        artifacts = {kind: {"path": name, "sha256": study.digest(folder / name)}
                     for kind, name in (("model", "model.pt"), ("predictions", "predictions.npz"), ("result", "result.json"))}
        study.write(folder / "completion_manifest.json", {"signature": condition_sig, "artifacts": artifacts})
        row = {"condition": condition, "condition_signature": condition_sig, "training_seed": seed, "result": result,
               "artifact_hashes": {path.name: study.digest(path) for path in folder.iterdir()}}
        study.write(folder / "record.json", row)
        rows.append(row)
    summary = {"state": "complete" if full else "incomplete_bins", "calibration_complete": full, "p": p,
               "planned_conditions": 202 + p, "completed_conditions": len(rows),
               "completed_ids": [r["condition"]["id"] for r in rows], "records": rows,
               "family_counts": dict(Counter(c["family"] for c in schedule)),
               "sampling_mode": "quota_bins", "bin_edges_percent": list(study.BINS),
               "bin_counts": {f: {str(b): sum(c["family"] == f and c.get("bin") == b for c in schedule)
                                  for b in range(5)} for f in ("anchor", "private")},
               "clean_linkage": clean, "clean_linkage_sha256": study.digest(directory / "clean_linkage.json"),
               "local_pooled": study.local_pooled_summary(rows, p)}
    study.write(directory / "summary.json", summary)
    return summary


class PipelineTests(unittest.TestCase):
    def specification(self):
        root = Path(tempfile.gettempdir()) / "casia_test_recipe"
        return notebook_api.study_spec("celeba", root / "raw", root / "cache", root / "output", "cpu")

    def test_schedule_all_p_and_invalid_bin_membership(self):
        self.assertEqual(sum(len(conditions(p)) for p in study.PARTICIPANTS), 1097)
        for p in study.PARTICIPANTS:
            study.validate_schedule(conditions(p), p)
        wrong = conditions(5)
        wrong[-1]["linkage_percent"] = 44.9
        with self.assertRaisesRegex(RuntimeError, "declared bin"):
            study.validate_schedule(wrong, 5)
        with self.assertRaisesRegex(RuntimeError, "composition"):
            study.validate_schedule(conditions(5, False), 5)
        study.validate_schedule(conditions(5, False), 5, complete=False)

    def test_revised_boundaries_are_left_closed(self):
        rows = conditions(2)
        for edge, target_bin in ((15., 1), (25., 2), (35., 3), (45., 4), (100., 4)):
            chosen = next(c for c in rows if c["family"] == "anchor" and c["bin"] == target_bin)
            changed = copy.deepcopy(rows)
            position = rows.index(chosen)
            changed[position]["linkage_percent"] = edge
            study.validate_schedule(changed, 2)
            changed[position]["bin"] = target_bin - 1
            with self.assertRaises(RuntimeError):
                study.validate_schedule(changed, 2)

    def test_no_fixed_draws_or_changed_quota_configuration(self):
        spec = self.specification()
        study.validate_spec(spec)
        for key, value in (("sampling_mode", "fixed_draws"), ("conditions_per_bin_per_family", 19),
                           ("h", 100), ("planned_training_fits", 1096),
                           ("identities_per_participant", 120), ("seed", 20261001),
                           ("dataset_seed", 20260712), ("reference_treatment", "raw_source"),
                           ("linkage_bin_edges_percent", [0, 15, 30, 45, 60, 100])):
            with self.assertRaisesRegex(RuntimeError, "fixed recipe"):
                study.validate_spec({**spec, key: value})

    def test_full_participant_gate_checks_real_artifacts_and_accepted_bins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = {"specification": self.specification()}
            summary = make_participant(root, 2, identity, full=True)
            self.assertEqual(pipeline._validate_participant(root / "study", 2, identity,
                             study.object_hash(identity)), summary)
            target = root / "study/p002/conditions/p002_anchor_0_0/model.pt"
            target.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "missing or changed"):
                pipeline._validate_participant(root / "study", 2, identity, study.object_hash(identity))

    def test_incomplete_terminal_is_checked_and_never_promoted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = {"specification": self.specification()}
            study.write(root / "study/run_identity.json", identity)
            summaries = [make_participant(root, p, identity, full=False) for p in study.PARTICIPANTS]
            receipt = {"state": "incomplete_bins", "planned_fits": 1097,
                       "completed_fits": sum(s["completed_conditions"] for s in summaries),
                       "study_signature": study.object_hash(identity),
                       "participant_summaries": [{k: v for k, v in s.items() if k != "records"} for s in summaries]}
            study.write(root / "study/incomplete.json", receipt)
            result = pipeline.validate_completion(root)
            self.assertEqual(result["state"], "incomplete_bins")
            self.assertEqual(result["completed_fits"], 97)
            study.write(root / "study/completion.json", {**receipt, "state": "complete", "completed_fits": 1097})
            with self.assertRaisesRegex(RuntimeError, "unique durable"):
                pipeline.validate_completion(root)
            (root / "study/incomplete.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "misstates"):
                pipeline.validate_completion(root)

    def test_locks_preserve_other_worker_and_release_own(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / ".pipeline.lock"
            lock.write_text('{"token":"other"}')
            with self.assertRaisesRegex(RuntimeError, "lock exists"):
                with pipeline.pipeline_worker(root):
                    pass
            self.assertEqual(lock.read_text(), '{"token":"other"}')
            lock.unlink()
            with self.assertRaisesRegex(RuntimeError, "simulate"):
                with pipeline.pipeline_worker(root):
                    self.assertTrue(lock.exists())
                    raise RuntimeError("simulate failure")
            self.assertFalse(lock.exists())

    def test_failed_child_cannot_create_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source"
            source.mkdir()
            output = directory / "new_run"
            config = directory / "experiment.json"
            spec = {**self.specification(), "source_root": str(source), "cache_root": str(directory / "cache"),
                    "module_dir": str(directory / "frozen")}
            study.write(config, spec)
            fake_dp = SimpleNamespace(configure=lambda spec: None,
                                      prepare_inputs=lambda spec, root, progress: str(source))
            child = SimpleNamespace(pid=222, stdout=iter(["child failed\n"]), wait=lambda **kwargs: 9, poll=lambda: 9)
            import data_pipeline
            with patch.object(data_pipeline, "configure", fake_dp.configure), patch.object(data_pipeline, "prepare_inputs", fake_dp.prepare_inputs), patch.object(pipeline.subprocess, "Popen", return_value=child):
                with self.assertRaisesRegex(RuntimeError, "Study exited 9"):
                    pipeline.run(output, config, device="cpu")
            self.assertFalse((output / "pipeline_completion.json").exists())
            self.assertEqual(study.read(output / "pipeline_status.json")["state"], "failed")
            self.assertFalse((output / ".pipeline.lock").exists())

    def test_condition_cache_release_follows_verified_fit_only(self):
        import torch
        import training
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            x = torch.zeros(10, 400)
            y = (torch.arange(10) % 2).float()
            arrays = {f"{split}_{key}": value for split in ("train", "val", "test")
                      for key, value in (("x", x), ("y", y))}
            released = []
            def release(ctx, condition):
                record = directory / "conditions" / condition["id"] / "record.json"
                self.assertTrue(record.exists())
                released.append(condition["id"])
            adapter = SimpleNamespace(build_arrays=lambda ctx, condition: dict(arrays), release_arrays=release)
            condition = {"id": "p002_c_gdp", "family": "c_gdp", "p": 2}
            import data_pipeline
            with patch.object(data_pipeline, "build_arrays", adapter.build_arrays), patch.object(data_pipeline, "release_arrays", adapter.release_arrays):
                row = study.train_one(None, condition, "root-signature", directory, 12, "cpu", lambda **event: None)
                self.assertEqual(row["result"]["status"], "complete")
                self.assertEqual(released, [condition["id"]])
                with patch.object(data_pipeline, "build_arrays", side_effect=AssertionError("must reuse durable receipt")):
                    study.train_one(None, condition, "root-signature", directory, 12, "cpu", lambda **event: None)
                failed = {**condition, "id": "p002_i_gdp", "family": "i_gdp"}
                with patch.object(training, "train_condition", side_effect=RuntimeError("interrupted fit")):
                    with self.assertRaisesRegex(RuntimeError, "interrupted fit"):
                        study.train_one(None, failed, "root-signature", directory, 12, "cpu", lambda **event: None)
                self.assertNotIn(failed["id"], released)


if __name__ == "__main__":
    unittest.main(verbosity=2)
