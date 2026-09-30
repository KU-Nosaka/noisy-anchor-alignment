"""Pure bookkeeping tests; no LFWA records, FaceNet, accelerator or training."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import data_pipeline as dp

try:
    import torch
except ImportError:
    torch = None


def seed(*parts):
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:15], 16)


def context():
    return SimpleNamespace(input_hash="test-input", cfg=SimpleNamespace(seed=20260928, n_participants=5),
                           base=SimpleNamespace(named_seed=seed))


def score(ctx, condition):
    return (10., 20., 30., 40., 55.)[condition["proposal"] % 5]


class PipelineTests(unittest.TestCase):
    def test_boundary_rule_and_invalid_scores(self):
        self.assertEqual([dp.linkage_bin(x) for x in (0, 14.9, 15, 24.9, 25, 34.9, 35, 44.9, 45, 100)],
                         [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        for bad in (-1, 101, float("nan"), float("inf")):
            with self.assertRaises(RuntimeError):
                dp.linkage_bin(bad)

    def test_controls_rerun_each_local_model_for_every_p(self):
        rows = [row for p in dp.PARTICIPANTS for row in dp.control_conditions(p)]
        self.assertEqual(len(rows), 97)
        self.assertEqual(len({row["id"] for row in rows}), 97)
        self.assertEqual(sum(row["family"] == "local" for row in rows), 87)

    def test_interrupted_calibration_resumes_without_rescoring(self):
        with tempfile.TemporaryDirectory() as directory:
            seen = []
            def interrupted(ctx, condition):
                if len(seen) == 13:
                    raise RuntimeError("simulated interruption")
                seen.append(condition["id"])
                return score(ctx, condition)
            with patch.object(dp, "_linkage", interrupted), self.assertRaisesRegex(RuntimeError, "simulated"):
                dp.calibrate(context(), directory)
            calls = []
            def resumed(ctx, condition):
                calls.append(condition["id"])
                return score(ctx, condition)
            with patch.object(dp, "_linkage", resumed):
                conditions = dp.calibrate(context(), directory)
            self.assertEqual(len(conditions), 200)
            self.assertEqual(len(calls), 187)
            self.assertFalse(set(seen) & set(calls))
            dp.validate_accepted(conditions, 5)
            with patch.object(dp, "_linkage", side_effect=AssertionError("must not rescore")):
                self.assertEqual(dp.calibrate(context(), directory), conditions)

    def test_changed_proposal_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(dp, "_linkage", side_effect=[10., RuntimeError("stop")]):
                with self.assertRaises(RuntimeError):
                    dp.calibrate(context(), directory)
            path = Path(directory) / "calibration/attempts/anchor_000000.json"
            row = dp.read(path)
            row["condition"]["scale"] *= 1.1
            dp.atomic_json(path, row)
            with patch.object(dp, "_linkage", score), self.assertRaisesRegex(RuntimeError, "Saved proposal changed"):
                dp.calibrate(context(), directory)

    def test_budget_exhaustion_never_claims_completed_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(dp, "MAX_PROPOSALS", 12), patch.object(dp, "_linkage", lambda *args: 10.):
                with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                    dp.calibrate(context(), directory)
            self.assertFalse((Path(directory) / "calibration/accepted_conditions.json").exists())
            self.assertEqual(len(list((Path(directory) / "calibration/attempts").glob("*.json"))), 24)
            self.assertTrue((Path(directory) / "calibration/anchor_limited_bins.json").exists())
            self.assertTrue((Path(directory) / "calibration/private_limited_bins.json").exists())
            self.assertTrue((Path(directory) / "calibration/partial_conditions.json").exists())


@unittest.skipIf(torch is None, "Torch is required for numerical protocol checks")
class NumericalPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(7)
        h, p = 400, 5
        self.ctx = context()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.ctx.out = Path(self.temporary.name)/"study"
        self.ctx.protocol_config = {"array_cache_root":str(Path(self.temporary.name)/"arrays"), "transform_batch_rows":2}
        self.ctx.blocks = {}
        for split, count in (("train", 5), ("val", 3), ("test", 4)):
            self.ctx.blocks[split] = (
                [torch.randn(count, h, dtype=torch.float32, generator=generator) for _ in range(p)],
                [torch.arange(count) % 2 for _ in range(p)], [])
        o0 = torch.diag(torch.where(torch.arange(h) % 2 == 0, 1., -1.)).double()
        self.ctx.deployment = {"rotations": [o0] + [torch.eye(h, dtype=torch.float64) for _ in range(4)],
                               "translations": [torch.linspace(-.5, .5, h).double(),
                                                *([torch.linspace(.5, -.5, h).double()] * 4)],
                               "anchor": torch.randn(8, h, dtype=torch.float64, generator=generator)}
        self.ctx.linkage = SimpleNamespace(query_splits=("train", "val", "test"),
                                           query_target_positions=(2, 0, 3))
        self.ctx.core = SimpleNamespace(invert_known_gdp=lambda q, o, psi:
                                       SimpleNamespace(reduced=(q - psi) @ o.T))

    def condition(self, family):
        return {"id": "synthetic", "p": 5, "family": family, "scale": .2,
                "proposal": 42, "input_hash": self.ctx.input_hash}

    def test_private_auditor_noise_matches_training_rows(self):
        condition = self.condition("private")
        arrays = dp.build_arrays(self.ctx, condition)
        recovered = dp._attack_reduced(self.ctx, condition)
        dep = self.ctx.deployment
        expected = []
        for split, position in zip(self.ctx.linkage.query_splits, self.ctx.linkage.query_target_positions):
            # Target participant 1 follows participant 0 in the concatenation.
            offset = len(self.ctx.blocks[split][0][0])
            upload = arrays[split + "_x"][offset + position].flatten().double()
            expected.append((upload - dep["translations"][0]) @ dep["rotations"][0].T)
        self.assertTrue(torch.allclose(recovered, torch.stack(expected), atol=4e-7, rtol=1e-6))
        repeated = dp.build_arrays(self.ctx, condition)
        self.assertTrue(torch.equal(arrays["train_x"], repeated["train_x"]))
        changed = dp.build_arrays(self.ctx, {**condition, "proposal": 43})
        self.assertFalse(torch.equal(arrays["train_x"], changed["train_x"]))

    def test_controls_and_local_shapes_keep_correct_participant_data(self):
        for family in ("c_gdp", "i_gdp", "local"):
            c = self.condition(family)
            if family == "local":
                c["participant"] = 1
            arrays = dp.build_arrays(self.ctx, c)
            self.assertEqual(arrays["train_x"].shape, (5 if family == "local" else 25, 400))
            self.assertEqual(arrays["train_y"].dtype, torch.float32)
            self.assertEqual(arrays["alignment_seconds"], 0.)
            if family == "local":
                self.assertTrue(torch.equal(arrays["train_x"].flatten(1), self.ctx.blocks["train"][0][1].float()))

    def test_anchor_uses_returned_spectral_rotation_without_iteration(self):
        captured = []
        def spectral(centered):
            captured.append(centered)
            return torch.eye(400, dtype=torch.float64).repeat(5, 1, 1), {"alignment_seconds": .01}
        import spectral as spectral_module
        with patch.object(spectral_module, "spectral_align", spectral):
            arrays = dp.build_arrays(self.ctx, self.condition("anchor"))
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].dtype, torch.float64)
        self.assertLess(float(captured[0].mean(1).abs().max()), 1e-12)
        self.assertEqual(arrays["metadata"]["alignment"], "spectral_only")
        self.assertEqual(arrays["metadata"]["spectral_svd_and_polar_seconds"], .01)

    def test_cached_array_checksum_and_release_are_condition_scoped(self):
        condition=self.condition('private')
        arrays=dp.build_arrays(self.ctx,condition)
        _,directory=dp._array_directory(self.ctx,condition)
        del arrays
        path=directory/'train_x.npy'
        with path.open('r+b') as stream:
            stream.seek(-1,2); byte=stream.read(1); stream.seek(-1,2); stream.write(bytes([byte[0]^1]))
        with self.assertRaisesRegex(RuntimeError,'checksum'):
            dp.build_arrays(self.ctx,condition)
        dp.release_arrays(self.ctx,condition)
        self.assertFalse(directory.exists())

    def test_chunked_cpu_normal_draws_exactly_match_full_h400_draw(self):
        for rows in (1,3,77,2051):
            shape=(rows,400); condition=self.condition('private')
            full=dp._noise(self.ctx,condition,shape,'train',1)
            generator=dp._noise_generator(self.ctx,condition,'train',1)
            chunks=[condition['scale']*torch.randn((min(31,rows-start),400),
                     dtype=torch.float64,generator=generator) for start in range(0,rows,31)]
            self.assertTrue(torch.equal(full,torch.cat(chunks)))


if __name__ == "__main__":
    unittest.main()
