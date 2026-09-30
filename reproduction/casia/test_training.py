"""CPU contract tests; the Colab notebook also runs training.pilot(device='cuda')."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np

import training


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sample(self):
        generator = torch.Generator().manual_seed(27)
        x = torch.randn(41, 400, generator=generator) * 0.1
        y = (torch.arange(41) % 2).float()
        return x[:17], y[:17], x[17:28], y[17:28], x[28:], y[28:]

    def test_architecture_and_padding_preserve_gradient(self):
        model = training._new_model(85, "cpu")
        self.assertEqual((model.conv1.in_channels, model.conv1.out_channels,
                          model.conv1.kernel_size), (1, 10, (5, 5)))
        self.assertEqual((model.conv2.in_channels, model.conv2.out_channels,
                          model.conv2.kernel_size), (10, 20, (5, 5)))
        self.assertEqual((model.fc1.in_features, model.fc1.out_features), (80, 50))
        self.assertEqual((model.fc2.in_features, model.fc2.out_features), (50, 1))
        self.assertEqual(tuple(model._convolve(torch.zeros(1, 1, 20, 20)).shape),
                         (1, 20, 2, 2))
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        x, y, *_ = self.sample()
        reference = training._new_model(85, "cpu")
        expected = torch.nn.functional.binary_cross_entropy_with_logits(reference(x), y)
        expected.backward()
        bx, by, mask = training._padded_batch(x, y, torch.arange(len(y)), 256, "cpu")
        actual = training._masked_loss(model(bx), by, mask)
        actual.backward()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)

    def test_resume_equals_uninterrupted_and_reuse_verifies_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            # Two train minibatches exercise permutation-generator restoration,
            # including the seven-record final minibatch at the fixed size256.
            generator = torch.Generator().manual_seed(486)
            x = torch.randn(287, 400, generator=generator) * 0.1
            y = (torch.arange(287) % 2).float()
            inputs = x[:263], y[:263], x[263:274], y[263:274], x[274:], y[274:]
            kwargs = dict(seed=212, condition_signature={"test": "resume"}, device="cpu")
            fresh = training.train_condition(*inputs, path / "fresh", **kwargs)

            class Pause(Exception):
                pass

            def pause(event):
                if event["stage"] == "epoch_complete" and event["epoch"] == 3:
                    raise Pause()

            with self.assertRaises(Pause):
                training.train_condition(*inputs, path / "resumed", **kwargs, progress=pause)
            resumed = training.train_condition(*inputs, path / "resumed", **kwargs)
            self.assertEqual(resumed["training"]["resumed_from_epoch"], 3)
            self.assertEqual(fresh["training"]["history"], resumed["training"]["history"])
            self.assertEqual(fresh["test"], resumed["test"])
            self.assertEqual(fresh["training"]["best_epoch"], resumed["training"]["best_epoch"])
            state1 = torch.load(path / "fresh/model.pt", weights_only=False)["state_dict"]
            state2 = torch.load(path / "resumed/model.pt", weights_only=False)["state_dict"]
            self.assertTrue(all(torch.equal(state1[k], state2[k]) for k in state1))
            before = (path / "resumed/result.json").read_bytes()
            training.train_condition(*inputs, path / "resumed", **kwargs)
            self.assertEqual(before, (path / "resumed/result.json").read_bytes())
            with (path / "resumed/predictions.npz").open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(RuntimeError, "altered"):
                training.train_condition(*inputs, path / "resumed", **kwargs)

    def test_signature_and_checkpoint_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            inputs = self.sample()

            class Pause(Exception):
                pass

            def pause(event):
                if event["stage"] == "epoch_complete":
                    raise Pause()

            with self.assertRaises(Pause):
                training.train_condition(*inputs, path, seed=1, condition_signature="A", device="cpu", progress=pause)
            with self.assertRaisesRegex(RuntimeError, "signature"):
                training.train_condition(*inputs, path, seed=1, condition_signature="B", device="cpu")
            manifest = json.loads((path / "checkpoint_manifest.json").read_text())
            with (path / manifest["artifacts"]["checkpoint"]["path"]).open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(RuntimeError, "altered"):
                training.train_condition(*inputs, path, seed=1, condition_signature="A", device="cpu")

    def test_tie_rule_threshold_and_all_records(self):
        # A learning rate of zero is confined to this unit test; public SETTINGS
        # and the production Adam factory are restored before any experiment.
        with tempfile.TemporaryDirectory() as temporary:
            original = training._optimizer
            training._optimizer = lambda model: torch.optim.Adam(model.parameters(), lr=0.0)
            try:
                result = training.train_condition(*self.sample(), temporary, seed=1,
                                                  condition_signature="tie", device="cpu")
            finally:
                training._optimizer = original
            self.assertEqual(result["training"]["best_epoch"], 1)
            self.assertEqual(result["test"]["n"], 13)
            self.assertEqual(len(result["training"]["history"]), 15)
            metrics = training._metrics(torch.tensor([0., 1.]), torch.tensor([0.499, 0.5]), 0.0)
            self.assertEqual(metrics["balanced_accuracy"], 1.0)

    def test_input_and_backend_validation(self):
        with self.assertRaises(ValueError):
            training._inputs(torch.zeros(2, 399), torch.zeros(2), "bad")
        with self.assertRaises(ValueError):
            training._inputs(torch.zeros(2, 4, 10, 10), torch.zeros(2), "bad")
        with self.assertRaises(ValueError):
            training._inputs(torch.full((2, 400), float("nan")), torch.zeros(2), "bad")
        with self.assertRaises(ValueError):
            training._inputs(torch.zeros(2, 400), torch.tensor([0., 2.]), "bad")
        with self.assertRaises(ValueError):
            training.runtime_probe("unsupported")
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(RuntimeError, "autocast"):
                training.runtime_probe("cpu")

    def test_native_rows_and_prescribed_reshape_have_identical_artifacts(self):
        flat_inputs = self.sample()
        image_inputs = [value.reshape(-1, 1, 20, 20) if index % 2 == 0 else value
                        for index, value in enumerate(flat_inputs)]
        self.assertEqual(tuple(image_inputs[0].shape), (17, 1, 20, 20))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            kwargs = dict(seed=325, condition_signature="shape-equivalence", device="cpu")
            image_result = training.train_condition(*image_inputs, output / "images", **kwargs)
            flat_result = training.train_condition(*flat_inputs, output / "flat", **kwargs)
            self.assertEqual(image_result["signature"], flat_result["signature"])
            self.assertEqual(image_result["training"]["history"], flat_result["training"]["history"])
            self.assertEqual(image_result["test"], flat_result["test"])
            image_state = torch.load(output / "images/model.pt", weights_only=False)["state_dict"]
            flat_state = torch.load(output / "flat/model.pt", weights_only=False)["state_dict"]
            self.assertTrue(all(torch.equal(image_state[key], flat_state[key]) for key in image_state))
            # Completed output is reusable across the two input representations.
            self.assertEqual(training.train_condition(*flat_inputs, output / "images", **kwargs)["signature"],
                             image_result["signature"])

    def test_existing_worker_lock_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            lock = path / ".training.lock"
            lock.write_text('{"owner":"another-worker"}')
            with self.assertRaisesRegex(RuntimeError, "Existing training lock"):
                training.train_condition(*self.sample(), path, seed=1,
                                          condition_signature="lock", device="cpu")
            self.assertEqual(lock.read_text(), '{"owner":"another-worker"}')

    def test_cpu_pilot(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = training.pilot(temporary, device="cpu")
            self.assertTrue(report["passed"])
            self.assertTrue(report["checks"]["all_test_records_retained"])
            self.assertTrue(report["checks"]["completed_artifacts_reverified"])

    def test_memmap_zero_copy_and_streaming_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.dat"
            mapped = np.memmap(path, dtype=np.float32, mode="w+", shape=(32771, 400))
            mapped[:] = 0.25
            mapped[-1, -1] = 0.5
            mapped.flush()
            source = torch.from_numpy(mapped)
            labels = (torch.arange(len(source)) % 2).float()
            checked, checked_labels = training._inputs(source, labels, "memmap")
            self.assertEqual(checked.data_ptr(), source.data_ptr())
            self.assertEqual(checked_labels.data_ptr(), labels.data_ptr())
            self.assertEqual(training._tensor_sha(checked), training._sha(path))
            batch, _, _ = training._padded_batch(checked, labels, torch.tensor([32770, 0]), 256, "cpu")
            self.assertEqual(float(batch[0, -1]), 0.5)
            mapped[-1, -1] = float("nan")
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                training._inputs(source, labels, "memmap")
            del checked, source, mapped


if __name__ == "__main__":
    unittest.main(verbosity=2)
