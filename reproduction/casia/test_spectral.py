"""Small numerical tests of the estimator, independent of experiment data."""

from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np
import torch

from spectral import spectral_align


def assert_close(actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float) -> None:
    # NumPy comparison avoids loading PyTorch's distributed-tensor test
    # helpers (and their optional symbolic-algebra dependencies) for CPU data.
    np.testing.assert_allclose(actual.detach().numpy(), expected.detach().numpy(), rtol=rtol, atol=atol)


def centered_random(p: int = 3, r: int = 17, h: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(4197)
    values = torch.randn(p, r, h, generator=generator, dtype=torch.float64)
    return values - values.mean(dim=1, keepdim=True)


def reference_spectral(uploads: torch.Tensor) -> torch.Tensor:
    """Direct transcription of the existing GPM initialization only."""
    h = uploads.shape[-1]
    _, _, vh = torch.linalg.svd(torch.cat(list(uploads), dim=1), full_matrices=False)
    rotations = []
    for block in vh[:h].mT.split(h, dim=0):
        u, _, vh_block = torch.linalg.svd(block, full_matrices=False)
        rotations.append(u @ vh_block)
    return torch.stack(rotations)


class SpectralAlignmentTests(unittest.TestCase):
    def test_matches_existing_initializer_including_global_gauge(self) -> None:
        uploads = centered_random()
        actual, diagnostics = spectral_align(uploads)
        assert_close(actual, reference_spectral(uploads), rtol=0, atol=0)
        self.assertEqual(actual.shape, (3, 4, 4))
        self.assertEqual(actual.dtype, torch.float64)
        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(diagnostics["estimator"], "spectral_only")
        self.assertNotIn("fixed_point_residual", diagnostics)
        self.assertNotIn("converged", diagnostics)
        json.dumps(diagnostics, allow_nan=False)

    def test_noiseless_uploads_align_up_to_one_common_gauge(self) -> None:
        generator = torch.Generator().manual_seed(723)
        common = torch.randn(19, 5, dtype=torch.float64, generator=generator)
        common -= common.mean(dim=0)
        source_rotations = torch.linalg.qr(
            torch.randn(4, 5, 5, dtype=torch.float64, generator=generator)
        ).Q
        uploads = torch.stack([common @ q for q in source_rotations])
        rotations, diagnostics = spectral_align(uploads)
        aligned = uploads @ rotations
        assert_close(aligned, aligned[0].expand_as(aligned), rtol=2e-13, atol=2e-13)
        assert_close(
            rotations.mT @ rotations,
            torch.eye(5, dtype=torch.float64).expand(4, -1, -1),
            rtol=1e-13,
            atol=1e-13,
        )
        self.assertEqual(diagnostics["numerical_rank"], 5)

    def test_projector_agrees_with_leading_gram_eigenspace(self) -> None:
        uploads = centered_random(p=3, r=13, h=3)
        d = torch.cat(list(uploads), dim=1)
        _, _, vh = torch.linalg.svd(d, full_matrices=False)
        _, eigenvectors = torch.linalg.eigh(d.mT @ d)
        spectral_vectors = vh[:3].mT
        gram_vectors = eigenvectors[:, -3:]
        assert_close(
            spectral_vectors @ spectral_vectors.mT,
            gram_vectors @ gram_vectors.mT,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_exactly_one_thin_svd_and_one_polar_per_participant(self) -> None:
        uploads = centered_random(p=5)
        original = torch.linalg.svd
        with mock.patch.object(torch.linalg, "svd", wraps=original) as svd:
            spectral_align(uploads)
        self.assertEqual(svd.call_count, 6)
        self.assertEqual(svd.call_args_list[0].args[0].shape, (17, 20))
        self.assertTrue(all(call.kwargs == {"full_matrices": False} for call in svd.call_args_list))
        self.assertTrue(all(call.args[0].shape == (4, 4) for call in svd.call_args_list[1:]))

    def test_numpy_noncontiguous_float32_and_input_preservation(self) -> None:
        array = centered_random().numpy().astype(np.float32)[:, ::-1, :]
        array -= array.mean(axis=1, keepdims=True)
        before = array.copy()
        rotations, diagnostics = spectral_align(array)
        np.testing.assert_array_equal(array, before)
        self.assertEqual(rotations.dtype, torch.float64)
        json.dumps(diagnostics, allow_nan=False)

    def test_deterministic_output_and_nonnegative_phase_timings(self) -> None:
        uploads = centered_random()
        first, diagnostic = spectral_align(uploads)
        second, _ = spectral_align(uploads)
        assert_close(first, second, rtol=0, atol=0)
        for key in (
            "svd_seconds", "polar_seconds", "alignment_seconds", "alignment_wall_seconds",
            "diagnostic_seconds", "validation_and_concatenation_seconds", "total_seconds",
        ):
            self.assertGreaterEqual(diagnostic[key], 0)
        self.assertEqual(diagnostic["alignment_seconds"], diagnostic["svd_seconds"] + diagnostic["polar_seconds"])
        self.assertGreaterEqual(diagnostic["total_seconds"], diagnostic["alignment_wall_seconds"])

    def test_rejects_invalid_input(self) -> None:
        valid = centered_random()
        nonfinite = valid.clone()
        nonfinite[0, 0, 0] = float("nan")
        rank_deficient = valid.clone()
        rank_deficient[:, :, 1:] = 0
        examples = (
            ([], TypeError),
            (valid[0], ValueError),
            (valid[:, :0], ValueError),
            (valid[:, :4], ValueError),
            (valid + 0.1, ValueError),
            (nonfinite, ValueError),
            (rank_deficient, ValueError),
            (torch.zeros(2, 7, 4, dtype=torch.float64), ValueError),
            (valid.to(torch.complex128), TypeError),
            (np.zeros((2, 7, 4), dtype=object), TypeError),
            (torch.ones(2, 7, 4, dtype=torch.bool), TypeError),
        )
        for inputs, exception in examples:
            with self.subTest(shape=getattr(inputs, "shape", None), exception=exception):
                with self.assertRaises(exception):
                    spectral_align(inputs)


if __name__ == "__main__":
    unittest.main()
