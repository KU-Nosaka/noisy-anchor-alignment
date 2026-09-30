"""CPU float64 spectral alignment, stopping before any GPM refinement.

This is the spectral initializer used by ``core_protocols.gpm_align``:
concatenate the centered anchor uploads horizontally, take the first h
right singular vectors, and project each participant block onto O(h).
The SVD basis (including its global gauge) is left unchanged.  In
particular, no reference-participant gauge fixing or iterative update is
applied.  The implementation never constructs the (p*h)-square Gram
matrix.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch


def _validated_input(centered: Any) -> tuple[torch.Tensor, float]:
    """Validate, without silently centering or changing the input gauge."""
    if isinstance(centered, torch.Tensor):
        if centered.device.type != "cpu":
            raise ValueError("Spectral alignment requires CPU input tensors.")
        if centered.is_complex() or centered.dtype == torch.bool:
            raise TypeError("Centered uploads must contain real numeric values.")
        source_epsilon = (
            torch.finfo(centered.dtype).eps
            if centered.is_floating_point()
            else torch.finfo(torch.float64).eps
        )
        uploads = centered.detach().to(dtype=torch.float64).contiguous()
    elif isinstance(centered, np.ndarray):
        if centered.dtype.kind not in "fiu":
            raise TypeError("Centered uploads must contain real numeric values.")
        source_epsilon = (
            float(np.finfo(centered.dtype).eps)
            if centered.dtype.kind == "f"
            else torch.finfo(torch.float64).eps
        )
        # Copy also supports NumPy arrays with negative strides/read-only views.
        uploads = torch.from_numpy(np.array(centered, dtype=np.float64, copy=True))
    else:
        raise TypeError("Expected a CPU torch.Tensor or numpy.ndarray.")

    if uploads.ndim != 3:
        raise ValueError("Expected centered uploads with shape (p, r, h).")
    p, r, h = uploads.shape
    if p < 1 or r < 1 or h < 1:
        raise ValueError("Participant, anchor-row, and feature counts must be positive.")
    if r <= h:
        raise ValueError("More than h anchor rows are needed for centered rank-h data.")
    if not bool(torch.isfinite(uploads).all()):
        raise ValueError("Centered uploads contain non-finite values.")

    # Centering error is scaled per participant, so a large upload cannot hide
    # a materially uncentered smaller upload.  Honor the source precision;
    # float32 inputs may already have accumulated float32 centering roundoff.
    scale = uploads.abs().amax(dim=(1, 2)).clamp_min(torch.finfo(torch.float64).tiny)
    relative_mean = uploads.mean(dim=1).abs().amax(dim=1) / scale
    centering_tolerance = 128.0 * max(source_epsilon, torch.finfo(torch.float64).eps)
    maximum_relative_mean = float(relative_mean.max())
    if maximum_relative_mean > centering_tolerance:
        raise ValueError(
            "Uploads must already be centered across anchor rows; "
            f"relative mean {maximum_relative_mean:.3g} exceeds "
            f"{centering_tolerance:.3g}."
        )
    return uploads, maximum_relative_mean


def spectral_align(centered: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return spectral-only rotations and JSON-serializable diagnostics.

    Args:
        centered: CPU torch tensor or NumPy array of shape ``(p, r, h)``.
            Each participant's upload must already have zero column means.

    Returns:
        Rotations of shape ``(p, h, h)``, on CPU in torch.float64, and a
        diagnostic dictionary.  Participants align using ``B_i @ R_i``.

    The rank check concerns the concatenation D=[B_1 ... B_p]: its h-th
    singular value must exceed the standard float64 numerical-rank
    tolerance.  No individual block-rank assumption is imposed.  A
    singular polar block has a non-unique orthogonal completion; its
    smallest singular value is reported for interpretation.

    ``alignment_seconds`` sums the timed SVD and block-polar phases.
    ``alignment_wall_seconds`` additionally includes the rank check and
    phase bookkeeping.  Validation, concatenation, and final diagnostics
    are reported separately; no objective or fixed-point iteration is
    evaluated just to produce diagnostics.
    """
    total_started = time.perf_counter()
    uploads, maximum_relative_mean = _validated_input(centered)
    p, r, h = uploads.shape
    concatenated = torch.cat(list(uploads.unbind(dim=0)), dim=1)
    validation_seconds = time.perf_counter() - total_started

    alignment_started = time.perf_counter()
    svd_started = time.perf_counter()
    # Keep this exact thin SVD rather than diagonalizing D @ D.T, which
    # squares the condition number and can choose a different basis/gauge.
    _, singular_values, vh = torch.linalg.svd(concatenated, full_matrices=False)
    svd_seconds = time.perf_counter() - svd_started
    rank_tolerance = float(singular_values[0]) * max(concatenated.shape) * torch.finfo(torch.float64).eps
    numerical_rank = int((singular_values > rank_tolerance).sum())
    if numerical_rank < h:
        raise ValueError(
            f"Concatenated uploads have numerical rank {numerical_rank}, below h={h}."
        )

    polar_started = time.perf_counter()
    leading_vectors = vh[:h].mT
    rotations = []
    block_minimum_singular_values = []
    # Use the same block order and unbatched polar SVDs as the existing
    # initializer, preserving its returned global coordinate system.
    for block in leading_vectors.split(h, dim=0):
        u_block, s_block, vh_block = torch.linalg.svd(block, full_matrices=False)
        rotations.append(u_block @ vh_block)
        block_minimum_singular_values.append(float(s_block[-1]))
    rotations_tensor = torch.stack(rotations, dim=0)
    polar_seconds = time.perf_counter() - polar_started
    alignment_wall_seconds = time.perf_counter() - alignment_started

    diagnostic_started = time.perf_counter()
    identity = torch.eye(h, dtype=torch.float64)
    orthogonality_errors = torch.linalg.matrix_norm(
        rotations_tensor.mT @ rotations_tensor - identity, ord="fro", dim=(-2, -1)
    )
    diagnostics: dict[str, Any] = {
        "estimator": "spectral_only",
        "algorithm": "thin_svd_leading_right_vectors_then_block_polar",
        "basis_gauge": "unmodified_svd_basis",
        "device": "cpu",
        "dtype": "float64",
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "participants": p,
        "anchor_rows": r,
        "embedding_dimension": h,
        "numerical_rank": numerical_rank,
        "rank_tolerance": rank_tolerance,
        "largest_singular_value": float(singular_values[0]),
        "retained_minimum_singular_value": float(singular_values[h - 1]),
        "maximum_relative_column_mean": maximum_relative_mean,
        "block_minimum_singular_values": block_minimum_singular_values,
        "maximum_orthogonality_error_frobenius": float(orthogonality_errors.max()),
        "validation_and_concatenation_seconds": validation_seconds,
        "svd_seconds": svd_seconds,
        "polar_seconds": polar_seconds,
        "alignment_seconds": svd_seconds + polar_seconds,
        "alignment_wall_seconds": alignment_wall_seconds,
    }
    diagnostics["diagnostic_seconds"] = time.perf_counter() - diagnostic_started
    diagnostics["total_seconds"] = time.perf_counter() - total_started
    return rotations_tensor, diagnostics
