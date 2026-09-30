"""Float64 PyTorch implementations of the I-GDP paper protocols and attacks.

Records are row vectors.  Every GDP transform is therefore a right action:
``X @ O`` and ``A @ O``.  The module intentionally keeps model training out of
scope so its algebraic self-tests can run quickly on CPU or CUDA.  The shared
empirical reduction is the reference's uncentered, unnormalized public Gaussian
random projection; reconstruction uses its Moore--Penrose inverse rather than
assuming an orthonormal basis.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple, Union

import torch

Tensor = torch.Tensor
TensorOrBlocks = Union[Tensor, Sequence[Tensor]]
PROTOCOL_DTYPE = torch.float64


@dataclass
class GaussianProjectionReducer:
    """Low-level public Gaussian linear reducer.

    ``F_pinv`` is cached because Gaussian projection columns are not
    orthonormal and hence the decoder is not generally ``F.T``.
    """

    F: Tensor
    singular_values: Tensor
    F_pinv: Tensor

    def transform(self, x: Tensor) -> Tensor:
        return _as_tensor(x, self.F.device) @ self.F

    def transform_blocks(self, blocks: Sequence[Tensor]) -> List[Tensor]:
        return [self.transform(x) for x in blocks]

    def decode(self, z: Tensor) -> Tensor:
        return _as_tensor(z, self.F.device) @ self.F_pinv

@dataclass
class GPMDiagnostics:
    """Numerical diagnostics for PA-I-GDP's block-Gram GPM."""

    iterations: int
    converged: bool
    elapsed_seconds: float
    anchor_singular_values: Tensor
    anchor_sigma_min: float
    anchor_sigma_max: float
    anchor_condition: float
    upload_singular_values: List[Tensor]
    gram_eigenvalues: Tensor
    eigengap: float
    relative_eigengap: float
    multistart_objective_gap: float
    gopp_residual: float
    relative_gopp_residual: float
    fixed_point_residual: float
    objective_history: List[float]
    step_history: List[float]


@dataclass
class ProtocolResult:
    """Protocol transcript plus the matrix supplied to the analyst's model."""

    name: str
    private_uploads: List[Tensor]
    collaboration_blocks: List[Tensor]
    collaboration: Tensor
    rotations: List[Tensor]
    translations: List[Tensor]
    anchor_uploads: Optional[List[Tensor]] = None
    alignment_rotations: Optional[List[Tensor]] = None
    private_noises: Optional[List[Tensor]] = None
    anchor_noises: Optional[List[Tensor]] = None
    diagnostics: Optional[GPMDiagnostics] = None
    reference: Optional[int] = None


@dataclass
class AttackResult:
    """Estimated target GDP parameters and reduced-space reconstruction."""

    rotation: Tensor
    translation: Tensor
    reduced: Tensor
    method: str


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy when present, and PyTorch deterministically."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def make_generator(seed: int, device: Union[str, torch.device] = "cpu") -> torch.Generator:
    """Return a seeded generator on ``device``."""

    dev = torch.device(device)
    return torch.Generator(device=dev).manual_seed(seed)


def _as_tensor(x: Tensor, device: Optional[torch.device] = None) -> Tensor:
    dev = x.device if device is None and isinstance(x, Tensor) else device
    return torch.as_tensor(x, dtype=PROTOCOL_DTYPE, device=dev)


def _blocks(blocks: Sequence[Tensor], device: Optional[torch.device] = None) -> List[Tensor]:
    if not blocks:
        raise ValueError("At least one participant block is required.")
    dev = torch.device(device) if device is not None else torch.as_tensor(blocks[0]).device
    out = [_as_tensor(x, dev) for x in blocks]
    h = out[0].shape[1]
    if any(x.ndim != 2 or x.shape[1] != h for x in out):
        raise ValueError("All participant blocks must be 2-D with the same feature dimension.")
    return out


def _randn(shape: Tuple[int, ...], ref: Tensor, generator: Optional[torch.Generator]) -> Tensor:
    return torch.randn(shape, dtype=PROTOCOL_DTYPE, device=ref.device, generator=generator)


def haar_orthogonal(
    h: int,
    *,
    generator: Optional[torch.Generator] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tensor:
    """Sample a Haar orthogonal matrix in O(h) by sign-corrected Gaussian QR."""

    z = torch.randn((h, h), dtype=PROTOCOL_DTYPE, device=device, generator=generator)
    q, r = torch.linalg.qr(z)
    signs = torch.where(torch.diagonal(r) >= 0, 1.0, -1.0).to(q)
    return q * signs.unsqueeze(0)


def sample_gdp_parameters(
    num_users: int,
    h: int,
    *,
    translation_scale: float = 1.0,
    generator: Optional[torch.Generator] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[List[Tensor], List[Tensor]]:
    """Sample independent ``(O_i, Psi_i)`` pairs."""

    rotations = [haar_orthogonal(h, generator=generator, device=device) for _ in range(num_users)]
    translations = [
        translation_scale
        * torch.randn((h,), dtype=PROTOCOL_DTYPE, device=device, generator=generator)
        for _ in range(num_users)
    ]
    return rotations, translations


def make_shared_gaussian_projection(
    raw_dimension: int,
    h: int,
    *,
    seed: int,
    device: Union[str, torch.device] = "cpu",
) -> GaussianProjectionReducer:
    """Create the public Gaussian map ``F`` with iid ``N(0, 1)`` entries.

    For row-oriented observations the sole reduction relation is
    ``X_tilde = X @ F``.  A tall Gaussian matrix has full column rank with
    probability one, and its cached decoder is
    ``F^dagger = (F.T @ F)^(-1) @ F.T`` (computed numerically by ``pinv``).
    """

    if raw_dimension < 1:
        raise ValueError("raw_dimension must be positive.")
    if not 1 <= h <= raw_dimension:
        raise ValueError("h must be between 1 and raw_dimension.")
    dev = torch.device(device)
    generator = make_generator(seed, dev)
    F = torch.randn(
        (raw_dimension, h),
        dtype=PROTOCOL_DTYPE,
        device=dev,
        generator=generator,
    )
    singular_values = torch.linalg.svdvals(F)
    tolerance = torch.finfo(PROTOCOL_DTYPE).eps * max(F.shape) * singular_values[0]
    if singular_values[-1] <= tolerance:
        raise RuntimeError("Seeded Gaussian projection was numerically rank deficient.")
    return GaussianProjectionReducer(
        F=F,
        singular_values=singular_values,
        F_pinv=torch.linalg.pinv(F),
    )


def reduce_blocks(blocks: Sequence[Tensor], F: Tensor) -> List[Tensor]:
    """Apply the shared composition ``Xtilde_i = X_i F``."""

    F = _as_tensor(F)
    return [_as_tensor(x, F.device) @ F for x in blocks]


def decode_reduced(z: Tensor, F: Tensor, F_pinv: Optional[Tensor] = None) -> Tensor:
    """Decode a reduced reconstruction as ``Xhat = Xtildehat F^dagger``."""

    F = _as_tensor(F)
    pinv = torch.linalg.pinv(F) if F_pinv is None else _as_tensor(F_pinv, F.device)
    return _as_tensor(z, F.device) @ pinv


def projection_floor(x: Tensor, F: Tensor) -> Tensor:
    """Return the public map's Moore--Penrose linear reconstruction control."""

    F = _as_tensor(F)
    return decode_reduced(_as_tensor(x, F.device) @ F, F)


def orthogonal_polar(matrix: Tensor) -> Tensor:
    """Project a square matrix onto O(h) using ``U V^T``."""

    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def blockwise_polar(stacked: Tensor, h: int) -> Tensor:
    """Apply the polar projection to each h-by-h vertical block."""

    if stacked.ndim != 2 or stacked.shape[1] != h or stacked.shape[0] % h:
        raise ValueError("Expected a (p*h)-by-h stacked matrix.")
    return torch.cat([orthogonal_polar(b) for b in stacked.split(h, dim=0)], dim=0)


def _parameters(
    p: int,
    h: int,
    ref: Tensor,
    rotations: Optional[Sequence[Tensor]],
    translations: Optional[Sequence[Tensor]],
    generator: Optional[torch.Generator],
    translation_scale: float,
) -> Tuple[List[Tensor], List[Tensor]]:
    if rotations is None or translations is None:
        sampled_o, sampled_p = sample_gdp_parameters(
            p,
            h,
            translation_scale=translation_scale,
            generator=generator,
            device=ref.device,
        )
        rotations = sampled_o if rotations is None else rotations
        translations = sampled_p if translations is None else translations
    if len(rotations) != p or len(translations) != p:
        raise ValueError("One rotation and translation are required per participant.")
    os_ = [_as_tensor(o, ref.device) for o in rotations]
    psis = [_as_tensor(psi, ref.device).reshape(h) for psi in translations]
    eye = torch.eye(h, dtype=PROTOCOL_DTYPE, device=ref.device)
    if any(o.shape != (h, h) or not torch.allclose(o.mT @ o, eye, atol=1e-8, rtol=1e-8) for o in os_):
        raise ValueError("Every O_i must be h-by-h and orthogonal.")
    return os_, psis


def _noise_blocks(
    blocks: Sequence[Tensor],
    scale: float,
    noises: Optional[Sequence[Tensor]],
    generator: Optional[torch.Generator],
) -> List[Tensor]:
    if noises is None:
        return [scale * _randn(tuple(x.shape), x, generator) for x in blocks]
    if len(noises) != len(blocks):
        raise ValueError("One noise matrix is required per participant.")
    out = [_as_tensor(e, x.device) for e, x in zip(noises, blocks)]
    if any(e.shape != x.shape for e, x in zip(out, blocks)):
        raise ValueError("Noise matrices must match their participant block shapes.")
    return out


def _result(
    name: str,
    uploads: List[Tensor],
    blocks: List[Tensor],
    rotations: List[Tensor],
    translations: List[Tensor],
    **kwargs,
) -> ProtocolResult:
    return ProtocolResult(
        name=name,
        private_uploads=uploads,
        collaboration_blocks=blocks,
        collaboration=torch.cat(blocks, dim=0),
        rotations=rotations,
        translations=translations,
        **kwargs,
    )


def c_gdp(
    x_blocks: Sequence[Tensor],
    *,
    rotation: Optional[Tensor] = None,
    translation: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
    translation_scale: float = 1.0,
) -> ProtocolResult:
    """Common GDP: all users upload ``Xtilde_i O + Psi``."""

    xs = _blocks(x_blocks)
    h = xs[0].shape[1]
    os_, psis = _parameters(1, h, xs[0], None if rotation is None else [rotation], None if translation is None else [translation], generator, translation_scale)
    O, psi = os_[0], psis[0]
    uploads = [x @ O + psi for x in xs]
    return _result("C-GDP", uploads, uploads, [O] * len(xs), [psi] * len(xs))


def c_gdp_an(
    x_blocks: Sequence[Tensor],
    sigma: float,
    *,
    rotation: Optional[Tensor] = None,
    translation: Optional[Tensor] = None,
    noises: Optional[Sequence[Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    translation_scale: float = 1.0,
) -> ProtocolResult:
    """Common GDP with iid private-upload Gaussian noise."""

    base = c_gdp(
        x_blocks,
        rotation=rotation,
        translation=translation,
        generator=generator,
        translation_scale=translation_scale,
    )
    noise = _noise_blocks(base.private_uploads, sigma, noises, generator)
    uploads = [y + e for y, e in zip(base.private_uploads, noise)]
    return _result("C-GDP-AN", uploads, uploads, base.rotations, base.translations, private_noises=noise)


def i_gdp(
    x_blocks: Sequence[Tensor],
    *,
    rotations: Optional[Sequence[Tensor]] = None,
    translations: Optional[Sequence[Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    translation_scale: float = 1.0,
) -> ProtocolResult:
    """Independent GDP with no analyst-side alignment."""

    xs = _blocks(x_blocks)
    os_, psis = _parameters(len(xs), xs[0].shape[1], xs[0], rotations, translations, generator, translation_scale)
    uploads = [x @ o + psi for x, o, psi in zip(xs, os_, psis)]
    return _result("I-GDP", uploads, uploads, os_, psis)


def aa_alignment(
    private_uploads: Sequence[Tensor],
    anchor_uploads: Sequence[Tensor],
    reference: int = 0,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """Apply the draft's AA map using ``Bbar_i.T @ Bbar_s``."""

    ys = _blocks(private_uploads)
    bs_upload = _blocks(anchor_uploads, ys[0].device)
    p, h = len(ys), ys[0].shape[1]
    if len(bs_upload) != p or not 0 <= reference < p:
        raise ValueError("Invalid number of anchors or reference index.")
    means = [b.mean(dim=0) for b in bs_upload]
    centered = [b - mean for b, mean in zip(bs_upload, means)]
    rotations: List[Tensor] = []
    eye = torch.eye(h, dtype=PROTOCOL_DTYPE, device=ys[0].device)
    for i in range(p):
        rotations.append(eye if i == reference else orthogonal_polar(centered[i].mT @ centered[reference]))
    ref_mean = means[reference]
    aligned = [(y - mean) @ r + ref_mean for y, mean, r in zip(ys, means, rotations)]
    return aligned, rotations, means


def aa_i_gdp(
    x_blocks: Sequence[Tensor],
    anchor: Tensor,
    *,
    reference: int = 0,
    rotations: Optional[Sequence[Tensor]] = None,
    translations: Optional[Sequence[Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    translation_scale: float = 1.0,
) -> ProtocolResult:
    """Noiseless anchor-aligned independent GDP."""

    xs = _blocks(x_blocks)
    A = _as_tensor(anchor, xs[0].device)
    p, h = len(xs), xs[0].shape[1]
    if A.ndim != 2 or A.shape[1] != h:
        raise ValueError("anchor must be r-by-h.")
    os_, psis = _parameters(p, h, xs[0], rotations, translations, generator, translation_scale)
    ys = [x @ o + psi for x, o, psi in zip(xs, os_, psis)]
    Bs = [A @ o + psi for o, psi in zip(os_, psis)]
    aligned, rs, _ = aa_alignment(ys, Bs, reference)
    return _result(
        "AA-I-GDP",
        ys,
        aligned,
        os_,
        psis,
        anchor_uploads=Bs,
        alignment_rotations=rs,
        reference=reference,
    )


def aa_i_gdp_an(
    x_blocks: Sequence[Tensor],
    anchor: Tensor,
    sigma: float,
    *,
    reference: int = 0,
    rotations: Optional[Sequence[Tensor]] = None,
    translations: Optional[Sequence[Tensor]] = None,
    noises: Optional[Sequence[Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    translation_scale: float = 1.0,
) -> ProtocolResult:
    """AA-I-GDP with Gaussian noise on private uploads only."""

    base = aa_i_gdp(
        x_blocks,
        anchor,
        reference=reference,
        rotations=rotations,
        translations=translations,
        generator=generator,
        translation_scale=translation_scale,
    )
    noise = _noise_blocks(base.private_uploads, sigma, noises, generator)
    ys = [y + e for y, e in zip(base.private_uploads, noise)]
    aligned, rs, _ = aa_alignment(ys, base.anchor_uploads or [], reference)
    return _result(
        "AA-I-GDP-AN",
        ys,
        aligned,
        base.rotations,
        base.translations,
        anchor_uploads=base.anchor_uploads,
        alignment_rotations=rs,
        private_noises=noise,
        reference=reference,
    )


def build_block_gram(centered_anchor_uploads: Sequence[Tensor]) -> Tensor:
    """Build PA-I-GDP blocks ``G_ij = Bbar_i.T @ Bbar_j``."""

    Bs = _blocks(centered_anchor_uploads)
    rows = [torch.cat([bi.mT @ bj for bj in Bs], dim=1) for bi in Bs]
    G = torch.cat(rows, dim=0)
    return 0.5 * (G + G.mT)


def gpm_align(
    centered_anchor_uploads: Sequence[Tensor],
    *,
    raw_anchor: Optional[Tensor] = None,
    max_iters: int = 200,
    tol: float = 1e-10,
    n_random_starts: int = 0,
    restart_seed: int = 0,
) -> Tuple[List[Tensor], GPMDiagnostics]:
    """Run spectral initialization and blockwise-polar GPM.

    The implementation deliberately avoids materializing the
    ``(p*h)``-by-``(p*h)`` block Gram matrix.  With the centered anchor
    uploads concatenated as ``D = [Bbar_1 ... Bbar_p]``, its action is
    ``G @ R = D.T @ (D @ R)``.  The leading eigenspace of ``G = D.T @ D``
    is obtained from a thin SVD of ``D``.
    """

    started = time.perf_counter()
    Bs = _blocks(centered_anchor_uploads)
    p, h = len(Bs), Bs[0].shape[1]
    D = torch.cat(Bs, dim=1)
    if D.shape[0] < h:
        raise ValueError("At least h centered anchor rows are required for spectral initialization.")
    _, singular_values, vh = torch.linalg.svd(D, full_matrices=False)
    gram_eigenvalues = torch.zeros(p * h, dtype=PROTOCOL_DTYPE, device=D.device)
    gram_eigenvalues[: singular_values.numel()] = singular_values.square()
    stacked = blockwise_polar(vh[:h].mT, h)

    def gram_apply(r: Tensor) -> Tensor:
        return D.mT @ (D @ r)

    def objective(r: Tensor) -> float:
        # trace(R.T @ D.T @ D @ R) = ||D @ R||_F^2.
        return float(torch.linalg.norm(D @ r).square().detach().cpu())

    objectives = [objective(stacked)]
    steps: List[float] = []
    converged = False
    for _ in range(max_iters):
        nxt = blockwise_polar(gram_apply(stacked), h)
        step = float(torch.linalg.norm(nxt - stacked).detach().cpu())
        steps.append(step)
        stacked = nxt
        objectives.append(objective(stacked))
        if step <= tol:
            converged = True
            break
    rs = list(stacked.split(h, dim=0))
    template = sum((b @ r for b, r in zip(Bs, rs)), torch.zeros_like(Bs[0])) / p
    residual = sum(float(torch.linalg.norm(template - b @ r).square().detach().cpu()) for b, r in zip(Bs, rs))
    denom = sum(float(torch.linalg.norm(b).square().detach().cpu()) for b in Bs)
    fixed = torch.linalg.norm(blockwise_polar(gram_apply(stacked), h) - stacked) / max(1.0, float(torch.linalg.norm(stacked).detach().cpu()))
    gap = (
        float((gram_eigenvalues[h - 1] - gram_eigenvalues[h]).detach().cpu())
        if gram_eigenvalues.numel() > h
        else float("nan")
    )
    lambda_h = abs(float(gram_eigenvalues[h - 1].detach().cpu()))
    top_scale = max(lambda_h, torch.finfo(PROTOCOL_DTYPE).eps)
    spectral_objective = objectives[-1]
    best_restart_objective = spectral_objective
    if n_random_starts < 0:
        raise ValueError("n_random_starts must be nonnegative.")
    if n_random_starts:
        restart_generator = make_generator(restart_seed, Bs[0].device)
        for _ in range(n_random_starts):
            trial = torch.cat(
                [haar_orthogonal(h, generator=restart_generator, device=Bs[0].device) for _ in range(p)],
                dim=0,
            )
            for _ in range(max_iters):
                trial_next = blockwise_polar(gram_apply(trial), h)
                trial_step = float(torch.linalg.norm(trial_next - trial).detach().cpu())
                trial = trial_next
                if trial_step <= tol:
                    break
            best_restart_objective = max(best_restart_objective, objective(trial))
    if raw_anchor is None:
        anchor_svals = torch.empty(0, dtype=PROTOCOL_DTYPE, device=Bs[0].device)
        anchor_sigma_min = anchor_sigma_max = anchor_condition = float("nan")
    else:
        A = _as_tensor(raw_anchor, Bs[0].device)
        anchor_svals = torch.linalg.svdvals(A - A.mean(dim=0))
        anchor_sigma_max = float(anchor_svals[0].detach().cpu())
        anchor_sigma_min = float(anchor_svals[-1].detach().cpu())
        anchor_condition = anchor_sigma_max / anchor_sigma_min if anchor_sigma_min > 0 else float("inf")
    diagnostics = GPMDiagnostics(
        iterations=len(steps),
        converged=converged,
        elapsed_seconds=time.perf_counter() - started,
        anchor_singular_values=anchor_svals,
        anchor_sigma_min=anchor_sigma_min,
        anchor_sigma_max=anchor_sigma_max,
        anchor_condition=anchor_condition,
        upload_singular_values=[torch.linalg.svdvals(b) for b in Bs],
        gram_eigenvalues=gram_eigenvalues,
        eigengap=gap,
        relative_eigengap=gap / top_scale,
        multistart_objective_gap=max(0.0, best_restart_objective - spectral_objective),
        gopp_residual=residual,
        relative_gopp_residual=residual / max(denom, torch.finfo(PROTOCOL_DTYPE).eps),
        fixed_point_residual=float(fixed.detach().cpu()),
        objective_history=objectives,
        step_history=steps,
    )
    return rs, diagnostics


def pa_i_gdp(
    x_blocks: Sequence[Tensor],
    anchor: Tensor,
    v: float,
    *,
    rotations: Optional[Sequence[Tensor]] = None,
    translations: Optional[Sequence[Tensor]] = None,
    anchor_noises: Optional[Sequence[Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    translation_scale: float = 1.0,
    max_iters: int = 200,
    tol: float = 1e-10,
    n_random_starts: int = 0,
    restart_seed: int = 0,
) -> ProtocolResult:
    """PA-I-GDP with noisy anchors and block-Gram generalized Procrustes."""

    xs = _blocks(x_blocks)
    A = _as_tensor(anchor, xs[0].device)
    p, h = len(xs), xs[0].shape[1]
    if A.ndim != 2 or A.shape[1] != h:
        raise ValueError("anchor must be r-by-h.")
    os_, psis = _parameters(p, h, xs[0], rotations, translations, generator, translation_scale)
    ys = [x @ o + psi for x, o, psi in zip(xs, os_, psis)]
    clean_Bs = [A @ o + psi for o, psi in zip(os_, psis)]
    noise = _noise_blocks(clean_Bs, v, anchor_noises, generator)
    Bs = [b + e for b, e in zip(clean_Bs, noise)]
    means = [b.mean(dim=0) for b in Bs]
    centered = [b - mean for b, mean in zip(Bs, means)]
    rs, diagnostics = gpm_align(
        centered,
        raw_anchor=A,
        max_iters=max_iters,
        tol=tol,
        n_random_starts=n_random_starts,
        restart_seed=restart_seed,
    )
    aligned = [(y - mean) @ r for y, mean, r in zip(ys, means, rs)]
    return _result(
        "PA-I-GDP",
        ys,
        aligned,
        os_,
        psis,
        anchor_uploads=Bs,
        alignment_rotations=rs,
        anchor_noises=noise,
        diagnostics=diagnostics,
    )


def _stack(x: TensorOrBlocks, device: Optional[torch.device] = None) -> Tensor:
    if isinstance(x, Tensor):
        return _as_tensor(x, device)
    return torch.cat(_blocks(x, device), dim=0)


def _center(x: Tensor) -> Tuple[Tensor, Tensor]:
    mean = x.mean(dim=0)
    return mean, x - mean


def attack_c_known_input(
    colluding_x: TensorOrBlocks,
    colluding_y: TensorOrBlocks,
    target_y: Tensor,
    *,
    estimator: Literal["exact", "mp", "op"] = "exact",
) -> AttackResult:
    """Known-input attack against C-GDP; ``mp`` uses a pseudoinverse on decoding."""

    Xq = _stack(colluding_x)
    Yq = _stack(colluding_y, Xq.device)
    if Xq.shape != Yq.shape:
        raise ValueError("Colluding raw/uploaded rows must be paired and shape-compatible.")
    xbar, Xc = _center(Xq)
    ybar, Yc = _center(Yq)
    if estimator == "op":
        O = orthogonal_polar(Xc.mT @ Yc)
    else:
        O = torch.linalg.pinv(Xc) @ Yc
    psi = ybar - O.mT @ xbar
    inverse = torch.linalg.pinv(O) if estimator == "mp" else O.mT
    reduced = (_as_tensor(target_y, Xq.device) - psi) @ inverse
    return AttackResult(O, psi, reduced, f"C-{estimator}")


def invert_known_gdp(target_y: Tensor, rotation: Tensor, translation: Tensor) -> AttackResult:
    """Invert a released GDP matrix when its orthogonal secret is revealed."""

    O = _as_tensor(rotation)
    psi = _as_tensor(translation, O.device)
    return AttackResult(O, psi, (_as_tensor(target_y, O.device) - psi) @ O.mT, "known-secret")


def attack_i_colluder_negative_control(
    target_y: Tensor,
    colluder_rotation: Tensor,
    colluder_translation: Tensor,
) -> AttackResult:
    """Apply a colluder's unrelated I-GDP secret to a noncolluding target."""

    result = invert_known_gdp(target_y, colluder_rotation, colluder_translation)
    result.method = "I-colluder-negative-control"
    return result


def attack_i_distribution_moment(
    colluder_reduced: TensorOrBlocks,
    target_reference_upload: TensorOrBlocks,
    target_y: Tensor,
) -> AttackResult:
    """Unpaired I-GDP distribution-alignment sensitivity attack.

    This attack is not an algebraic inversion theorem.  It uses reduced raw
    records revealed by a colluder as a reference population and estimates a
    non-colluding target's orthogonal transform from the eigensystems of the
    two centered covariance matrices.  Eigenvector signs are resolved with
    standardized third moments, falling back to fifth moments when skewness is
    nearly zero.  It is threat-model compatible for iid participant samples
    but relies on the additional population-matching assumption and must be
    reported as a distribution-prior sensitivity.
    """

    Xq = _stack(colluder_reduced)
    Yref = _stack(target_reference_upload, Xq.device)
    Ytarget = _as_tensor(target_y, Xq.device)
    if Xq.ndim != 2 or Yref.ndim != 2 or Xq.shape[1] != Yref.shape[1]:
        raise ValueError("Reference reduced data and target uploads need the same feature dimension.")
    if Ytarget.ndim != 2 or Ytarget.shape[1] != Xq.shape[1]:
        raise ValueError("Target reconstruction rows need the same feature dimension.")
    if len(Xq) < 2 or len(Yref) < 2:
        raise ValueError("Distribution alignment requires at least two rows per reference sample.")

    xbar, Xc = _center(Xq)
    ybar, Yc = _center(Yref)
    cov_x = Xc.mT @ Xc / max(1, len(Xc) - 1)
    cov_y = Yc.mT @ Yc / max(1, len(Yc) - 1)
    eval_x, Ux = torch.linalg.eigh(cov_x)
    eval_y, Uy = torch.linalg.eigh(cov_y)
    order_x = torch.argsort(eval_x, descending=True)
    order_y = torch.argsort(eval_y, descending=True)
    Ux = Ux[:, order_x]
    Uy = Uy[:, order_y]

    px = Xc @ Ux
    py = Yc @ Uy
    eps = torch.finfo(PROTOCOL_DTYPE).eps**0.5
    px = px / px.square().mean(dim=0).sqrt().clamp_min(eps)
    py = py / py.square().mean(dim=0).sqrt().clamp_min(eps)
    skew_x, skew_y = px.pow(3).mean(dim=0), py.pow(3).mean(dim=0)
    fifth_x, fifth_y = px.pow(5).mean(dim=0), py.pow(5).mean(dim=0)
    use_fifth = (skew_x.abs() + skew_y.abs()) < 1.0e-3
    odd_x = torch.where(use_fifth, fifth_x, skew_x)
    odd_y = torch.where(use_fifth, fifth_y, skew_y)
    signs = torch.where(odd_x * odd_y >= 0, 1.0, -1.0).to(Xq)

    O = Ux @ torch.diag(signs) @ Uy.mT
    psi = ybar - xbar @ O
    reduced = (Ytarget - psi) @ O.mT
    return AttackResult(O, psi, reduced, "I-distribution-moment")


def attack_aa_known_anchor(anchor: Tensor, target_B: Tensor, target_y: Tensor) -> AttackResult:
    """Draft pseudoinverse known-anchor attack against AA-I-GDP or AA-I-GDP-AN."""

    A = _as_tensor(anchor)
    B = _as_tensor(target_B, A.device)
    a, Ac = _center(A)
    b, Bc = _center(B)
    O = torch.linalg.pinv(Ac) @ Bc
    psi = b - O.mT @ a
    reduced = (_as_tensor(target_y, A.device) - psi) @ O.mT
    return AttackResult(O, psi, reduced, "AA-known-anchor")


def attack_pa_mp(anchor: Tensor, target_B: Tensor, target_y: Tensor) -> AttackResult:
    """Unconstrained Moore--Penrose known-anchor attack against PA-I-GDP."""

    A = _as_tensor(anchor)
    B = _as_tensor(target_B, A.device)
    a, Ac = _center(A)
    b, Bc = _center(B)
    O = torch.linalg.pinv(Ac) @ Bc
    psi = b - O.mT @ a
    reduced = (_as_tensor(target_y, A.device) - psi) @ torch.linalg.pinv(O)
    return AttackResult(O, psi, reduced, "PA-MP")


def attack_pa_op(anchor: Tensor, target_B: Tensor, target_y: Tensor) -> AttackResult:
    """Orthogonal-Procrustes known-anchor attack against PA-I-GDP."""

    A = _as_tensor(anchor)
    B = _as_tensor(target_B, A.device)
    a, Ac = _center(A)
    b, Bc = _center(B)
    O = orthogonal_polar(Ac.mT @ Bc)
    psi = b - O.mT @ a
    reduced = (_as_tensor(target_y, A.device) - psi) @ O.mT
    return AttackResult(O, psi, reduced, "PA-OP")


def attack_pa_alignment_calibration(
    anchor: Tensor,
    target_B: Tensor,
    target_y: Tensor,
    target_alignment: Tensor,
    colluder_rotations: Sequence[Tensor],
    colluder_alignments: Sequence[Tensor],
) -> AttackResult:
    """Calibrate PA's global rotation from colluders' revealed local transforms."""

    if not colluder_rotations or len(colluder_rotations) != len(colluder_alignments):
        raise ValueError("Matching nonempty colluder rotation/alignment lists are required.")
    A = _as_tensor(anchor)
    B = _as_tensor(target_B, A.device)
    q_sum = sum(
        (_as_tensor(o, A.device) @ _as_tensor(r, A.device) for o, r in zip(colluder_rotations, colluder_alignments)),
        torch.zeros_like(_as_tensor(colluder_rotations[0], A.device)),
    )
    Q = orthogonal_polar(q_sum)
    Rj = _as_tensor(target_alignment, A.device)
    O = Q @ Rj.mT
    a = A.mean(dim=0)
    b = B.mean(dim=0)
    psi = b - O.mT @ a
    reduced = (_as_tensor(target_y, A.device) - psi) @ O.mT
    return AttackResult(O, psi, reduced, "PA-AM")


def _relative_error(x: Tensor, y: Tensor) -> float:
    num = torch.linalg.norm(x - y)
    den = torch.clamp(torch.linalg.norm(y), min=1.0)
    return float((num / den).detach().cpu())


def affine_orthogonal_registration(
    source: Tensor,
    target: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, float]:
    """Register row embeddings by one right-orthogonal map and common shift."""

    x = _as_tensor(source)
    y = _as_tensor(target, x.device)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("source and target must be same-shaped matrices.")
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    rotation = orthogonal_polar((x - x_mean).mT @ (y - y_mean))
    translation = y_mean - x_mean @ rotation
    registered = x @ rotation + translation
    return registered, rotation, translation, _relative_error(registered, y)


def _assert_close(x: Tensor, y: Tensor, name: str, tol: float = 1e-9) -> float:
    error = _relative_error(x, y)
    if error > tol:
        raise AssertionError(f"{name} relative error {error:.3e} exceeds {tol:.3e}.")
    return error


def run_algebraic_self_tests(
    device: Union[str, torch.device] = "cpu",
    seed: int = 20260711,
) -> dict:
    """Run synthetic orientation, attack, GPM, and C/AA equivalence gates."""

    seed_everything(seed)
    dev = torch.device(device)
    gen = make_generator(seed, dev)
    p, raw_d, h, r = 3, 9, 4, 8
    ns = [12, 11, 10]
    raw = [torch.randn((n, raw_d), dtype=PROTOCOL_DTYPE, device=dev, generator=gen) for n in ns]
    reducer = make_shared_gaussian_projection(raw_d, h, seed=seed + 17, device=dev)
    xs = reducer.transform_blocks(raw)

    # A centred, well-conditioned anchor with A^T A proportional to I.
    g = torch.randn((r, h), dtype=PROTOCOL_DTYPE, device=dev, generator=gen)
    q, _ = torch.linalg.qr(g - g.mean(dim=0), mode="reduced")
    A = q * (r**0.5)
    anchor_svals = torch.linalg.svdvals(A - A.mean(dim=0))
    anchor_rank = int(torch.linalg.matrix_rank(A - A.mean(dim=0)).detach().cpu())
    if anchor_rank != h or float(anchor_svals[-1].detach().cpu()) <= 1e-10:
        raise AssertionError("Centered anchor is not full column rank.")
    os_, psis = sample_gdp_parameters(p, h, generator=gen, device=dev)
    s = 0

    aa = aa_i_gdp(xs, A, reference=s, rotations=os_, translations=psis)
    common = c_gdp(xs, rotation=os_[s], translation=psis[s])
    exact_error = _assert_close(aa.collaboration, common.collaboration, "C/AA equality")
    r_error = max(_relative_error(rr, oi.mT @ os_[s]) for rr, oi in zip(aa.alignment_rotations or [], os_))
    if r_error > 1e-9:
        raise AssertionError(f"AA relative-rotation error {r_error:.3e}.")

    sigma = 0.17
    shared_noise = [sigma * _randn(tuple(x.shape), x, gen) for x in xs]
    coupled_local = [g_i @ rr.mT for g_i, rr in zip(shared_noise, aa.alignment_rotations or [])]
    aa_an = aa_i_gdp_an(
        xs,
        A,
        sigma,
        reference=s,
        rotations=os_,
        translations=psis,
        noises=coupled_local,
    )
    common_an = c_gdp_an(
        xs,
        sigma,
        rotation=os_[s],
        translation=psis[s],
        noises=shared_noise,
    )
    noisy_error = _assert_close(aa_an.collaboration, common_an.collaboration, "coupled C-AN/AA-AN equality")

    c_attack = attack_c_known_input(xs[0], common.private_uploads[0], common.private_uploads[1])
    c_attack_error = _assert_close(c_attack.reduced, xs[1], "C attack")
    aa_attack = attack_aa_known_anchor(A, aa.anchor_uploads[1], aa.private_uploads[1])  # type: ignore[index]
    aa_attack_error = _assert_close(aa_attack.reduced, xs[1], "AA attack")
    aa_an_attack = attack_aa_known_anchor(A, aa_an.anchor_uploads[1], aa_an.private_uploads[1])  # type: ignore[index]
    aa_an_endpoint = xs[1] + coupled_local[1] @ os_[1].mT
    aa_an_endpoint_error = _assert_close(
        aa_an_attack.reduced,
        aa_an_endpoint,
        "AA-AN attack noise endpoint",
    )

    independent = i_gdp(xs, rotations=os_, translations=psis)
    i_negative = attack_i_colluder_negative_control(
        independent.private_uploads[1], os_[0], psis[0]
    )
    i_negative_error = _relative_error(i_negative.reduced, xs[1])
    if i_negative_error < 1e-3:
        raise AssertionError("I-GDP colluder negative control unexpectedly reconstructed the target.")

    pa = pa_i_gdp(
        xs,
        A,
        0.0,
        rotations=os_,
        translations=psis,
        generator=gen,
        max_iters=100,
        tol=1e-12,
        n_random_starts=1,
        restart_seed=seed + 1,
    )
    if pa.diagnostics is None:
        raise AssertionError("PA diagnostics are missing.")
    if not pa.diagnostics.converged:
        raise AssertionError("Noiseless PA GPM failed to converge.")
    if pa.diagnostics.relative_gopp_residual > 1e-10:
        raise AssertionError(
            f"PA relative GOPP residual {pa.diagnostics.relative_gopp_residual:.3e}."
        )
    if pa.diagnostics.fixed_point_residual > 1e-10:
        raise AssertionError(
            f"PA fixed-point residual {pa.diagnostics.fixed_point_residual:.3e}."
        )
    if pa.diagnostics.anchor_sigma_min <= 1e-10:
        raise AssertionError("PA diagnostic reports a rank-deficient centered anchor.")
    effective = [o @ rr for o, rr in zip(os_, pa.alignment_rotations or [])]
    pa_sync_error = max(_relative_error(qi, effective[0]) for qi in effective)
    if pa_sync_error > 1e-8:
        raise AssertionError(f"PA noiseless synchronization error {pa_sync_error:.3e}.")
    _, _, _, pa_c_registration_error = affine_orthogonal_registration(
        pa.collaboration, common.collaboration
    )
    if pa_c_registration_error > 1e-8:
        raise AssertionError(
            f"PA v=0/C affine-orthogonal registration error {pa_c_registration_error:.3e}."
        )
    pa_op = attack_pa_op(A, pa.anchor_uploads[1], pa.private_uploads[1])  # type: ignore[index]
    pa_op_error = _assert_close(pa_op.reduced, xs[1], "PA OP attack", tol=1e-8)
    pa_mp = attack_pa_mp(A, pa.anchor_uploads[1], pa.private_uploads[1])  # type: ignore[index]
    pa_mp_error = _assert_close(pa_mp.reduced, xs[1], "PA MP attack", tol=1e-8)
    pa_am = attack_pa_alignment_calibration(
        A,
        pa.anchor_uploads[1],  # type: ignore[index]
        pa.private_uploads[1],
        pa.alignment_rotations[1],  # type: ignore[index]
        [os_[0]],
        [pa.alignment_rotations[0]],  # type: ignore[index]
    )
    pa_am_error = _assert_close(pa_am.reduced, xs[1], "PA AM attack", tol=1e-8)

    decoded = decode_reduced(xs[1], reducer.F, reducer.F_pinv)
    floor = projection_floor(raw[1], reducer.F)
    decode_error = _assert_close(decoded, floor, "Gaussian projection decode")
    return {
        "c_aa_relative_error": exact_error,
        "c_aa_an_coupled_relative_error": noisy_error,
        "aa_rotation_relative_error": r_error,
        "c_attack_relative_error": c_attack_error,
        "aa_attack_relative_error": aa_attack_error,
        "aa_an_attack_endpoint_relative_error": aa_an_endpoint_error,
        "i_colluder_negative_control_relative_error": i_negative_error,
        "pa_sync_relative_error": pa_sync_error,
        "pa_c_affine_registration_relative_error": pa_c_registration_error,
        "pa_mp_relative_error": pa_mp_error,
        "pa_op_relative_error": pa_op_error,
        "pa_am_relative_error": pa_am_error,
        "grp_decode_relative_error": decode_error,
        "grp_matrix_sigma_min": float(reducer.singular_values[-1].detach().cpu()),
        "grp_matrix_condition": float(
            (reducer.singular_values[0] / reducer.singular_values[-1]).detach().cpu()
        ),
        "centered_anchor_rank": anchor_rank,
        "centered_anchor_sigma_min": float(anchor_svals[-1].detach().cpu()),
        "pa_gopp_residual": pa.diagnostics.gopp_residual if pa.diagnostics else float("nan"),
        "pa_relative_gopp_residual": pa.diagnostics.relative_gopp_residual if pa.diagnostics else float("nan"),
        "pa_fixed_point_residual": pa.diagnostics.fixed_point_residual if pa.diagnostics else float("nan"),
        "pa_converged": pa.diagnostics.converged if pa.diagnostics else False,
    }


def run_self_tests(
    device: Union[str, torch.device] = "cpu",
    seed: int = 20260711,
) -> dict:
    """Run all algebra gates and add a notebook-friendly success flag."""

    metrics = run_algebraic_self_tests(device=device, seed=seed)
    return {"all_passed": True, **metrics}


__all__ = [
    "PROTOCOL_DTYPE",
    "GaussianProjectionReducer",
    "GPMDiagnostics",
    "ProtocolResult",
    "AttackResult",
    "seed_everything",
    "make_generator",
    "haar_orthogonal",
    "sample_gdp_parameters",
    "make_shared_gaussian_projection",
    "reduce_blocks",
    "decode_reduced",
    "projection_floor",
    "orthogonal_polar",
    "blockwise_polar",
    "c_gdp",
    "c_gdp_an",
    "i_gdp",
    "aa_alignment",
    "aa_i_gdp",
    "aa_i_gdp_an",
    "build_block_gram",
    "gpm_align",
    "pa_i_gdp",
    "attack_c_known_input",
    "invert_known_gdp",
    "attack_i_colluder_negative_control",
    "attack_i_distribution_moment",
    "attack_aa_known_anchor",
    "attack_pa_mp",
    "attack_pa_op",
    "attack_pa_alignment_calibration",
    "affine_orthogonal_registration",
    "run_algebraic_self_tests",
    "run_self_tests",
]
