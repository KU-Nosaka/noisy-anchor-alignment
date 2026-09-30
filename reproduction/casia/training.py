"""Reproducible h=400 utility training with the original CelebA CNN.

The production backend is CUDA float32 on Colab T4. Inputs, initialization,
shuffling, metrics and portable checkpoints use CPU tensors. Public entry points
never tune the fixed 15-epoch recipe. Artifact paths are condition-relative.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import socket
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VERSION = "casia-allp-h400-spectral-utility-rebinned-v2"
SETTINGS = {
    "input_dimensions": 400, "input_shape": [1, 20, 20],
    "architecture": "original CelebA CNN: Conv1-10k5, ReLU, MaxPool2, Conv10-20k5, ReLU, MaxPool2, Linear80-50, ReLU, Linear50-1",
    "convolution_channels": [10, 20], "kernel_size": 5,
    "pool_size": 2, "flatten_dimension": 80, "hidden_units": 50,
    "utility_adapter": "row-major reshape of 400 protected or aligned coordinates to 1x20x20; no padding or normalization",
    "epochs": 15, "batch_size": 256, "eval_batch_size": 512,
    "optimizer": "Adam", "learning_rate": 1e-3, "weight_decay": 0.0,
    "betas": [0.9, 0.999], "epsilon": 1e-8,
    "loss": "unweighted_binary_cross_entropy_with_logits",
    "selection": "maximum_validation_balanced_accuracy_earliest_tie",
    "threshold": 0.5, "dtype": "float32", "xla_matmul_precision": "highest",
    "last_batch": "zero_padded_masked_sum_divided_by_valid_count",
    "initialization": "cpu_seeded", "permutations": "cpu_generator",
    "xla_adam_capturable": True, "cuda_tf32": False,
}
_XLA: Any = None
_XLA_INITIALIZED = False


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(_json(value) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return copy.deepcopy(value)


def _emit(progress: Any, stage: str, **fields: Any) -> None:
    if progress is None:
        return
    value = {"stage": stage, "updated_unix": time.time(), **fields}
    if callable(progress):
        progress(value)
    elif hasattr(progress, "write"):
        progress.write(**value)
    else:
        raise TypeError("progress must be callable or provide write(**fields)")


def _major_minor(version: str) -> str:
    return ".".join(version.split("+")[0].split(".")[:2])


def runtime_probe(device: str = "cuda") -> dict[str, Any]:
    """Validate the requested backend; never silently fall back to CPU."""
    global _XLA, _XLA_INITIALIZED
    if device in ("cpu", "cuda") and torch.is_autocast_enabled(device):
        raise RuntimeError("Ambient autocast would alter the specified float32 recipe")
    report: dict[str, Any] = {
        "requested_device": device, "torch": torch.__version__,
        "numpy": np.__version__, "python": platform.python_version(),
        "torch_threads": torch.get_num_threads(), "training_dtype": "float32",
        "autocast": False,
    }
    if device == "cpu":
        return {**report, "device": "cpu", "backend": "cpu", "precision": "float32"}
    if device == "cuda":
        # Must precede the first CUDA context initialization for deterministic GEMM.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        return {
            **report, "device": "cuda", "backend": "cuda", "precision": "float32",
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
    if device != "xla":
        raise ValueError("This experiment supports only 'cpu', 'cuda', and single-device 'xla'.")
    for name in ("XLA_USE_BF16", "XLA_DOWNCAST_BF16"):
        if os.environ.get(name, "").lower() not in ("", "0", "false", "no"):
            raise RuntimeError(f"{name} would alter the specified float32 recipe")
    import torch_xla
    import torch_xla.backends as backends
    import torch_xla.runtime as runtime

    if _major_minor(torch.__version__) != _major_minor(torch_xla.__version__):
        raise RuntimeError("torch and torch_xla major/minor versions must match")
    if runtime.device_type() != "TPU":
        raise RuntimeError("XLA must be backed by a TPU for this experiment")
    if runtime.global_device_count() != 1:
        raise RuntimeError("This recipe requires exactly one global TPU device")
    if not _XLA_INITIALIZED:
        backends.set_mat_mul_precision("highest")
        _XLA_INITIALIZED = True
    if backends.get_mat_mul_precision() != "highest":
        raise RuntimeError("XLA convolution/matmul precision changed during the run")
    _XLA = torch_xla
    actual = torch_xla.device()
    if actual.type != "xla":
        raise RuntimeError("torch_xla.device() did not return an XLA device")
    return {
        **report, "device": str(actual), "backend": "xla",
        "torch_xla": torch_xla.__version__, "global_devices": 1,
        "precision": backends.get_mat_mul_precision(),
        "precision_note": "TPU highest uses multiple BF16 products, approximately FP32; not bitwise CPU equivalence.",
    }


def _sync(device: str, wait: bool = False) -> None:
    if device == "xla":
        _XLA.sync(wait=wait)
    elif device == "cuda":
        torch.cuda.synchronize()


def _xla_metrics(device: str) -> dict[str, Any]:
    if device != "xla":
        return {"available": False, "reason": f"{device} backend"}
    import torch_xla.debug.metrics as metrics

    result: dict[str, Any] = {"available": True, "report": metrics.short_metrics_report()}
    result["counters"] = {
        name: metrics.counter_value(name) for name in metrics.counter_names()
        if name.startswith("aten::") or "Compile" in name
    }
    result["metrics"] = {}
    for name in ("CompileTime", "ExecuteTime", "TransferToDeviceTime", "TransferFromDeviceTime"):
        value = metrics.metric_data(name)
        if value is not None:
            # Samples are intentionally omitted; cumulative counts/totals stay compact.
            result["metrics"][name] = {"total_samples": value[0], "accumulator": value[1]}
    return result


class BinaryReferenceCNN(nn.Module):
    """Same layers, coordinate order and activations as the h=400 CelebA CNN."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, dtype=torch.float32)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, dtype=torch.float32)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(80, 50, dtype=torch.float32)
        self.fc2 = nn.Linear(50, 1, dtype=torch.float32)

    def _convolve(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        return self.pool(F.relu(self.conv2(x)))

    def forward_features(self, x):
        if x.ndim == 2 and x.shape[1] == 400:
            x = x.reshape(-1, 1, 20, 20)
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 20, 20):
            raise ValueError("CNN expects N-by-400 or N-by-1-by-20-by-20 inputs")
        return F.relu(self.fc1(torch.flatten(self._convolve(x), 1)))

    def forward(self, x):
        return self.fc2(self.forward_features(x)).squeeze(1)


def _new_model(seed: int,device: str) -> BinaryReferenceCNN:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = BinaryReferenceCNN()
    return model.to(device=device,dtype=torch.float32)


def _optimizer(model: nn.Module) -> torch.optim.Adam:
    return torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, foreach=False,
        # On XLA, leave the changing step/bias-correction values as device tensor
        # inputs. A host scalar step embeds new constants and can recompile each
        # batch. PyTorch's single-tensor capturable Adam supports XLA.
        capturable=next(model.parameters()).device.type == "xla",
    )


def _inputs(x: Any, y: Any, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    # Existing contiguous CPU float32 tensor/memmap storage is retained. Only
    # minibatches travel to CUDA; input checking never allocates an N-by-400 mask.
    x = torch.as_tensor(x).detach().cpu().to(torch.float32).contiguous()
    y = torch.as_tensor(y).detach().cpu().to(torch.float32).flatten().contiguous()
    # Hash both supported representations in the same coordinate order.
    if x.ndim == 4 and tuple(x.shape[1:]) == (1, 20, 20):
        x = x.reshape(x.shape[0], 400)
    if x.ndim != 2 or x.shape[1] != 400 or x.shape[0] != y.numel() or not len(y):
        raise ValueError(f"{name} requires nonempty N-by-400 or N-by-1-by-20-by-20 inputs and N labels")
    for start in range(0, len(x), 16384):
        if not bool(torch.isfinite(x[start:start + 16384]).all()):
            raise ValueError(f"{name} contains nonfinite values")
    if not bool(torch.isfinite(y).all()):
        raise ValueError(f"{name} contains nonfinite values")
    if not bool(((y == 0) | (y == 1)).all()):
        raise ValueError(f"{name} labels must be binary")
    return x, y


def _tensor_sha(value: torch.Tensor) -> str:
    # Hash identical C-order bytes in bounded chunks, preserving old signatures
    # without creating a second multi-gigabyte Python bytes object.
    digest = hashlib.sha256()
    value = value.detach().cpu()
    if value.ndim == 0:
        value = value.reshape(1)
    for start in range(0, len(value), 16384):
        array = value[start:start + 16384].contiguous().numpy()
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _padded_batch(x: torch.Tensor, y: torch.Tensor, indices: torch.Tensor,
                  batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = len(indices)
    if not 0 < valid <= batch_size:
        raise ValueError("A padded batch must contain 1..batch_size real records")
    bx = torch.zeros((batch_size, 400), dtype=torch.float32)
    by = torch.zeros(batch_size, dtype=torch.float32)
    mask = torch.zeros(batch_size, dtype=torch.float32)
    bx[:valid] = x[indices]
    by[:valid] = y[indices]
    mask[:valid] = 1.0
    return bx.to(device), by.to(device), mask.to(device)


def _masked_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * mask).sum() / mask.sum()


def _metrics(y: torch.Tensor, probability: torch.Tensor, loss: float) -> dict[str, Any]:
    labels = y.long().cpu()
    probability = probability.float().cpu()
    prediction = (probability >= 0.5).long()
    tn = int(((labels == 0) & (prediction == 0)).sum())
    fp = int(((labels == 0) & (prediction == 1)).sum())
    fn = int(((labels == 1) & (prediction == 0)).sum())
    tp = int(((labels == 1) & (prediction == 1)).sum())
    return {
        "n": len(labels), "loss": float(loss),
        "accuracy": float((labels == prediction).float().mean()),
        "balanced_accuracy": 0.5 * (tn / max(1, tn + fp) + tp / max(1, tp + fn)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "positive_prevalence": float(labels.float().mean()),
    }


@torch.no_grad()
def _evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: str) -> tuple[dict[str, Any], torch.Tensor]:
    model.eval()
    probabilities = []
    total_loss = 0.0
    for start in range(0, len(y), SETTINGS["eval_batch_size"]):
        ids = torch.arange(start, min(start + SETTINGS["eval_batch_size"], len(y)))
        bx, by, mask = _padded_batch(x, y, ids, SETTINGS["eval_batch_size"], device)
        logits = model(bx)
        loss = (F.binary_cross_entropy_with_logits(logits, by, reduction="none") * mask).sum()
        probability = torch.sigmoid(logits)
        _sync(device)
        # One deliberately bounded transfer per evaluation batch; no records dropped.
        probabilities.append(probability.detach().cpu()[:len(ids)])
        total_loss += float(loss.detach().cpu())
    probability = torch.cat(probabilities)
    if not bool(torch.isfinite(probability).all()) or not math.isfinite(total_loss):
        raise RuntimeError("Nonfinite evaluation loss/probabilities")
    return _metrics(y, probability, total_loss / len(y)), probability


@contextmanager
def _condition_lock(output: Path):
    path = output / ".training.lock"
    token = {"pid": os.getpid(), "host": socket.gethostname(), "token": uuid.uuid4().hex}
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Existing training lock requires verified stale-worker recovery: {path}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_json(token))
        yield
    finally:
        if path.is_file() and json.loads(path.read_text()) == token:
            path.unlink()


def _verified_manifest(output: Path, manifest: dict[str, Any], signature: str) -> None:
    if manifest.get("signature") != signature:
        raise RuntimeError("Saved condition/settings/input signature differs")
    for artifact in manifest["artifacts"].values():
        filename = artifact["path"]
        if Path(filename).name != filename:
            raise RuntimeError("Artifact path must be a simple relative filename")
        path = output / filename
        if not path.is_file() or _sha(path) != artifact["sha256"]:
            raise RuntimeError(f"Saved artifact is missing or altered: {filename}")


def _save_checkpoint(output: Path, checkpoint: dict[str, Any], signature: str) -> tuple[str, str]:
    filename = f"checkpoint_epoch_{checkpoint['completed_epoch']:03d}.pt"
    path = output / filename
    _atomic_torch(path, _cpu(checkpoint))
    digest = _sha(path)
    _atomic_json(output / "checkpoint_manifest.json", {
        "signature": signature, "completed_epoch": checkpoint["completed_epoch"],
        "artifacts": {"checkpoint": {"path": filename, "sha256": digest}},
    })
    # The committed manifest is now durable. Keep only its checkpoint and its
    # immediate predecessor; deletion is scoped to filenames we generate.
    old_epoch = checkpoint["completed_epoch"] - 2
    if old_epoch >= 1:
        (output / f"checkpoint_epoch_{old_epoch:03d}.pt").unlink(missing_ok=True)
    return filename, digest


def _timing_template() -> dict[str, Any]:
    return {
        "training_seconds": 0.0, "validation_seconds": 0.0,
        "checkpoint_seconds": 0.0, "final_evaluation_seconds": 0.0,
        "cold_first_training_step_including_compile_seconds": None,
        "subsequent_training_steps_seconds": 0.0, "subsequent_training_steps": 0,
        "note": "Step timing is synchronized wall time including transfers; first step includes any compilation. Later steps may also compile. XLA CompileTime metrics are reported separately and cumulatively per process.",
    }


def train_condition(train_x: Any, train_y: Any, val_x: Any, val_y: Any,
                    test_x: Any, test_y: Any, output_dir: str | Path, *, seed: int,
                    condition_signature: Any, device: str = "cuda", progress: Any = None) -> dict[str, Any]:
    """Train or exactly resume one fixed-recipe condition; verify completed outputs."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    backend = runtime_probe(device)
    train_x, train_y = _inputs(train_x, train_y, "train")
    val_x, val_y = _inputs(val_x, val_y, "validation")
    test_x, test_y = _inputs(test_x, test_y, "test")
    input_hashes = {name: _tensor_sha(value) for name, value in (
        ("train_x", train_x), ("train_y", train_y), ("val_x", val_x),
        ("val_y", val_y), ("test_x", test_x), ("test_y", test_y),
    )}
    identity = {
        "version": VERSION, "condition_signature": condition_signature,
        "settings": SETTINGS, "seed": int(seed), "input_hashes": input_hashes,
        "source_sha256": _sha(Path(__file__)),
        "backend": {k: backend[k] for k in ("backend", "precision", "torch")},
    }
    if device == "xla":
        identity["backend"]["torch_xla"] = backend["torch_xla"]
    signature = _digest(identity)
    with _condition_lock(output):
        return _train_locked(train_x, train_y, val_x, val_y, test_x, test_y,
                             output, seed, signature, identity, backend, device, progress)


def _train_locked(tx, ty, vx, vy, sx, sy, output, seed, signature, identity, backend, device, progress):
    result_path = output / "result.json"
    if result_path.is_file():
        manifest_path = output / "completion_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("Result exists without its completion manifest; manual recovery required")
        manifest = json.loads(manifest_path.read_text())
        _verified_manifest(output, manifest, signature)
        result = json.loads(result_path.read_text())
        if result["signature"] != signature:
            raise RuntimeError("Result signature mismatch")
        _emit(progress, "condition_reused", output_dir=str(output), signature=signature)
        return result
    _emit(progress, "training_start", output_dir=str(output), signature=signature, backend=backend)
    model = _new_model(seed, device)
    optimizer = _optimizer(model)
    permutation_seed = int.from_bytes(hashlib.sha256(f"{seed}:minibatches".encode()).digest()[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(permutation_seed)
    history = []
    best_score = -math.inf
    best_epoch = 0
    best_state = _cpu(model.state_dict())
    completed_epoch = 0
    timing = _timing_template()
    resumed_epoch = 0
    checkpoint_manifest = output / "checkpoint_manifest.json"
    if checkpoint_manifest.is_file():
        manifest = json.loads(checkpoint_manifest.read_text())
        _verified_manifest(output, manifest, signature)
        checkpoint = torch.load(output / manifest["artifacts"]["checkpoint"]["path"], map_location="cpu", weights_only=False)
        if checkpoint["signature"] != signature or checkpoint["identity"] != identity:
            raise RuntimeError("Checkpoint provenance mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        generator.set_state(checkpoint["permutation_generator_state"])
        completed_epoch = int(checkpoint["completed_epoch"])
        if not 0 <= completed_epoch <= SETTINGS["epochs"]:
            raise RuntimeError("Checkpoint epoch is outside the fixed recipe")
        resumed_epoch = completed_epoch
        history = checkpoint["history"]
        best_score, best_epoch = checkpoint["best_score"], checkpoint["best_epoch"]
        best_state, timing = checkpoint["best_state"], checkpoint["timing"]
        _emit(progress, "training_resumed", completed_epoch=completed_epoch, output_dir=str(output))
    metrics_before = _xla_metrics(device)
    for epoch in range(completed_epoch + 1, SETTINGS["epochs"] + 1):
        model.train()
        permutation = torch.randperm(len(ty), generator=generator)
        training_started = time.perf_counter()
        # A fixed scalar accumulator stays a graph input rather than an unbounded
        # history of all previous batches, because every optimizer step is synced.
        loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        for start in range(0, len(ty), SETTINGS["batch_size"]):
            step_started = time.perf_counter()
            ids = permutation[start:start + SETTINGS["batch_size"]]
            bx, by, mask = _padded_batch(tx, ty, ids, SETTINGS["batch_size"], device)
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_loss(model(bx), by, mask)
            loss.backward()
            optimizer.step()
            loss_sum = loss_sum + loss.detach() * mask.sum()
            _sync(device, wait=True)
            elapsed = time.perf_counter() - step_started
            if timing["cold_first_training_step_including_compile_seconds"] is None:
                timing["cold_first_training_step_including_compile_seconds"] = elapsed
            else:
                timing["subsequent_training_steps_seconds"] += elapsed
                timing["subsequent_training_steps"] += 1
        train_loss = float(loss_sum.detach().cpu()) / len(ty)
        timing["training_seconds"] += time.perf_counter() - training_started
        if not math.isfinite(train_loss):
            raise RuntimeError("Nonfinite training loss")
        evaluate_started = time.perf_counter()
        validation, _ = _evaluate(model, vx, vy, device)
        timing["validation_seconds"] += time.perf_counter() - evaluate_started
        history.append({"epoch": epoch, "train_loss": train_loss, "validation": validation})
        if validation["balanced_accuracy"] > best_score:
            best_score = validation["balanced_accuracy"]
            best_epoch = epoch
            best_state = _cpu(model.state_dict())
        checkpoint_started = time.perf_counter()
        _save_checkpoint(output, {
            "signature": signature, "identity": identity, "completed_epoch": epoch,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "best_state": best_state, "best_score": best_score, "best_epoch": best_epoch,
            "permutation_generator_state": generator.get_state(),
            "permutation_seed": permutation_seed, "permutation_sha256": _tensor_sha(permutation),
            "history": history, "timing": timing,
            "rng_policy": "CPU initialization and serialized CPU permutation generator; no stochastic model layers",
        }, signature)
        timing["checkpoint_seconds"] += time.perf_counter() - checkpoint_started
        _emit(progress, "epoch_complete", epoch=epoch, epochs=SETTINGS["epochs"],
              best_epoch=best_epoch, train_loss=train_loss, validation=validation,
              output_dir=str(output))
    model.load_state_dict(best_state)
    evaluation_started = time.perf_counter()
    validation, validation_probabilities = _evaluate(model, vx, vy, device)
    test, test_probabilities = _evaluate(model, sx, sy, device)
    timing["final_evaluation_seconds"] += time.perf_counter() - evaluation_started
    model_path, predictions_path = output / "model.pt", output / "predictions.npz"
    if model_path.exists():
        existing = torch.load(model_path, map_location="cpu", weights_only=False)
        if existing["signature"] != signature or any(not torch.equal(existing["state_dict"][k], v) for k, v in best_state.items()):
            raise RuntimeError("Existing final model disagrees with resumed checkpoint")
    else:
        _atomic_torch(model_path, {"signature": signature, "state_dict": best_state,
                                   "settings": SETTINGS, "best_epoch": best_epoch})
    prediction_values = {
        "validation_labels": vy.numpy(), "validation_probabilities": validation_probabilities.numpy(),
        "test_labels": sy.numpy(), "test_probabilities": test_probabilities.numpy(),
    }
    if predictions_path.exists():
        with np.load(predictions_path, allow_pickle=False) as existing:
            if set(existing.files) != set(prediction_values) or any(not np.array_equal(existing[k], v) for k, v in prediction_values.items()):
                raise RuntimeError("Existing final predictions disagree with resumed checkpoint")
    else:
        temporary = output / ("predictions." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **prediction_values)
            os.replace(temporary, predictions_path)
        finally:
            temporary.unlink(missing_ok=True)
    result = {
        "status": "complete", "version": VERSION, "signature": signature,
        "identity": identity, "runtime": backend, "training": {
            "epochs": SETTINGS["epochs"], "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_score, "history": history,
            "resumed_from_epoch": resumed_epoch,
        },
        "validation": validation, "test": test, "timing": timing,
        "xla_metrics_before": metrics_before, "xla_metrics_after": _xla_metrics(device),
        "artifacts": {
            "model": {"path": model_path.name, "sha256": _sha(model_path)},
            "predictions": {"path": predictions_path.name, "sha256": _sha(predictions_path)},
        },
        "completed_unix": time.time(),
    }
    _atomic_json(result_path, result)
    _atomic_json(output / "completion_manifest.json", {
        "signature": signature,
        "artifacts": {**result["artifacts"], "result": {"path": result_path.name, "sha256": _sha(result_path)}},
    })
    _emit(progress, "condition_complete", output_dir=str(output), test=test,
          best_epoch=best_epoch, timing=timing)
    return result


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str,
                  *, atol: float, rtol: float) -> float:
    difference = float((actual.detach().cpu() - expected.detach().cpu()).abs().max())
    if not torch.allclose(actual.detach().cpu(), expected.detach().cpu(), atol=atol, rtol=rtol):
        raise RuntimeError(f"{label} disagrees with CPU: maximum absolute difference {difference}")
    return difference


def pilot(output_dir: str | Path, device: str = "cuda") -> dict[str, Any]:
    """Synthetic compatibility gate, including a real interrupted/resumed fit.

    The fit has 19 train, 11 validation and 13 test records; it exercises masked
    tails and the complete 15-epoch recipe without using research data.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    backend = runtime_probe(device)
    generator = torch.Generator().manual_seed(73921)
    x = torch.randn(43, 400, generator=generator) * 0.1
    y = (torch.arange(43) % 2).float()
    ids = torch.arange(19)
    cpu_model, model = _new_model(381, "cpu"), _new_model(381, device)
    cpu_opt, opt = _optimizer(cpu_model), _optimizer(model)
    bx, by, mask = _padded_batch(x, y, ids, SETTINGS["batch_size"], device)
    cbx, cby, cmask = _padded_batch(x, y, ids, SETTINGS["batch_size"], "cpu")
    cpu_logits, logits = cpu_model(cbx), model(bx)
    cpu_loss, loss = _masked_loss(cpu_logits, cby, cmask), _masked_loss(logits, by, mask)
    _sync(device, wait=True)
    checks = {
        "forward_max_absolute_error": _assert_close(logits, cpu_logits, "forward", atol=5e-5, rtol=5e-4),
        "loss_absolute_error": _assert_close(loss, cpu_loss, "loss", atol=5e-5, rtol=5e-4),
        "partial_batch_loss_error": _assert_close(cpu_loss, F.binary_cross_entropy_with_logits(cpu_model(x[:19]), y[:19]), "masked tail", atol=1e-7, rtol=1e-6),
    }
    cpu_loss.backward(); loss.backward()
    cpu_opt.step(); opt.step(); _sync(device, wait=True)
    checks["adam_step_max_absolute_error"] = max(
        _assert_close(parameter, dict(cpu_model.named_parameters())[name], f"Adam {name}", atol=2e-4, rtol=1e-3)
        for name, parameter in model.named_parameters()
    )
    del model, cpu_model, opt, cpu_opt
    fit_output = output / "resume_check"

    class PilotInterruption(Exception):
        pass

    def interrupt_after_checkpoint(event):
        if event["stage"] == "epoch_complete" and event["epoch"] == 1:
            raise PilotInterruption()

    arguments = (x[:19], y[:19], x[19:30], y[19:30], x[30:], y[30:], fit_output)
    kwargs = {"seed": 912, "condition_signature": {"pilot": VERSION}, "device": device}
    if not (fit_output / "checkpoint_manifest.json").exists() and not (fit_output / "result.json").exists():
        try:
            train_condition(*arguments, **kwargs, progress=interrupt_after_checkpoint)
        except PilotInterruption:
            pass
        else:
            raise RuntimeError("Pilot did not exercise epoch-boundary interruption")
    resumed = train_condition(*arguments, **kwargs)
    if resumed["training"]["resumed_from_epoch"] != 1:
        raise RuntimeError("Pilot did not resume from its first epoch checkpoint")
    checks["resumed_from_epoch"] = resumed["training"]["resumed_from_epoch"]
    checks["all_test_records_retained"] = resumed["test"]["n"] == 13
    checks["completed_artifacts_reverified"] = train_condition(*arguments, **kwargs)["signature"] == resumed["signature"]
    # Exact interrupted/uninterrupted equivalence is checked on the actual
    # requested backend, rather than inferred from successful resumption.
    fresh_output = output / "uninterrupted_check"
    fresh = train_condition(*arguments[:-1], fresh_output, **kwargs)
    if fresh["training"]["history"] != resumed["training"]["history"] or fresh["test"] != resumed["test"]:
        raise RuntimeError("Pilot resumed training differs from uninterrupted training")
    fresh_state = torch.load(fresh_output / "model.pt", map_location="cpu", weights_only=False)["state_dict"]
    resumed_state = torch.load(fit_output / "model.pt", map_location="cpu", weights_only=False)["state_dict"]
    if set(fresh_state) != set(resumed_state) or any(not torch.equal(fresh_state[k], resumed_state[k]) for k in fresh_state):
        raise RuntimeError("Pilot resumed model differs from uninterrupted model")
    checks["resume_matches_uninterrupted_exactly"] = True
    report = {"passed": True, "backend": backend, "checks": checks,
              "timing": resumed["timing"], "xla_metrics": _xla_metrics(device),
              "note": "Synthetic correctness/compatibility gate; not a full-dataset throughput estimate."}
    _atomic_json(output / "pilot_report.json", report)
    return report
