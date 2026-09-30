"""Frozen CASIA-WebFace FaceNet clean linkage; no optimization or utility fit.

Prepared cohorts/galleries are selected before this module sees auditor scores.
The primary h400 condition decodes both queries and withheld references. Raw/raw
and decoded-query/raw-reference diagnostics reuse the exact same galleries.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as TF

VERSION = 'casia-clean-linkage-20260928-v1'
CHECKPOINT_URL = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt'
CHECKPOINT_NAME = '20180408-102900-casia-webface.pt'
CHECKPOINT_BYTES = 115887415
CHECKPOINT_SHA256 = '7a67afdbbc995fce5e10128675e318799a70698c2f433ba75dd7eb9a2f096e7d'
STATE_SHA256 = '769c635b91a84600f222b46fca15bc6b29d80299e71b69e5eb37cbe15c7ad582'
H = 400
RAW_SHAPE = (3, 64, 64)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def lock_json(path, value):
    path = Path(path)
    if path.exists():
        require(json.loads(path.read_text(encoding='utf-8')) == value, f'Incompatible clean-audit resume: {path.name}')
    else:
        atomic_json(path, value)


def named_seed(seed, *parts):
    # Exact original CelebA named_seed encoding for primitive values.
    digest = hashlib.sha256(json.dumps([int(seed), *parts]).encode()).digest()
    return int.from_bytes(digest[:8], 'little') & ((1 << 63) - 1)


def checkpoint(cache_dir, progress=None):
    """Fetch only the official model checkpoint; never fetch CASIA images."""
    import requests
    directory = Path(cache_dir); directory.mkdir(parents=True, exist_ok=True)
    path = directory / CHECKPOINT_NAME
    if path.exists():
        require(path.stat().st_size == CHECKPOINT_BYTES and file_sha(path) == CHECKPOINT_SHA256,
                'Existing CASIA checkpoint failed size/SHA256; not overwritten')
        return path
    part = path.with_name(path.name + '.part')
    require(not part.exists(), 'Interrupted model partial exists; preserve/inspect it before retry')
    if progress:
        progress(stage='downloading_casia_weights', expected_bytes=CHECKPOINT_BYTES)
    digest = hashlib.sha256(); received = 0
    with requests.get(CHECKPOINT_URL, stream=True, timeout=(30, 180), headers={'Accept-Encoding': 'identity'}) as response:
        response.raise_for_status()
        require(response.status_code == 200, 'Unexpected CASIA checkpoint HTTP status')
        if response.headers.get('Content-Length'):
            require(int(response.headers['Content-Length']) == CHECKPOINT_BYTES, 'Official checkpoint length changed')
        with part.open('xb') as output:
            for chunk in response.iter_content(4 * 1024 * 1024):
                if not chunk:
                    continue
                require(received + len(chunk) <= CHECKPOINT_BYTES, 'Checkpoint body exceeds pinned size')
                output.write(chunk); digest.update(chunk); received += len(chunk)
            output.flush(); os.fsync(output.fileno())
    require(received == CHECKPOINT_BYTES and digest.hexdigest() == CHECKPOINT_SHA256,
            'Downloaded CASIA checkpoint failed pinned SHA256; partial preserved')
    require(not path.exists(), 'Checkpoint appeared during download; refusing overwrite')
    os.rename(part, path)
    return path


def load_casia(cache_dir, device='cuda', progress=None):
    """Load exact 2.6.0 architecture and frozen normalized 512D embedding path."""
    require(importlib.metadata.version('facenet-pytorch') == '2.6.0', 'Use facenet-pytorch==2.6.0')
    from facenet_pytorch import InceptionResnetV1
    require(device in ('cpu', 'cuda'), 'Invalid auditor device')
    if device == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        require(torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0), 'Requested T4 auditor unavailable')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    path = checkpoint(cache_dir, progress)
    state = torch.load(path, map_location='cpu', weights_only=True)
    require(state['logits.weight'].shape == (10575, 512), 'Wrong CASIA classification head')
    model = InceptionResnetV1(pretrained=None, classify=True, num_classes=10575)
    model.load_state_dict(state, strict=True)
    model.classify = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    require(digest.hexdigest() == STATE_SHA256, 'CASIA model tensor-state fingerprint differs')
    metadata = {'architecture': 'InceptionResnetV1', 'package': 'facenet-pytorch==2.6.0',
                'architecture_file_sha256': file_sha(inspect.getfile(InceptionResnetV1)),
                'pretrained': 'casia-webface', 'checkpoint': CHECKPOINT_NAME,
                'checkpoint_bytes': CHECKPOINT_BYTES, 'checkpoint_sha256': CHECKPOINT_SHA256,
                'checkpoint_url': CHECKPOINT_URL, 'state_sha256': STATE_SHA256,
                'hash_provenance': 'Locally computed official GitHub asset SHA256; publisher digest unavailable',
                'embedding_dim': 512, 'frozen': True,
                'preprocessing': 'RGB [0,1], bilinear 160x160 align_corners=False, (255*x-127.5)/128',
                'output': 'L2-normalized embeddings; strict torch.argsort single-winner cosine ranking',
                'device': device, 'dtype': 'float32', 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'accelerator': torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU',
                'identity_overlap_caveat': 'CASIA celebrity pretraining may overlap benchmark people; no non-overlap claim'}
    return model.to(device), metadata


@torch.inference_mode()
def embeddings(model, images, device='cuda', batch_size=128):
    require(isinstance(images, torch.Tensor) and images.ndim == 4 and tuple(images.shape[1:]) == RAW_SHAPE,
            'Auditor input must be NCHW RGB64')
    require(images.dtype in (torch.uint8, torch.float32), 'Auditor inputs must be uint8 or float32')
    require(batch_size > 0 and len(images) > 0, 'Empty auditor input or invalid batch size')
    if images.dtype != torch.uint8:
        require(bool(torch.isfinite(images).all()) and float(images.min()) >= 0 and float(images.max()) <= 1,
                'Floating auditor images must be finite [0,1]')
    require(not model.training and not any(p.requires_grad for p in model.parameters()), 'Auditor must be frozen/eval')
    rows = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device=device, dtype=torch.float32)
        if images.dtype == torch.uint8:
            batch = batch / 255.
        batch = TF.interpolate(batch, size=(160, 160), mode='bilinear', align_corners=False)
        batch = (batch * 255. - 127.5) / 128.
        value = model(batch)
        require(value.shape == (len(batch), 512) and bool(torch.isfinite(value).all()), 'Invalid FaceNet embedding output')
        rows.append(value.cpu())
    return TF.normalize(torch.cat(rows), dim=1)


def as_raw(value, name):
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.from_numpy(np.asarray(value))
    require(tensor.dtype == torch.uint8 and tensor.ndim == 4 and tuple(tensor.shape[1:]) == RAW_SHAPE,
            f'{name} must be NCHW RGB64 uint8')
    return tensor.contiguous()


def validate_inputs(queries, references, query_ids, reference_ids, gallery_indices, query_names, reference_names):
    query_ids, reference_ids = np.asarray(query_ids), np.asarray(reference_ids)
    galleries = np.asarray(gallery_indices)
    require(query_ids.ndim == reference_ids.ndim == 1 and len(query_ids) == len(queries)
            and len(reference_ids) == len(references), 'Identity array shape differs')
    require(len(set(query_ids.tolist())) == len(query_ids), 'Need one query per identity')
    require(len(set(reference_ids.tolist())) == len(reference_ids), 'Need one reference per identity')
    require(set(query_ids.tolist()) <= set(reference_ids.tolist()), 'A query lacks its withheld reference')
    require(len(query_names) == len(queries) and len(reference_names) == len(references), 'Filename counts differ')
    require(len(set(query_names)) == len(query_names) and len(set(reference_names)) == len(reference_names), 'Repeated image filename')
    require(not (set(query_names) & set(reference_names)), 'Query/reference image leakage')
    require(galleries.shape == (len(query_ids), 10, 10) and galleries.dtype.kind in 'iu',
            'Expected ten fixed ten-way galleries per query')
    require(bool(((galleries >= 0) & (galleries < len(reference_ids))).all()), 'Gallery index out of range')
    for q, groups in enumerate(galleries):
        for gallery in groups:
            require(len(set(gallery.tolist())) == 10, 'Gallery repeats a candidate')
            require(int(np.sum(reference_ids[gallery] == query_ids[q])) == 1, 'Gallery lacks unique genuine identity')
    by_identity = {identity: i for i, identity in enumerate(reference_ids.tolist())}
    duplicate_genuine = [int(q) for q, identity in enumerate(query_ids.tolist())
                         if torch.equal(queries[q], references[by_identity[identity]])]
    require(not duplicate_genuine, f'Identical RGB64 query/reference pixels invalidate cross-image linkage: {duplicate_genuine}')
    return query_ids, reference_ids, galleries.astype(np.int64, copy=False)


def validate_basis(basis, metadata):
    f = basis.detach().cpu() if isinstance(basis, torch.Tensor) else torch.from_numpy(np.asarray(basis))
    require(f.dtype == torch.float32 and f.shape == (12288, H) and bool(torch.isfinite(f).all()),
            'Expected finite float32 public F400 with shape12288x400')
    require(metadata.get('uncentered') is True, 'Public basis must explicitly declare uncentered fitting')
    residual = float(torch.linalg.matrix_norm(f.double().T @ f.double() - torch.eye(H, dtype=torch.float64)))
    require(residual < 1e-3, 'Public basis is not orthonormal')
    if metadata.get('basis_matrix_sha256'):
        require(metadata['basis_matrix_sha256'] == tensor_sha(f), 'Public basis matrix checksum differs')
    return f.contiguous(), residual


def decode(raw, basis, reduced=None, batch_size=512):
    """Row-major RGB/255 projection and float32 decoding, matching CelebA."""
    require(raw.device.type == basis.device.type == 'cpu', 'Projection/decoding must use CPU')
    if reduced is not None:
        reduced = reduced.detach().cpu() if isinstance(reduced, torch.Tensor) else torch.from_numpy(np.asarray(reduced))
        require(reduced.dtype == torch.float32 and reduced.shape == (len(raw), H)
                and bool(torch.isfinite(reduced).all()), 'Invalid cached reduced queries')
    output = torch.empty((len(raw), *RAW_SHAPE), dtype=torch.float32)
    for start in range(0, len(raw), batch_size):
        end = start + batch_size
        z = reduced[start:end] if reduced is not None else (raw[start:end].float().flatten(1) / 255.) @ basis
        output[start:end] = (z @ basis.T).reshape(-1, *RAW_SHAPE).clamp(0, 1)
    return output


def identity_bootstrap(values, seed, resamples=10000):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and len(values) > 0 and bool(np.isfinite(values).all()), 'Invalid bootstrap observations')
    require(resamples >= 200, 'Need at least200 bootstrap replicates')
    rng = np.random.default_rng(seed); means = []
    remaining = resamples
    while remaining:
        count = min(500, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means.append(values[indices].mean(axis=1)); remaining -= count
    return np.quantile(np.concatenate(means), [.025, .975]).tolist()


def score_embeddings(query_embeddings, reference_embeddings, query_ids, reference_ids, galleries, seed=20260713):
    """Preserve CPU torch.argsort tie behavior from the manuscript scorer."""
    q, r = query_embeddings.cpu(), reference_embeddings.cpu()
    require(q.ndim == r.ndim == 2 and q.shape[1] == r.shape[1] == 512, 'Expected512D embeddings')
    require(bool(torch.isfinite(q).all()) and bool(torch.isfinite(r).all()), 'Embedding contains nonfinite values')
    require(torch.allclose(q.norm(dim=1), torch.ones(len(q)), atol=1e-5, rtol=0)
            and torch.allclose(r.norm(dim=1), torch.ones(len(r)), atol=1e-5, rtol=0), 'Embeddings must be normalized')
    ranks, successes = [], []
    for index, groups in enumerate(galleries):
        row_ranks, row_successes = [], []
        for gallery in groups:
            genuine = int(np.flatnonzero(reference_ids[gallery] == query_ids[index])[0])
            scores = r[torch.as_tensor(gallery)] @ q[index]
            order = torch.argsort(scores, descending=True)
            rank = int(torch.nonzero(order == genuine, as_tuple=False)[0]) + 1
            row_ranks.append(rank); row_successes.append(int(rank == 1))
        ranks.append(row_ranks); successes.append(row_successes)
    rank_array = np.asarray(ranks, dtype=np.int64); success_array = np.asarray(successes, dtype=np.int64)
    per_identity = success_array.mean(axis=1)
    ci = identity_bootstrap(per_identity, named_seed(seed, 'linkage-ci', 'cross_image_identity'))
    return {'top1': float(success_array.mean()), 'top1_percent': float(100. * success_array.mean()),
            'top3': float((rank_array <= 3).mean()), 'mrr': float((1. / rank_array).mean()),
            'chance_top1': .1, 'ci95_identity_cluster': ci, 'ci95_percent': [100. * x for x in ci],
            'n_identities': len(q), 'episodes_per_identity': 10, 'n_episodes': int(success_array.size),
            'bootstrap_unit': 'identity', 'bootstrap_resamples': 10000,
            'bootstrap_seed': named_seed(seed, 'linkage-ci', 'cross_image_identity'),
            'identity_top1': per_identity.tolist(), 'episode_successes': successes, 'episode_ranks': ranks,
            'tie_policy': 'single winner from torch.argsort(descending=True), unchanged candidate order'}


def _write_identity_csv(path, query_ids, query_names, metrics):
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    conditions = list(metrics)
    fields = ['identity', 'query_filename']
    for condition in conditions:
        fields.extend([condition + '_successes', condition + '_episodes', condition + '_top1']
                      + [condition + f'_episode_{i + 1}' for i in range(10)])
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for i, identity in enumerate(query_ids.tolist()):
            row = {'identity': identity, 'query_filename': query_names[i]}
            for condition, score in metrics.items():
                outcomes = score['episode_successes'][i]
                row.update({condition + '_successes': sum(outcomes), condition + '_episodes': 10,
                            condition + '_top1': score['identity_top1'][i]})
                row.update({condition + f'_episode_{j + 1}': outcome for j, outcome in enumerate(outcomes)})
            writer.writerow(row)
    os.replace(temporary, path)


def evaluate_audit(*, queries, references, query_ids, reference_ids, gallery_indices,
                   query_names, reference_names, basis, basis_metadata, model, auditor_metadata,
                   provenance, out_dir, query_reduced=None, seed=20260713,
                   device='cuda', batch_size=128, progress=None):
    """Run three paired clean conditions and persist checksummed per-identity results."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    raw_q, raw_r = as_raw(queries, 'queries'), as_raw(references, 'references')
    qi, ri, galleries = validate_inputs(raw_q, raw_r, query_ids, reference_ids, gallery_indices,
                                        query_names, reference_names)
    f, orthogonality = validate_basis(basis, basis_metadata)
    require(auditor_metadata.get('pretrained') == 'casia-webface' and auditor_metadata.get('frozen') is True
            and auditor_metadata.get('checkpoint_sha256') == CHECKPOINT_SHA256
            and auditor_metadata.get('state_sha256') == STATE_SHA256, 'Primary audit requires the pinned CASIA model')
    if query_reduced is not None:
        query_reduced = torch.as_tensor(query_reduced).cpu().contiguous()
    identity = {'version': VERSION, 'seed': seed, 'provenance': provenance, 'basis_metadata': basis_metadata,
                'auditor': auditor_metadata, 'query_ids': qi.tolist(), 'reference_ids': ri.tolist(),
                'query_names': list(query_names), 'reference_names': list(reference_names),
                'query_raw_sha256': tensor_sha(raw_q), 'reference_raw_sha256': tensor_sha(raw_r),
                'basis_sha256': tensor_sha(f), 'gallery_sha256': hashlib.sha256(galleries.tobytes()).hexdigest(),
                'query_reduced_sha256': tensor_sha(query_reduced) if query_reduced is not None else None,
                'query_decode_source': 'verified cached reduced query rows' if query_reduced is not None else 'CPU float32 raw RGB projection',
                'reference_decode_source': 'CPU float32 raw RGB projection', 'batch_size': batch_size,
                'conditions': ['raw_raw', 'decoded_decoded', 'decoded_raw'], 'selection_uses_auditor_scores': False}
    lock_json(out / 'input_lock.json', identity)
    identity_hash = canonical_hash(identity)
    if (out / 'completion.json').exists():
        saved = json.loads((out / 'completion.json').read_text())
        require(saved['input_hash'] == identity_hash, 'Completed clean audit inputs differ')
        for name, checksum in saved['artifacts'].items():
            require(file_sha(out / name) == checksum, f'Completed artifact differs: {name}')
        return json.loads((out / 'results.json').read_text())
    started = time.perf_counter()
    decoded_q = decode(raw_q, f, query_reduced)
    decoded_r = decode(raw_r, f)
    embedded = {}
    for key, images in (('raw_query', raw_q), ('raw_reference', raw_r),
                        ('decoded_query', decoded_q), ('decoded_reference', decoded_r)):
        if progress:
            progress(stage='embedding', condition=key, images=len(images))
        embedded[key] = embeddings(model, images, device, batch_size)
    metrics = {}
    for name, query_key, reference_key in (
            ('raw_raw', 'raw_query', 'raw_reference'),
            ('decoded_decoded', 'decoded_query', 'decoded_reference'),
            ('decoded_raw', 'decoded_query', 'raw_reference')):
        metrics[name] = score_embeddings(embedded[query_key], embedded[reference_key], qi, ri, galleries, seed)
    csv_path = out / 'per_identity.csv'
    _write_identity_csv(csv_path, qi, list(query_names), metrics)
    result = {'version': VERSION, 'input_hash': identity_hash, 'completed_utc': utc(),
              'primary_condition': 'decoded_decoded', 'metrics': metrics,
              'basis_orthogonality_residual': orthogonality,
              'embedding_sha256': {key: tensor_sha(value) for key, value in embedded.items()},
              'elapsed_seconds': time.perf_counter() - started, 'utility_training_executed': False,
              'noise_protection_applied': False,
              'interpretation': 'Clean cross-image linkage sensitivity; no guaranteed pretraining identity disjointness'}
    atomic_json(out / 'results.json', result)
    atomic_json(out / 'completion.json', {'version': VERSION, 'input_hash': identity_hash,
                'state': 'complete', 'completed_utc': utc(),
                'artifacts': {'results.json': file_sha(out / 'results.json'),
                              'per_identity.csv': file_sha(csv_path), 'input_lock.json': file_sha(out / 'input_lock.json')}})
    if progress:
        progress(stage='complete', input_hash=identity_hash,
                 primary_top1_percent=metrics['decoded_decoded']['top1_percent'])
    return result
