"""Paired MP/AM/OP audit built from raw dataset inputs; no utility fitting.

500 unconditional Uniform(0,.08) scales, shared across attacks and datasets.
Each draw creates fresh participant-specific Gaussian anchor perturbations.
The zero-noise control is separate. Protocol algebra is CPU float64; the
production auditor is the pinned CASIA-WebFace model on a T4 in float32.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import socket
import time
import uuid

import numpy as np
import torch

try:
    from . import casia_linkage as casia
    from .spectral import spectral_align
except ImportError:
    import casia_linkage as casia
    from spectral import spectral_align

VERSION = 'casia-p10-mp-am-op-uniform500-20260930-v1'
DRAW_COUNT, P, H, ANCHOR_ROWS = 500, 10, 400, 1600
TARGET, COLLUDER, UPPER, FACE_BATCH = 1, 0, .08, 128
ATTACKS = ('MP', 'AM', 'OP')
HERE = Path(__file__).resolve().parent
PRODUCTION_FILES = ('attack_comparison.py', 'casia_linkage.py', 'spectral.py')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    with temporary.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def lock_json(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, f'Immutable input/schedule/code identity changed: {path.name}')
    else:
        atomic_json(path, value)


@contextmanager
def worker_lock(out, recover_stale=False):
    """One worker per output. Recovery requires an explicit CLI flag."""
    path = Path(out)/'worker.lock'
    if recover_stale and path.exists():
        # Caller must first verify the former runtime is stopped. Never silently
        # reclaim a lock from another host or from an unknown process.
        owner = read(path)
        if owner.get('host') == socket.gethostname():
            try:
                os.kill(int(owner['pid']), 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('Cannot recover a lock whose local process is still alive')
        archive = Path(out)/('recovered-lock-' + uuid.uuid4().hex + '.json')
        os.rename(path, archive)
    try:
        with path.open('x', encoding='utf8') as f:
            json.dump({'host':socket.gethostname(), 'pid':os.getpid(), 'created_utc':utc()}, f)
            f.flush(); os.fsync(f.fileno())
    except FileExistsError as exc:
        raise RuntimeError('Existing worker.lock: stop the prior worker before explicit --recover-stale-lock') from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def make_schedule(seed=20260930):
    scale_seed = casia.named_seed(seed, VERSION, 'uniform-scales')
    rng = np.random.Generator(np.random.PCG64(scale_seed))
    scales = rng.uniform(0., UPPER, DRAW_COUNT)
    require(bool(((scales > 0) & (scales < UPPER)).all()), 'Unexpected floating endpoint in uniform schedule')
    draws = [{'draw':i, 'scale':float(v), 'noise_seeds':[casia.named_seed(seed, VERSION, 'anchor-noise', i, k) for k in range(P)]}
             for i,v in enumerate(scales)]
    require(len({s for d in draws for s in d['noise_seeds']}) == DRAW_COUNT*P, 'Noise seed collision')
    return {'version':VERSION, 'seed':int(seed), 'distribution':'Uniform(0,0.08)',
            'scale_generator':'numpy.Generator(PCG64)', 'scale_seed':scale_seed,
            'draw_count':DRAW_COUNT, 'participants':P, 'paired_across_attacks':True,
            'paired_across_datasets':True, 'adaptive_selection':False, 'bin_acceptance':False,
            'noise_generator':'independent named CPU torch.Generator seeds; torch.randn float64 (1600,400)',
            'draws':draws}


def gaussian_anchors(design, condition):
    a = design['anchor']; rotations = design['rotations'][:P]; shifts = design['translations'][:P]
    uploads, noise_hashes = [], []
    for participant, (o, psi) in enumerate(zip(rotations, shifts)):
        if condition['scale'] == 0:
            noise = torch.zeros_like(a)
        else:
            generator = torch.Generator(device='cpu').manual_seed(condition['noise_seeds'][participant])
            noise = condition['scale'] * torch.randn(a.shape, dtype=torch.float64, generator=generator)
        noise_hashes.append(tensor_sha(noise))
        uploads.append(a @ o + psi + noise)
    return torch.stack(uploads), noise_hashes


def pinv_policy(matrix):
    require(matrix.dtype == torch.float64 and matrix.device.type == 'cpu', 'MP requires CPU float64')
    return {'function':'torch.linalg.pinv', 'hermitian':False, 'atol':0.,
            'rtol':max(matrix.shape)*torch.finfo(matrix.dtype).eps,
            'cutoff_rule':'max(atol, sigma_max * rtol); retain singular values strictly above cutoff',
            'arguments':'No explicit atol/rtol; exact PyTorch float64 defaults; no ridge regularization'}


def am_rotation(colluder_rotation, alignments, target=TARGET, colluder=COLLUDER):
    return colluder_rotation @ alignments[colluder] @ alignments[target].T


def recover_attacks(anchor, uploads, target_upload, colluder_rotation, target=TARGET, colluder=COLLUDER):
    """Each estimator consumes the SAME anchor uploads and target private upload.

    The stored A is centered and unit-isometric. MP uses A.T @ centered B
    followed by its Moore-Penrose inverse; OP uses the unrestricted O(h)
    polar factor; AM uses O_c R_c R_i.T from the unmodified spectral gauge.
    The tiny finite-precision anchor mean is retained in the original
    translation formula psi_hat = mean(B_i) - mean(A) @ O_hat.
    """
    require(anchor.dtype == uploads.dtype == target_upload.dtype == colluder_rotation.dtype == torch.float64,
            'Protocol tensors must be float64')
    require(all(x.device.type == 'cpu' for x in (anchor, uploads, target_upload, colluder_rotation)), 'Protocol algebra must run on CPU')
    means = uploads.mean(1); centered = uploads - means[:,None,:]
    a_mean = anchor.mean(0)
    matrix = anchor.T @ centered[target]
    u, values, vh = torch.linalg.svd(matrix, full_matrices=False)
    policy = pinv_policy(matrix); cutoff = float(values[0])*policy['rtol']
    inverse = torch.linalg.pinv(matrix, hermitian=False)
    alignments, diagnostics = spectral_align(centered)
    rotations = {'MP':matrix, 'OP':u@vh, 'AM':am_rotation(colluder_rotation, alignments, target, colluder)}
    estimates, offsets = {}, {}
    for name in ATTACKS:
        rotation = rotations[name]
        offsets[name] = means[target] - a_mean @ rotation
        estimates[name] = (target_upload - offsets[name]) @ (inverse if name == 'MP' else rotation.T)
        require(bool(torch.isfinite(estimates[name]).all()), f'Non-finite {name} reconstruction; draw is not discarded')
    rank = int((values > cutoff).sum())
    extra = {'spectral':diagnostics,
             'anchor_mean_max_abs':float(a_mean.abs().max()),
             'translation_policy':'mean(B_i) - mean(A) @ O_hat (theory mean(A)=0)',
             'MP':{**policy, 'effective_cutoff':cutoff, 'numerical_rank':rank,
                   'sigma_min':float(values[-1]), 'sigma_max':float(values[0]),
                   'condition_number':float(values[0]/values[-1]) if values[-1] > 0 else None,
                   'all_singular_values':values.tolist(), 'rank_truncated':rank < matrix.shape[1]},
             'rotation_sha256':{name:tensor_sha(value) for name,value in rotations.items()},
             'translation_sha256':{name:tensor_sha(value) for name,value in offsets.items()},
             'alignment_rotations_sha256':tensor_sha(alignments)}
    return estimates, extra


def decode(reduced, basis):
    return (reduced.float() @ basis.T).reshape(-1,3,64,64).clamp(0,1)


def gallery_success(query_embeddings, reference_embeddings, gallery_indices, genuine_positions):
    """Exact original CPU dot products and one torch.argsort winner per lineup."""
    successes, winners = [], []
    for query in range(len(query_embeddings)):
        row, row_winners = [], []
        for episode in range(gallery_indices.shape[1]):
            scores = torch.stack([reference_embeddings[int(i)] for i in gallery_indices[query,episode]]) @ query_embeddings[query]
            winner = int(torch.argsort(scores, descending=True)[0])
            row.append(int(winner == int(genuine_positions[query,episode]))); row_winners.append(winner)
        successes.append(row); winners.append(row_winners)
    array = np.asarray(successes, dtype=np.int64)
    return {'linkage_percent':float(array.mean()*100), 'correct_trials':int(array.sum()), 'trials':int(array.size),
            'correct_per_identity':array.sum(1).tolist(), 'trial_successes':successes, 'winner_positions':winners}


def validate_record(record, identity_hash, condition, *, dataset=None, schedule_hash=None, genuine_positions=None):
    require(record.get('version') == VERSION and record.get('training_performed') is False, 'Saved record protocol role differs')
    if dataset is not None:
        require(record.get('dataset') == dataset, 'Saved record dataset differs')
    if schedule_hash is not None:
        require(record.get('schedule_hash') == schedule_hash, 'Saved record schedule hash differs')
    require(record.get('identity_hash') == identity_hash and record.get('condition') == condition, 'Saved record input/schedule differs')
    require(record.get('record_sha256') == canonical({k:v for k,v in record.items() if k!='record_sha256'}), 'Saved record checksum differs')
    require(set(record.get('attacks',{})) == set(ATTACKS), 'Incomplete paired-attack record')
    for name,result in record['attacks'].items():
        trial = np.asarray(result['trial_successes'])
        require(trial.shape == (100,10) and bool(np.isin(trial,[0,1]).all()), f'Invalid {name} trial outcomes')
        require(result['trials'] == 1000 and result['correct_trials'] == int(trial.sum()), 'Trial total differs')
        require(result['correct_per_identity'] == trial.sum(1).tolist(), 'Identity totals differ')
        require(abs(result['linkage_percent']-100*float(trial.mean())) < 1e-10, 'Linkage rate differs')
        winners = np.asarray(result['winner_positions'])
        require(winners.shape == (100,10) and winners.dtype.kind in 'iu'
                and bool(np.isin(winners,range(10)).all()), 'Invalid gallery winner positions')
        if genuine_positions is not None:
            require(np.array_equal(trial, (winners == np.asarray(genuine_positions)).astype(np.int64)), 'Winners and trial outcomes disagree')


def commit_record(path, record):
    record = {**record, 'record_sha256':canonical(record)}
    require(not Path(path).exists(), 'Refusing to overwrite an immutable record')
    atomic_json(path,record)
    return record


def summarize(records, identity_hash, schedule_hash, complete=False):
    ordered = sorted(records,key=lambda r:r['condition']['draw'])
    values = {attack:np.asarray([r['attacks'][attack]['linkage_percent'] for r in ordered]) for attack in ATTACKS}
    stats = {attack:{'mean_linkage_percent':float(v.mean()) if len(v) else None,
                     'sample_sd':float(v.std(ddof=1)) if len(v)>1 else None,
                     'minimum':float(v.min()) if len(v) else None,'maximum':float(v.max()) if len(v) else None} for attack,v in values.items()}
    return {'version':VERSION,'state':'complete' if complete else 'partial','identity_hash':identity_hash,'schedule_hash':schedule_hash,
            'planned_draws':DRAW_COUNT,'completed_draws':len(ordered),'zero_control_included_in_statistics':False,
            'statistics':stats,'draws':[{'draw':r['condition']['draw'],'scale':r['condition']['scale'],
                       **{attack:r['attacks'][attack]['linkage_percent'] for attack in ATTACKS}} for r in ordered],
            'record_hashes':{f'draw_{r["condition"]["draw"]:04d}.json':r['record_sha256'] for r in ordered},
            'training_performed':False,'updated_utc':utc()}


def _study_path(output_root):
    root = Path(output_root).resolve()
    return root if (root / 'input_identity.json').is_file() else root / 'study'


def _context_inputs(ctx, dataset):
    """Extract only audit rows from a newly prepared context, never a prior study."""
    require(dataset in ('celeba', 'vggface2'), 'Unknown dataset')
    require(ctx.cfg.n_participants == P and ctx.cfg.reduced_dim == H,
            'Attack comparison fixes p=10 and h=400')
    F = ctx.data.reducer.F.detach().float().cpu().contiguous()
    query = ctx.query_reduced.detach().double().cpu().contiguous()
    bank = {key: ctx.deployment[key] for key in ('anchor', 'rotations', 'translations')}
    require(F.shape == (12288, H) and query.shape == (100, H), 'Wrong basis/query dimensions')
    require(bool(torch.isfinite(F).all()) and bool(torch.isfinite(query).all()), 'Nonfinite basis/query values')
    require(ctx.projection.get('uncentered') is True, 'The public projection must be uncentered')
    require(float(torch.linalg.matrix_norm(F.double().T @ F.double() - torch.eye(H))) < 1e-3,
            'The public basis is not orthonormal')
    a = bank['anchor']
    require(a.shape == (ANCHOR_ROWS, H) and a.dtype == torch.float64 and a.device.type == 'cpu',
            'Expected a CPU float64 1600x400 anchor')
    require(float(a.mean(0).abs().max()) < 1e-10
            and torch.allclose(a.T @ a, torch.eye(H, dtype=torch.float64), rtol=0, atol=1e-10),
            'The anchor must be centered and unit-isometric')
    require(len(bank['rotations']) == len(bank['translations']) == P, 'Expected ten GDP parameter pairs')
    for rotation, shift in zip(bank['rotations'], bank['translations']):
        require(rotation.shape == (H, H) and shift.shape == (H,)
                and rotation.dtype == shift.dtype == torch.float64
                and rotation.device.type == shift.device.type == 'cpu', 'Wrong GDP shape/dtype/device')
        require(torch.allclose(rotation.T @ rotation, torch.eye(H, dtype=torch.float64), rtol=0, atol=1e-10)
                and bool(torch.isfinite(shift).all()), 'Invalid GDP parameters')
    refs = ctx.data.references
    reference_ids = list(map(int, refs.identities.tolist()))
    query_ids = list(map(int, ctx.linkage.query_identities))
    require(len(reference_ids) == len(set(reference_ids)) == 1000
            and len(query_ids) == len(set(query_ids)) == 100, 'Wrong audit identity counts')
    reference = refs.raw_uint8.detach().cpu().contiguous()
    require(reference.dtype == torch.uint8 and reference.shape == (1000, 3, 64, 64),
            'Expected 1,000 reserved reference RGB64 images')
    query_names = [ctx.data.split(split).filenames[index]
                   for split, index in zip(ctx.linkage.query_splits, ctx.linkage.query_split_indices)]
    require(len(set(query_names)) == 100 and not set(query_names) & set(refs.filenames),
            'Queries and reserved references must be different images')
    lookup = {identity: i for i, identity in enumerate(reference_ids)}
    gallery_indices, genuine = [], []
    for identity, episodes in zip(query_ids, ctx.linkage.cross_galleries):
        require(identity in lookup and len(episodes) == 10, 'Missing genuine reference or gallery episode')
        rows, positions = [], []
        for tokens in episodes:
            require(len(tokens) == 10 and all(token[0] == 'reference' for token in tokens),
                    'Expected a ten-way cross-image reference gallery')
            row = [lookup[int(token[1])] for token in tokens]
            require(len(set(row)) == 10 and row.count(lookup[identity]) == 1,
                    'Duplicate candidates or wrong genuine-reference count')
            rows.append(row); positions.append(row.index(lookup[identity]))
        gallery_indices.append(rows); genuine.append(positions)
    require(len(gallery_indices) == 100, 'Expected 100 protected queries')
    metadata = read(Path(ctx.out) / 'face_embedder_manifest.json')
    require(metadata['pretrained'] == 'casia-webface' and metadata['frozen'] is True
            and metadata['checkpoint_sha256'] == casia.CHECKPOINT_SHA256
            and metadata['state_sha256'] == casia.STATE_SHA256, 'Wrong or unfrozen auditor')
    source = {
        'preparation': 'raw data -> frozen cohort/public SVD -> p10 audit; no prior experiment outputs',
        'context_input_hash': ctx.input_hash,
        'allocation_seed': int(ctx.cfg.seed), 'design_seed': 20260928,
        'basis_tensor_sha256': tensor_sha(F), 'query_tensor_sha256': tensor_sha(query),
        'reference_pixels_sha256': tensor_sha(reference),
        'anchor_tensor_sha256': tensor_sha(a),
        'rotations_tensor_sha256': [tensor_sha(value) for value in bank['rotations']],
        'translations_tensor_sha256': [tensor_sha(value) for value in bank['translations']],
        'query_identities': query_ids, 'reference_identities': reference_ids,
        'query_filenames': list(query_names), 'reference_filenames': list(refs.filenames),
        'gallery_indices': gallery_indices, 'genuine_positions': genuine,
        'linkage_manifest_hash': ctx.linkage.manifest_hash,
        'source_identity': ctx.source_identity, 'projection': ctx.projection,
    }
    return F, query, reference, bank, torch.tensor(gallery_indices), torch.tensor(genuine), metadata, source


def run_attack_comparison(dataset, data_root, cache_root, output_root, device='cuda', seed=20260930,
                          max_new_draws=None, progress=None, recover_stale_lock=False):
    """Prepare from raw data, then run or resume exactly 500 paired attack draws.

    The public preparation API creates all cohorts, features and GDP parameters.
    It verifies caches when reused. Only the official frozen model weights are
    downloaded automatically; callers supply the dataset and annotation files.
    CPU execution is diagnostic only. A finite ``max_new_draws`` permits a
    smoke run/resume without changing the immutable 500-draw schedule.
    """
    try:
        from . import notebook_api
    except ImportError:
        import notebook_api
    require(dataset in ('celeba', 'vggface2'), 'Unknown dataset')
    require(device in ('cuda', 'cpu'), 'Device must be cuda or cpu')
    require(max_new_draws is None or isinstance(max_new_draws, int) and 0 <= max_new_draws <= DRAW_COUNT,
            'Invalid maximum new draws')
    root, raw = Path(output_root).resolve(), Path(data_root).resolve()
    require(root != raw and not root.is_relative_to(raw) and not raw.is_relative_to(root),
            'Results must be separate from raw dataset inputs')
    root.mkdir(parents=True, exist_ok=True)
    study = root / 'study'; study.mkdir(exist_ok=True)
    with worker_lock(study, recover_stale_lock):
        torch.set_num_threads(4)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        if progress: progress(stage='preparing_raw_dataset', dataset=dataset)
        notebook_api.prepare_dataset(dataset, raw, cache_root, device=device, progress=progress)
        ctx = notebook_api.load_context(dataset, raw, cache_root, root / 'context',
                                        p=P, device=device, progress=progress)
        F, query, references, bank, galleries, genuine, model_metadata, source = _context_inputs(ctx, dataset)
        model = ctx.face_model
        schedule = make_schedule(seed); schedule_hash = canonical(schedule)
        lock_json(study / 'schedule.json', schedule)
        identity = {
            'version': VERSION, 'dataset': dataset, 'p': P, 'h': H, 'anchor_rows': ANCHOR_ROWS,
            'draw_count': DRAW_COUNT, 'zero_control': 'separate,excluded from500', 'source': source,
            'schedule_hash': schedule_hash, 'source_sha256': {name: sha(HERE / name) for name in PRODUCTION_FILES},
            'model': model_metadata, 'protocol_dtype': 'CPU float64', 'auditor_dtype': 'float32',
            'device': device, 'audit_role': 'production T4' if device == 'cuda' else 'CPU diagnostic only',
            'cuda_device_name': torch.cuda.get_device_name(0) if device == 'cuda' else None,
            'torch_cuda_version': torch.version.cuda, 'torch_version': torch.__version__,
            'numpy_version': np.__version__, 'python_version': platform.python_version(),
            'cpu_threads': 4, 'face_batch_size': FACE_BATCH, 'reference_treatment': 'projected_decoded_clipped',
            'scorer': 'original CPU gallery dot-products; single torch.argsort(descending=True)[0]',
            'alignment': 'exact thin SVD then block polar; no GPM refinement',
            'MP_pinv_policy': pinv_policy(torch.empty(H, H, dtype=torch.float64)), 'training_performed': False,
        }
        lock_json(study / 'input_identity.json', identity); identity_hash = canonical(identity)
        if (study / 'completion.json').exists():
            # A terminal study is immutable. Missing/corrupt records are errors,
            # not an invitation to regenerate them and change their receipts.
            return inspect_results(root)
        reference_pixels = decode(references.float().flatten(1).div(255.) @ F, F)
        reference_embeddings = casia.embeddings(model, reference_pixels, device=device, batch_size=FACE_BATCH)
        lock_json(study / 'reference_embeddings.json', {'identity_hash': identity_hash,
                                                       'sha256': tensor_sha(reference_embeddings)})
        target_upload = query @ bank['rotations'][TARGET] + bank['translations'][TARGET]
        directory = study / 'records'; directory.mkdir(exist_ok=True)
        expected = {f'draw_{i:04d}.json' for i in range(DRAW_COUNT)}
        require({path.name for path in directory.glob('*.json')} <= expected, 'Unexpected draw record files')
        zero = {'draw': -1, 'scale': 0., 'noise_seeds': None, 'kind': 'separate zero-noise control'}
        records, new_count = [], 0
        for condition in [zero] + schedule['draws']:
            is_zero = condition['draw'] == -1
            path = study / 'zero_control.json' if is_zero else directory / f'draw_{condition["draw"]:04d}.json'
            if path.exists():
                record = read(path)
                validate_record(record, identity_hash, condition, dataset=dataset,
                                schedule_hash=schedule_hash, genuine_positions=genuine)
                if not is_zero: records.append(record)
                continue
            if not is_zero and max_new_draws is not None and new_count >= max_new_draws:
                continue  # Inspect existing later records, even after the new-draw limit.
            started = time.perf_counter()
            atomic_json(study / 'status.json', {'state': 'running', 'dataset': dataset,
                'draw': condition['draw'], 'completed_draws': len(records), 'planned_draws': DRAW_COUNT,
                'scale': condition['scale'], 'updated_utc': utc(), 'training_performed': False})
            uploads, noise_hashes = gaussian_anchors(bank, condition)
            recovered, diagnostics = recover_attacks(bank['anchor'], uploads, target_upload, bank['rotations'][COLLUDER])
            results = {}
            for attack in ATTACKS:
                if is_zero:
                    require(torch.allclose(recovered[attack], query, atol=1e-10, rtol=1e-10),
                            f'{attack} failed noiseless algebra gate')
                pixels = decode(recovered[attack], F)
                embeddings = casia.embeddings(model, pixels, device=device, batch_size=FACE_BATCH)
                result = gallery_success(embeddings, reference_embeddings, galleries, genuine)
                result.update({'reduced_nrmse': float(torch.linalg.norm(recovered[attack] - query) / torch.linalg.norm(query)),
                    'reduced_sha256': tensor_sha(recovered[attack]), 'decoded_pixel_sha256': tensor_sha(pixels),
                    'query_embedding_sha256': tensor_sha(embeddings)})
                results[attack] = result
            record = commit_record(path, {
                'version': VERSION, 'dataset': dataset, 'identity_hash': identity_hash, 'schedule_hash': schedule_hash,
                'condition': condition, 'noise_sha256_by_participant': noise_hashes,
                'anchor_uploads_sha256': tensor_sha(uploads), 'target_private_upload_sha256': tensor_sha(target_upload),
                'attacks': results, 'diagnostics': diagnostics, 'elapsed_seconds': time.perf_counter() - started,
                'completed_utc': utc(), 'training_performed': False,
            })
            validate_record(record, identity_hash, condition, dataset=dataset,
                            schedule_hash=schedule_hash, genuine_positions=genuine)
            if not is_zero: records.append(record); new_count += 1
            event = {'stage': 'attack_draw_completed', 'dataset': dataset, 'draw': condition['draw'],
                'scale': condition['scale'], 'scores': {name: results[name]['linkage_percent'] for name in ATTACKS},
                'completed_draws': len(records), 'planned_draws': DRAW_COUNT,
                'seconds': round(record['elapsed_seconds'], 3)}
            print(json.dumps(event), flush=True)
            if progress: progress(**event)
            atomic_json(study / 'summary.json', summarize(records, identity_hash, schedule_hash))
        complete = len(records) == DRAW_COUNT
        summary = summarize(records, identity_hash, schedule_hash, complete)
        atomic_json(study / 'summary.json', summary)
        zero_record = read(study / 'zero_control.json')
        validate_record(zero_record, identity_hash, zero, dataset=dataset,
                        schedule_hash=schedule_hash, genuine_positions=genuine)
        if complete:
            lock_json(study / 'completion.json', {'state': 'complete', 'identity_hash': identity_hash,
                'schedule_hash': schedule_hash, 'draw_count': DRAW_COUNT,
                'zero_control_record_sha256': zero_record['record_sha256'],
                'record_hashes': summary['record_hashes'], 'training_performed': False})
        atomic_json(study / 'status.json', {'state': 'complete' if complete else 'partial_by_requested_limit',
            'dataset': dataset, 'completed_draws': len(records), 'planned_draws': DRAW_COUNT,
            'updated_utc': utc(), 'training_performed': False})
    return inspect_results(root, require_complete=complete)


def inspect_results(output_root, require_complete=True):
    """Recompute numeric results from immutable records; do not run any models."""
    study = _study_path(output_root)
    identity, schedule = read(study / 'input_identity.json'), read(study / 'schedule.json')
    identity_hash, schedule_hash = canonical(identity), canonical(schedule)
    require(identity['version'] == VERSION and identity['p'] == P and identity['h'] == H
            and identity['anchor_rows'] == ANCHOR_ROWS and identity['training_performed'] is False,
            'Wrong comparison protocol')
    require(schedule == make_schedule(schedule['seed']) and identity['schedule_hash'] == schedule_hash,
            'Stored schedule differs from prescribed uniform paired schedule')
    require(set(identity['source_sha256']) == set(PRODUCTION_FILES), 'Wrong source roster')
    for name, digest in identity['source_sha256'].items():
        require(sha(HERE / name) == digest, f'Source changed since execution: {name}')
    require(identity['model']['pretrained'] == 'casia-webface' and identity['model']['frozen'] is True
            and identity['model']['checkpoint_sha256'] == casia.CHECKPOINT_SHA256
            and identity['model']['state_sha256'] == casia.STATE_SHA256, 'Wrong auditor pins')
    genuine = np.asarray(identity['source']['genuine_positions'])
    require(genuine.shape == (100, 10) and genuine.dtype.kind in 'iu'
            and bool(np.isin(genuine, range(10)).all()), 'Invalid frozen genuine positions')
    records = []
    expected = {f'draw_{i:04d}.json' for i in range(DRAW_COUNT)}
    actual = {path.name for path in (study / 'records').glob('*.json')}
    require(actual <= expected, 'Unexpected draw records')
    for condition in schedule['draws']:
        path = study / 'records' / f'draw_{condition["draw"]:04d}.json'
        if not path.exists(): continue
        record = read(path)
        validate_record(record, identity_hash, condition, dataset=identity['dataset'],
                        schedule_hash=schedule_hash, genuine_positions=genuine)
        require(datetime.fromisoformat(record['completed_utc']).tzinfo is not None,
                'Draw timestamp must include its time zone')
        records.append(record)
    complete = len(records) == DRAW_COUNT
    if require_complete: require(complete, 'All 500 draws are required before final figures')
    zero_condition = {'draw': -1, 'scale': 0., 'noise_seeds': None, 'kind': 'separate zero-noise control'}
    zero = read(study / 'zero_control.json')
    validate_record(zero, identity_hash, zero_condition, dataset=identity['dataset'],
                    schedule_hash=schedule_hash, genuine_positions=genuine)
    derived = summarize(records, identity_hash, schedule_hash, complete)
    saved = read(study / 'summary.json')
    require({k: v for k, v in saved.items() if k != 'updated_utc'}
            == {k: v for k, v in derived.items() if k != 'updated_utc'}, 'Summary differs from committed records')
    if complete:
        completion = read(study / 'completion.json')
        require(completion == {'state': 'complete', 'identity_hash': identity_hash,
            'schedule_hash': schedule_hash, 'draw_count': DRAW_COUNT,
            'zero_control_record_sha256': zero['record_sha256'],
            'record_hashes': derived['record_hashes'], 'training_performed': False}, 'Completion receipt differs')
    counts = np.asarray([[r['attacks'][a]['correct_trials'] for a in ATTACKS] for r in records], dtype=np.int64)
    wins = {}
    if len(counts):
        highest = counts == counts.max(axis=1, keepdims=True); tie_sizes = highest.sum(1)
        wins = {attack: {'strict_highest_draws': int((highest[:, j] & (tie_sizes == 1)).sum()),
                        'tied_highest_draws': int((highest[:, j] & (tie_sizes > 1)).sum())}
                for j, attack in enumerate(ATTACKS)}
    return {'dataset': identity['dataset'], 'state': 'complete' if complete else 'partial',
        'completed_draws': len(records), 'planned_draws': DRAW_COUNT, 'identity_hash': identity_hash,
        'schedule_hash': schedule_hash, 'study_directory': str(study), 'audit_role': identity['audit_role'],
        'statistics': derived['statistics'], 'highest_draw_counts': wins,
        'zero_control_excluded': True, 'zero_linkage_percent': {a: zero['attacks'][a]['linkage_percent'] for a in ATTACKS},
        'training_performed': False,
        'record_byte_sha256': {f'draw_{r["condition"]["draw"]:04d}.json': sha(study / 'records' / f'draw_{r["condition"]["draw"]:04d}.json')
                               for r in records}}


def export_figures(output_root, figure_root=None, resamples=10000):
    """Export verified numeric observations and paired-bootstrap spline figures."""
    try:
        from .attack_plotting import export_figures as render
    except ImportError:
        from attack_plotting import export_figures as render
    return render(output_root, figure_root=figure_root, resamples=resamples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['celeba', 'vggface2'])
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=20260930)
    parser.add_argument('--max-new-draws', type=int)
    parser.add_argument('--recover-stale-lock', action='store_true')
    arguments = parser.parse_args()
    print(json.dumps(run_attack_comparison(**vars(arguments)), indent=2))
