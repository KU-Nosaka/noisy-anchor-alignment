"""Fresh matched h400 protocol design and CASIA audit finalization.

Dataset adapters own immutable raw-data row selection and projection. GDP banks
and the verified frozen CASIA auditor are generated/downloaded by this code.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import os

H, ANCHOR_ROWS, PMAX = 400, 1600, 50


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def atomic_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def design_hashes(payload, dp):
    return {key: tensor_hash(value) if key == 'anchor' else
            dp.canonical_hash([tensor_hash(x) for x in value]) for key, value in payload.items()}


def fresh_design(core, base, output, seed):
    """Generate and verify the matched nested GDP bank from the fixed seed."""
    import torch
    import data_pipeline as dp
    directory = Path(output)
    path, receipt_path = directory / 'design.pt', directory / 'receipt.json'
    signature = {'version': 'casia-full-shared-design-h400-pmax50-v1', 'seed': seed,
                 'h': H, 'anchor_rows': ANCHOR_ROWS, 'participants': PMAX,
                 'anchor_design': 'centered_unit_isometric',
                 'seed_rule': 'named_seed(seed,casia-full-shared-design-v1,fresh-gdp400|fresh-anchor400)',
                 'dataset_and_basis_independent': True, 'historical_parameters_reused': False}
    if path.exists():
        dp.require(receipt_path.exists(), 'Design exists without committed receipt')
        receipt = dp.read(receipt_path)
        dp.require(receipt['identity'] == signature and receipt['file_sha256'] == dp.sha(path),
                   'Fresh shared GDP design identity/checksum differs')
        payload = torch.load(path, map_location='cpu', weights_only=False)
    else:
        dp.require(not receipt_path.exists(), 'Design receipt has no tensor file')
        generator = core.make_generator(base.named_seed(seed, 'casia-full-shared-design-v1', 'fresh-gdp400'), 'cpu')
        rotations, translations = core.sample_gdp_parameters(PMAX, H, generator=generator, device='cpu')
        generator = torch.Generator().manual_seed(base.named_seed(seed, 'casia-full-shared-design-v1', 'fresh-anchor400'))
        raw = torch.randn((ANCHOR_ROWS, H), generator=generator, dtype=torch.float64)
        q, r = torch.linalg.qr(raw - raw.mean(0), mode='reduced')
        signs = torch.where(torch.diagonal(r) < 0, -1., 1.)
        payload = {'anchor': (q * signs).contiguous(), 'rotations': rotations, 'translations': translations}
        atomic_torch(path, payload)
        receipt = {'identity': signature, 'file_sha256': dp.sha(path), 'tensor_hashes': design_hashes(payload, dp)}
        dp.atomic_json(receipt_path, receipt)
    dp.require(receipt['tensor_hashes'] == design_hashes(payload, dp), 'Design tensor hash differs')
    anchor = payload['anchor']
    dp.require(anchor.shape == (ANCHOR_ROWS, H) and anchor.dtype == torch.float64
               and bool(torch.isfinite(anchor).all()), 'Wrong anchor shape/dtype/finiteness')
    eye = torch.eye(H, dtype=torch.float64)
    dp.require(float(anchor.mean(0).abs().max()) < 1e-10 and
               torch.allclose(anchor.T @ anchor, eye, rtol=0, atol=1e-10), 'Anchor is not centered unit-isometric')
    dp.require(len(payload['rotations']) == len(payload['translations']) == PMAX, 'Wrong GDP bank capacity')
    for rotation, translation in zip(payload['rotations'], payload['translations']):
        dp.require(rotation.shape == (H,H) and rotation.dtype == torch.float64
                   and translation.numel() == H and translation.dtype == torch.float64
                   and bool(torch.isfinite(translation).all())
                   and torch.allclose(rotation.T @ rotation, eye, rtol=0, atol=1e-10), 'Invalid GDP parameters')
    return payload, receipt, {str(path): dp.sha(path), str(receipt_path): dp.sha(receipt_path)}


def private_rms(reduced, batch_rows=4096):
    total, count = 0., 0
    for start in range(0, len(reduced), batch_rows):
        part = reduced[start:start+batch_rows].double()
        total += float(part.square().sum())
        count += part.numel()
    return (total / count) ** .5


def finalize_context(inputs, out_dir, audit_device, spec, progress=None):
    import json
    import torch
    import data_pipeline as dp
    import casia_linkage as casia
    ctx, out = inputs, Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ctx.out, ctx.protocol_config = out, dict(spec)
    p = ctx.cfg.n_participants
    dp.require(ctx.cfg.reduced_dim == H and ctx.decoder.shape == (H,12288), 'Expected h400 pixel decoder')
    dp.require(ctx.projection.get('uncentered') is True, 'Centered public projection is unsupported')
    dp.require(len(ctx.linkage.query_identities) == len(set(ctx.linkage.query_identities)) == 100,
               'Exactly 100 distinct protected query identities required')
    dp.require(ctx.linkage.episodes == 10 and ctx.linkage.n_candidates == 10, 'Expected ten-by-ten galleries')
    ctx.audit_cfg = SimpleNamespace(**vars(ctx.cfg)); ctx.audit_cfg.device = audit_device
    dp.lock_json(out / 'face_linkage_manifest.json', json.loads(json.dumps(ctx.linkage.to_dict())))
    dp.lock_json(out / 'derived_split_hashes.json', ctx.derived_splits)
    if progress:
        progress(stage='fresh_design_and_casia', p=p)
    study_root = Path(spec.get('study_root', out.parent.parent))
    design, receipt, design_files = fresh_design(ctx.core, ctx.base,
            spec.get('design_root', study_root / 'shared_design'), int(spec['seed']))
    design_reuse = {'policy': 'fresh seeded bank using the evaluated shared-design namespace',
                    'prior_calibration_or_utility_results_reused': False,
                    'new_bins_percent': list(dp.BINS)}
    cache = Path(spec.get('model_cache_dir', study_root / 'model_assets'))
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / casia.CHECKPOINT_NAME
    ctx.face_model, face_meta = casia.load_casia(cache, device=audit_device,
                  progress=(lambda **kw: progress(**kw)) if progress else None)
    dp.lock_json(out / 'face_embedder_manifest.json', face_meta)
    ctx.deployment = {'rotations': design['rotations'][:p], 'translations': design['translations'][:p],
        'anchor': design['anchor'], 'anchor_rms': float(design['anchor'].square().mean().sqrt()),
        'anchor_spec': receipt['identity'], 'deployment_spec': receipt,
        'private_rms': private_rms(ctx.data.train.reduced)}
    tokens = sorted({token for groups in ctx.linkage.cross_galleries for gallery in groups for token in gallery})
    ctx.gallery = {}
    f = ctx.data.reducer.F.float().cpu()
    with torch.no_grad():
        for start in range(0, len(tokens), ctx.cfg.face_batch_size):
            batch = tokens[start:start + ctx.cfg.face_batch_size]
            pixels = torch.stack([ctx.base._candidate_image(ctx.data, token) for token in batch])
            images = (((pixels.float().flatten(1) / 255.) @ f) @ f.T).reshape(-1,3,64,64).clamp(0,1)
            ctx.gallery.update(zip(batch, ctx.base._face_embeddings(ctx.face_model, images, ctx.audit_cfg)))
    files = {**ctx.source_files, **design_files, str(destination): dp.sha(destination)}
    ctx.input_lock = {'version': dp.VERSION, 'dataset': spec['dataset'], 'p': p, 'h': H,
        'files': files, 'source_identity': ctx.source_identity, 'projection': ctx.projection,
        'derived_splits': ctx.derived_splits, 'gallery_hash': ctx.linkage.manifest_hash,
        'fresh_design': receipt, 'design_reuse': design_reuse, 'face_state': face_meta['state_sha256'],
        'auditor_checkpoint_sha256': casia.CHECKPOINT_SHA256, 'study_seed': dp.STUDY_SEED,
        'allocation_seed': ctx.cfg.seed, 'noise_dtype': 'float64', 'protocol_device': 'cpu',
        'bin_edges_percent': list(dp.BINS), 'per_family_per_bin': dp.PER_BIN,
        'max_proposals_per_family': dp.MAX_PROPOSALS,
        'noise_scale_units': 'absolute Gaussian standard deviation',
        'private_noise_attack': 'legitimate colluder-known shared GDP secret inversion',
        'anchor_noise_attack': 'PA-OP using public anchor correspondence',
        'linkage_metric': 'decoded-query/decoded-reference; CASIA normalized512 cosine; torch.argsort single winner',
        'utility': 'original TinyCNN h400 reshaped1x20x20; fresh15epochs,Adam1e-3,batch256',
        'fresh_noise_and_utility_fits': True}
    ctx.provenance, ctx.input_hash = ctx.input_lock, dp.canonical_hash(ctx.input_lock)
    dp.lock_json(out / 'input_lock.json', ctx.input_lock)
    ctx.execution = {'audit_device': audit_device, 'audit_dtype': 'float32', 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'audit_accelerator': torch.cuda.get_device_name(0) if audit_device == 'cuda' else 'CPU',
        'protocol_device': 'cpu', 'protocol_dtype': 'float64'}
    dp.lock_json(out / 'execution_context.json', ctx.execution)
    if progress:
        progress(stage='independent_clean_linkage_validation', p=p)
    actual = dp._linkage(ctx, {'family':'private', 'scale':0., 'proposal':0})
    validation = dp._validate_clean_linkage_replay(ctx, {'actual_percent':actual})
    dp.lock_json(out / 'clean_linkage_validation.json', validation)
    values = validation['production_identity_top1']
    ci = casia.identity_bootstrap(values, ctx.base.named_seed(ctx.cfg.seed, 'linkage-ci', 'cross_image_identity'),
                                   ctx.cfg.bootstrap_resamples)
    ctx.clean_linkage = {'input_hash':ctx.input_hash, 'p':p, 'h':H, 'clean_linkage_percent':actual,
        'identity_top1':values, 'identity_bootstrap_95ci_percent':[100.*x for x in ci], 'chance_percent':10.,
        'auditor':'CASIA-WebFace InceptionResnetV1', 'validation_sha256':dp.sha(out/'clean_linkage_validation.json'),
        'same_runtime_independent_replay_passed':True, 'upper_bin_contains_clean_score':actual >= dp.BINS[-2],
        'upper_bin_lower_bound_percent':dp.BINS[-2],
        'upper_bin_note':'Diagnostic only; no monotonicity/maximum claim; finite bin quotas unchanged'}
    dp.lock_json(out / 'clean_linkage.json', ctx.clean_linkage)
    ctx.clean_linkage_validation = validation
    ctx.clean_linkage_preflight = ctx.clean_linkage
    ctx.clean_linkage_hash = dp.sha(out / 'clean_linkage.json')
    return ctx
