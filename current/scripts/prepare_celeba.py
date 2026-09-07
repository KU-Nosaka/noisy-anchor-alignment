"""Fresh-input initialization used verbatim by the public CelebA_P50 notebook.

Call after its data/model/reporting definitions have loaded, and before shared
deployment preparation. This helper does not train a model, build a projection,
score linkage, or modify a directory that already has a protocol manifest.
"""

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


def initialize_parallel_inputs(config, scientific_code_sha256, *,
                               source_factory, auditor_factory, runtime):
    """Initialize a fresh legacy-input directory; return False for existing runs.

    Factories are the unchanged prepare_source and FaceAuditor definitions from
    the reported experiment. runtime is torch. An existing protocol is handled
    entirely by the original preparation/resume validation, without mutation.
    """
    root = Path(config['output_root'])
    if (root / 'protocol.json').exists():
        return False
    # A cache can survive a failed download, but unidentified result or metadata
    # files must never be adopted as a fresh run or silently overwritten.
    if root.exists():
        unexpected = [p.name for p in root.iterdir() if p.name != 'cache']
        if unexpected:
            raise RuntimeError('Cannot initialize a nonempty experiment without '
                               f'a protocol manifest: {sorted(unexpected)}')
    if not runtime.cuda.is_available():
        raise RuntimeError('Select a GPU runtime before CelebA preparation.')
    identity = {'config': config, 'code_sha256': scientific_code_sha256}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    environment = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch': runtime.__version__, 'cuda': runtime.version.cuda,
        'gpu': runtime.cuda.get_device_name(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'pandas', 'torch', 'torchvision',
                      'facenet-pytorch', 'Pillow', 'matplotlib')},
    }
    root.mkdir(parents=True, exist_ok=True)
    lock = root / 'initialization.lock'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    try:
        source = source_factory(Path(config['old_cache']), root / 'cache')
        source_manifest = source['source_manifest']
        del source
        old_torch = Path(config['old_cache']) / 'torch_home'
        new_torch = root / 'cache' / 'torch_home'
        if old_torch.is_dir():
            for src in old_torch.rglob('*.pt'):
                dst = new_torch / src.relative_to(old_torch)
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
        auditor = auditor_factory(device='cuda', torchcache=new_torch)
        auditor_metadata = dict(auditor.metadata)
        del auditor
        gc.collect()
        runtime.cuda.empty_cache()
        seeds = len(config['seeds'])
        draws = config['realizations']
        counts = len(config['participant_counts'])
        fits = seeds * (counts * (len(config['anchor_scales']) * draws + 3)
                        + len(config['private_sigmas']) * draws
                        + max(config['participant_counts']))
        rows = seeds * (counts * (len(config['anchor_scales']) * draws + 4)
                        + len(config['private_sigmas']) * draws)
        now = datetime.now(timezone.utc).isoformat()
        payloads = {
            'environment.json': environment,
            'runtime_sessions.json': [{**environment, 'initialized_utc': now}],
            'source_manifest.json': source_manifest,
            'auditor.json': auditor_metadata,
            'progress.json': {
                'status': 'initialized', 'completed_fits': 0,
                'completed_local_fits': 0, 'completed_local_aggregates': 0,
                'completed_result_rows': 0, 'scheduled_fits': fits,
                'scheduled_result_rows': rows, 'current': None, 'error': None,
                'session_elapsed_seconds': 0.0, 'updated_utc': now,
            },
            'initialization.json': {
                'schema': 'celeba-public-fresh-inputs-v1',
                'initialized_utc': now, 'scientific_code_sha256': scientific_code_sha256,
                'scope': 'Fresh input manifests only; no utility fits, projections, or attack scores.',
            },
            # Commit the protocol last, after all dependencies are complete.
            'protocol.json': {**identity, 'signature': signature,
                              'expected_fits': fits, 'expected_result_rows': rows},
        }
        with TemporaryDirectory(prefix='.initialization-', dir=root) as temp:
            temp = Path(temp)
            for name, payload in payloads.items():
                (temp / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
            for name in payloads:
                os.replace(temp / name, root / name)
        print(f'Initialized fresh CelebA inputs: 0/{fits} fits; no training performed.', flush=True)
        return True
    finally:
        lock.unlink()
