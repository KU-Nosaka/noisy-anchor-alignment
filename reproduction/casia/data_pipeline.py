"""Verified face-benchmark inputs and fresh, linkage-conditioned noise draws.

Protocol algebra uses CPU float64; the frozen CASIA auditor uses CUDA float32. Only
``build_arrays`` invokes spectral alignment; calibration never invokes an
alignment optimizer or sees utility outcomes. This module never trains a model.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

VERSION = "casia-rebinned-h400-bin20-20260928-v2"
STUDY_SEED = 20260928
BINS = (0., 15., 25., 35., 45., 100.)
PER_BIN = 20
PARTICIPANTS = (2, 5, 10, 20, 50)
UPPER = {"anchor": .08, "private": 3.}
MAX_PROPOSALS = 1500  # per family; failure preserves every evaluated proposal
MODULES = ("core_protocols.py", "celeba_public_svd_identity100_runner.py")
PROTOCOL_CONFIG = {}


def configure(spec):
    """One dataset per process; fixed scientific quota protocol."""
    global VERSION, STUDY_SEED, PROTOCOL_CONFIG
    require(spec.get('sampling_mode', 'quota_bins') == 'quota_bins', 'Only authorized quota bins supported')
    require(tuple(spec.get('participants', PARTICIPANTS)) == PARTICIPANTS, 'Participant sweep differs')
    require(tuple(spec.get('linkage_bin_edges_percent', spec.get('bin_edges', BINS))) == BINS, 'Absolute linkage bins differ')
    require(int(spec.get('conditions_per_bin_per_family', spec.get('per_bin', PER_BIN))) == PER_BIN,
            'Twenty observations per bin required')
    require(int(spec.get('max_proposals_per_family', MAX_PROPOSALS)) == MAX_PROPOSALS,
            'Finite proposal budget differs')
    VERSION = str(spec['version']) + '-' + str(spec['dataset']).lower()
    STUDY_SEED = int(spec['seed'])
    PROTOCOL_CONFIG = dict(spec)


def prepare_inputs(spec, root, progress=None):
    configure(spec)
    module = importlib.import_module(spec['input_adapter'])
    module.prepare_inputs(spec['source_root'], spec['cache_root'], spec['module_dir'],
                          protocol_config=spec, progress=progress)
    return str(spec['source_root'])


def load_context(source_root, out_dir, p, module_dir, audit_device='cuda',
                 protocol_config=None, cache_root=None, progress=None):
    spec = protocol_config or PROTOCOL_CONFIG
    configure(spec)
    require(p in PARTICIPANTS, 'Unsupported participant count')
    adapter = importlib.import_module(spec['input_adapter'])
    inputs = adapter.load_frozen_data(source_root, p, module_dir, out_dir,
                                      cache_root=cache_root, progress=progress)
    from context_common import finalize_context
    return finalize_context(inputs, out_dir, audit_device, spec, progress)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def lock_json(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, f"Incompatible resume: {path}")
    else:
        atomic_json(path, value)


def linkage_bin(percent):
    value = round(float(percent), 10)
    require(math.isfinite(value) and 0 <= value <= 100, "Invalid linkage percentage")
    return next(i for i in range(5) if value < BINS[i + 1] or i == 4)


def control_conditions(p):
    require(p in PARTICIPANTS, "Unsupported participant count")
    return ([{"id": f"p{p:03d}_{family}", "p": p, "family": family,
              "scale": 0., "role": "control"} for family in ("c_gdp", "i_gdp")]
            + [{"id": f"p{p:03d}_local_{i:03d}", "p": p, "family": "local",
                "participant": i, "scale": 0., "role": "control"} for i in range(p)])


def _validate_clean_linkage_replay(context, audit):
    """Replay known-secret inversion against actual clean benchmark features.

    This validation-only path never changes production reconstructions,
    galleries, noise observations, or utility inputs.  The production reference
    cache is rebuilt independently using the declared CPU float32 decoding and
    identical reference batch boundaries.  Missing source-image entries are
    computed only to satisfy the frozen evaluator's unused source-image branch.
    """
    import torch
    ctx = context
    condition = {"family": "private", "scale": 0., "proposal": 0}
    known_reduced = _attack_reduced(ctx, condition)
    clean_reduced = ctx.query_reduced
    require(torch.allclose(known_reduced, clean_reduced, rtol=1e-10, atol=1e-10),
            "Known-secret clean recovery differs from true query features")
    o, psi = ctx.deployment["rotations"][0], ctx.deployment["translations"][0]
    colluding_x = ctx.blocks["train"][0][0].double()
    colluding_y = colluding_x @ o + psi
    target_uploads = {split: ctx.blocks[split][0][1].double() @ o + psi
                      for split in ("train", "val", "test")}
    target_queries = torch.stack([target_uploads[split][position]
                        for split, position in zip(ctx.linkage.query_splits,
                                                   ctx.linkage.query_target_positions)])
    centered_colluder = colluding_x - colluding_x.mean(0)
    colluder_rank = int(torch.linalg.matrix_rank(centered_colluder))
    learned_exact = {"training_rows": len(colluding_x), "centered_rank": colluder_rank,
                     "required_rank": clean_reduced.shape[1],
                     "status": "unavailable_rank_deficient",
                     "note": "Private-noise audit uses colluder-known shared secret; no learned-transform claim"}
    if colluder_rank == clean_reduced.shape[1]:
        exact_reduced = ctx.core.attack_c_known_input(colluding_x, colluding_y,
                                                     target_queries, estimator="exact").reduced
        learned_exact.update(status="checked", maximum_absolute_error=float((exact_reduced-clean_reduced).abs().max()),
                             allclose_1e_8=bool(torch.allclose(exact_reduced, clean_reduced, rtol=1e-8, atol=1e-8)))
    reference_tokens = sorted({token for groups in ctx.linkage.cross_galleries
                               for gallery in groups for token in gallery})
    all_tokens = {token for task in (ctx.linkage.source_galleries, ctx.linkage.cross_galleries)
                  for groups in task for gallery in groups for token in gallery}
    require(set(ctx.gallery) == set(reference_tokens), "Production reference gallery token set differs")
    replay_gallery = {}
    f = ctx.data.reducer.F.to(device="cpu", dtype=torch.float32)
    with torch.no_grad():
        for tokens in (reference_tokens, sorted(all_tokens - set(reference_tokens))):
            for start in range(0, len(tokens), ctx.cfg.face_batch_size):
                batch = tokens[start:start + ctx.cfg.face_batch_size]
                pixels = torch.stack([ctx.base._candidate_image(ctx.data, token) for token in batch])
                x = pixels.to(device="cpu", dtype=torch.float32).flatten(1) / 255.
                images = ((x @ f) @ f.T).reshape(-1, *ctx.base.RAW_SHAPE).clamp(0, 1)
                embeddings = ctx.base._face_embeddings(ctx.face_model, images, ctx.audit_cfg)
                replay_gallery.update(zip(batch, embeddings))
        require(all(torch.equal(ctx.gallery[token], replay_gallery[token]) for token in reference_tokens),
                "Production reference gallery differs from independent clean replay")
        known_images = (known_reduced.float() @ ctx.decoder).reshape(-1, *ctx.base.RAW_SHAPE).clamp(0, 1)
        production_embeddings = ctx.base._face_embeddings(ctx.face_model, known_images, ctx.audit_cfg)
        production_identity_top1 = []
        successes_all = []
        for query, groups in enumerate(ctx.linkage.cross_galleries):
            genuine = ("reference", int(ctx.linkage.query_identities[query]))
            successes = []
            for gallery in groups:
                scores = torch.stack([ctx.gallery[token] for token in gallery]) @ production_embeddings[query]
                winner = int(torch.argsort(scores, descending=True)[0])
                successes.append(int(winner == gallery.index(genuine)))
            require(bool(successes), "Clean validation has an empty query gallery")
            production_identity_top1.append(sum(successes) / len(successes))
            successes_all.extend(successes)
        production_percent = 100. * sum(successes_all) / len(successes_all)
        require(production_percent == audit["actual_percent"],
                "Clean production aggregation differs from original linkage path")
        clean_images = (clean_reduced.float() @ ctx.decoder).reshape(-1, *ctx.base.RAW_SHAPE).clamp(0, 1)
        frozen = ctx.base._evaluate_face_linkage(clean_images, ctx.linkage, replay_gallery,
                        ctx.face_model, ctx.audit_cfg)["cross_image_identity"]
    require(production_identity_top1 == list(frozen["identity_top1"]),
            "Clean production per-query scores differ from independent frozen evaluator")
    # The decisions above agree exactly; this tolerance only handles distinct
    # floating representations of the same integer success fraction.
    frozen_percent = 100. * float(frozen["top1"])
    require(math.isclose(production_percent, frozen_percent, rel_tol=0., abs_tol=1e-12),
            "Clean independent evaluator aggregate disagrees with its per-query scores")
    return {"version": "clean-casia-h400-independent-replay-20260928-v1", "passed": True,
            "input_hash": ctx.input_hash, "p": ctx.cfg.n_participants, "h": 400,
            "execution": ctx.execution,
            "validation_adaptation": "Known-secret replay vs true clean benchmark rows; C-exact learned inversion only diagnostic when colluder has full centered rank",
            "known_secret_maximum_absolute_error": float((known_reduced-clean_reduced).abs().max()),
            "known_secret_allclose_rtol_atol": 1e-10,
            "learned_c_exact_diagnostic": learned_exact,
            "reference_embeddings_exactly_equal": True, "per_query_scores_exactly_equal": True,
            "production_percent": production_percent, "frozen_replay_percent": frozen_percent,
            "production_identity_top1": production_identity_top1,
            "frozen_replay_identity_top1": list(frozen["identity_top1"]),
            "reference_replay": "CPU float32 F400 projection and F400.T decoding; original cross-reference batches",
            "source_image_branch": "decoded source entries for frozen evaluator; source scores unused",
            "historical_results_used": False}


def context_provenance(context):
    """Immutable data/scientific identity plus explicit accelerator/runtime identity."""
    return {"input_hash": context.input_hash, "scientific": context.input_lock,
            "execution": context.execution,
            "clean_linkage_sha256": context.clean_linkage_hash}


input_provenance = context_provenance


def _noise(context, condition, shape, split, participant):
    import torch
    seed = context.base.named_seed(STUDY_SEED, VERSION, "noise", condition["family"],
                                   condition["proposal"], split, participant)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return float(condition["scale"]) * torch.randn(tuple(shape), generator=generator, dtype=torch.float64)


def _attack_reduced(context, condition):
    """Recover target query rows from the exact same noise used for utility."""
    import torch
    ctx, dep, family = context, context.deployment, condition["family"]
    if family == "anchor":
        a, o, psi = dep["anchor"], dep["rotations"][1], dep["translations"][1]
        noisy_anchor = a @ o + psi + _noise(ctx, condition, a.shape, "anchor", 1)
        queries = ctx.query_reduced.double() @ o + psi
        recovered = ctx.core.attack_pa_op(a, noisy_anchor, queries).reduced
    else:
        o, psi = dep["rotations"][0], dep["translations"][0]
        uploads = {}
        for split in ("train", "val", "test"):
            x = ctx.blocks[split][0][1].double()
            uploads[split] = x @ o + psi + _noise(ctx, condition, x.shape, split, 1)
        queries = torch.stack([uploads[split][position] for split, position in
                               zip(ctx.linkage.query_splits, ctx.linkage.query_target_positions)])
        recovered = ctx.core.invert_known_gdp(queries, o, psi).reduced
    return recovered


def _linkage(context, condition):
    import torch
    ctx = context
    recovered = _attack_reduced(ctx, condition)
    images = (recovered.float() @ ctx.decoder).reshape(-1, 3, 64, 64).clamp(0, 1)
    with torch.no_grad():
        queries = ctx.base._face_embeddings(ctx.face_model, images, ctx.audit_cfg)
    successes = []
    for index, galleries in enumerate(ctx.linkage.cross_galleries):
        genuine = ("reference", int(ctx.linkage.query_identities[index]))
        for gallery in galleries:
            scores = torch.stack([ctx.gallery[token] for token in gallery]) @ queries[index]
            winner = int(torch.argsort(scores, descending=True)[0])
            successes.append(int(winner == gallery.index(genuine)))
    return 100. * sum(successes) / len(successes)


def proposal_range(family, attempts, counts, proposal):
    """Recorded pilot/adaptive proposal; acceptance uses linkage, never utility."""
    upper = UPPER[family]
    if proposal < 30:
        # Six broad, disjoint ranges repeatedly cover very low to high noise.
        edges = [0., upper / 128, upper / 32, upper / 8, upper / 2, upper * .75, upper]
        j = proposal % 6
        return edges[j], edges[j + 1], "fixed-six-range-pilot", None
    target = min((i for i in range(5) if counts[i] < PER_BIN), key=lambda i: (counts[i], i))
    if proposal % 5 == 0:
        return 0., upper, "full-range-exploration", target
    candidates = [a["condition"]["scale"] for a in attempts
                  if BINS[target] - 5 <= a["linkage_percent"] <= BINS[target + 1] + 5]
    if len(candidates) >= 2:
        low, high = min(candidates), max(candidates)
        pad = max((high - low) * .15, upper / 2048)
        return max(0., low - pad), min(upper, high + pad), "linkage-guided-uniform", target
    center = (BINS[target] + BINS[target + 1]) / 2
    nearest = min(attempts, key=lambda a: abs(a["linkage_percent"] - center))["condition"]["scale"]
    return max(0., nearest / 2), min(upper, max(nearest * 2, upper / 2048)), "nearest-linkage-uniform", target


def validate_accepted(conditions, p):
    require(len(conditions) == 200, "Need 200 accepted noisy observations")
    require(len({c["id"] for c in conditions}) == 200, "Duplicate condition IDs")
    for family in UPPER:
        rows = [c for c in conditions if c["family"] == family]
        require(len(rows) == 100, "Need 100 observations per noise family")
        for b in range(5):
            require(sum(c["bin"] == b for c in rows) == PER_BIN, "Need 20 observations per bin")
        for c in rows:
            require(c["p"] == p and c["bin"] == linkage_bin(c["linkage_percent"]), "Wrong p or bin")
            require(math.isfinite(c["scale"]) and 0 < c["scale"] <= UPPER[family], "Invalid noise scale")


class CalibrationIncomplete(RuntimeError):
    def __init__(self, conditions):
        super().__init__("Calibration budget exhausted; limited-bin receipts and accepted observations preserved")
        self.conditions = conditions


def calibrate(context, out_dir, progress=None):
    """Fresh rejection calibration, resumable at every evaluated proposal."""
    import numpy as np
    ctx, out = context, Path(out_dir) / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    lock_json(out / "design.json", {"version": VERSION, "input_hash": ctx.input_hash,
              "bin_edges": list(BINS), "per_bin": PER_BIN, "max_proposals_per_family": MAX_PROPOSALS,
              "selection": "first 20 accepted in each bin; linkage-only; no utility selection",
              "proposal_policy": "30 fixed six-range pilot; thereafter four guided uniforms per one global uniform",
              "noise_seed_rule": "named_seed(base_seed,version,noise,family,proposal,split,participant); CPU float64"})
    finished = out / "accepted_conditions.json"
    if finished.is_file():
        saved = read(finished)
        require(saved["input_hash"] == ctx.input_hash, "Calibration input changed")
        require(saved["conditions_sha256"] == canonical_hash(saved["conditions"]), "Calibration checksum differs")
        validate_accepted(saved["conditions"], ctx.cfg.n_participants)
        return saved["conditions"]
    accepted = []
    for family in UPPER:
        attempts, counts = [], [0] * 5
        for proposal in range(MAX_PROPOSALS):
            if counts == [PER_BIN] * 5:
                break
            low, high, policy, target = proposal_range(family, attempts, counts, proposal)
            rng = np.random.default_rng(ctx.base.named_seed(STUDY_SEED, VERSION, "scale", family, proposal))
            scale = max(float(rng.uniform(low, high)), float(np.nextafter(0., 1.)))
            condition = {"id": f"p{ctx.cfg.n_participants:03d}_{family}_{proposal:06d}",
                         "p": ctx.cfg.n_participants, "family": family, "proposal": proposal,
                         "scale": scale, "role": "noise", "proposal_interval": [low, high],
                         "proposal_policy": policy, "target_bin": target,
                         "training_seed": STUDY_SEED}
            path = out / "attempts" / f"{family}_{proposal:06d}.json"
            if path.exists():
                row = read(path)
                require(row["condition"] == condition and row["input_hash"] == ctx.input_hash,
                        f"Saved proposal changed: {path}")
            else:
                started = time.perf_counter()
                value = _linkage(ctx, condition)
                row = {"condition": condition, "linkage_percent": value,
                       "bin": linkage_bin(value), "input_hash": ctx.input_hash,
                       "audit_seconds": time.perf_counter() - started}
                row["result_sha256"] = canonical_hash(row)
                atomic_json(path, row)
            require(row.get("result_sha256") == canonical_hash(
                    {k: v for k, v in row.items() if k != "result_sha256"}),
                    f"Saved audit checksum differs: {path}")
            require(row["bin"] == linkage_bin(row["linkage_percent"]), "Saved proposal bin differs")
            attempts.append(row)
            if counts[row["bin"]] < PER_BIN:
                accepted.append({**condition, "linkage_percent": row["linkage_percent"],
                                 "bin": row["bin"], "audit_seconds": row["audit_seconds"],
                                 "input_hash": ctx.input_hash})
                counts[row["bin"]] += 1
            status = {"p": ctx.cfg.n_participants, "family": family, "proposals": proposal + 1,
                      "accepted_per_bin": counts, "accepted_total": len(accepted), "stage": "calibration"}
            atomic_json(out / "status.json", status)
            if progress:
                progress(status)
        if counts != [PER_BIN] * 5:
            report = {"state": "incomplete", "p": ctx.cfg.n_participants, "family": family,
                      "accepted_per_bin": counts, "requested_per_bin": PER_BIN,
                      "missing_per_bin": [PER_BIN-count for count in counts],
                      "max_proposals": MAX_PROPOSALS, "input_hash": ctx.input_hash,
                      "reason": "fixed proposal budget exhausted; bin boundaries/quotas unchanged"}
            atomic_json(out / (family + "_limited_bins.json"), report)
            atomic_json(out / "partial_conditions.json", {"input_hash": ctx.input_hash,
                        "conditions": accepted, "conditions_sha256": canonical_hash(accepted)})
            if progress:
                progress(dict(report, state="running", stage="calibration_limited_bins"))
    if len(accepted) != 200:
        raise CalibrationIncomplete(accepted)

    validate_accepted(accepted, ctx.cfg.n_participants)
    atomic_json(finished, {"input_hash": ctx.input_hash, "conditions": accepted,
                          "conditions_sha256": canonical_hash(accepted)})
    return accepted


def _noise_generator(context, condition, split, participant):
    import torch
    seed = context.base.named_seed(STUDY_SEED, VERSION, 'noise', condition['family'],
                                   condition['proposal'], split, participant)
    return torch.Generator(device='cpu').manual_seed(seed)


def _array_directory(context, condition):
    root = Path(context.protocol_config.get('array_cache_root',
                   context.out.parent / 'regenerable_arrays')).resolve()
    key = canonical_hash({'condition':condition, 'input_hash':context.input_hash})
    directory = (root / f'p{context.cfg.n_participants:03d}' / key).resolve()
    require(directory.is_relative_to(root) and directory != root, 'Array cache escapes owned root')
    return root, directory


def build_arrays(context, condition):
    """Transform in float64 batches, committing one float32 condition to memmaps.

    At h400 each random chunk has a multiple-of-16 element count, preserving
    Torch CPU normal-generator consumption versus each original full block.
    Only the current condition is cached; its verified fit permits release.
    """
    import numpy as np
    import torch
    ctx, dep, family = context, context.deployment, condition['family']
    require(condition['p'] == ctx.cfg.n_participants, 'Condition participant count differs')
    require(family in ('anchor','private','c_gdp','i_gdp','local'), 'Unknown family')
    if family in UPPER:
        require(condition.get('input_hash') == ctx.input_hash, 'Uncalibrated or mismatched condition')
    root, directory = _array_directory(ctx, condition)
    directory.mkdir(parents=True, exist_ok=True)
    signature = {'condition':condition, 'input_hash':ctx.input_hash,
                 'preparation':'chunked-float64-to-float32-h400-v1'}
    lock_json(directory / 'identity.json', signature)
    receipt_path = directory / 'arrays.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt['identity'] == signature, 'Array cache identity differs')
        for name, digest in receipt['file_hashes'].items():
            require(name in {f'{s}_{k}.npy' for s in ('train','val','test') for k in ('x','y')}, 'Unexpected cached array file')
            require(sha(directory / name) == digest, 'Cached transformed array checksum differs')
    else:
        started = time.perf_counter()
        diagnostics, elapsed, upload_seconds, spectral_seconds = {}, 0., 0., 0.
        if family == 'anchor':
            from spectral import spectral_align
            anchor = dep['anchor']
            uploads = [anchor @ o + psi + _noise(ctx, condition, anchor.shape, 'anchor', i)
                for i,(o,psi) in enumerate(zip(dep['rotations'],dep['translations']))]
            means = [b.mean(0) for b in uploads]
            centered = torch.stack([b-mean for b,mean in zip(uploads,means)])
            del uploads
            upload_seconds = time.perf_counter() - started
            spectral_started = time.perf_counter()
            rotations, diagnostics = spectral_align(centered)
            del centered
            spectral_seconds = time.perf_counter() - spectral_started
            elapsed = time.perf_counter() - started
        batch_rows = int(ctx.protocol_config.get('transform_batch_rows', 2048))
        require(batch_rows > 0, 'Invalid transform batch size')
        file_hashes = {}
        for split in ('train','val','test'):
            xs, ys, _ = ctx.blocks[split]
            participants = [int(condition['participant'])] if family == 'local' else list(range(len(xs)))
            require(all(0 <= i < ctx.cfg.n_participants for i in participants), 'Invalid local participant')
            n = sum(len(xs[i]) for i in participants)
            x_path, y_path = directory / f'{split}_x.npy', directory / f'{split}_y.npy'
            x_tmp, y_tmp = x_path.with_suffix('.npy.tmp'), y_path.with_suffix('.npy.tmp')
            x_map = np.lib.format.open_memmap(x_tmp, mode='w+', dtype=np.float32, shape=(n,400))
            y_map = np.lib.format.open_memmap(y_tmp, mode='w+', dtype=np.float32, shape=(n,))
            offset = 0
            for i in participants:
                x, labels = xs[i], ys[i]
                require(x.ndim == 2 and x.shape[1] == 400 and len(labels) == len(x), 'Invalid h400 block')
                generator = _noise_generator(ctx, condition, split, i) if family == 'private' else None
                for start in range(0,len(x),batch_rows):
                    end = min(start+batch_rows,len(x)); part = x[start:end].double()
                    if family == 'local':
                        value = part
                    elif family in ('c_gdp','private'):
                        value = part @ dep['rotations'][0] + dep['translations'][0]
                        if family == 'private':
                            value += float(condition['scale']) * torch.randn(part.shape, generator=generator, dtype=torch.float64)
                    else:
                        value = part @ dep['rotations'][i] + dep['translations'][i]
                        if family == 'anchor':
                            value = (value - means[i]) @ rotations[i]
                    require(bool(torch.isfinite(value).all()), 'Nonfinite transformed features')
                    x_map[offset+start:offset+end] = value.float().numpy()
                    y_map[offset+start:offset+end] = labels[start:end].float().numpy()
                offset += len(x)
            x_map.flush(); y_map.flush(); del x_map,y_map
            os.replace(x_tmp,x_path); os.replace(y_tmp,y_path)
            file_hashes[x_path.name], file_hashes[y_path.name] = sha(x_path), sha(y_path)
        metadata = {'condition':condition,'input_hash':ctx.input_hash,
            'alignment':'spectral_only' if family == 'anchor' else 'none', 'spectral':diagnostics,
            'array_preparation_seconds':time.perf_counter()-started,
            'alignment_seconds_scope':'anchor upload generation, centering, and spectral call including diagnostics',
            'anchor_upload_and_centering_seconds':upload_seconds,'spectral_call_wall_seconds':spectral_seconds,
            'spectral_svd_and_polar_seconds':float(diagnostics.get('alignment_seconds',0.)),
            'protocol_dtype':'float64','training_input_dtype':'float32','transform_batch_rows':batch_rows,
            'storage':'regenerable CPU float32 npy memmaps; no full raw or transformed float64 concatenation'}
        receipt = {'identity':signature,'file_hashes':file_hashes,'metadata':metadata,'alignment_seconds':elapsed}
        atomic_json(receipt_path,receipt)
    output = {f'{split}_{kind}':torch.from_numpy(np.load(directory/f'{split}_{kind}.npy',mmap_mode='c'))
              for split in ('train','val','test') for kind in ('x','y')}
    output.update(metadata=receipt['metadata'],alignment_seconds=receipt['alignment_seconds'])
    return output


def release_arrays(context, condition):
    """Caller must release tensor views after a durable verified fit receipt."""
    import gc
    import shutil
    root, directory = _array_directory(context, condition)
    if directory.exists():
        identity = read(directory / 'identity.json')
        require(identity['condition'] == condition and identity['input_hash'] == context.input_hash,
                'Refuse to release another condition cache')
        require(directory.is_relative_to(root) and directory.parent.parent == root,
                'Refuse to release cache outside the exact owned condition directory')
        gc.collect()
        shutil.rmtree(directory)


def release_context(context):
    import gc
    import torch
    context.face_model = None
    context.gallery = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
