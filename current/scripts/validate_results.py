"""Validate complete current-manuscript schedules and seed-level summaries."""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NAA = 'NAA-GDP'
PRIVATE = 'C-GDP-private-noise'
KEYS = ['method', 'participant_count', 'noise_scale']
METRICS = ['balanced_accuracy', 'linkage_rate']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_schedule(raw, participants, private_p):
    expected = []
    for seed in range(5):
        for p in participants:
            expected.extend((seed, p, method, 0, 0) for method in
                            ['AA-GDP', 'C-GDP', 'I-GDP', 'Local-only'])
            for method in [NAA] + ([PRIVATE] if p == private_p else []):
                expected.extend((seed, p, method, level, realization)
                                for level in range(1, 11) for realization in range(4))
    cols = ['outer_seed', 'participant_count', 'method', 'level_index', 'realization']
    require(Counter(raw[cols].itertuples(index=False, name=None)) == Counter(expected),
            'Incomplete, duplicate, or unexpected seed/participant/method/level/realization schedule')
    require(raw.run_id.is_unique, 'Duplicate utility run ID')
    scale = np.where(raw.method == PRIVATE, raw.level_index * .3, raw.level_index * .006)
    np.testing.assert_allclose(raw.noise_scale, scale, rtol=0, atol=1e-12)
    require(raw.balanced_accuracy.between(0, 1).all(), 'Invalid utility measurement')
    utility_only = raw.method.isin(['I-GDP', 'Local-only'])
    require(raw.loc[utility_only, 'linkage_rate'].isna().all(),
            'Utility-only controls must retain unmeasured linkage as NaN')
    require(raw.loc[~utility_only, 'linkage_rate'].between(0, 1).all(), 'Missing or invalid measured linkage')
    for key, rows in raw.groupby(['outer_seed', 'participant_count']):
        require(rows.deployment_hash.nunique() == 1, f'Mixed deployments in {key}')
    require(raw.auditor_sha256.nunique() == 1, 'Mixed frozen auditors')


def check_summary(raw, saved, keys=KEYS, metrics=METRICS, sd_suffix='seed_sd'):
    seed = raw.groupby(keys + ['outer_seed'])[metrics].mean()
    require(seed.groupby(keys).size().eq(5).all(), 'Summary has fewer or more than five seed means')
    computed = seed.groupby(keys).agg(['mean', 'std'])
    computed.columns = ['_'.join(col) for col in computed.columns]
    reference = saved.set_index(keys).sort_index()
    computed = computed.sort_index()
    require(reference.index.is_unique and reference.index.equals(computed.index), 'Summary groups differ from raw results')
    for metric in metrics:
        for stat, suffix in [('mean', 'mean'), ('std', sd_suffix)]:
            np.testing.assert_allclose(computed[f'{metric}_{stat}'], reference[f'{metric}_{suffix}'],
                                       rtol=0, atol=1e-12, equal_nan=True,
                                       err_msg=f'Saved {metric} {suffix} disagrees with seed aggregation')
    if 'n_seeds' in saved:
        require(saved.n_seeds.eq(5).all(), 'Saved summary seed count differs')
    if 'final' in saved:
        require(saved.final.eq(True).all(), 'Provisional summary')
    return seed.reset_index()


def check_lfwa(folder):
    raw = pd.read_csv(folder/'raw_results.csv')
    attacks = pd.read_csv(folder/'attack_results.csv')
    check_schedule(raw, [5], 5)
    require(len(raw) == 420, 'LFWA must contain 420 result rows')
    check_summary(raw, pd.read_csv(folder/'summary.csv'))
    require(len(attacks) == 830 and not attacks.duplicated(['run_id', 'attack']).any(), 'LFWA attack count or IDs differ')
    expected = []
    for row in raw.itertuples():
        if row.method in [NAA, 'AA-GDP', 'C-GDP']:
            expected.extend((row.run_id, attack) for attack in ['MP', 'AM', 'OP'])
        elif row.method == PRIVATE:
            expected.append((row.run_id, 'known-common-transform'))
    require(Counter(attacks[['run_id', 'attack']].itertuples(index=False, name=None)) == Counter(expected),
            'LFWA attacks do not cover the complete utility schedule')
    joined = attacks.merge(raw[['run_id'] + KEYS + ['outer_seed', 'realization', 'linkage_rate']],
                           on='run_id', validate='many_to_one', suffixes=('', '_utility'))
    for key in KEYS + ['outer_seed', 'realization']:
        np.testing.assert_array_equal(joined[key], joined[f'{key}_utility'])
    main = joined.attack.isin(['OP', 'known-common-transform'])
    np.testing.assert_allclose(joined.loc[main, 'linkage_rate'], joined.loc[main, 'linkage_rate_utility'], rtol=0, atol=1e-12)
    check_summary(attacks, pd.read_csv(folder/'attack_summary.csv'), KEYS + ['attack'], ['linkage_rate'], 'std')
    naa = raw[raw.method == NAA]
    require(naa.gpm_converged.isin([True, False]).all(), 'LFWA has missing stopping flags')
    protocol = json.loads((folder/'protocol.json').read_text())['config']
    cap, tolerance = protocol['gpm_max_iters'], protocol['gpm_tolerance']
    converged = naa.gpm_converged.eq(True)
    require(naa.gpm_iterations.between(1, cap).all(), 'LFWA iteration count exceeds configured bounds')
    require(naa.loc[converged, 'gpm_last_step'].le(tolerance).all(), 'LFWA converged flag disagrees with stopping tolerance')
    require(naa.loc[~converged, 'gpm_iterations'].eq(cap).all(), 'LFWA capped flag disagrees with iteration limit')
    progress = json.loads((folder/'progress.json').read_text())
    require(progress['status'] == 'complete' and progress['completed_fits'] == 440
            and progress['completed_result_rows'] == 420 and progress['error'] is None,
            'LFWA progress does not certify the complete schedule')
    return {'result_rows': len(raw), 'fits': 440, 'attack_rows': len(attacks), 'gpm_converged': int(converged.sum()), 'gpm_capped': int((~converged).sum())}


def check_celeba(folder, gpm_path=None):
    raw = pd.read_csv(folder/'raw_results.csv')
    check_schedule(raw, [2, 5, 10, 20, 50], 10)
    require(len(raw) == 1300, 'CelebA must contain 1300 result rows')
    check_summary(raw, pd.read_csv(folder/'summary.csv'))
    naa = raw[raw.method == NAA]
    require(naa.groupby(['outer_seed', 'level_index', 'realization']).linkage_rate.nunique().eq(1).all(),
            'OP linkage differs across paired participant counts')
    attack = pd.read_csv(folder/'attack_results.csv')
    require(len(attack) == 1400 and attack.attack_id.is_unique, 'CelebA posthoc attack count or IDs differ')
    require(Counter(attack.attack) == Counter({'AM': 1000, 'MP': 200, 'OP': 200}), 'Unexpected posthoc attack coverage')
    require(attack.extra_utility_fits.eq(0).all(), 'Posthoc replay unexpectedly reports utility training')
    joined = pd.read_csv(folder/'privacy_utility_by_attack.csv')
    require(len(joined) == 3250 and not joined.duplicated(['run_id', 'attack']).any(), 'Joined attack rows differ')
    joined_utility = joined.merge(raw[['run_id'] + KEYS + ['outer_seed', 'realization', 'balanced_accuracy']],
                                  on='run_id', validate='many_to_one', suffixes=('', '_utility'))
    require(len(joined_utility) == len(joined), 'Attack table references missing utility rows')
    for key in KEYS + ['outer_seed', 'realization', 'balanced_accuracy']:
        np.testing.assert_array_equal(joined_utility[key], joined_utility[f'{key}_utility'])
    check_summary(joined, pd.read_csv(folder/'attack_summary.csv'), KEYS + ['attack'])
    gpm_path = Path(gpm_path) if gpm_path is not None else folder/'celeba_gpm_diagnostics.csv'
    gpm = pd.read_csv(gpm_path).set_index('run_id').sort_index()
    target = naa[naa.participant_count == 10].set_index('run_id').sort_index()
    require(len(gpm) == 200 and gpm.index.is_unique and gpm.index.equals(target.index), 'GPM rows do not match primary CelebA NAA runs')
    for field, raw_field in [('iterations', 'gpm_iterations'), ('last_step', 'gpm_last_step'), ('elapsed_seconds', 'alignment_seconds')]:
        np.testing.assert_allclose(gpm[field], target[raw_field], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(gpm.converged, target.gpm_converged)
    require(gpm.converged.isin([True, False]).all(), 'CelebA has missing stopping flags')
    require(gpm.iterations.between(1, 500).all(), 'CelebA iteration count exceeds configured bounds')
    require(gpm.loc[gpm.converged.eq(True), 'last_step'].le(1e-7).all(), 'CelebA converged flag disagrees with stopping tolerance')
    require(gpm.loc[gpm.converged.eq(False), 'iterations'].eq(500).all(), 'CelebA capped flag disagrees with iteration limit')
    require(np.isfinite(gpm.fixed_point_residual).all() and gpm.fixed_point_residual.gt(0).all(), 'Invalid fixed-point residual')
    progress = json.loads((folder/'progress.json').read_text())
    require(progress['utility_complete'] and progress['completed_fits'] == 1525
            and progress['completed_main_result_rows'] == 1300, 'CelebA progress does not certify completion')
    return {'result_rows': len(raw), 'fits': 1525, 'posthoc_attack_rows': len(attack),
            'primary_gpm_converged': int(gpm.converged.eq(True).sum()),
            'primary_gpm_capped': int(gpm.converged.eq(False).sum())}


def check_equivalence(folder):
    raw = pd.read_csv(folder/'equivalence_raw.csv')
    require(len(raw) == 800 and raw.passed.eq(True).all(), 'Equivalence check did not pass all 800 comparisons')
    identity = ['profile', 'dimension', 'participants', 'anchor_rows', 'seed', 'reference']
    require(raw.groupby(identity).size().eq(5).all() and len(raw.groupby(identity)) == 160,
            'Equivalence design must contain 160 deployments at five coupled noise levels')
    require(not raw.duplicated(identity + ['sigma']).any(), 'Duplicate equivalence comparison')
    return {'deployments': 160, 'paired_comparisons': 800, 'all_passed': True}


def locate_gpm(results):
    candidates = [results/'figures'/'manuscript_inputs'/'celeba_gpm_diagnostics.csv',
                  results/'celeba_parallel'/'merged'/'celeba_gpm_diagnostics.csv']
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError('Missing CelebA GPM diagnostics: expected ' + str(candidates[0])
                            + ' (or ' + str(candidates[1]) + '). Generate them with the figure notebook.')


def require_inputs(results):
    files = {
        'lfwa_full_sweep': ['raw_results.csv', 'summary.csv', 'attack_results.csv',
                            'attack_summary.csv', 'protocol.json', 'progress.json'],
        'celeba_parallel/merged': ['raw_results.csv', 'summary.csv', 'attack_results.csv',
                                  'privacy_utility_by_attack.csv', 'attack_summary.csv', 'progress.json'],
        'equivalence': ['equivalence_raw.csv'],
    }
    missing = [str(results/folder/name) for folder, names in files.items() for name in names
               if not (results/folder/name).is_file()]
    if missing:
        raise FileNotFoundError('Saved results are not bundled with this checkout. Supply a completed '
                               'directory with --results-root. Missing inputs: ' + ', '.join(missing))
    return locate_gpm(results)


def validate_all(current=None, results_root=None):
    current = Path(current) if current is not None else Path(__file__).resolve().parents[1]
    results = Path(results_root).resolve() if results_root is not None else current/'results'
    gpm_path = require_inputs(results)
    return {'lfwa': check_lfwa(results/'lfwa_full_sweep'),
            'celeba': check_celeba(results/'celeba_parallel'/'merged', gpm_path),
            'equivalence': check_equivalence(results/'equivalence')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, help='Separately supplied saved results directory; defaults to current/results')
    args = parser.parse_args()
    try:
        evidence = validate_all(results_root=args.results_root)
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        parser.error(str(exc))
    print(json.dumps(evidence, indent=2))
