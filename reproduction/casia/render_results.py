"""Export calibrated-bin results with actual counts and explicit completion state."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

VERSION = 'casia-rebinned-figures-20260928-v2'
BINS = (0, 15, 25, 35, 45, 100)
PS = (2, 5, 10, 20, 50)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    root = Path(root)
    identity = json.loads((root / 'pipeline_identity.json').read_text())
    if tuple(identity['configuration'].get('linkage_bin_edges_percent', ())) != BINS:
        raise RuntimeError('Figure configuration does not use the revised bins')
    dataset = identity['configuration'].get('dataset_label', identity['configuration']['dataset'])
    target = root / 'figures'
    target.mkdir(exist_ok=True)
    condition_rows, statistics_rows, controls, inputs = [], [], [], {}
    for p in PS:
        path = root / 'study' / f'p{p:03d}' / 'summary.json'
        if not path.is_file():
            raise RuntimeError(f'Missing terminal p={p} summary; no full-sweep figure emitted')
        summary = json.loads(path.read_text())
        inputs[str(path.relative_to(root))] = sha(path)
        if summary['p'] != p:
            raise RuntimeError('Participant summary does not match its folder')
        if tuple(summary.get('bin_edges_percent', ())) != BINS:
            raise RuntimeError('Participant summary does not use the revised bins')
        seen = set()
        for record in summary['records']:
            c, result = record['condition'], record['result']
            if c['id'] in seen:
                raise RuntimeError('Duplicate condition in a summary')
            seen.add(c['id'])
            test = result['test']
            utility = float(test['balanced_accuracy']) * 100
            if c['family'] in ('anchor', 'private'):
                score = float(c['linkage_percent'])
                b = next(i for i in range(5) if score < BINS[i + 1] or i == 4)
                if b != c['bin'] or not 0 <= score <= 100:
                    raise RuntimeError('Condition linkage does not match its calibrated bin')
                condition_rows.append({'p': p, 'family': c['family'], 'id': c['id'],
                    'noise_scale': c['scale'], 'bin': b, 'linkage_percent': score,
                    'test_balanced_accuracy_percent': utility, 'test_n': test['n']})
            elif c['family'] in ('i_gdp', 'c_gdp'):
                controls.append({'p': p, 'method': c['family'], 'test_balanced_accuracy_percent': utility,
                                 'test_n': test['n'], 'n_models': 1})
        local = summary['local_pooled']
        controls.append({'p': p, 'method': 'local_pooled', 'test_balanced_accuracy_percent':
                         float(local['test_balanced_accuracy']) * 100, 'test_n': local['n'], 'n_models': p})
        for family in ('anchor', 'private'):
            for b in range(5):
                values = [r for r in condition_rows if r['p'] == p and r['family'] == family and r['bin'] == b]
                ys = [r['test_balanced_accuracy_percent'] for r in values]
                xs = [r['linkage_percent'] for r in values]
                statistics_rows.append({'p': p, 'family': family, 'bin': b,
                    'bin_low_percent': BINS[b], 'bin_high_percent': BINS[b + 1],
                    'n': len(values), 'target_n': 20, 'quota_met': len(values) == 20,
                    'mean_linkage_percent': statistics.mean(xs) if xs else None,
                    'mean_utility_percent': statistics.mean(ys) if ys else None,
                    'utility_sample_sd_percent': statistics.stdev(ys) if len(ys) > 1 else None})
    save_csv(target / 'conditions.csv', condition_rows, ['p','family','id','noise_scale','bin','linkage_percent','test_balanced_accuracy_percent','test_n'])
    save_csv(target / 'binned_statistics.csv', statistics_rows, ['p','family','bin','bin_low_percent','bin_high_percent','n','target_n','quota_met','mean_linkage_percent','mean_utility_percent','utility_sample_sd_percent'])
    save_csv(target / 'control_utility.csv', controls, ['p','method','test_balanced_accuracy_percent','test_n','n_models'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(16.0, 3.5), sharex=True, sharey=True, constrained_layout=True)
    colors = {'anchor': '#2369A8', 'private': '#C65D25'}
    labels = {'anchor': 'Anchor noise (PA-OP)', 'private': 'Private noise (known-secret)'}
    for ax, p in zip(axes, PS):
        for family in ('anchor', 'private'):
            records = [r for r in statistics_rows if r['p'] == p and r['family'] == family and r['n']]
            ax.errorbar([r['mean_linkage_percent'] for r in records],
                        [r['mean_utility_percent'] for r in records],
                        yerr=[r['utility_sample_sd_percent'] or 0 for r in records],
                        fmt='o', ms=4, capsize=3, color=colors[family], label=labels[family])
        counts = {f: [r['n'] for r in statistics_rows if r['p'] == p and r['family'] == f]
                  for f in ('anchor', 'private')}
        ax.set_title(f'p = {p}')
        ax.text(.02, .02, 'n by bin\nA: ' + '/'.join(map(str, counts['anchor'])) +
                '\nP: ' + '/'.join(map(str, counts['private'])), transform=ax.transAxes, fontsize=7)
        ax.axvline(10, color='#777777', linestyle=':', linewidth=1)
        for edge in BINS[1:-1]:
            ax.axvline(edge, color='#DDDDDD', linewidth=.6, zorder=0)
        ax.set(xlim=(0, 100), ylim=(0, 100), xlabel='Identity linkage (%)')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Smiling balanced accuracy (%)')
    axes[-1].legend(fontsize=7, loc='upper right')
    complete = all(r['quota_met'] for r in statistics_rows)
    title = 'Calibrated privacy–utility results' if complete else 'Partial calibrated results — bin quotas incomplete'
    fig.suptitle(f'{dataset}: {title}')
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(target / f'privacy_utility_all_p.{suffix}', dpi=200)
    plt.close(fig)
    caption = ('Points are per-bin means over the actual accepted noise conditions; bars are sample standard deviations of utility across those conditions. '
               'They are not confidence intervals, uncertainty across cohorts, or variation across independent training seeds. '
               'The x coordinate is mean observed ten-way linkage, and the dotted line marks 10% chance. '
               'A/P counts list anchor/private conditions in the five user-approved revised bins: [0,15), [15,25), [25,35), [35,45), [45,100]. Empty bins are omitted, not interpolated. '
               'I-GDP, C-GDP and pooled Local utility are reported separately in control_utility.csv. '
               'A full completion claim additionally requires the pipeline completion receipt and its artifact checks.\n')
    (target / 'caption.txt').write_text(caption, encoding='utf-8')
    from manuscript_plots import render
    render(target, identity['configuration']['dataset'], statistics_rows, controls)
    provenance = {'version': VERSION, 'bin_edges_percent': list(BINS), 'summary_inputs_sha256': inputs, 'renderer_sha256': sha(__file__),
                  'manuscript_style_renderer_sha256': sha(Path(__file__).with_name('manuscript_plots.py')),
                  'manuscript_error_bars': 'sample SD; separate calibration at each participant count',
                  'missing_bins': 'omitted, with actual counts in binned_statistics.csv',
                  'all_bin_quotas_met': complete, 'error_bars': 'utility sample SD across accepted noise conditions',
                  'artifacts_sha256': {p.name: sha(p) for p in sorted(target.iterdir()) if p.is_file() and p.name != 'figure_provenance.json'}}
    (target / 'figure_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'figures': str(target), 'all_bin_quotas_met': complete, 'noise_fits': len(condition_rows)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    summarize(parser.parse_args().root)
