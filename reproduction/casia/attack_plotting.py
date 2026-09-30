"""Numeric-only figures for a completed, verified uniform500 attack comparison."""
from pathlib import Path
import csv

import numpy as np

try:
    from . import attack_comparison as attack
except ImportError:
    import attack_comparison as attack


def spline_design(x):
    """The published truncated-power cubic basis with five interior knots."""
    values = np.asarray(x, dtype=np.float64).reshape(-1) / attack.UPPER
    return np.column_stack([np.ones_like(values), values, values**2, values**3]
        + [np.maximum(values - knot, 0.)**3 for knot in np.linspace(0., 1., 7)[1:-1]])


def paired_spline(x, y, resamples=10000, seed=20260930):
    """One bootstrap multiplicity vector is shared by all three attack curves."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    attack.require(x.shape == (500,) and y.shape == (500, 3), 'Expected 500 paired observations')
    attack.require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite spline observations')
    attack.require(((x > 0) & (x < attack.UPPER)).all(), 'Zero/end-point controls do not enter spline fits')
    attack.require(((y >= 0) & (y <= 100)).all(), 'Linkage is a percentage in [0,100]')
    attack.require(isinstance(resamples, int) and resamples >= 1, 'Positive integer bootstrap count required')
    design = spline_design(x); grid = np.linspace(0., attack.UPPER, 201); gd = spline_design(grid)
    gram = design.T @ design
    ridge = 1e-8 * max(float(np.trace(gram) / design.shape[1]), 1.)
    penalty = np.eye(design.shape[1]) * ridge; penalty[0, 0] = 0.
    estimate = gd @ np.linalg.solve(gram + penalty, design.T @ y)
    rng = np.random.default_rng(seed)
    predictions = np.empty((resamples, len(grid), 3), dtype=np.float64)
    means = np.empty((resamples, 3), dtype=np.float64)
    for start in range(0, resamples, 256):
        stop = min(start + 256, resamples)
        counts = rng.multinomial(len(x), np.full(len(x), 1 / len(x)), size=stop-start).astype(np.float64)
        bg = np.einsum('bn,np,nq->bpq', counts, design, design, optimize=True)
        br = np.einsum('bn,np,nk->bpk', counts, design, y, optimize=True)
        beta = np.linalg.solve(bg + penalty[None, :, :], br)
        predictions[start:stop] = np.einsum('gp,bpk->bgk', gd, beta)
        means[start:stop] = counts @ y / len(x)
    lower, upper = np.quantile(predictions, [.025, .975], axis=0)
    return {'x': grid, 'estimate': estimate, 'lower': lower, 'upper': upper,
            'bootstrap_means': means, 'ridge': ridge, 'resamples': resamples, 'seed': seed}


def export_figures(output_root, figure_root=None, resamples=10000):
    inspection = attack.inspect_results(output_root)
    attack.require(inspection['audit_role'] == 'production T4', 'Final figures require production T4 receipts')
    study = Path(inspection['study_directory'])
    destination = Path(figure_root) if figure_root else study.parent / 'figures'
    destination = destination.resolve()
    attack.require(destination != study and not destination.is_relative_to(study)
                   and not study.is_relative_to(destination), 'Figure output must be separate from study receipts')
    destination.mkdir(parents=True, exist_ok=True)
    summary = attack.read(study / 'summary.json')
    identity = attack.read(study / 'input_identity.json')
    x = np.array([row['scale'] for row in summary['draws']])
    y = np.array([[row[name] for name in attack.ATTACKS] for row in summary['draws']])
    fit = paired_spline(x, y, resamples=resamples)
    stem = f'{inspection["dataset"]}_p010_mp_am_op_uniform500'
    observations = destination / f'{stem}_observations.csv'
    with observations.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream); writer.writerow(['draw', 'anchor_noise_scale', *attack.ATTACKS])
        writer.writerows((row['draw'], row['scale'], *(row[name] for name in attack.ATTACKS)) for row in summary['draws'])
    coordinates = destination / f'{stem}_spline.csv'
    with coordinates.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream); writer.writerow(['anchor_noise_scale', 'attack', 'estimate', 'lower95', 'upper95'])
        for i, scale in enumerate(fit['x']):
            for j, name in enumerate(attack.ATTACKS):
                writer.writerow([scale, name, fit['estimate'][i,j], fit['lower'][i,j], fit['upper'][i,j]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 15,
        'axes.labelsize': 18, 'legend.fontsize': 14, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    colors = {'MP': '#0072B2', 'AM': '#009E73', 'OP': '#E69F00'}
    fig, ax = plt.subplots(figsize=(8.1, 5.4))
    for j, name in enumerate(attack.ATTACKS):
        ax.scatter(x, y[:,j], s=9, color=colors[name], alpha=.18, linewidths=0)
        ax.fill_between(fit['x'], fit['lower'][:,j].clip(0,100), fit['upper'][:,j].clip(0,100),
                        color=colors[name], alpha=.17, linewidth=0)
        ax.plot(fit['x'], fit['estimate'][:,j].clip(0,100), color=colors[name], linewidth=2.2, label=name)
    ax.axhline(10., color='#8C9197', linestyle='--', linewidth=1.2)
    handles = [Line2D([], [], color=colors[name], linewidth=2.2, label=name) for name in attack.ATTACKS]
    handles.append(Line2D([], [], color='#8C9197', linestyle='--', linewidth=1.2, label='Random choice (10%)'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,1.2), ncol=4, frameon=False)
    ax.set(xlim=(0,.08), ylim=(0,100), xlabel=r'Anchor-noise standard deviation $v$',
           ylabel='Identity Linkage Accuracy (%)')
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', color='#E5E8EB', linewidth=.7)
    ax.set_axisbelow(True); fig.tight_layout()
    paths = {}
    for extension in ('png', 'pdf', 'svg'):
        path = destination / f'{stem}.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight'); paths[extension] = str(path)
    plt.close(fig)
    metadata = {
        'dataset': inspection['dataset'], 'verified_draws': 500, 'identity_hash': inspection['identity_hash'],
        'schedule_hash': inspection['schedule_hash'], 'audit_role': inspection['audit_role'],
        'source_sha256': identity['source_sha256'], 'auditor': identity['model'],
        'zero_control_excluded': True, 'linkage_or_utility_selection': False,
        'bootstrap_resamples': resamples, 'bootstrap_seed': fit['seed'], 'ridge': fit['ridge'],
        'uncertainty': 'pointwise percentile intervals from paired noise-draw resampling, conditional on fixed deployment',
        'spline': 'cubic truncated-power regression, five interior knots; descriptive only',
        'display': 'points are observed rates; fitted curves/bands clipped to [0,100] for display; raw CSV coordinates retained',
        'statistics': inspection['statistics'], 'highest_draw_counts': inspection['highest_draw_counts'],
        'mean_draw_bootstrap_95ci_percent': {name: np.quantile(fit['bootstrap_means'][:,j],[.025,.975]).tolist()
                                          for j,name in enumerate(attack.ATTACKS)},
        'paired_mean_difference_95ci_pp': {f'{a}_minus_{b}': np.quantile(
            fit['bootstrap_means'][:,attack.ATTACKS.index(a)] - fit['bootstrap_means'][:,attack.ATTACKS.index(b)], [.025,.975]).tolist()
            for a,b in [('OP','MP'),('OP','AM'),('AM','MP')]},
        'record_byte_sha256': inspection['record_byte_sha256'], 'generated_utc': attack.utc(),
        'files': {**{extension: {'path': path, 'sha256': attack.sha(path)} for extension,path in paths.items()},
                  'observations': {'path': str(observations), 'sha256': attack.sha(observations)},
                  'spline': {'path': str(coordinates), 'sha256': attack.sha(coordinates)}},
    }
    provenance = destination / f'{stem}_provenance.json'; attack.atomic_json(provenance, metadata)
    return {**paths, 'observations': str(observations), 'spline': str(coordinates), 'provenance': str(provenance)}
