"""Cost-matched study: accuracy and attack success against communication rounds and client calls.

Reads results_sync/results/full_<dataset>/s*/{recovery,continuation}.csv (every method in one run per
seed; zero-round methods continue for delay-1 extra rounds). Writes
  manuscript/figures/fig_frontier.pdf
  manuscript/sections/table_baselines.tex
  analysis/out/cost_matched_cells.csv
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CELLS = [('full_cifar10', 16, 'CIFAR-10, delay 16', 44.7e6), ('full_fmnist', 32, 'Fashion-MNIST, delay 32', 95.8e3)]
TABLE_CELLS = [('full_cifar10', 16, 'CIFAR-10', 44.7e6), ('full_fmnist', 16, 'Fashion-MNIST', 95.8e3),
               ('full_fmnist', 32, 'Fashion-MNIST', 95.8e3)]
STYLE = {  # colour, marker, label
    'filter': ('#D55E00', 'x', 'Filter'), 'rollback': ('#E69F00', 'o', 'Rollback'),
    'FT5': ('#CC79A7', 'D', 'Server fine-tuning (5 ep.)'), 'FT1': ('#CC79A7', 'd', 'Server fine-tuning (1 ep.)'),
    'S4': ('#56B4E9', 'v', 'Sampled $s{=}4$'), 'P4': ('#0072B2', '^', 'Paired $s{=}4$'),
    'FedRecover': ('#009E73', 's', 'FedRecover-style'), 'FedEraser': ('#999999', 'P', 'FedEraser-style'),
    'Crab': ('#882255', '*', 'Crab-style'), 'CrabStored': ('#AA4499', 'X', 'Crab-style, stored anchor'),
    'exact': ('#000000', 'o', 'Exact replay'),
}
FIG_METHODS = ['filter', 'rollback', 'FT5', 'S4', 'P4', 'FedRecover', 'FedEraser', 'Crab', 'CrabStored', 'exact']
TABLE_METHODS = ['filter', 'rollback', 'FT1', 'FT5', 'S4', 'P4', 'FedRecover', 'FedEraser', 'Crab', 'CrabStored',
                 'exact']

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})


def load(run, name):
    paths = sorted(glob.glob(str(ROOT / 'results_sync/results' / run / 's*' / name)))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()


def trajectories(run, delay):
    """Per seed and method: recovery point plus continuation points with both clocks."""
    rec, con = load(run, 'recovery.csv'), load(run, 'continuation.csv')
    if rec.empty or con.empty or delay not in set(rec['delay']) or delay not in set(con['delay']):
        return pd.DataFrame()
    rec = rec[rec['delay'] == delay].assign(continuation=0, communication_rounds=lambda f: f['serial_rounds'],
                                           total_client_calls=lambda f: f['calls'])
    con = con[con['delay'] == delay]
    cols = ['seed', 'method', 'continuation', 'communication_rounds', 'total_client_calls', 'accuracy', 'asr']
    return pd.concat([rec[cols], con[cols]], ignore_index=True)


def storage(method, delay, retained, vector_bytes, rec):
    """Bytes of stored history the method reads (uncompressed float32 vectors)."""
    if method in ('filter', 'FT1', 'FT5'):
        return 0
    if method in ('rollback', 'exact', 'S4'):
        return vector_bytes
    if method in ('P4', 'FedRecover'):
        return (delay * retained + delay) * vector_bytes
    if method == 'FedEraser':
        return int(np.ceil(delay / 2)) * vector_bytes
    if method in ('Crab', 'CrabStored'):
        selected = max(1, int(delay * .6))
        return (selected * max(1, int(retained * .7)) + delay) * vector_bytes
    return np.nan


def human(nbytes):
    if not np.isfinite(nbytes) or nbytes == 0:
        return '0'
    for unit, scale in [('GB', 1e9), ('MB', 1e6), ('kB', 1e3)]:
        if nbytes >= scale:
            return f'{nbytes / scale:.1f}\\,{unit}'
    return f'{nbytes:.0f}\\,B'


ACC_METHODS = ['rollback', 'S4', 'P4', 'FedRecover', 'FedEraser', 'Crab', 'exact']
ASR_METHODS = ['filter', 'FT5', 'CrabStored', 'rollback', 'exact']


def figure():
    """Accuracy of backdoor-free methods against both clocks; attack success of every kind of method."""
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    seen = {}
    for row, (run, delay, title, _) in enumerate(CELLS):
        traj = trajectories(run, delay)
        if traj.empty:
            continue
        for col, (x, y, methods) in enumerate([('comm', 'acc', ACC_METHODS), ('calls', 'acc', ACC_METHODS),
                                               ('comm', 'asr', ASR_METHODS)]):
            ax = axes[row][col]
            for method in methods:
                g = traj[traj['method'] == method]
                if g.empty:
                    continue
                m = g.groupby('continuation').agg(comm=('communication_rounds', 'mean'),
                                                  calls=('total_client_calls', 'mean'),
                                                  acc=('accuracy', 'mean'), asr=('asr', 'mean')).reset_index()
                color, marker, label = STYLE[method]
                xs = m[x].to_numpy(dtype=float)
                if x == 'calls':
                    xs = np.maximum(xs, 1)
                line, = ax.plot(xs, 100 * m[y], color=color, lw=1.1, ls='--' if method == 'exact' else '-',
                                label=label)
                ax.plot(xs[:1], 100 * m[y].to_numpy()[:1], marker=marker, color=color, ms=4, ls='none')
                seen.setdefault(label, line)
            ax.grid(alpha=.25, lw=.5)
            ax.set_title('abcdef'[3 * row + col], loc='left', fontweight='bold')
        axes[row][0].set_ylabel(f'{title}\nclean accuracy (%)')
        axes[row][2].set_ylabel('attack success (%)')
        axes[row][1].set_xscale('log')
        acc = 100 * traj[traj['method'].isin(ACC_METHODS)].groupby(['method', 'continuation'])['accuracy'].mean()
        for ax in axes[row][:2]:
            ax.set_ylim(acc.min() - 1, acc.max() + .5)
    for col, xl in enumerate(['Communication rounds since detection', 'Client calls since detection',
                              'Rounds since detection']):
        axes[-1][col].set_xlabel(xl)
    order = [STYLE[m][2] for m in ['rollback', 'S4', 'P4', 'FedRecover', 'FedEraser', 'Crab', 'exact', 'filter',
                                   'FT5', 'CrabStored'] if STYLE[m][2] in seen]
    fig.legend([seen[k] for k in order], order, frameon=False, ncol=5, loc='lower center')
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(ROOT / 'manuscript' / 'figures' / 'fig_frontier.pdf')


def sign_flip(d):
    import itertools
    d = np.asarray([x for x in d if np.isfinite(x) and x != 0])
    if d.size == 0:
        return float('nan')
    total = sum(abs(np.dot(s, np.abs(d))) >= abs(d.sum()) - 1e-12 for s in itertools.product([-1, 1], repeat=d.size))
    return total / 2 ** d.size


def table():
    rows, lines = [], []
    for run, delay, label, vbytes in TABLE_CELLS:
        traj = trajectories(run, delay)
        if traj.empty:
            continue
        rec = traj[traj['continuation'] == 0].pivot_table(index='seed', columns='method', values=['accuracy', 'asr'])
        c16 = traj[traj['continuation'] == 16].pivot_table(index='seed', columns='method', values=['accuracy', 'asr'])
        # Latency-matched: zero-round methods at continuation delay-1+16, others at continuation 16.
        lat = traj[traj['continuation'] == delay - 1 + 16].pivot_table(index='seed', columns='method', values='accuracy')
        # Rollback-and-continue accuracy at the same communication round (linear interpolation per seed).
        rb = traj[traj['method'] == 'rollback'].sort_values('communication_rounds')
        def rollback_at(seed, rounds):
            g = rb[rb['seed'] == seed]
            if rounds > g['communication_rounds'].max():  # beyond the observed rollback trajectory
                return float('nan')
            return float(np.interp(rounds, g['communication_rounds'], g['accuracy']))
        end16 = traj[traj['continuation'] == 16]
        calls = traj[traj['continuation'] == 0].groupby('method')['total_client_calls'].mean()
        serial = traj[traj['continuation'] == 0].groupby('method')['communication_rounds'].mean()
        n = rec.shape[0]
        for method in TABLE_METHODS:
            if ('accuracy', method) not in rec.columns:
                continue
            d16 = 100 * (c16['accuracy'][method] - c16['accuracy']['exact']) if method != 'exact' else None
            dlat = None
            if method in lat.columns and serial[method] == 0:
                dlat = 100 * (lat[method] - c16['accuracy']['exact'])
            m16 = end16[end16['method'] == method]
            if method in ('filter', 'FT1', 'FT5'):  # zero-round methods compared at the exact-replay clock
                m16 = traj[(traj['method'] == method) & (traj['continuation'] == delay - 1 + 16)]
            drb = pd.Series({r.seed: 100 * (r.accuracy - rollback_at(r.seed, r.communication_rounds))
                             for r in m16.itertuples()}) if method != 'rollback' else None
            rows.append(dict(run=run, delay=delay, method=method, n=n, calls=calls[method], serial=serial[method],
                             drb=np.nan if drb is None or drb.isna().any() else drb.mean(),
                             drb_wins=np.nan if drb is None or drb.isna().any() else int((drb > 0).sum()),
                             drb_p=np.nan if drb is None or drb.isna().any() else sign_flip(drb.values),
                             storage=storage(method, delay, 40, vbytes, rec),
                             acc_rec=100 * rec['accuracy'][method].mean(), asr_rec=100 * rec['asr'][method].mean(),
                             acc16=100 * c16['accuracy'][method].mean(), asr16=100 * c16['asr'][method].mean(),
                             d16=np.nan if d16 is None else d16.mean(),
                             d16_wins=np.nan if d16 is None else int((d16 > 0).sum()),
                             d16_p=np.nan if d16 is None else sign_flip(d16.values),
                             dlat=np.nan if dlat is None else dlat.mean(),
                             dlat_wins=np.nan if dlat is None else int((dlat > 0).sum()),
                             dlat_p=np.nan if dlat is None else sign_flip(dlat.dropna().values)))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / 'analysis' / 'out' / 'cost_matched_cells.csv', index=False)
    lines = ['\\begin{table*}[t]', '\\centering',
             '\\caption{Cost-matched study (means over nine held-out seeds). Calls and serial rounds are spent during '
             'recovery; storage is the history the method reads. Accuracy (Acc.) and attack success (ASR) are given at the '
             'recovery step and after 16 honest rounds. $\\Delta$ is the paired difference from exact replay after 16 '
             'honest rounds (pp, seeds with a positive difference in parentheses); $\\Delta_{\\mathrm{rb}}$ is the paired '
             'difference from rollback-and-continue at the same number of communication rounds since detection (serial '
             'recovery rounds plus 16; methods without recovery rounds are taken at the exact-replay round count).}',
             '\\label{tab:baselines}', '\\resizebox{\\textwidth}{!}{%', '\\begin{tabular}{llrrrrrrrrr}', '\\toprule',
             ' & & & & & \\multicolumn{2}{c}{Recovery step} & \\multicolumn{2}{c}{After 16 rounds} & & \\\\',
             '\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}',
             'Cell & Method & Calls & Serial & Storage & Acc. & ASR & Acc. & ASR & $\\Delta$ & $\\Delta_{\\mathrm{rb}}$\\\\',
             '\\midrule']
    for (run, delay), g in out.groupby(['run', 'delay'], sort=False):
        label = dict((r, l) for r, _, l, _ in TABLE_CELLS)[run]
        first = True
        for r in g.itertuples():
            cell = f'{label}, $\\Delta={delay}$' if first else ''
            first = False
            d = '--' if not np.isfinite(r.d16) else f'{r.d16:+.2f} ({int(r.d16_wins)})'
            dl = '--' if not np.isfinite(r.drb) else f'{r.drb:+.2f} ({int(r.drb_wins)})'
            lines.append(' & '.join([cell, STYLE[r.method][2], f'{r.calls:.0f}', f'{r.serial:.1f}'.rstrip('0').rstrip('.'),
                                     human(r.storage), f'{r.acc_rec:.1f}', f'{r.asr_rec:.1f}', f'{r.acc16:.1f}',
                                     f'{r.asr16:.1f}', d, dl]) + '\\\\')
        lines.append('\\midrule')
    lines[-1] = '\\bottomrule'
    lines += ['\\end{tabular}}', '\\end{table*}']
    (ROOT / 'manuscript' / 'sections' / 'table_baselines.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    pd.set_option('display.width', 220)
    print(out.round(2).to_string())


if __name__ == '__main__':
    figure()
    table()
