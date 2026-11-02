"""Publication figures generated directly from mirrored result CSVs (held-out seeds only).

fig_trajectories.pdf: ASR, clean accuracy and disagreement with exact replay against honest rounds after detection.
fig_budget.pdf: recovery-time accuracy against extra client calls, sampled versus paired replay.
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manuscript' / 'figures'
EXCLUDE = {931, 1771}
COLOR = {'filter': '#D55E00', 'rollback': '#E69F00', 'S4': '#56B4E9', 'P4': '#0072B2',
         'exact': '#000000', 'S': '#56B4E9', 'P': '#0072B2', 'FedRecover': '#009E73'}
LABEL = {'filter': 'Filter', 'rollback': 'Rollback', 'S4': 'Sampled ($s{=}4$)',
         'P4': 'Paired ($s{=}4$)', 'exact': 'Exact replay'}
CELLS = [('dr_main_cifar_a', 16, 'CIFAR-10, delay 16'), ('dr_main_fmnist_s3', 32, 'Fashion-MNIST, delay 32')]

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})


def load(run, name):
    frames = [pd.read_csv(p) for p in sorted(glob.glob(str(ROOT / 'results_sync/results' / run / 's*' / name)))]
    frame = pd.concat(frames, ignore_index=True)
    return frame[~frame['seed'].isin(EXCLUDE)]


def trajectories():
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 5.6), sharex='col')
    rows = [('asr', 'Attack success (%)'), ('accuracy', 'Clean accuracy (%)'),
            ('disagreement', 'Disagreement with\nexact replay (%)')]
    for col, (run, delay, title) in enumerate(CELLS):
        rec = load(run, 'recovery.csv')
        con = load(run, 'continuation.csv')
        rec = rec[rec['delay'] == delay].assign(continuation=0)
        con = con[con['delay'] == delay].rename(columns={'disagreement_same_continuation': 'disagreement'})
        cols = ['seed', 'method', 'continuation', 'accuracy', 'asr', 'disagreement']
        both = pd.concat([rec[cols], con[cols]])
        for row, (metric, ylabel) in enumerate(rows):
            ax = axes[row][col]
            for method in ['filter', 'rollback', 'S4', 'P4', 'exact']:
                if metric == 'disagreement' and method == 'exact':
                    continue
                g = both[both['method'] == method].groupby('continuation')[metric]
                m, s = 100 * g.mean(), 100 * g.std()
                ax.plot(m.index, m.values, color=COLOR[method], lw=1.4,
                        ls='--' if method == 'exact' else '-', label=LABEL[method])
                ax.fill_between(m.index, (m - s).values, (m + s).values, color=COLOR[method], alpha=.12, lw=0)
            ax.set_ylabel(ylabel)
            if metric == 'disagreement':
                ax.set_yscale('log')
                ticks = [t for t in (0.3, 0.5, 1, 2, 5, 10, 20) if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
                ax.set_yticks(ticks)
                ax.set_yticklabels([f'{t:g}' for t in ticks])
                ax.minorticks_off()
            if row == 0:
                ax.set_title(title)
            if row == len(rows) - 1:
                ax.set_xlabel('Honest rounds after detection')
            ax.grid(alpha=.25, lw=.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc='lower center')
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUT / 'fig_trajectories.pdf')
    plt.close(fig)


def budget():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, (run, delay, title) in zip(axes, CELLS):
        rec = load(run, 'recovery.csv')
        rec = rec[rec['delay'] == delay]
        g = rec.groupby('method').agg(acc=('accuracy', 'mean'), sd=('accuracy', 'std'), calls=('calls', 'mean'))
        for family in ['S', 'P']:
            names = [f'{family}{k}' for k in (2, 4, 8) if f'{family}{k}' in g.index]
            if names:
                sub = g.loc[names]
                ax.errorbar(sub['calls'], 100 * sub['acc'], yerr=100 * sub['sd'], color=COLOR[family], marker='o',
                            ms=3, lw=1.2, capsize=2, label='Sampled' if family == 'S' else 'Paired (history)')
        for name, marker in [('exact', 's'), ('FedRecover', '^')]:
            if name in g.index:
                ax.errorbar(g.loc[name, 'calls'], 100 * g.loc[name, 'acc'], yerr=100 * g.loc[name, 'sd'],
                            color=COLOR[name], marker=marker, ms=4, lw=0, elinewidth=1, capsize=2,
                            label='Exact replay' if name == 'exact' else 'FedRecover-style')
        for name, ls in [('filter', ':'), ('rollback', '-.')]:
            if name in g.index:
                ax.axhline(100 * g.loc[name, 'acc'], color=COLOR[name], ls=ls, lw=1, label=LABEL[name])
        ax.set_xscale('log')
        ax.set_xlabel('Extra client calls during recovery')
        ax.set_ylabel('Accuracy at recovery (%)')
        ax.set_title(title)
        ax.grid(alpha=.25, lw=.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=6, loc='lower center')
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(OUT / 'fig_budget.pdf')
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    trajectories()
    budget()
    print('figures written to', OUT)
