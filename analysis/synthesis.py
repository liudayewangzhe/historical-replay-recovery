"""Cross-setting synthesis of the durable-backdoor recovery runs.

For every (run, delay) cell and held-out seed it computes
  progress  = clean accuracy of the attacked model minus that of the onset checkpoint,
  exact_progress = exact replay minus the onset checkpoint at the recovery step (the honest
              progress contained in the window, which rollback forfeits),
  deficit   = rollback minus exact replay after the common honest continuation,
  gap_rec / gap_cont = paired minus direct sampling (s=4) at recovery / after continuation.
Outputs analysis/out/synthesis_cells.csv, analysis/out/synthesis_seeds.csv and
manuscript/figures/fig_synthesis.pdf.
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
EXCLUDE = {931, 1771}
CELLS = [
    ('dr_main_cifar_a', 'CIFAR-10', [8, 16]),
    ('cohort_cifar10', 'CIFAR-10 (replication)', [16, 32]),
    ('cohort_cifar100', 'CIFAR-100', [8, 16]),
    ('dr_main_fmnist_s3', 'Fashion-MNIST', [8, 16, 32, 64]),
    ('hetero_fmnist_a01', 'Fashion-MNIST, Dir(0.1)', [16, 32]),
    ('dr_neurotoxin2_fmnist', 'Fashion-MNIST, Neurotoxin-style', [16, 32]),
]
FAMILY = {'CIFAR-10': ('#0072B2', 'o'), 'CIFAR-10 (replication)': ('#56B4E9', 's'),
          'CIFAR-100': ('#CC79A7', 'D'), 'Fashion-MNIST': ('#D55E00', '^'),
          'Fashion-MNIST, Dir(0.1)': ('#E69F00', 'v'), 'Fashion-MNIST, Neurotoxin-style': ('#009E73', 'P')}
CONTINUATION = 16

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})


def load(run, name):
    frames = [pd.read_csv(p) for p in sorted(glob.glob(str(ROOT / 'results_sync/results' / run / 's*' / name)))]
    frame = pd.concat(frames, ignore_index=True)
    return frame[~frame['seed'].isin(EXCLUDE)]


def ci95(x):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return float('nan')
    return float(stats.t.ppf(.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))


def seed_rows():
    rows = []
    for run, label, delays in CELLS:
        rec = load(run, 'recovery.csv')
        con = load(run, 'continuation.csv')
        con = con[con['continuation'] == CONTINUATION]
        for delay in delays:
            r = rec[rec['delay'] == delay].pivot_table(index='seed', columns='method', values='accuracy')
            c = con[con['delay'] == delay].pivot_table(index='seed', columns='method', values='accuracy')
            a = con[con['delay'] == delay].pivot_table(index='seed', columns='method', values='asr')
            for seed in r.index:
                rows.append(dict(
                    run=run, setting=label, delay=delay, seed=seed,
                    progress=r.loc[seed, 'filter'] - r.loc[seed, 'rollback'],
                    exact_progress=r.loc[seed, 'exact'] - r.loc[seed, 'rollback'],
                    deficit=c.loc[seed, 'rollback'] - c.loc[seed, 'exact'],
                    gap_rec=r.loc[seed, 'P4'] - r.loc[seed, 'S4'],
                    gap_cont=c.loc[seed, 'P4'] - c.loc[seed, 'S4'],
                    s4_cont=c.loc[seed, 'S4'] - c.loc[seed, 'exact'],
                    p4_cont=c.loc[seed, 'P4'] - c.loc[seed, 'exact'],
                    filter_asr=a.loc[seed, 'filter'], exact_asr=a.loc[seed, 'exact']))
    return pd.DataFrame(rows)


def cell_table(seeds):
    out = []
    for (run, label, delay), g in seeds.groupby(['run', 'setting', 'delay'], sort=False):
        row = dict(run=run, setting=label, delay=delay, n=len(g))
        for col in ['progress', 'exact_progress', 'deficit', 'gap_rec', 'gap_cont', 's4_cont', 'p4_cont', 'filter_asr', 'exact_asr']:
            row[col] = g[col].mean()
            row[col + '_ci'] = ci95(g[col])
        row['deficit_neg'] = int((g['deficit'] < 0).sum())
        row['gap_rec_pos'] = int((g['gap_rec'] > 0).sum())
        out.append(row)
    return pd.DataFrame(out)


def figure(seeds, cells):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axes[0]
    # Rollback and exact replay never read the attacked history, so the Neurotoxin-style cohort
    # (same checkpoints as the main Fashion-MNIST cohort) duplicates those points; it is omitted here.
    seeds = seeds[seeds['run'] != 'dr_neurotoxin2_fmnist']
    for label, (color, marker) in FAMILY.items():
        if label.endswith('Neurotoxin-style'):
            continue
        g = seeds[seeds['setting'] == label]
        ax.scatter(100 * g['exact_progress'], 100 * g['deficit'], s=11, color=color, marker=marker,
                   alpha=.8, lw=0, label=label)
    x, y = 100 * seeds['exact_progress'].to_numpy(), 100 * seeds['deficit'].to_numpy()
    slope = float(np.dot(x, y) / np.dot(x, x))
    grid = np.linspace(0, x.max() * 1.05, 20)
    ax.plot(grid, -grid, color='0.55', lw=.7, ls=':')
    ax.plot(grid, slope * grid, color='k', lw=.8, ls='--')
    ax.axhline(0, color='0.6', lw=.5)
    ax.text(.62 * grid[-1], -.62 * grid[-1] - .35, 'no catch-up', fontsize=6, color='0.45', ha='left', va='top',
            rotation=-38, rotation_mode='anchor')
    ax.set_xlabel('Honest progress inside the window (pp)\nexact replay minus onset checkpoint')
    ax.set_ylabel('Rollback minus exact after\n16 honest rounds (pp)')
    ax.set_title('a  Rollback forfeits window progress', loc='left')
    handles, names = ax.get_legend_handles_labels()
    ax.grid(alpha=.25, lw=.5)

    ax = axes[1]
    labels = [f"{r.setting}, d={r.delay}" for r in cells.itertuples()]
    ypos = np.arange(len(cells))[::-1]
    ax.errorbar(100 * cells['gap_rec'], ypos + .15, xerr=100 * cells['gap_rec_ci'], fmt='o', ms=3.5,
                color='#0072B2', elinewidth=.9, capsize=1.5, label='At the recovery step')
    ax.errorbar(100 * cells['gap_cont'], ypos - .15, xerr=100 * cells['gap_cont_ci'], fmt='o', ms=3.5,
                mfc='white', color='#D55E00', elinewidth=.9, capsize=1.5, label='After 16 honest rounds')
    ax.axvline(0, color='0.4', lw=.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=6.3)
    ax.set_xlabel('Paired minus direct sampling, $s{=}4$ (pp)\nmean and 95% CI over seeds')
    ax.set_title('b  The history advantage is transient', loc='left')
    ax.legend(frameon=False, loc='upper right')
    ax.grid(alpha=.25, lw=.5, axis='x')
    fig.legend(handles, names, frameon=False, loc='lower left', bbox_to_anchor=(.01, 0), ncol=3,
               handletextpad=.2, columnspacing=.8)
    fig.tight_layout(w_pad=1.0, rect=(0, .1, 1, 1))
    fig.savefig(ROOT / 'manuscript' / 'figures' / 'fig_synthesis.pdf')
    return slope


def main():
    seeds = seed_rows()
    cells = cell_table(seeds)
    out = ROOT / 'analysis' / 'out'
    seeds.to_csv(out / 'synthesis_seeds.csv', index=False)
    cells.to_csv(out / 'synthesis_cells.csv', index=False)
    slope = figure(seeds, cells)
    distinct = seeds[seeds['run'] != 'dr_neurotoxin2_fmnist']
    r = stats.pearsonr(distinct['exact_progress'], distinct['deficit'])
    r_obs = stats.pearsonr(seeds['progress'], seeds['exact_progress'])
    print(cells[['setting', 'delay', 'n', 'progress', 'exact_progress', 'deficit', 'deficit_neg', 'gap_rec', 'gap_cont',
                 'filter_asr']].round(4).to_string())
    print(f'slope through origin {slope:.3f}; pearson(progress, deficit) r={r.statistic:.3f} p={r.pvalue:.2g}; '
          f'pearson(visible progress, exact progress) r={r_obs.statistic:.3f}; n={len(distinct)}')


if __name__ == '__main__':
    main()
