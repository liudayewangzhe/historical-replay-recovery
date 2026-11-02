"""Interval estimates, equivalence tests and pooled analyses for the recurring contrasts.

For every (cohort, delay) cell and contrast it reports the mean paired difference (pp), the 95%
t-interval, the exact sign-flip p-value, the 90% interval used for two one-sided tests against an
equivalence margin of 0.5 pp, and win counts. Pooled analyses average each seed's differences over
the delays of its cohort and treat cohort-seed pairs as units.
Outputs analysis/out/stats_cells.csv and analysis/out/stats_pooled.csv.
"""
import glob
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {931, 1771}
MARGIN = 0.5
COHORTS = [
    ('dr_main_cifar_a', 'CIFAR-10', [8, 16]),
    ('cohort_cifar10', 'CIFAR-10 (replication)', [16, 32]),
    ('cohort_cifar100', 'CIFAR-100', [8, 16]),
    ('dr_main_fmnist_s3', 'Fashion-MNIST', [8, 16, 32, 64]),
    ('hetero_fmnist_a01', 'Fashion-MNIST, Dir(0.1)', [16, 32]),
    ('dr_neurotoxin2_fmnist', 'Fashion-MNIST, Neurotoxin-style', [16, 32]),
]
# (name, stage, method, reference); stage 'rec' is the recovery step, 'cont' after 16 honest rounds.
CONTRASTS = [
    ('P4-S4', 'rec', 'P4', 'S4'), ('P4-S4', 'cont', 'P4', 'S4'),
    ('V4-S4', 'rec', 'V4', 'S4'), ('V4-S4', 'cont', 'V4', 'S4'),
    ('P4-rollback', 'rec', 'P4', 'rollback'), ('S4-rollback', 'rec', 'S4', 'rollback'),
    ('rollback-exact', 'cont', 'rollback', 'exact'), ('S4-exact', 'cont', 'S4', 'exact'),
    ('P4-exact', 'cont', 'P4', 'exact'), ('FedRecover-exact', 'cont', 'FedRecover', 'exact'),
]
INDEPENDENT = ['dr_main_cifar_a', 'cohort_cifar10', 'cohort_cifar100', 'dr_main_fmnist_s3', 'hetero_fmnist_a01']


def load(run, name):
    frames = [pd.read_csv(p) for p in sorted(glob.glob(str(ROOT / 'results_sync/results' / run / 's*' / name)))]
    frame = pd.concat(frames, ignore_index=True)
    return frame[~frame['seed'].isin(EXCLUDE)]


def sign_flip(d):
    d = np.asarray([x for x in d if np.isfinite(x) and x != 0])
    if d.size == 0:
        return float('nan')
    if d.size > 20:  # Monte Carlo for large pooled samples
        rng = np.random.default_rng(0)
        signs = rng.choice([-1, 1], size=(200000, d.size))
        return float(np.mean(np.abs(signs @ np.abs(d)) >= abs(d.sum()) - 1e-12))
    total = sum(abs(np.dot(s, np.abs(d))) >= abs(d.sum()) - 1e-12 for s in itertools.product([-1, 1], repeat=d.size))
    return total / 2 ** d.size


def interval(d, level):
    d = np.asarray(d, dtype=float)
    half = stats.t.ppf(.5 + level / 2, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size)
    return d.mean() - half, d.mean() + half


def accuracy_tables(run, delays):
    rec = load(run, 'recovery.csv')
    con = load(run, 'continuation.csv')
    con = con[con['continuation'] == 16]
    out = {}
    for delay in delays:
        out[('rec', delay)] = rec[rec['delay'] == delay].pivot_table(index='seed', columns='method', values='accuracy')
        out[('cont', delay)] = con[con['delay'] == delay].pivot_table(index='seed', columns='method', values='accuracy')
    return out


def main():
    cells, seed_rows = [], []
    for run, label, delays in COHORTS:
        tables = accuracy_tables(run, delays)
        for name, stage, method, ref in CONTRASTS:
            for delay in delays:
                t = tables[(stage, delay)]
                if method not in t or ref not in t:
                    continue
                d = 100 * (t[method] - t[ref]).dropna()
                lo95, hi95 = interval(d, .95)
                lo90, hi90 = interval(d, .90)
                cells.append(dict(cohort=run, setting=label, delay=delay, contrast=name, stage=stage, n=d.size,
                                  mean_pp=d.mean(), ci95_lo=lo95, ci95_hi=hi95, ci90_lo=lo90, ci90_hi=hi90,
                                  equivalent=bool(-MARGIN < lo90 and hi90 < MARGIN), wins=int((d > 0).sum()),
                                  p=sign_flip(d)))
                for seed, value in d.items():
                    seed_rows.append(dict(cohort=run, delay=delay, contrast=name, stage=stage, seed=seed, diff_pp=value))
    cells = pd.DataFrame(cells)
    seeds = pd.DataFrame(seed_rows)
    # Holm within the family of post-continuation history contrasts of the two main cohorts.
    fam = cells[(cells['contrast'].isin(['P4-S4', 'V4-S4'])) & (cells['stage'] == 'cont') &
                (cells['cohort'].isin(['dr_main_cifar_a', 'dr_main_fmnist_s3']))].sort_values('p')
    m = len(fam)
    adjusted = np.maximum.accumulate([min(1, (m - i) * p) for i, p in enumerate(fam['p'])])
    cells.loc[fam.index, 'holm_main'] = adjusted
    out = ROOT / 'analysis' / 'out'
    cells.to_csv(out / 'stats_cells.csv', index=False)

    pooled = []
    for name, stage, _, _ in CONTRASTS:
        sub = seeds[(seeds['contrast'] == name) & (seeds['stage'] == stage) & (seeds['cohort'].isin(INDEPENDENT))]
        if sub.empty:
            continue
        units = sub.groupby(['cohort', 'seed'])['diff_pp'].mean()
        rng = np.random.default_rng(1)
        boot = [rng.choice(units.values, units.size).mean() for _ in range(20000)]
        pooled.append(dict(contrast=name, stage=stage, units=units.size, mean_pp=units.mean(),
                           boot95_lo=np.percentile(boot, 2.5), boot95_hi=np.percentile(boot, 97.5),
                           positive=int((units > 0).sum()), p=sign_flip(units.values)))
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(out / 'stats_pooled.csv', index=False)
    pd.set_option('display.width', 200)
    show = cells[cells['contrast'].isin(['P4-S4', 'P4-rollback', 'S4-rollback'])]
    print(show[['setting', 'delay', 'contrast', 'stage', 'n', 'mean_pp', 'ci95_lo', 'ci95_hi', 'ci90_lo', 'ci90_hi',
                'equivalent', 'wins', 'p']].round(3).to_string())
    print(cells.dropna(subset=['holm_main'])[['setting', 'delay', 'contrast', 'mean_pp', 'p', 'holm_main']].round(3).to_string())
    print(pooled.round(3).to_string())


if __name__ == '__main__':
    main()
