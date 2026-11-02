"""Paired comparison of historical replay against direct sampling at equal client budget.

For each dataset and delay, computes the per-seed difference P4 - S4 (and V4 - S4) in
clean accuracy at recovery and after the final continuation index, with the exact
two-sided sign-flip test. Reads the primary analysis (calibration seeds excluded).
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {931, 1771}


def sign_flip_p(d):
    d = np.asarray([x for x in d if np.isfinite(x) and x != 0], float)
    if d.size == 0:
        return float('nan')
    obs = abs(d.sum())
    hits = sum(abs(np.dot(s, np.abs(d))) >= obs - 1e-12 for s in itertools.product([-1, 1], repeat=d.size))
    return hits / 2 ** d.size


def run(name, rec_path, con_path):
    rec = pd.read_csv(rec_path)
    con = pd.read_csv(con_path)
    rec = rec[~rec.seed.isin(EXCLUDE)]
    con = con[~con.seed.isin(EXCLUDE)]
    last = con.groupby(['delay', 'method'])['continuation'].transform('max')
    fin = con[con.continuation == last]
    rows = []
    for stage, frame in [('recovery', rec), ('continued', fin)]:
        for delay, g in frame.groupby('delay'):
            piv = g.pivot_table(index='seed', columns='method', values='accuracy', aggfunc='first')
            for a, b in [('P4', 'S4'), ('V4', 'S4'), ('V4', 'P4')]:
                if a in piv and b in piv:
                    d = (piv[a] - piv[b]).dropna()
                    rows.append(dict(dataset=name, stage=stage, delay=int(delay), contrast=f'{a}-{b}',
                                     n=len(d), mean_pp=100 * d.mean(), wins=int((d > 0).sum()),
                                     losses=int((d < 0).sum()), p_two_sided=sign_flip_p(d.values)))
    return rows


def main():
    rows = []
    rows += run('CIFAR-10', ROOT / 'analysis/out/dr_main_cifar_a/recovery.csv' if False else
                ROOT / 'results_sync/results/dr_main_cifar_a', None) if False else []
    for name, run_dir in [('CIFAR-10', 'dr_main_cifar_a'), ('FMNIST', 'dr_main_fmnist_s3')]:
        base = ROOT / 'results_sync' / 'results' / run_dir
        rec = pd.concat([pd.read_csv(p) for p in sorted(base.glob('s*/recovery.csv'))], ignore_index=True)
        con = pd.concat([pd.read_csv(p) for p in sorted(base.glob('s*/continuation.csv'))], ignore_index=True)
        tmp_r = ROOT / 'analysis' / 'out' / run_dir / '_rec_all.csv'
        tmp_c = ROOT / 'analysis' / 'out' / run_dir / '_con_all.csv'
        tmp_r.parent.mkdir(parents=True, exist_ok=True)
        rec.to_csv(tmp_r, index=False)
        con.to_csv(tmp_c, index=False)
        rows += run(name, tmp_r, tmp_c)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / 'analysis' / 'out' / 'history_vs_direct.csv', index=False)
    print(out.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
