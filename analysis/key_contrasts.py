"""Key paired contrasts for one recovery run (held-out seeds): recovery step and end of continuation.

Usage: python key_contrasts.py <run> [--exclude seeds...]
Prints, per delay: filter ASR, rollback/S4/P4/FedRecover minus exact replay, P4 minus S4 (accuracy, pp),
with win/loss counts and exact two-sided sign-flip p-values, plus mean client calls.
"""
import argparse
import glob
import itertools

import numpy as np
import pandas as pd


def sign_flip(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    if not len(d):
        return float('nan')
    obs = abs(d.sum())
    return sum(abs(np.dot(s, np.abs(d))) >= obs - 1e-12 for s in itertools.product([-1, 1], repeat=len(d))) / 2 ** len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--exclude', type=int, nargs='*', default=[931, 1771])
    a = ap.parse_args()
    base = f'results_sync/results/{a.run}'
    rec = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(base + '/s*/recovery.csv'))])
    con = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(base + '/s*/continuation.csv'))])
    rec, con = rec[~rec.seed.isin(a.exclude)], con[~con.seed.isin(a.exclude)]
    last = con.groupby(['delay', 'method'])['continuation'].transform('max')
    fin = con[con.continuation == last]
    print(a.run, 'seeds', sorted(rec.seed.unique().tolist()))
    for delay in sorted(rec.delay.unique()):
        for stage, frame in [('recovery', rec), ('continued', fin)]:
            f = frame[frame.delay == delay]
            acc = f.pivot_table(index='seed', columns='method', values='accuracy')
            asr = f.pivot_table(index='seed', columns='method', values='asr')
            line = [f'd{delay} {stage:9s} n={len(acc)} ASR% filter {100 * asr["filter"].mean():.1f} exact {100 * asr["exact"].mean():.1f}',
                    f'acc% exact {100 * acc["exact"].mean():.2f}']
            for m, ref in [('rollback', 'exact'), ('S4', 'exact'), ('P4', 'exact'), ('FedRecover', 'exact'), ('P4', 'S4')]:
                if m in acc and ref in acc:
                    d = acc[m] - acc[ref]
                    line.append(f'{m}-{ref} {100 * d.mean():+.2f} ({int((d > 0).sum())}/{int((d < 0).sum())}, p={sign_flip(d):.3f})')
            print(' | '.join(line))
        calls = rec[rec.delay == delay].groupby('method')['calls'].mean().round(0).to_dict()
        print(f'd{delay} recovery calls', {k: int(v) for k, v in calls.items()})


if __name__ == '__main__':
    main()
