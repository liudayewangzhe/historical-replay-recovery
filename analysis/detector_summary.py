"""Summarise real-detector runs: detection quality and paired contrasts against the true exclusion target."""
import glob
import itertools
import sys

import numpy as np
import pandas as pd


def sign_flip(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    if not len(d):
        return float('nan')
    obs = abs(d.sum())
    return sum(abs(np.dot(s, np.abs(d))) >= obs - 1e-12 for s in itertools.product([-1, 1], repeat=len(d))) / 2 ** len(d)


def main(run):
    base = f'results_sync/results/{run}'
    det = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(base + '/s*/detection.csv'))])
    det = det[det.stage == 'detection']
    print('seeds', len(det), '| alarms', det['alarm_offset'].notna().sum(),
          '| delay', det['delay'].dropna().astype(int).tolist(),
          '| detected', det['true_positive'].dropna().astype(int).tolist(),
          '| false pos', det['false_positive'].dropna().astype(int).tolist())
    recs = sorted(glob.glob(base + '/s*/recovery.csv'))
    if not recs:
        print('no recovery files (no alarm)')
        return
    rec = pd.concat([pd.read_csv(p) for p in recs])
    con = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(base + '/s*/continuation.csv'))])
    fin = con[con.continuation == con.continuation.max()]
    for stage, frame in [('recovery', rec), ('continued', fin)]:
        acc = frame.pivot_table(index='seed', columns='method', values='accuracy')
        asr = frame.pivot_table(index='seed', columns='method', values='asr')
        print(f'--- {stage} (n={len(acc)}): mean ASR', {m: round(100 * asr[m].mean(), 1) for m in asr.columns})
        for m in [c for c in acc.columns if c != 'exact_target']:
            d = acc[m] - acc['exact_target']
            print(f'  {m:15s} acc-target {100 * d.mean():+.2f} pp  W/L {int((d > 0).sum())}/{int((d < 0).sum())}  p={sign_flip(d):.3f}')
        if {'P4', 'S4'} <= set(acc.columns):
            d = acc['P4'] - acc['S4']
            print(f'  P4-S4 {100 * d.mean():+.2f} pp  W/L {int((d > 0).sum())}/{int((d < 0).sum())}  p={sign_flip(d):.3f}')
    if 'late_true' in fin.columns:
        print('late detections in continuation (true/all):',
              fin.groupby('method')[['late_true', 'late_flagged']].mean().round(2).to_dict('index'))


if __name__ == '__main__':
    main(sys.argv[1])
