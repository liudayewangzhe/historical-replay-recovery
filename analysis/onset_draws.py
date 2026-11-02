"""Fashion-MNIST robustness: late onset estimates and repeated replay-sample draws.

late_onset_fmnist/k{2,4,8}: recovery restarts k rounds after the true onset (delay 32); 'oracle' is
the true exclusion trajectory. draws_fmnist/r{1,2}: the same seeds with two further sample draws,
combined with draw 0 from the main Fashion-MNIST cohort. Writes analysis/out/onset_draws.txt.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results_sync' / 'results'


def load(pattern):
    paths = sorted(glob.glob(str(RES / pattern)))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()


def main():
    lines = []
    for k in (2, 4, 8):
        rec = load(f'late_onset_fmnist/k{k}/s*/recovery.csv')
        con = load(f'late_onset_fmnist/k{k}/s*/continuation.csv')
        if rec.empty or con.empty:
            continue
        c16 = con[con['continuation'] == 16]
        for stage, frame in (('recovery', rec), ('after 16', c16)):
            acc = frame.pivot_table(index='seed', columns='method', values='accuracy')
            asr = frame.pivot_table(index='seed', columns='method', values='asr')
            parts = [f'k={k} {stage} (n={len(acc)}):']
            for m in ['oracle', 'rollback', 'S4', 'P4', 'exact', 'filter']:
                if m in acc:
                    d = 100 * (acc[m] - acc['oracle']).mean() if 'oracle' in acc else float('nan')
                    parts.append(f'{m} acc {100 * acc[m].mean():.2f} (vs oracle {d:+.2f}) ASR {100 * asr[m].mean():.1f}')
            lines.append('; '.join(parts))
    draws = []
    main_rec = load('dr_main_fmnist_s3/s*/recovery.csv')
    main_con = load('dr_main_fmnist_s3/s*/continuation.csv')
    for r, (rec, con) in enumerate([(main_rec, main_con)] +
                                   [(load(f'draws_fmnist/r{i}/s*/recovery.csv'), load(f'draws_fmnist/r{i}/s*/continuation.csv'))
                                    for i in (1, 2)]):
        if rec.empty:
            continue
        rec = rec[rec['delay'].isin([16, 32]) & ~rec['seed'].isin([1771])]
        con = con[con['delay'].isin([16, 32]) & (con['continuation'] == 16) & ~con['seed'].isin([1771])]
        for stage, frame in (('rec', rec), ('cont', con)):
            p = frame.pivot_table(index=['seed', 'delay'], columns='method', values='accuracy')
            if {'P4', 'S4'} <= set(p.columns):
                for (seed, delay), v in (100 * (p['P4'] - p['S4'])).items():
                    draws.append(dict(draw=r, seed=seed, delay=delay, stage=stage, gap=v))
    if draws:
        d = pd.DataFrame(draws)
        for (delay, stage), g in d.groupby(['delay', 'stage']):
            per_seed = g.groupby('seed')['gap']
            within = per_seed.std(ddof=1).mean()
            between = per_seed.mean().std(ddof=1)
            lines.append(f'draws delay {delay} {stage}: draws {sorted(g.draw.unique())}, mean P4-S4 {g.gap.mean():+.2f} pp, '
                         f'positive {int((g.gap > 0).sum())}/{len(g)}, within-seed sd {within:.2f}, '
                         f'between-seed sd of seed means {between:.2f}')
    text = '\n'.join(lines)
    (ROOT / 'analysis' / 'out' / 'onset_draws.txt').write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
