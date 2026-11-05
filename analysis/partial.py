"""Partial-participation cohort: summary table of the recurring contrasts per rate and delay.

Reads results_sync/results/partial_cifar10_r{02,05}/s*/{recovery,continuation}.csv and writes
manuscript/sections/table_partial.tex and analysis/out/partial_cells.csv.
"""
import glob
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = [('partial_cifar10_r02', 0.2), ('partial_cifar10_r05', 0.5)]
CONTRASTS = [('rollback', 'exact', 'cont'), ('S4', 'exact', 'cont'), ('P4', 'exact', 'cont'),
             ('S2', 'exact', 'cont'), ('P2', 'exact', 'cont'), ('P4', 'S4', 'rec'), ('P4', 'S4', 'cont'),
             ('P2', 'S2', 'rec'), ('P2', 'S2', 'cont')]


def load(run, name):
    paths = sorted(glob.glob(str(ROOT / 'results_sync/results' / run / 's*' / name)))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()


def sign_flip(d):
    d = np.asarray([x for x in d if np.isfinite(x) and x != 0])
    if d.size == 0:
        return float('nan')
    return sum(abs(np.dot(s, np.abs(d))) >= abs(d.sum()) - 1e-12
               for s in itertools.product([-1, 1], repeat=d.size)) / 2 ** d.size


def main():
    rows = []
    for run, rate in RUNS:
        rec, con = load(run, 'recovery.csv'), load(run, 'continuation.csv')
        if rec.empty or con.empty:
            continue
        for delay in sorted(rec['delay'].unique()):
            r = rec[rec['delay'] == delay].pivot_table(index='seed', columns='method', values='accuracy')
            c = con[(con['delay'] == delay) & (con['continuation'] == 16)]
            ca = c.pivot_table(index='seed', columns='method', values='accuracy')
            casr = c.pivot_table(index='seed', columns='method', values='asr')
            seeds = sorted(set(r.index) & set(ca.index))
            row = dict(rate=rate, delay=delay, n=len(seeds),
                       filter_asr=100 * casr.loc[seeds, 'filter'].mean(), exact_asr=100 * casr.loc[seeds, 'exact'].mean(),
                       exact_acc=100 * ca.loc[seeds, 'exact'].mean(),
                       exact_calls=rec[(rec['delay'] == delay) & (rec['method'] == 'exact')]['calls'].mean(),
                       s4_calls=rec[(rec['delay'] == delay) & (rec['method'] == 'S4')]['calls'].mean())
            for a, b, stage in CONTRASTS:
                t = (r if stage == 'rec' else ca).loc[seeds]
                if a not in t or b not in t:
                    continue
                d = 100 * (t[a] - t[b])
                key = f'{a}-{b}_{stage}'
                row[key] = d.mean()
                row[key + '_wins'] = int((d > 0).sum())
                row[key + '_p'] = sign_flip(d.values)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / 'analysis' / 'out' / 'partial_cells.csv', index=False)
    if out.empty:
        return
    f = lambda r, k: '--' if k not in r or not np.isfinite(r[k]) else \
        f"{0.0 if abs(r[k]) < .005 else r[k]:+.2f}".replace('-', '$-$') + f" ({int(r[k + '_wins'])})"
    lines = ['\\begin{table}[!htbp]', '\\centering',
             '\\caption{Partial participation on CIFAR-10 (a fraction $q$ of the 50 clients is active in each round; '
             'nine held-out seeds per cell). Filter ASR after 16 honest rounds, client calls of exact replay during '
             'recovery, and paired differences in pp (seeds with a positive difference in parentheses), after 16 honest '
             'rounds unless marked as recovery step.}',
             '\\label{tab:partial}', '\\scriptsize\\setlength{\\tabcolsep}{3.5pt}',
             '\\begin{tabular}{rrrrrrrrr}', '\\toprule',
             ' & & Filter & Exact & Rollback & S2 & \\multicolumn{2}{c}{P4 $-$ S4} & P2 $-$ S2\\\\',
             '\\cmidrule(lr){7-8}',
             '$q$ & $\\Delta$ & ASR (\\%) & calls & $-$ exact & $-$ exact & recovery & after 16 & after 16\\\\', '\\midrule']
    for _, r in out.iterrows():
        lines.append(' & '.join([f"{r['rate']:.1f}", f"{int(r['delay'])}", f"{r['filter_asr']:.1f}",
                                 f"{r['exact_calls']:.0f}", f(r, 'rollback-exact_cont'), f(r, 'S2-exact_cont'),
                                 f(r, 'P4-S4_rec'), f(r, 'P4-S4_cont'), f(r, 'P2-S2_cont')]) + '\\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (ROOT / 'manuscript' / 'sections' / 'table_partial.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    pd.set_option('display.width', 220)
    print(out.round(2).T.to_string())


if __name__ == '__main__':
    main()
