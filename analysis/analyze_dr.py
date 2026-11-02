"""Aggregate durable-backdoor recovery runs: recovery tables, continuation tables and figures.

Reads the mirrored CSV files under results_sync/results/<run>/s<seed>/ and writes
summary tables (CSV and LaTeX fragments) and figures under analysis/out/<run>/.
Paired comparisons use the same training seed for every method; the exact
two-sided sign-flip test enumerates all sign assignments of paired differences.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(run_dir):
    recovery, continuation = [], []
    for seed_dir in sorted(run_dir.glob('s*')):
        if not seed_dir.is_dir():
            continue
        for name, bucket in [('recovery.csv', recovery), ('continuation.csv', continuation)]:
            path = seed_dir / name
            if path.exists() and path.stat().st_size > 0:
                bucket.append(pd.read_csv(path))
    rec = pd.concat(recovery, ignore_index=True) if recovery else pd.DataFrame()
    con = pd.concat(continuation, ignore_index=True) if continuation else pd.DataFrame()
    return rec, con


def sign_flip_p(differences):
    """Exact two-sided sign-flip p-value for paired differences."""
    d = np.asarray([x for x in differences if np.isfinite(x) and x != 0], dtype=float)
    if d.size == 0:
        return float('nan')
    observed = abs(d.sum())
    count = 0
    total = 0
    for signs in itertools.product([-1, 1], repeat=d.size):
        total += 1
        if abs(np.dot(signs, np.abs(d))) >= observed - 1e-12:
            count += 1
    return count / total


def paired_summary(frame, value, reference, method_order):
    """Mean paired difference (method minus reference) with W/T/L and exact p."""
    rows = []
    pivot = frame.pivot_table(index='seed', columns='method', values=value, aggfunc='first')
    if reference not in pivot.columns:
        return pd.DataFrame()
    for method in method_order:
        if method == reference or method not in pivot.columns:
            continue
        pair = pivot[[method, reference]].dropna()
        if pair.empty:
            continue
        diff = pair[method] - pair[reference]
        rows.append(dict(method=method, reference=reference, n=len(diff),
                         mean_diff=diff.mean(), sd_diff=diff.std(ddof=1) if len(diff) > 1 else np.nan,
                         wins=int((diff > 0).sum()), ties=int((diff == 0).sum()), losses=int((diff < 0).sum()),
                         p_two_sided=sign_flip_p(diff.values)))
    return pd.DataFrame(rows)


def fmt(mean, sd, digits=3):
    if pd.isna(mean):
        return 'n/a'
    if pd.isna(sd):
        return f'{mean:.{digits}f}'
    return f'{mean:.{digits}f} ± {sd:.{digits}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', help='run directory name under results_sync/results, e.g. dr_main_cifar_a')
    parser.add_argument('--reference', default='filter')
    parser.add_argument('--exclude', type=int, nargs='*', default=[931, 1771],
                        help='calibration seeds used for attack-strength pilots; excluded from primary analysis')
    parser.add_argument('--tag', default='', help='suffix for the output folder, e.g. _all_seeds')
    args = parser.parse_args()
    run_dir = ROOT / 'results_sync' / 'results' / args.run
    out = ROOT / 'analysis' / 'out' / (args.run + args.tag)
    out.mkdir(parents=True, exist_ok=True)
    rec, con = load(run_dir)
    if args.exclude:
        rec = rec[~rec['seed'].isin(args.exclude)].reset_index(drop=True)
        con = con[~con['seed'].isin(args.exclude)].reset_index(drop=True)
    if rec.empty:
        print('no recovery rows yet')
        return
    method_order = ['filter', 'rollback', 'S2', 'S4', 'S8', 'P2', 'P4', 'P8', 'V4', 'Fit4', 'FedRecover',
                    'FedEraser', 'Crab', 'CrabStored', 'FT1', 'FT5', 'oracle', 'exact']
    report = {}

    # Recovery-time table: one row per (delay, method).
    summary = (rec.groupby(['delay', 'method'])
                  .agg(n=('seed', 'nunique'), acc_mean=('accuracy', 'mean'), acc_sd=('accuracy', 'std'),
                       asr_mean=('asr', 'mean'), asr_sd=('asr', 'std'),
                       dis_mean=('disagreement', 'mean'), dis_sd=('disagreement', 'std'),
                       calls=('calls', 'mean'), serial_rounds=('serial_rounds', 'mean'),
                       seconds=('seconds', 'mean'))
                  .reset_index())
    summary.to_csv(out / 'recovery_summary.csv', index=False)
    rows = []
    for delay, group in summary.groupby('delay'):
        for _, r in group.set_index('method').loc[[m for m in method_order if m in set(group['method'])]].reset_index().iterrows():
            rows.append(dict(delay=delay, method=r['method'], n=r['n'],
                             accuracy=fmt(r['acc_mean'], r['acc_sd'], 2), asr=fmt(r['asr_mean'], r['asr_sd'], 2),
                             disagreement=fmt(r['dis_mean'], r['dis_sd'], 3), calls=round(r['calls'], 1),
                             serial_rounds=round(r['serial_rounds'], 1)))
    pd.DataFrame(rows).to_csv(out / 'recovery_table.csv', index=False)

    # Paired comparisons against the reference at recovery time.
    paired = []
    for delay, group in rec.groupby('delay'):
        for value in ['accuracy', 'asr', 'disagreement']:
            ps = paired_summary(group, value, args.reference, method_order)
            if not ps.empty:
                ps.insert(0, 'metric', value)
                ps.insert(0, 'delay', delay)
                paired.append(ps)
    if paired:
        pd.concat(paired).to_csv(out / f'recovery_paired_vs_{args.reference}.csv', index=False)

    # Continuation-time table at the last continuation index.
    if not con.empty:
        # The common continuation length; latency-matched runs continue zero-round methods further.
        common = con.groupby(['delay', 'method'])['continuation'].transform('max').groupby(con['delay']).transform('min')
        final = con[con['continuation'] == common].copy()
        cf = (final.groupby(['delay', 'method'])
                   .agg(n=('seed', 'nunique'), acc_mean=('accuracy', 'mean'), acc_sd=('accuracy', 'std'),
                        asr_mean=('asr', 'mean'), asr_sd=('asr', 'std'),
                        comm=('communication_rounds', 'mean'), calls=('total_client_calls', 'mean'))
                   .reset_index())
        cf.to_csv(out / 'continuation_final.csv', index=False)
        paired_c = []
        for delay, group in final.groupby('delay'):
            for value in ['accuracy', 'asr']:
                ps = paired_summary(group, value, args.reference, method_order)
                if not ps.empty:
                    ps.insert(0, 'metric', value)
                    ps.insert(0, 'delay', delay)
                    paired_c.append(ps)
        if paired_c:
            pd.concat(paired_c).to_csv(out / f'continuation_paired_vs_{args.reference}.csv', index=False)

        # Figure: ASR and accuracy trajectories after detection.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, len(con['delay'].unique()), figsize=(4.2 * len(con['delay'].unique()), 6.4),
                                 squeeze=False)
        for col, delay in enumerate(sorted(con['delay'].unique())):
            sub = con[con['delay'] == delay]
            for metric_row, metric in enumerate(['asr', 'accuracy']):
                ax = axes[metric_row][col]
                for method in method_order:
                    if method not in set(sub['method']):
                        continue
                    traj = sub[sub['method'] == method].groupby('continuation')[metric]
                    m, s = traj.mean(), traj.std()
                    ax.plot(m.index, m.values, marker='o', ms=3, label=method)
                    ax.fill_between(m.index, (m - s).values, (m + s).values, alpha=.08)
                ax.set_title(f'delay {delay}')
                ax.set_xlabel('honest rounds after detection')
                ax.set_ylabel('ASR' if metric == 'asr' else 'clean accuracy')
                ax.grid(alpha=.3)
        axes[0][0].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out / 'continuation_trajectories.png', dpi=160)
        plt.close(fig)

    report['seeds'] = sorted(int(s) for s in rec['seed'].unique())
    report['delays'] = sorted(int(d) for d in rec['delay'].unique())
    report['methods'] = sorted(rec['method'].unique().tolist())
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    print(pd.read_csv(out / 'recovery_table.csv').to_string(index=False))


if __name__ == '__main__':
    main()
