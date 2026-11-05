"""Generate LaTeX tables for the manuscript directly from analysis outputs.

Every number in the generated fragments is read from analysis/out/<run>/*.csv, so the
manuscript never contains hand-copied values. Output: manuscript/sections/tables_<run>.tex
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHOD_LABEL = {
    'filter': 'Filter', 'rollback': 'Rollback', 'S2': 'Sampled $s{=}2$', 'S4': 'Sampled $s{=}4$',
    'S8': 'Sampled $s{=}8$', 'P2': 'Paired $s{=}2$', 'P4': 'Paired $s{=}4$', 'P8': 'Paired $s{=}8$',
    'V4': 'Capped paired $s{=}4$', 'Fit4': 'Fitted $s{=}4$', 'FedRecover': 'FedRecover-style',
    'FedEraser': 'FedEraser-style', 'Crab': 'Crab-style', 'CrabStored': 'Crab-style (stored anchor)',
    'FT1': 'Server fine-tuning, 1 epoch', 'FT5': 'Server fine-tuning, 5 epochs',
    'exact': 'Exact replay',
}
ORDER = list(METHOD_LABEL)


def pct(x):
    return '--' if pd.isna(x) else f'{100 * x:.1f}'


def pm(mean, sd):
    if pd.isna(mean):
        return '--'
    if pd.isna(sd):
        return f'{100 * mean:.1f}'
    return f'{100 * mean:.1f}$\\pm${100 * sd:.1f}'


def recovery_table(run, delay, summary):
    sub = summary[summary['delay'] == delay].set_index('method')
    lines = [
        '\\begin{tabular}{lrrrrr}', '\\toprule',
        'Method & Clean acc.\\ (\\%) & ASR (\\%) & Disagree.\\ (\\%) & Client calls & Serial rounds\\\\', '\\midrule',
    ]
    for method in ORDER:
        if method not in sub.index:
            continue
        r = sub.loc[method]
        lines.append(f"{METHOD_LABEL[method]} & {pm(r['acc_mean'], r['acc_sd'])} & {pm(r['asr_mean'], r['asr_sd'])} & "
                     f"{pct(r['dis_mean'])} & {r['calls']:.0f} & {r['serial_rounds']:.0f}\\\\")
    lines += ['\\bottomrule', '\\end{tabular}}']
    lines[0] = '\\resizebox{\\columnwidth}{!}{%\n' + lines[0]
    return '\n'.join(lines)


def continuation_table(run, delay, final, paired_exact):
    sub = final[final['delay'] == delay].set_index('method')
    pe = paired_exact[(paired_exact['delay'] == delay) & (paired_exact['metric'] == 'accuracy')].set_index('method') \
        if not paired_exact.empty else pd.DataFrame()
    pa = paired_exact[(paired_exact['delay'] == delay) & (paired_exact['metric'] == 'asr')].set_index('method') \
        if not paired_exact.empty else pd.DataFrame()
    lines = [
        '\\begin{tabular}{lrrrr}', '\\toprule',
        'Method & Acc.\\ (\\%) & $\\Delta$Acc vs exact (pp) & ASR (\\%) & $p$ (acc)\\\\', '\\midrule',
    ]
    for method in ORDER:
        if method not in sub.index:
            continue
        r = sub.loc[method]
        if method == 'exact':
            d, p = '--', '--'
        else:
            d = f"{0.0 if abs(100 * pe.loc[method, 'mean_diff']) < .005 else 100 * pe.loc[method, 'mean_diff']:+.2f}".replace('-', '$-$') if method in pe.index else '--'
            p = f"{pe.loc[method, 'p_two_sided']:.3f}" if method in pe.index else '--'
        lines.append(f"{METHOD_LABEL[method]} & {pm(r['acc_mean'], r['acc_sd'])} & {d} & "
                     f"{pm(r['asr_mean'], r['asr_sd'])} & {p}\\\\")
    lines += ['\\bottomrule', '\\end{tabular}}']
    lines[0] = '\\resizebox{\\columnwidth}{!}{%\n' + lines[0]
    return '\n'.join(lines)


TITLE = {'dr_main_cifar_a': 'CIFAR-10 (main study)', 'dr_main_fmnist_s3': 'Fashion-MNIST (main study)',
         'full_cifar10': 'CIFAR-10 (cost-matched study)', 'full_fmnist': 'Fashion-MNIST (cost-matched study)',
         'partial_cifar10_r02': 'CIFAR-10, participation 0.2', 'partial_cifar10_r05': 'CIFAR-10, participation 0.5',
         'cohort_cifar10': 'CIFAR-10 (replication)', 'cohort_cifar100': 'CIFAR-100',
         'hetero_fmnist_a01': 'Fashion-MNIST, Dirichlet 0.1'}


def float_table(caption, label, body):
    return f'\\begin{{table}}[!htbp]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n{body}\n\\end{{table}}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args = parser.parse_args()
    out_dir = ROOT / 'analysis' / 'out' / args.run
    summary = pd.read_csv(out_dir / 'recovery_summary.csv')
    final = pd.read_csv(out_dir / 'continuation_final.csv')
    pe_path = out_dir / 'continuation_paired_vs_exact.csv'
    paired_exact = pd.read_csv(pe_path) if pe_path.exists() else pd.DataFrame()
    fragments = []
    name = TITLE.get(args.run, args.run)
    for delay in sorted(summary['delay'].unique()):
        fragments.append(float_table(f'{name}, delay {delay}: models at the recovery step (mean $\\pm$ standard deviation over seeds).',
                                     f'tab:{args.run}-rec-{delay}', recovery_table(args.run, delay, summary)))
        fragments.append(float_table(f'{name}, delay {delay}: after 16 honest continuation rounds; differences and exact sign-flip $p$-values are paired against exact replay.',
                                     f'tab:{args.run}-cont-{delay}', continuation_table(args.run, delay, final, paired_exact)))
    target = ROOT / 'manuscript' / 'sections' / f'tables_{args.run}.tex'
    target.write_text('\n\n'.join(fragments) + '\n', encoding='utf-8')
    print(f'wrote {target}')


if __name__ == '__main__':
    main()
