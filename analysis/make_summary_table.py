"""Main-text summary table across cohorts and delays, generated from analysis outputs.

Writes manuscript/sections/table_summary.tex. Differences are paired over seeds (pp);
the win count is the number of seeds in which the difference is positive.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
cells = pd.read_csv(ROOT / 'analysis' / 'out' / 'stats_cells.csv')
synth = pd.read_csv(ROOT / 'analysis' / 'out' / 'synthesis_cells.csv')
ORDER = ['dr_main_cifar_a', 'cohort_cifar10', 'cohort_cifar100', 'dr_main_fmnist_s3', 'hetero_fmnist_a01',
         'dr_neurotoxin2_fmnist']
LABEL = {'dr_main_cifar_a': 'CIFAR-10', 'cohort_cifar10': 'CIFAR-10 repl.', 'cohort_cifar100': 'CIFAR-100',
         'dr_main_fmnist_s3': 'F-MNIST', 'hetero_fmnist_a01': 'F-MNIST Dir(0.1)',
         'dr_neurotoxin2_fmnist': 'F-MNIST Neurotoxin'}


def get(cohort, delay, contrast, stage):
    r = cells[(cells.cohort == cohort) & (cells.delay == delay) & (cells.contrast == contrast) & (cells.stage == stage)]
    return r.iloc[0] if len(r) else None


def fmt(r, wins=True):
    if r is None:
        return '--'
    text = f"{r.mean_pp:+.2f}"
    return f"{text} ({int(r.wins)}/{int(r.n)})" if wins else text


def main():
    lines = [
        '\\begin{table*}[t]', '\\centering',
        '\\caption{Summary over cohorts and delays (training-matched, 16 honest rounds after recovery unless stated). '
        'Filter ASR is the attack success after filtering and 16 honest rounds; cells marked $^\\ast$ are durable '
        '(ASR $\\ge 30\\%$). Differences are paired means in percentage points with the number of seeds in which '
        'the difference is positive. $\\Delta$Acc$_{\\mathrm{win}}$ is the honest progress inside the window '
        '(exact replay minus the onset checkpoint at the recovery step).}',
        '\\label{tab:summary}', '\\resizebox{\\textwidth}{!}{%',
        '\\begin{tabular}{llrrrrrrr}', '\\toprule',
        ' & & Filter & $\\Delta$Acc$_{\\mathrm{win}}$ & Rollback & S4 & P4 & \\multicolumn{2}{c}{P4 $-$ S4}\\\\',
        '\\cmidrule(lr){8-9}',
        'Cohort & $\\Delta$ & ASR (\\%) & (pp) & $-$ exact & $-$ exact & $-$ exact & at recovery & after 16 rounds\\\\',
        '\\midrule',
    ]
    for cohort in ORDER:
        sub = synth[synth.run == cohort]
        for _, s in sub.iterrows():
            d = int(s.delay)
            star = '$^\\ast$' if s.filter_asr >= .3 else ''
            lines.append(' & '.join([
                LABEL[cohort], f'{d}{star}', f'{100 * s.filter_asr:.1f}', f'{100 * s.exact_progress:.2f}',
                fmt(get(cohort, d, 'rollback-exact', 'cont')), fmt(get(cohort, d, 'S4-exact', 'cont')),
                fmt(get(cohort, d, 'P4-exact', 'cont')), fmt(get(cohort, d, 'P4-S4', 'rec')),
                fmt(get(cohort, d, 'P4-S4', 'cont'))]) + '\\\\')
        if cohort != ORDER[-1]:
            lines.append('\\addlinespace[2pt]')
    lines += ['\\bottomrule', '\\end{tabular}}', '\\end{table*}']
    target = ROOT / 'manuscript' / 'sections' / 'table_summary.tex'
    target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('wrote', target)


if __name__ == '__main__':
    main()
