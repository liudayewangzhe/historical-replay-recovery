"""Main contrasts on the server validation split versus the test split (cost-matched study).

Server fine-tuning trains on the validation split and is excluded. Writes analysis/out/val_check.csv.
"""
import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTRASTS = [('rollback', 'exact', 'cont'), ('S4', 'exact', 'cont'), ('P4', 'exact', 'cont'),
             ('P4', 'S4', 'rec'), ('P4', 'S4', 'cont'), ('FedRecover', 'exact', 'cont')]


def load(run, name):
    paths = sorted(glob.glob(str(ROOT / 'results_sync/results' / run / 's*' / name)))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


rows = []
for run, delays in [('full_cifar10', [16]), ('full_fmnist', [16, 32])]:
    rec, con = load(run, 'recovery.csv'), load(run, 'continuation.csv')
    for delay in delays:
        for a, b, stage in CONTRASTS:
            f = rec[rec['delay'] == delay] if stage == 'rec' else con[(con['delay'] == delay) & (con['continuation'] == 16)]
            row = dict(run=run, delay=delay, contrast=f'{a}-{b}', stage=stage)
            for metric in ['accuracy', 'val_accuracy']:
                t = f.pivot_table(index='seed', columns='method', values=metric)
                d = 100 * (t[a] - t[b])
                row[metric] = d.mean()
                row[metric + '_wins'] = int((d > 0).sum())
                row['n'] = len(d)
            rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(ROOT / 'analysis' / 'out' / 'val_check.csv', index=False)
print(out.round(2).to_string())
