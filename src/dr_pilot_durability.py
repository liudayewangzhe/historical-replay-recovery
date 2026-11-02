"""Pilot: backdoor effectiveness during an attack window and its persistence after filtering.

No recovery is run here. The pilot only calibrates attack parameters on one
checkpoint; recovery methods are evaluated later on separate checkpoints.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch

import run_covariance as rc
from dr_lib import attack_round, neurotoxin_mask, setup


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='data')
    p.add_argument('--out', required=True)
    p.add_argument('--attack-rounds', type=int, default=32)
    p.add_argument('--filter-rounds', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=4)
    p.add_argument('--poison-fraction', type=float, default=.5)
    p.add_argument('--scale', type=float, default=1.)
    p.add_argument('--neurotoxin', type=float, default=0.)
    p.add_argument('--patch', type=int, default=4)
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    engine, args, state, ck = setup(a.checkpoint, a.data, device, a.attack_rounds + a.filter_rounds,
                                    a.poison_fraction, patch=a.patch, threads=a.threads)
    (out / 'config.json').write_text(json.dumps(dict(vars(a), onset=args.onset, seed=args.seed,
                                                     dataset=args.dataset), indent=2))
    rows = []
    begin = time.monotonic()

    def log(phase, offset):
        row = dict(phase=phase, offset=offset, round=args.onset + offset,
                   elapsed=time.monotonic() - begin, **engine.evaluate(state))
        rows.append(row)
        with (out / 'durability.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(json.dumps(row), flush=True)

    log('onset', 0)
    previous = None
    for j in range(a.attack_rounds):
        mask = neurotoxin_mask(previous, a.neurotoxin)
        aggregate, _ = attack_round(engine, state, args.onset + j, args, a.scale, mask)
        previous = aggregate.clone()
        state -= aggregate
        if (j + 1) % a.eval_every == 0:
            log('attack', j + 1)
    honest = list(range(args.malicious, args.clients))
    for c in range(a.filter_rounds):
        t = args.onset + a.attack_rounds + c
        state -= rc.aggregate_stream(engine, state, t, honest)
        if (c + 1) % a.eval_every == 0:
            log('filter', a.attack_rounds + c + 1)


if __name__ == '__main__':
    main()
