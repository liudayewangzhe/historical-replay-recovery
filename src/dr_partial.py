"""Recovery from a durable backdoor under per-round partial client participation.

From the onset checkpoint on, every round activates a uniformly random subset of
`rate * clients` clients. The participation schedule is drawn once per seed and is
shared by the attacked window, exact exclusion replay, every sampled replay and the
honest continuation, so all branches see the same active clients in each round.
Aggregates are equal-weight means over the active clients (attacked run) or over the
active retained clients (exclusion reference and recovery). Replay estimators sample
from the active retained clients of each round, the only clients with stored uploads.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_paired import CLIENT_OFFSET
from dr_lib import setup


def write(path, rows):
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


def mean_delta(engine, state, t, clients):
    total = torch.zeros_like(state)
    for m in clients:
        total.add_(engine.delta(state, t, m))
    return total / len(clients)


def replay(engine, start, window, onset, kind, size, permutations):
    """Exact (kind 'exact'), direct sampling ('S'), paired ('P') or capped paired ('V')."""
    device = start.device
    state = start - window[0]['h'].to(device)  # stored retained aggregate is exact at the shared start
    calls = engine.calls
    v0 = window[0]['variance']
    for j in range(1, len(window)):
        record = window[j]
        retained = record['retained']
        if not retained:
            continue
        t = onset + j
        if kind == 'exact' or size >= len(retained):
            estimate = mean_delta(engine, state, t, retained)
        else:
            chosen = permutations[j][:size]
            new = mean_delta(engine, state, t, chosen)
            if kind == 'S':
                estimate = new
            else:
                beta = 1.0 if kind == 'P' else (min(1.0, (v0 / record['variance']) ** .5) if record['variance'] else 1.0)
                old = sum(record['updates'][m].to(device) for m in chosen) / len(chosen)
                estimate = new + beta * (record['h'].to(device) - old)
        state = state - estimate
    return state, engine.calls - calls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='data')
    p.add_argument('--out', required=True)
    p.add_argument('--rate', type=float, default=.2)
    p.add_argument('--delays', type=int, nargs='+', default=[8, 16])
    p.add_argument('--poison-fraction', type=float, default=.5)
    p.add_argument('--scale', type=float, default=1.)
    p.add_argument('--patch', type=int, default=4)
    p.add_argument('--methods', nargs='+', default=['filter', 'rollback', 'S2', 'S4', 'P2', 'P4', 'V4', 'exact'])
    p.add_argument('--continue-methods', nargs='+', default=['filter', 'rollback', 'S2', 'S4', 'P2', 'P4', 'exact'])
    p.add_argument('--continuation', type=int, default=16)
    p.add_argument('--eval-every', type=int, default=2)
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    dmax = max(a.delays)
    horizon = dmax + a.continuation
    engine, args, start, ck = setup(a.checkpoint, a.data, device, horizon, a.poison_fraction,
                                    patch=a.patch, threads=a.threads)
    seed = args.seed
    active_count = max(1, min(args.clients, round(a.rate * args.clients)))
    prng = np.random.default_rng(seed + 823331)
    active = [sorted(prng.choice(args.clients, active_count, replace=False).tolist()) for _ in range(horizon)]
    retained_at = [[m for m in act if m >= args.malicious] for act in active]
    rng = np.random.default_rng(seed + CLIENT_OFFSET)
    permutations = [rng.permutation(r).tolist() for r in retained_at]
    (out / 'config.json').write_text(json.dumps(dict(
        vars(a), seed=seed, onset=args.onset, dataset=args.dataset, lr=args.lr, clients=args.clients,
        malicious=args.malicious, active_per_round=active_count,
        checkpoint_test_accuracy=ck.get('official_test_accuracy'), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        protocol='oracle identities; per-round uniform participation shared by all branches'), indent=2))
    (out / 'participation.json').write_text(json.dumps(active))
    begin = time.monotonic()

    def stamp(row):
        print(json.dumps(dict(elapsed=round(time.monotonic() - begin, 1), **row)), flush=True)

    # Attacked window: equal-weight FedAvg over the active clients; attackers poison when active.
    state = start.clone()
    window, attack_rows = [], []
    for j in range(dmax):
        t = args.onset + j
        total = torch.zeros_like(state)
        updates = {}
        for m in active[j]:
            poisoned = m < args.malicious
            d = engine.delta(state, t, m, poison=poisoned)
            if poisoned:
                d = d * a.scale
            else:
                updates[m] = d.detach().cpu()
            total.add_(d)
        retained = retained_at[j]
        h = (sum(updates[m] for m in retained) / len(retained)) if retained else torch.zeros(state.numel())
        variance = (sum((updates[m] - h).square().sum().item() for m in retained) / len(retained)) if retained else 0.0
        window.append(dict(w=state.detach().cpu().clone(), h=h, updates=updates, retained=retained,
                           attackers=len(active[j]) - len(retained), variance=variance))
        state -= total / len(active[j])
        window[-1]['after'] = state.detach().cpu().clone()
        if (j + 1) % a.eval_every == 0 or j + 1 in a.delays:
            attack_rows.append(dict(seed=seed, rate=a.rate, offset=j + 1, attackers_active=window[-1]['attackers'],
                                    **engine.evaluate(state)))
            write(out / 'attack.csv', attack_rows)
            stamp(dict(stage='attack', **attack_rows[-1]))

    recovery_rows, continuation_rows = [], []
    for delay in a.delays:
        part = window[:delay]
        end = args.onset + delay
        common = dict(dataset=args.dataset, seed=seed, attack='backdoor', rate=a.rate, delay=delay)
        states, costs = {}, {}
        tic = time.perf_counter()
        exact, calls = replay(engine, start, part, args.onset, 'exact', 0, permutations)
        states['exact'] = exact
        costs['exact'] = dict(calls=calls, serial_rounds=delay - 1, seconds=time.perf_counter() - tic)
        for method in a.methods:
            if method in states:
                continue
            tic = time.perf_counter()
            if method in ('filter', 'rollback'):
                states[method] = (part[-1]['after'].to(device) if method == 'filter' else start).clone()
                costs[method] = dict(calls=0, serial_rounds=0, seconds=0.0)
            else:
                s, calls = replay(engine, start, part, args.onset, method[0], int(method[1:]), permutations)
                states[method] = s
                costs[method] = dict(calls=calls, serial_rounds=delay - 1, seconds=time.perf_counter() - tic)
        for method, s in states.items():
            recovery_rows.append(dict(common, method=method, finite=bool(torch.isfinite(s).all()), **costs[method],
                                      **engine.metrics(s, exact, 'backdoor')))
            stamp(dict(stage='recovery', **recovery_rows[-1]))
        write(out / 'recovery.csv', recovery_rows)

        reference = {}
        order = ['exact'] + [m for m in a.continue_methods if m != 'exact' and m in states]
        for method in order:
            s = states[method].clone()
            calls = costs[method]['calls']
            for c in range(1, a.continuation + 1):
                retained = retained_at[delay + c - 1]
                if retained:
                    s -= mean_delta(engine, s, end + c - 1, retained)
                    calls += len(retained)
                if c % a.eval_every == 0 or c == a.continuation:
                    ev = engine.evaluate(s)
                    if method == 'exact':
                        reference[c] = s.detach().cpu().clone()
                    dis = (engine.predictions(s) != engine.predictions(reference[c].to(device))).float().mean().item()
                    continuation_rows.append(dict(common, method=method, continuation=c,
                                                  communication_rounds=costs[method]['serial_rounds'] + c,
                                                  total_client_calls=calls, disagreement_same_continuation=dis, **ev))
            write(out / 'continuation.csv', continuation_rows)
            stamp(dict(stage='continued', **continuation_rows[-1]))
        del states, reference
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
