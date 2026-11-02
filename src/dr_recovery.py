"""Recovery from a durable backdoor on validation-selected checkpoints.

For one checkpoint, the script builds the attacked window history with every
upload kept in host memory, computes the exclusion reference by exact replay,
applies the recovery estimators, and continues honest training from each
recovered model. Every method continues for the same number of honest rounds;
replay serial rounds are reported as a cost, not as extra training.
"""
import argparse
import csv
import fcntl
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import run_covariance as rc
from run_paired import CLIENT_OFFSET
from fedrecover_adapter import replay_fedrecover
from dr_baselines import replay_crab, replay_federaser, server_finetune
from dr_lib import attack_round, neurotoxin_mask, setup


class DeviceRows:
    """Index CPU-stored uploads and return them on the accelerator."""

    def __init__(self, rows, device):
        self.rows, self.device = rows, device

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.rows[index].to(self.device)
        return self.rows[index]


def device_history(history, device):
    return [dict(w=r['w'].to(device), h=r['h'].to(device), updates=DeviceRows(r['updates'], device))
            for r in history]


def write(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='data')
    p.add_argument('--out', required=True)
    p.add_argument('--delays', type=int, nargs='+', default=[8, 16])
    p.add_argument('--poison-fraction', type=float, default=.5)
    p.add_argument('--scale', type=float, default=1.)
    p.add_argument('--neurotoxin', type=float, default=0.)
    p.add_argument('--patch', type=int, default=4)
    p.add_argument('--methods', nargs='+',
                   default=['filter', 'rollback', 'S2', 'S4', 'S8', 'P2', 'P4', 'P8',
                            'V4', 'Fit4', 'FedRecover', 'exact'])
    p.add_argument('--continue-methods', nargs='+',
                   default=['filter', 'rollback', 'S4', 'P4', 'V4', 'Fit4', 'FedRecover', 'exact'])
    p.add_argument('--continuation', type=int, default=16)
    p.add_argument('--eval-every', type=int, default=2)
    p.add_argument('--replay-seed-offset', type=int, default=0)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--start-offset', type=int, default=0,
                   help='late onset estimate: recovery starts k rounds after the true onset; the true '
                        'exclusion trajectory is reported as method "oracle" and used as the reference')
    p.add_argument('--latency-matched', action='store_true',
                   help='zero-round methods also continue for delay-1 extra rounds (equal communication rounds)')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    dmax = max(a.delays)
    engine, args, start, ck = setup(a.checkpoint, a.data, device, 2 * dmax - 1 + a.continuation,
                                    a.poison_fraction, patch=a.patch, threads=a.threads)
    args.history_cpu, args.diagnostics, args.initial_beta = True, False, 0.0
    seed = args.seed
    honest = list(range(args.malicious, args.clients))
    rng = np.random.default_rng(seed + CLIENT_OFFSET + a.replay_seed_offset)
    permutations = [rng.permutation(honest) for _ in range(dmax - 1)]
    (out / 'config.json').write_text(json.dumps(dict(
        vars(a), seed=seed, onset=args.onset, dataset=args.dataset, lr=args.lr,
        clients=args.clients, malicious=args.malicious, local_epochs=args.local_epochs,
        checkpoint_validation_accuracy=ck.get('validation_accuracy'),
        checkpoint_test_accuracy=ck.get('official_test_accuracy'),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        protocol='oracle identities, onset-aligned window; replay schedules coupled with the attacked run'),
        indent=2))
    begin = time.monotonic()

    def stamp(row):
        print(json.dumps(dict(elapsed=round(time.monotonic() - begin, 1), **row)), flush=True)

    # Attacked window; all uploads are retained for the replay estimators.
    state = start.clone()
    history, attack_rows, previous = [], [], None
    p_dim = state.numel()
    for j in range(dmax):
        store = torch.empty((args.clients, p_dim), dtype=state.dtype)  # malicious rows stay unused and untouched
        mask = neurotoxin_mask(previous, a.neurotoxin)
        w = state.detach().cpu().clone()
        aggregate, h = attack_round(engine, state, args.onset + j, args, a.scale, mask, store)
        previous = aggregate.clone()
        state -= aggregate
        h_cpu = h.detach().cpu()
        variance = sum((store[m] - h_cpu).square().sum().item() for m in honest) / len(honest)
        history.append(dict(w=w, h=h_cpu, updates=store, variance=variance,
                            after=state.detach().cpu().clone()))
        if (j + 1) % a.eval_every == 0 or j + 1 in a.delays:
            attack_rows.append(dict(seed=seed, offset=j + 1, history_variance=variance,
                                    **engine.evaluate(state)))
            write(out / 'attack.csv', attack_rows)
            stamp(dict(stage='attack', **attack_rows[-1]))

    recovery_rows, continuation_rows = [], []
    k = a.start_offset
    for delay in a.delays:
        end = args.onset + delay
        corrupt = history[delay - 1]['after'].to(device)
        common = dict(dataset=args.dataset, seed=seed, attack='backdoor', delay=delay)
        states, costs = {}, {}
        if k:
            # The server believes the attack began k rounds late: every method restarts from the
            # attacked model w_{a+k} and replays rounds a+k..a+delay-1 only.
            oracle, cost, _, _ = rc.replay(engine, start, history[:delay], honest, 'exact', permutations, args, common)
            states['oracle'] = oracle
            costs['oracle'] = dict(calls=cost['extra_client_updates'], serial_rounds=delay - 1,
                                   seconds=cost['recovery_seconds'])
            start_m = history[k]['w'].to(device)
            margs = SimpleNamespace(**vars(args))
            margs.onset = args.onset + k
            perms = permutations[k:]
        else:
            start_m, margs, perms = start, args, permutations
        window = history[k:delay]
        exact, cost, _, _ = rc.replay(engine, start_m, window, honest, 'exact', perms, margs, common)
        states['exact'] = exact
        costs['exact'] = dict(calls=cost['extra_client_updates'], serial_rounds=delay - k - 1,
                              seconds=cost['recovery_seconds'])
        reference_state = states['oracle'] if k else exact
        for method in a.methods:
            if method in states:
                continue
            tic = time.perf_counter()
            if method in ['filter', 'rollback']:
                states[method] = (corrupt if method == 'filter' else start_m).clone()
                costs[method] = dict(calls=0, serial_rounds=0, seconds=0.0)
            elif method.startswith('FedRecover'):
                # Optional schedule suffix, e.g. FedRecover_w1_p20: warm-up 1 round, correction period 20.
                schedule = dict(warmup=3, period=10)
                for part in method.split('_')[1:]:
                    schedule[{'w': 'warmup', 'p': 'period'}[part[0]]] = int(part[1:])
                before = engine.calls
                # FedRecover-style keeps curvature pairs for every client on the GPU; a file lock keeps
                # concurrent runs from entering this phase together (it waits, results are unchanged).
                with open('/tmp/dr_fedrecover.lock', 'w') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    state_fr, info, trace = replay_fedrecover(engine, start_m, device_history(window, device),
                                                              honest, margs.onset, **schedule)
                    torch.cuda.empty_cache()
                states[method] = state_fr
                costs[method] = dict(calls=engine.calls - before,
                                     serial_rounds=sum(1 for r in trace if r['calls'] > 0),
                                     seconds=time.perf_counter() - tic)
            elif method.startswith('FT'):
                # Filtered model fine-tuned on the server's clean split for the given number of epochs.
                states[method] = server_finetune(engine, corrupt, int(method[2:]), seed)
                costs[method] = dict(calls=0, serial_rounds=0, seconds=time.perf_counter() - tic)
            elif method in ('FedEraser', 'Crab', 'CrabStored'):
                if method == 'FedEraser':
                    state_b, info = replay_federaser(engine, start_m, window, honest, margs.onset)
                else:
                    state_b, info = replay_crab(engine, start_m, window, honest, margs.onset,
                                                anchor='rollback' if method == 'Crab' else 'stored')
                states[method] = state_b
                costs[method] = dict(calls=info['calls'], serial_rounds=info['serial_rounds'],
                                     seconds=time.perf_counter() - tic)
            else:
                state_m, cost, _, _ = rc.replay(engine, start_m, window, honest, method,
                                                perms, margs, common)
                states[method] = state_m
                costs[method] = dict(calls=cost['extra_client_updates'], serial_rounds=delay - k - 1,
                                     seconds=cost['recovery_seconds'])
            # Release cached blocks between methods so that two runs can share one GPU.
            torch.cuda.empty_cache()
        for method, s in states.items():
            finite = bool(torch.isfinite(s).all())
            metrics = engine.metrics(s, reference_state, 'backdoor') if finite else dict(
                accuracy=float('nan'), asr=float('nan'), disagreement=float('nan'),
                parameter_residual=float('nan'))
            recovery_rows.append(dict(common, method=method, finite=finite, **costs[method], **metrics))
            stamp(dict(stage='recovery', **recovery_rows[-1]))
        write(out / 'recovery.csv', recovery_rows)

        # Continuation with honest clients. The exact branch is computed first so that
        # the other branches can be compared with it at the same continuation index.
        reference = {}
        ref_name = 'oracle' if k else 'exact'
        order = [ref_name] + [m for m in a.continue_methods if m != ref_name and m in states]
        if k and 'exact' in states and 'exact' not in order:
            order.append('exact')
        for method in order:
            s = states[method].clone()
            # Every method continues the same number of honest rounds after detection;
            # replay serial rounds are a separate cost column, not extra training.
            rounds = a.continuation
            if a.latency_matched and costs[method]['serial_rounds'] == 0:
                rounds += delay - 1
            for c in range(1, rounds + 1):
                s -= rc.aggregate_stream(engine, s, end + c - 1, honest)
                if c % a.eval_every == 0 or c == rounds:
                    finite = bool(torch.isfinite(s).all())
                    ev = engine.evaluate(s) if finite else dict(accuracy=float('nan'), asr=float('nan'))
                    if method == ref_name:
                        reference[c] = s.detach().cpu().clone()
                    dis = float('nan')
                    if finite and c in reference:
                        ref = reference[c].to(device)
                        dis = (engine.predictions(s) != engine.predictions(ref)).float().mean().item()
                    comm = costs[method]['serial_rounds'] + c
                    calls = costs[method]['calls'] + c * len(honest)
                    continuation_rows.append(dict(common, method=method, continuation=c,
                                                  communication_rounds=comm, total_client_calls=calls,
                                                  disagreement_same_continuation=dis, **ev))
            write(out / 'continuation.csv', continuation_rows)
            stamp(dict(stage='continued', **continuation_rows[-1]))
            torch.cuda.empty_cache()
        del states, reference
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
