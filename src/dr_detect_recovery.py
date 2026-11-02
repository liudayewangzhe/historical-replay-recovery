"""Real-detector recovery for a durable backdoor on validation-selected checkpoints.

Protocol, per checkpoint:
1. Calibration: 20 attack-free rounds from the checkpoint, using batch schedules
   that the attack window never uses (a warm-up period a server can observe).
   Each round gives a robust z-score per client: the log distance of its upload to the
   coordinate-wise median upload, standardised by the median and MAD over clients.
   The threshold is the 99.5th percentile of all calibration z-scores.
2. Attack: the shared-patch backdoor from the checkpoint round (onset). All uploads are kept.
3. Detection: a client is flagged after two consecutive rounds above the threshold;
   the alarm is the first round in which at least two clients are flagged.
   Flagged identities are excluded from later rounds.
4. Recovery window: onset-aligned, rounds [onset, d) with d = alarm + 1 (realised delay).
   The server uses the retained set R (all clients except the flagged ones).
5. References:
   - target: exact replay with the TRUE attackers excluded from onset (the exclusion trajectory);
   - detected-exact: exact replay with the DETECTED retained set R (isolates recovery error
     from detection error).
6. Continuation: honest training for C rounds after d with the retained set for every method.
   Attackers that were missed keep following the attack protocol in replay and continuation.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import run_covariance as rc
from dr_lib import setup
from run_paired import CLIENT_OFFSET


def lower_median(x, dim):
    """Lower median via sorting; deterministic on CUDA, unlike median with indices."""
    count = x.shape[dim]
    return torch.sort(x, dim=dim).values.narrow(dim, (count - 1) // 2, 1).squeeze(dim)


def robust_z(uploads):
    """Per-client robust z-score of log distance to the coordinate-wise median upload."""
    median = lower_median(uploads, 0)
    log_distance = torch.log((uploads - median).norm(dim=1) + 1e-12)
    centre = lower_median(log_distance, 0)
    scale = lower_median((log_distance - centre).abs(), 0) * 1.4826 + 1e-12
    return ((log_distance - centre) / scale).cpu().numpy()


def write(path, rows):
    """Write rows with the union of keys, so rows of different stages can share one file."""
    if rows:
        fieldnames = []
        for row in rows:
            fieldnames += [k for k in row if k not in fieldnames]
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
            writer.writeheader()
            writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='data')
    p.add_argument('--out', required=True)
    p.add_argument('--scale', type=float, default=3.)
    p.add_argument('--poison-fraction', type=float, default=.5)
    p.add_argument('--patch', type=int, default=4)
    p.add_argument('--calib-rounds', type=int, default=20)
    p.add_argument('--max-window', type=int, default=32)
    p.add_argument('--continuation', type=int, default=16)
    p.add_argument('--methods', nargs='+', default=['filter', 'rollback', 'S4', 'P4', 'V4', 'exact'])
    p.add_argument('--eval-every', type=int, default=4)
    p.add_argument('--replay-seed-offset', type=int, default=0)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--continued-detection', type=int, default=1,
                   help='keep running the detector after the first alarm and exclude newly flagged clients')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    calib_base = 2 * a.max_window + a.continuation + 10
    rounds_after_onset = calib_base + a.calib_rounds  # schedule must cover absolute index onset+calib_base+calib-1
    engine, args, start, ck = setup(a.checkpoint, a.data, device, rounds_after_onset,
                                    a.poison_fraction, patch=a.patch, threads=a.threads)
    args.history_cpu, args.diagnostics, args.initial_beta = True, False, 0.0  # window uploads stay in host memory
    onset, seed = args.onset, args.seed
    malicious = list(range(args.malicious))
    true_honest = list(range(args.malicious, args.clients))
    all_clients = list(range(args.clients))
    attacked_ids = set(malicious)

    # Wrap the local-update map so that attackers keep following the attack protocol whenever
    # they are recomputed (replay and continuation), with their saved batch schedules.
    raw_delta = engine.delta

    def attacked_delta(w, t, m, poison=False):
        d = raw_delta(w, t, m, poison=poison or (m in attacked_ids))
        return d * a.scale if m in attacked_ids else d

    engine.delta = attacked_delta

    rng = np.random.default_rng(seed + CLIENT_OFFSET + a.replay_seed_offset)
    begin = time.monotonic()
    log = []

    def stamp(row):
        row = dict(elapsed=round(time.monotonic() - begin, 1), **row)
        log.append(row)
        print(json.dumps(row), flush=True)

    # 1. Calibration on attack-free rounds from the checkpoint, with unused batch schedules.
    thr_pool = []
    state = start.clone()
    for c in range(a.calib_rounds):
        t = onset + calib_base + c
        uploads = torch.stack([raw_delta(state, t, m) for m in all_clients])
        thr_pool.append(robust_z(uploads))
        state -= uploads.mean(0)
    threshold = float(np.percentile(np.concatenate(thr_pool), 99.5))
    stamp(dict(stage='calibration', seed=seed, threshold=threshold, calib_rounds=a.calib_rounds))

    # 2-3. Attack with detection; keep all uploads of the window.
    state = start.clone()
    consecutive = np.zeros(args.clients, dtype=int)
    window = []
    alarm = None
    flagged = []
    for j in range(a.max_window):
        t = onset + j
        w_before = state.clone().cpu()
        uploads = torch.stack([engine.delta(state, t, m, poison=m in attacked_ids) for m in all_clients])
        z = robust_z(uploads)
        consecutive = np.where(z > threshold, consecutive + 1, 0)
        flagged_now = np.flatnonzero(consecutive >= 2).tolist()
        window.append(dict(w=w_before, updates=uploads.cpu(), after=None))
        state -= uploads.mean(0)
        window[-1]['after'] = state.clone().cpu()
        if len(flagged_now) >= 2:
            alarm = j
            flagged = flagged_now
            break
    if alarm is None:
        stamp(dict(stage='detection', seed=seed, alarm=None, note='no alarm within the window'))
        write(out / 'detection.csv', log)
        return
    delay = alarm + 1
    end = onset + delay
    retained = [m for m in all_clients if m not in flagged]
    fp = len([m for m in flagged if m not in attacked_ids])
    tp = len([m for m in flagged if m in attacked_ids])
    stamp(dict(stage='detection', seed=seed, alarm_offset=alarm, delay=delay, flagged=len(flagged),
               true_positive=tp, false_positive=fp, missed=len(attacked_ids) - tp, threshold=threshold))
    write(out / 'detection.csv', log)

    # Records for each branch are built from the stored uploads of the window.
    def branch_records(members):
        records = []
        for r in window[:delay]:
            ups = r['updates']
            records.append(dict(w=r['w'], h=ups[members].mean(0),
                                updates=ups, variance=float(
                                    (ups[members] - ups[members].mean(0)).square().sum(1).mean().item()),
                                after=r['after']))
        return records

    rng_perm = {'detected': np.array(retained), 'target': np.array(true_honest)}
    permutations = {k: [rng.permutation(v) for _ in range(max(1, delay - 1))] for k, v in rng_perm.items()}
    corrupt = window[delay - 1]['after'].to(device)
    detected_records = branch_records(retained)
    target_records = branch_records(true_honest)

    # The target and detected-exact replays use the same sampled identities as the estimators.
    target_state, target_cost, _, _ = rc.replay(engine, start, target_records, true_honest, 'exact',
                                                permutations['target'], args, {})
    exact_det, exact_cost, _, _ = rc.replay(engine, start, detected_records, retained, 'exact',
                                            permutations['detected'], args, {})
    states, costs = {'exact_target': target_state, 'exact_detected': exact_det}, {}
    for method in a.methods:
        if method == 'exact':
            continue
        if method == 'filter':
            states[method] = corrupt.clone()
            costs[method] = dict(calls=0, serial_rounds=0)
        elif method == 'rollback':
            states[method] = start.clone()
            costs[method] = dict(calls=0, serial_rounds=0)
        else:
            s, cost, _, _ = rc.replay(engine, start, detected_records, retained, method,
                                      permutations['detected'], args, {})
            states[method] = s
            costs[method] = dict(calls=cost['extra_client_updates'], serial_rounds=delay - 1)
    costs['exact_target'] = dict(calls=target_cost['extra_client_updates'], serial_rounds=delay - 1)
    costs['exact_detected'] = dict(calls=exact_cost['extra_client_updates'], serial_rounds=delay - 1)

    recovery_rows = []
    for name, s in states.items():
        ref = target_state
        metrics = engine.metrics(s, ref, 'backdoor')
        ev = engine.evaluate(s)
        recovery_rows.append(dict(seed=seed, delay=delay, method=name, accuracy=ev['accuracy'], asr=ev['asr'],
                                  disagreement=metrics['disagreement'], residual=metrics['parameter_residual'],
                                  calls=costs[name]['calls'], serial_rounds=costs[name]['serial_rounds']))
        stamp(dict(stage='recovery', **recovery_rows[-1]))
    write(out / 'recovery.csv', recovery_rows)

    # 6. Continuation with the retained set (target uses the true honest set).
    continuation_rows = []
    reference = {}
    order = ['exact_target'] + [m for m in ['exact_detected'] + a.methods if m not in ('exact', 'exact_target')]
    for name in order:
        s = states[name].clone()
        members = list(true_honest if name == 'exact_target' else retained)
        cons = np.zeros(args.clients, dtype=int)
        late = []
        for c in range(1, a.continuation + 1):
            t = end + c - 1
            if name == 'exact_target' or not a.continued_detection:
                s -= rc.aggregate_stream(engine, s, t, members)
            else:
                # The detector keeps running: flagged uploads are dropped and the clients excluded.
                ups = torch.stack([engine.delta(s, t, m) for m in members])
                z = robust_z(ups)
                for i, m in enumerate(members):
                    cons[m] = cons[m] + 1 if z[i] > threshold else 0
                keep = [i for i, m in enumerate(members) if cons[m] < 2]
                newly = [m for m in members if cons[m] >= 2]
                s -= ups[keep].mean(0)
                del ups
                if newly:
                    late += newly
                    members = [m for m in members if m not in newly]
            if c % a.eval_every == 0 or c == a.continuation:
                ev = engine.evaluate(s)
                if name == 'exact_target':
                    reference[c] = s.detach().cpu().clone()
                dis = float('nan')
                if c in reference and name != 'exact_target':
                    dis = (engine.predictions(s) != engine.predictions(reference[c].to(device))).float().mean().item()
                continuation_rows.append(dict(seed=seed, delay=delay, method=name, continuation=c,
                                              accuracy=ev['accuracy'], asr=ev['asr'],
                                              disagreement_same_continuation=dis,
                                              late_flagged=len(late),
                                              late_true=len([m for m in late if m in attacked_ids])))
        stamp(dict(stage='continued', seed=seed, method=name, continuation=a.continuation,
                   accuracy=continuation_rows[-1]['accuracy'], asr=continuation_rows[-1]['asr']))
    write(out / 'continuation.csv', continuation_rows)
    (out / 'config.json').write_text(json.dumps(dict(vars(a), seed=seed, onset=onset, threshold=threshold,
                                                     alarm_offset=alarm, delay=delay, flagged=flagged,
                                                     retained=retained), indent=2))


if __name__ == '__main__':
    main()
