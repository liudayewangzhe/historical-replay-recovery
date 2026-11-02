"""Windowed FedRecover-style adaptation of Alg. 1--3 to local-update maps.

Source: https://arxiv.org/html/2210.10936v1. Original FedRecover uses
gradients; this protocol uses two-step (or epoch) client update maps.
First saved retained aggregate is already exact and incurs no new calls.
"""
import math
import time

import torch

from run_recovery import sync


def hessian_product(s, y, v):
    """Compact limited-memory BFGS Hessian-vector product, not its inverse."""
    S = torch.stack(s, dim=1)
    Y = torch.stack(y, dim=1)
    A = S.T @ Y
    sigma = (s[-1] @ y[-1]) / (s[-1] @ s[-1])
    D = torch.diag(torch.diag(A))
    L = torch.tril(A, diagonal=-1)
    M = torch.cat((torch.cat((-D, L.T), dim=1),
                   torch.cat((L, sigma * (S.T @ S)), dim=1)), dim=0)
    rhs = torch.cat((Y.T @ v, sigma * S.T @ v))
    p = torch.linalg.solve(M, rhs)
    return sigma * v - torch.cat((Y, sigma * S), dim=1) @ p


def historical_threshold(history, honest, tolerance):
    values = []
    for record in history:
        # CUDA kthvalue has no deterministic implementation in PyTorch;
        # this calibration uses saved history and executes once on the CPU.
        samples = record['updates'][honest].detach().cpu().abs().flatten()
        k = max(1, min(samples.numel(), math.ceil((1 - tolerance) * samples.numel())))
        values.append(torch.kthvalue(samples, k).values.item())
    return max(values)


def replay_fedrecover(engine, start, history, honest, onset, warmup=3,
                      period=10, final=1, memory=2, tolerance=1e-6, storage_profile=None):
    """History-based L-BFGS prediction, exact corrections and full cost trace."""
    if warmup < 1 or period < 1 or final < 0 or memory < 1:
        raise ValueError('invalid recovery schedule')
    buffers = {m: [] for m in honest}
    state = start.clone()
    calls0 = engine.calls
    exact_rounds = abnormal = insufficient = 0
    minibatches = 0
    trace = []
    sync(start.device)
    begin = time.perf_counter()
    threshold = historical_threshold(history, honest, tolerance)
    for j, record in enumerate(history):
        if j == 0:
            state -= record['h']
            trace.append({'step': j, 'mode': 'saved_exact', 'calls': 0})
            continue
        displacement = state - record['w']
        exact_all = j < warmup or j >= len(history) - final or (j-warmup+1) % period == 0
        exact_rounds += int(exact_all)
        estimates = []
        reasons = {'scheduled': 0, 'abnormal': 0, 'insufficient': 0}
        for m in honest:
            old = record['updates'][m]
            reason = 'scheduled' if exact_all else None
            pairs = buffers[m]
            if reason is None and len(pairs) < memory:
                reason = 'insufficient'
            if reason is None:
                try:
                    correction = hessian_product([p[0] for p in pairs], [p[1] for p in pairs], displacement)
                    predicted = old + correction
                except torch.linalg.LinAlgError:
                    reason = 'insufficient'
                if reason is None and (not torch.isfinite(predicted).all() or predicted.abs().max().item() > threshold):
                    reason = 'abnormal'
            if reason is not None:
                exact = engine.delta(state, onset+j, m)
                minibatches += len(engine.schedule[onset+j][m])
                reasons[reason] += 1
                difference = exact - old
                # A BFGS curvature pair needs a nonzero denominator and positive curvature.
                if (displacement @ difference).item() > 0 and (displacement @ displacement).item() > 0:
                    pairs.append((displacement.clone(), difference.clone()))
                    del pairs[:-memory]
                predicted = exact
            estimates.append(predicted)
        abnormal += reasons['abnormal']
        insufficient += reasons['insufficient']
        if storage_profile is not None:
            storage_profile.append(sum(t.numel()*t.element_size()
                for pairs in buffers.values() for pair in pairs for t in pair))
        state -= torch.stack(estimates).mean(0)
        trace.append({'step': j, 'mode': 'exact_all' if exact_all else 'estimate',
                      'calls': sum(reasons.values()), **reasons})
    sync(start.device)
    return state, {'extra_client_updates': engine.calls-calls0,
                   'local_sgd_minibatches': minibatches,
                   'scheduled_exact_rounds': exact_rounds,
                   'abnormal_client_updates': abnormal,
                   'insufficient_curvature_client_updates': insufficient,
                   'abnormality_threshold': threshold,
                   'recovery_seconds': time.perf_counter()-begin}, trace
