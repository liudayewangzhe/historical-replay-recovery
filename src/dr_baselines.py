"""Windowed FedEraser and Crab-style baselines for the durable-backdoor protocol.

Both methods start from the onset checkpoint, which is the rollback point shared by all
methods, and read only the stored uploads of retained clients.

FedEraser (Liu et al., IWQoS 2021; calibration as in serverEraser.py of the Crab code):
every `interval`-th stored round is replayed; retained clients run a fraction of their
local work from the current model, and the mean calibration update takes the stored
mean-update norm and the new direction, layer by layer. Skipped rounds are not replayed.

Crab-style (Jiang et al., TIFS 2025; serverCrab.py of github.com/shenjiyuan123/
Adaptive_Recovery): rounds are selected by the KL divergence between consecutive stored
global models and clients by cosine similarity with the stored global model. Each selected
round applies the two-call calibration of the official code. With anchor='rollback' the
calibration starts from the current recovered model, following the paper's adaptive
rollback to a model not yet affected by the attackers; with anchor='stored' it starts from
the stored global model of each selected round as the released code does, so the output
is the calibrated stored model of the last processed round.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def layer_shapes(model):
    return [p.shape for p in model.parameters()]


def split_layers(vector, shapes):
    out, offset = [], 0
    for shape in shapes:
        count = math.prod(shape)
        out.append(vector[offset:offset + count].reshape(shape))
        offset += count
    return out


def calibrate(old_update, new_update, shapes):
    """Layer-wise historical norm with the new direction (FedEraser Eq. 1)."""
    parts = []
    for old, new in zip(split_layers(old_update, shapes), split_layers(new_update, shapes)):
        norm = torch.linalg.vector_norm(new)
        parts.append((new * (torch.linalg.vector_norm(old) / norm) if norm > 0 else torch.zeros_like(new)).flatten())
    return torch.cat(parts)


def mean_delta(engine, state, t, clients, fraction=1.0):
    """Mean client update; `fraction` keeps the first part of each local minibatch list."""
    total, minibatches = torch.zeros_like(state), 0
    for m in clients:
        saved = engine.schedule[t][m]
        n = max(1, math.ceil(len(saved) * fraction))
        engine.schedule[t][m] = saved[:n]
        try:
            total.add_(engine.delta(state, t, m))
        finally:
            engine.schedule[t][m] = saved
        minibatches += n
    return total / len(clients), minibatches


def replay_federaser(engine, start, window, honest, onset, interval=2, fraction=0.5):
    device = start.device
    shapes = layer_shapes(engine.model)
    retained = list(range(0, len(window), interval))
    state = start - window[0]['h'].to(device)  # first stored aggregate is exact at the shared start
    calls, minibatches = engine.calls, 0
    for j in retained[1:]:
        new, n = mean_delta(engine, state, onset + j, honest, fraction)
        minibatches += n
        state -= calibrate(window[j]['h'].to(device), new, shapes)
    return state, dict(calls=engine.calls - calls, serial_rounds=len(retained) - 1, minibatches=minibatches)


def kl_divergences(window, shapes, device):
    """Official Crab score: KL between per-layer softmaxes of consecutive global models."""
    models = [r['w'] for r in window] + [window[-1]['after']]
    scores = []
    for j in range(len(window)):
        prior = split_layers(models[j].to(device), shapes)
        now = split_layers(models[j + 1].to(device), shapes)
        kl = sum(F.kl_div(F.log_softmax(b, dim=-1), F.softmax(a, dim=-1), reduction='sum')
                 for a, b in zip(prior, now))
        scores.append(float(kl.item()))
    return scores


def select_clients(record, honest, shapes, ratio, device):
    """Retained clients whose stored local models are most similar to the global model."""
    w = record['w'].to(device)
    global_layers = split_layers(w, shapes)
    similarity = []
    for m in honest:
        client_layers = split_layers(w - record['updates'][m].to(device), shapes)
        cos = [F.cosine_similarity(g, c, dim=1).mean().item()
               for g, c in zip(global_layers, client_layers) if g.ndim > 1]
        similarity.append(float(np.mean(cos)))
    count = max(1, int(len(honest) * ratio))
    order = np.argsort(similarity)[::-1][:count]
    return [honest[i] for i in order]


def replay_crab(engine, start, window, honest, onset, round_ratio=0.6, client_ratio=0.7, anchor='rollback'):
    device = start.device
    shapes = layer_shapes(engine.model)
    scores = kl_divergences(window, shapes, device)
    count = max(1, int(len(window) * round_ratio))
    by_score = np.argsort(scores)[::-1][:count].tolist()
    order = sorted(by_score) if anchor == 'rollback' else by_score
    calls, state = engine.calls, start.clone()
    for j in order:
        record = window[j]
        chosen = select_clients(record, honest, shapes, client_ratio, device)
        base = state if anchor == 'rollback' else record['w'].to(device)
        first, _ = mean_delta(engine, base, onset + j, chosen)
        moved = base - first
        second, _ = mean_delta(engine, moved, onset + j, chosen)
        old = sum(record['updates'][m].to(device) for m in chosen) / len(chosen)
        state = moved - calibrate(old, second, shapes)
    return state, dict(calls=engine.calls - calls, serial_rounds=2 * len(order), selected_rounds=len(order))


def server_finetune(engine, state, epochs, seed):
    """Fine-tune a model on the server's held-out clean split; no client computation.

    Plain SGD with the clients' learning rate and batch size and a fixed shuffling seed.
    """
    vector_to_parameters(state.clone(), engine.model.parameters())
    engine.model.train()
    generator = torch.Generator().manual_seed(seed)
    n = len(engine.vy)
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator).to(engine.vy.device)
        for i in range(0, n, engine.args.batch):
            ids = order[i:i + engine.args.batch]
            engine.model.zero_grad(set_to_none=True)
            engine.loss(engine.model(engine.vx[ids]), engine.vy[ids]).backward()
            with torch.no_grad():
                for p in engine.model.parameters():
                    p.add_(p.grad, alpha=-engine.args.lr)
    return parameters_to_vector(engine.model.parameters()).detach().clone()
