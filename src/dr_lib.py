"""Shared setup for durable-backdoor recovery studies on validation-selected checkpoints.

The module reuses the existing Engine, partition and schedule code so that the
client split, local batches and model match the saved federated checkpoint.
It adds a configurable white trigger patch and Neurotoxin-style projection of
malicious updates onto coordinates rarely changed by recent global updates.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn.utils import vector_to_parameters

import run_covariance as rc
from run_recovery import Engine, partition
from holdout_data import datasets, normalization, tensors

CIFAR_MEAN = (.4914, .4822, .4465)
CIFAR_STD = (.247, .243, .261)


def resnet18_gn(num_classes=10):
    from torchvision.models import resnet18
    model = resnet18(num_classes=num_classes, norm_layer=lambda c: nn.GroupNorm(8, c))
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


class TriggerEngine(Engine):
    """Engine whose poisoned batches and triggered evaluation share one patch."""

    def __init__(self, args, x, y, tx, ty, schedule, white, patch):
        super().__init__(args, x, y, tx, ty, schedule)
        self.white = white.view(1, -1, 1, 1)
        self.patch = patch

    def stamp(self, x):
        x[:, :, -self.patch:, -self.patch:] = self.white

    def delta(self, w, t, m, poison=False):
        vector_to_parameters(w.clone(), self.model.parameters())
        self.model.train()
        for ids in self.schedule[t][m]:
            bx, by = self.x[ids], self.y[ids]
            if poison:
                bx, by = bx.clone(), by.clone()
                n = max(1, int(len(by) * self.args.poison_fraction))
                self.stamp(bx[:n])
                by[:n] = self.args.target
            self.model.zero_grad(set_to_none=True)
            self.loss(self.model(bx), by).backward()
            with torch.no_grad():
                for p in self.model.parameters():
                    p.add_(p.grad, alpha=-self.args.lr)
        self.calls += 1
        return w - self.vector()

    @torch.no_grad()
    def predictions(self, w, trigger=False):
        vector_to_parameters(w.clone(), self.model.parameters())
        self.model.eval()
        out = []
        for x in self.tx.split(512):
            if trigger:
                x = x.clone()
                self.stamp(x)
            out.append(self.model(x).argmax(1))
        return torch.cat(out)

    def evaluate(self, w):
        """Clean accuracy and attack success on non-target test inputs (and on the server split)."""
        if not torch.isfinite(w).all():
            return dict(accuracy=float('nan'), asr=float('nan'))
        pred = self.predictions(w)
        eligible = self.ty != self.args.target
        asr = (self.predictions(w, True)[eligible] == self.args.target).float().mean().item()
        return dict(accuracy=(pred == self.ty).float().mean().item(), asr=asr, **self.validation(w))

    @torch.no_grad()
    def validation(self, w):
        """Accuracy and attack success on the server's held-out split, a check against test-set adaptivity."""
        if getattr(self, 'vx', None) is None:
            return {}
        tx, ty = self.tx, self.ty
        self.tx, self.ty = self.vx, self.vy
        try:
            pred = self.predictions(w)
            eligible = self.ty != self.args.target
            asr = (self.predictions(w, True)[eligible] == self.args.target).float().mean().item()
            return dict(val_accuracy=(pred == self.ty).float().mean().item(), val_asr=asr)
        finally:
            self.tx, self.ty = tx, ty

    def metrics(self, w, reference, attack):
        return dict(super().metrics(w, reference, attack), **self.validation(w))


def setup(checkpoint_path, data_root, device, rounds_after_onset, poison_fraction=.5,
          target=0, patch=4, malicious=10, threads=4):
    """Load a checkpoint with its exact split, schedule and model."""
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    name = ck['dataset']
    onset = ck['round']
    args = SimpleNamespace(
        dataset=name, clients=ck['clients'], malicious=malicious, alpha=ck['alpha'],
        batch=ck['batch'], lr=ck['lr'], local_epochs=ck['local_epochs'],
        local_steps=ck['local_steps'], rounds=onset + rounds_after_onset,
        poison_fraction=poison_fraction, target=target, threads=threads,
        onset=onset, seed=ck['seed'])
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    train, test = datasets(name, data_root)
    x, y = tensors(train, ck['train_indices'].numpy(), device)
    tx, ty = tensors(test, None, device)
    stats = normalization(name)
    if stats is not None:
        mean_t, std_t = stats
        mean = torch.tensor(mean_t, device=device).view(1, 3, 1, 1)
        std = torch.tensor(std_t, device=device).view(1, 3, 1, 1)
        x.sub_(mean).div_(std)
        tx.sub_(mean).div_(std)
        white = (1 - torch.tensor(mean_t, device=device)) / torch.tensor(std_t, device=device)
    else:
        white = torch.ones(1, device=device)
    seed = ck['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    parts = partition(y.cpu().numpy(), args.clients, args.alpha, seed)
    schedule = rc.schedule_for_run(parts, args, seed, device)
    engine = TriggerEngine(args, x, y, tx, ty, schedule, white, patch)
    # The server's held-out clean split (used for checkpoint selection) for server-side baselines.
    vx, vy = tensors(train, ck['validation_indices'].numpy(), device)
    if stats is not None:
        vx.sub_(mean).div_(std)
    engine.vx, engine.vy = vx, vy
    if name in ('cifar10', 'cifar100'):
        engine.model = resnet18_gn(100 if name == 'cifar100' else 10).to(device)
    start = ck['state'].to(device)
    return engine, args, start, ck


def neurotoxin_mask(previous_update, ratio):
    """Neurotoxin-style mask: keep only the fraction `ratio` of coordinates that changed least.

    The attacker projects its update onto rarely updated coordinates (Zhang et al., ICML 2022).
    Coordinates are ranked by the magnitude of the previous global update; the smallest
    `ratio` fraction is kept and all other coordinates are zeroed.
    """
    if previous_update is None or ratio <= 0:
        return None
    magnitude = previous_update.detach().abs().cpu()
    k = max(1, int(ratio * magnitude.numel()))
    threshold = torch.kthvalue(magnitude, k).values.item()
    return (previous_update.abs() <= threshold).to(previous_update.dtype)


def attack_round(engine, state, t, args, scale, mask, store=None):
    """One attacked FedAvg round; optionally stores every upload on the CPU."""
    total = torch.zeros_like(state)
    honest_sum = torch.zeros_like(state)
    for m in range(args.clients):
        poisoned = m < args.malicious
        d = engine.delta(state, t, m, poison=poisoned)
        if poisoned:
            if mask is not None:
                d = d * mask
            d = d * scale
        else:
            honest_sum.add_(d)
        total.add_(d)
        if store is not None and not poisoned:
            # Only honest uploads are read by the recovery estimators; malicious rows
            # are never stored, which keeps the host-memory footprint within the container limit.
            store[m].copy_(d.detach(), non_blocking=False)
    return total / args.clients, honest_sum / (args.clients - args.malicious)
