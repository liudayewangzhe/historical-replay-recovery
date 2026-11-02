"""Controlled-delay recovery experiment; all reported values come from this run."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class SmallCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AvgPool2d(2), nn.Flatten(),
            nn.Linear(32 * (3 if channels == 1 else 4) ** 2, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.net(x)


def load_data(args, device):
    if args.smoke:
        gen = torch.Generator().manual_seed(909)
        x = torch.rand(400, 1, 28, 28, generator=gen)
        y = torch.arange(400) % 10
        # A learnable location and intensity signature, not a benchmark result.
        for i in range(len(x)):
            x[i, 0, (int(y[i]) // 5) * 5: (int(y[i]) // 5) * 5 + 3,
              (int(y[i]) % 5) * 3: (int(y[i]) % 5) * 3 + 2] = 2
        return x[:300].to(device), y[:300].to(device), x[300:].to(device), y[300:].to(device)
    from torchvision.datasets import FashionMNIST, CIFAR10
    cls = FashionMNIST if args.dataset == 'fashionmnist' else CIFAR10
    train, test = cls(args.data, train=True, download=True), cls(args.data, train=False, download=True)
    def convert(ds):
        x = torch.as_tensor(np.asarray(ds.data).copy())
        x = x.unsqueeze(1) if x.ndim == 3 else x.permute(0, 3, 1, 2)
        return x.float().div_(255), torch.as_tensor(ds.targets).long()
    x, y = convert(train)
    tx, ty = convert(test)
    rng = np.random.default_rng(12345)
    take = rng.permutation(len(y))[:int(len(y) * args.fraction)]
    x, y = x[take], y[take]
    if args.test_samples:
        take = rng.permutation(len(ty))[:args.test_samples]
        tx, ty = tx[take], ty[take]
    return x.to(device), y.to(device), tx.to(device), ty.to(device)


def partition(labels, clients, alpha, seed):
    rng = np.random.default_rng(seed + 3101)
    parts = [[] for _ in range(clients)]
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        rng.shuffle(ids)
        proportions = rng.dirichlet(np.full(clients, alpha))
        split = np.split(ids, (np.cumsum(proportions)[:-1] * len(ids)).astype(int))
        for m, ids_m in enumerate(split):
            parts[m].extend(ids_m.tolist())
    # Dirichlet draws may leave a client empty; move one existing sample.
    for part in parts:
        if not part:
            donor = max(parts, key=len)
            part.append(donor.pop())
    return [np.asarray(p, dtype=np.int64) for p in parts]


def make_schedule(parts, args, seed, device):
    rng = np.random.default_rng(seed + 7201)
    return [[torch.tensor(rng.choice(p, size=(args.local_steps, args.batch), replace=len(p) < args.batch),
                          dtype=torch.long, device=device) for p in parts]
            for _ in range(args.rounds)]


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


class Engine:
    def __init__(self, args, x, y, tx, ty, schedule):
        self.args, self.x, self.y, self.tx, self.ty, self.schedule = args, x, y, tx, ty, schedule
        self.model = SmallCNN(x.shape[1]).to(x.device)
        self.loss = nn.CrossEntropyLoss()
        self.calls = 0

    def vector(self):
        return parameters_to_vector(self.model.parameters()).detach().clone()

    def delta(self, w, t, m, poison=False):
        vector_to_parameters(w.clone(), self.model.parameters())
        self.model.train()
        for ids in self.schedule[t][m]:
            bx, by = self.x[ids], self.y[ids]
            if poison:
                bx, by = bx.clone(), by.clone()
                n = max(1, int(len(by) * self.args.poison_fraction))
                bx[:n, :, -3:, -3:] = 1
                by[:n] = self.args.target
            self.model.zero_grad(set_to_none=True)
            self.loss(self.model(bx), by).backward()
            with torch.no_grad():
                for p in self.model.parameters():
                    p.add_(p.grad, alpha=-self.args.lr)
        self.calls += 1
        return w - self.vector()

    def aggregate(self, w, t, clients):
        return torch.stack([self.delta(w, t, m) for m in clients]).mean(0)

    @torch.no_grad()
    def predictions(self, w, trigger=False):
        vector_to_parameters(w.clone(), self.model.parameters())
        self.model.eval()
        preds = []
        for x in self.tx.split(512):
            if trigger:
                x = x.clone()
                x[:, :, -3:, -3:] = 1
            preds.append(self.model(x).argmax(1))
        return torch.cat(preds)

    def metrics(self, w, reference, attack):
        pred = self.predictions(w)
        acc = (pred == self.ty).float().mean().item()
        disagreement = (pred != self.predictions(reference)).float().mean().item()
        asr = float('nan')
        if attack == 'backdoor':
            eligible = self.ty != self.args.target
            asr = (self.predictions(w, True)[eligible] == self.args.target).float().mean().item()
        return {'accuracy': acc, 'asr': asr, 'disagreement': disagreement,
                'parameter_residual': torch.linalg.vector_norm(w - reference).item()}


def secant_transport(history, j, displacement, args):
    """Regularized aggregate secant map, with exact low-rank operator-norm cap."""
    if len(history) < 2 or torch.linalg.vector_norm(displacement).item() == 0:
        return torch.zeros_like(displacement), 0.0
    # The complete retained buffer is available at detection. Use pairs nearest j.
    pairs = sorted(range(len(history) - 1), key=lambda k: abs(k - j))[:args.rank]
    s = torch.stack([history[k+1]['w'] - history[k]['w'] for k in pairs], 1)
    y = torch.stack([history[k+1]['h'] - history[k]['h'] for k in pairs], 1)
    gram = s.T @ s
    scale = torch.trace(gram).item() / len(pairs)
    ridge = args.ridge * max(scale, 1e-12)
    system = gram + ridge * torch.eye(len(pairs), device=s.device)
    # QR reduces the d-by-d operator norm to a rank-by-rank SVD.
    _, rs = torch.linalg.qr(s, mode='reduced')
    _, ry = torch.linalg.qr(y, mode='reduced')
    norm = torch.linalg.matrix_norm(ry @ torch.linalg.solve(system, rs.T), ord=2).item()
    factor = min(1.0, args.jacobian_cap / max(norm, 1e-12))
    return factor * (y @ torch.linalg.solve(system, s.T @ displacement)), norm * factor


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    x, y, tx, ty = load_data(args, device)
    metadata = dict(vars(args), torch=torch.__version__, python=platform.python_version(),
                    gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None,
                    train_samples=len(y), test_samples=len(ty),
                    protocol='controlled_oracle_detection_full_participation',
                    update='w_minus_local_SGD_model; equal_client_weights',
                    timing='synchronized_server_recovery_and_client_compute_wall_seconds')
    (out / 'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    results, trajectories = [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        parts = partition(y.cpu().numpy(), args.clients, args.alpha, seed)
        schedule = make_schedule(parts, args, seed, device)
        engine = Engine(args, x, y, tx, ty, schedule)
        all_clients = list(range(args.clients))
        honest = list(range(args.malicious, args.clients))
        start = engine.vector()
        for t in range(args.onset):
            start -= engine.aggregate(start, t, all_clients)
        # A no-attack/all-client control is distinct from the exclusion target.
        clean = start.clone()
        clean_states = {args.onset: clean.clone()}
        for t in range(args.onset, args.rounds):
            clean -= engine.aggregate(clean, t, all_clients)
            if (t+1-args.onset in args.delays) or t+1 == args.rounds:
                clean_states[t+1] = clean.clone()
        for attack in args.attacks:
            # One attacked prefix up to max delay is shared by the delay sweep.
            attacked = start.clone()
            full_history = []
            for t in range(args.onset, args.onset + max(args.delays)):
                deltas = [engine.delta(attacked, t, m,
                          poison=attack == 'backdoor' and m < args.malicious)
                          for m in all_clients]
                h = torch.stack(deltas[args.malicious:]).mean(0)
                if attack == 'signflip':
                    for m in range(args.malicious):
                        deltas[m] = -args.sign_scale * deltas[m]
                elif attack == 'ipm':
                    for m in range(args.malicious):
                        deltas[m] = -args.ipm_epsilon * h
                elif attack == 'backdoor':
                    for m in range(args.malicious):
                        deltas[m] *= args.backdoor_scale
                full_history.append({'w': attacked.clone(), 'h': h.clone(),
                                     'updates': torch.stack(deltas)})
                attacked -= torch.stack(deltas).mean(0)
                full_history[-1]['after'] = attacked.clone()
            for delay in args.delays:
                end = args.onset + delay
                history = full_history[:delay]
                corrupt = history[-1]['after'].clone()
                # Exact replay recomputes honest updates along the exclusion path.
                exact = start.clone()
                calls0 = engine.calls
                sync(device)
                clock0 = time.perf_counter()
                for j, t in enumerate(range(args.onset, end)):
                    exact -= engine.aggregate(exact, t, honest)
                sync(device)
                exact_seconds = time.perf_counter() - clock0
                exact_calls = engine.calls - calls0
                recovered = {'filter': corrupt.clone(), 'rollback': start.clone()}
                elapsed = {'filter': 0.0, 'rollback': 0.0}
                norms = []
                for method in ['E1', 'E2']:
                    sync(device)
                    clock0 = time.perf_counter()
                    v = start.clone()
                    for j, record in enumerate(history):
                        correction = torch.zeros_like(v)
                        if method == 'E2':
                            correction, op_norm = secant_transport(history, j, v - record['w'], args)
                            norms.append(op_norm)
                        v -= record['h'] + correction
                    sync(device)
                    elapsed[method] = time.perf_counter() - clock0
                    recovered[method] = v
                recovered['exact'] = exact.clone()
                elapsed['exact'] = exact_seconds
                recovered['clean'] = clean_states[end].clone()
                elapsed['clean'] = 0.0
                denom = torch.linalg.vector_norm(corrupt-exact).item()
                buffer_bytes = delay * (args.clients + 1) * len(start) * start.element_size()
                base = {'dataset': args.dataset if not args.smoke else 'synthetic_smoke',
                        'seed': seed, 'attack': attack, 'delay': delay,
                        'onset': args.onset, 'detected_round': end,
                        'buffer_bytes': buffer_bytes, 'parameters': len(start)}
                for method, state in recovered.items():
                    metrics = engine.metrics(state, exact, attack)
                    row = dict(base, method=method, stage='recovery', round=end,
                               recovery_fraction=(1 - metrics['parameter_residual']/denom) if denom else float('nan'),
                               recovery_seconds=elapsed[method],
                               extra_client_updates=exact_calls if method == 'exact' else 0,
                               max_secant_operator_norm=max(norms, default=0.0) if method == 'E2' else 0.0,
                               **metrics)
                    results.append(row)
                    trajectories.append(dict(base, method=method, round=end, **metrics))
                # All recovered paths use the same honest schedule after detection.
                # The clean all-client path was computed once above.
                recovered.pop('clean')
                for t in range(end, args.rounds):
                    for method in recovered:
                        recovered[method] -= engine.aggregate(recovered[method], t, honest)
                    if (t+1-end) % args.eval_every == 0 or t+1 == args.rounds:
                        for method, state in recovered.items():
                            trajectories.append(dict(base, method=method, round=t+1,
                                **engine.metrics(state, recovered['exact'], attack)))
                final_ref = recovered['exact']
                recovered['clean'] = clean_states[args.rounds]
                for method, state in recovered.items():
                    metrics = engine.metrics(state, final_ref, attack)
                    results.append(dict(base, method=method, stage='final', round=args.rounds,
                               recovery_fraction=float('nan'), recovery_seconds=elapsed[method],
                               extra_client_updates=exact_calls if method == 'exact' else 0,
                               max_secant_operator_norm=max(norms, default=0.0) if method == 'E2' else 0.0,
                               **metrics))
                write_csv(out / 'results.csv', results)
                write_csv(out / 'trajectories.csv', trajectories)
                print(json.dumps({'seed': seed, 'attack': attack, 'delay': delay,
                       'final_accuracy': {m: round(r['accuracy'], 4) for m, r in
                         [(row['method'], row) for row in results[-6:]]},
                       'rows_saved': len(results)}), flush=True)
    return results


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['fashionmnist', 'cifar10', 'cifar100'], default='fashionmnist')
    p.add_argument('--data', default='./data')
    p.add_argument('--out', default='./results')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    p.add_argument('--attacks', nargs='+', choices=['signflip', 'ipm', 'backdoor'], default=['signflip','ipm','backdoor'])
    p.add_argument('--delays', type=int, nargs='+', default=[2, 4, 8])
    p.add_argument('--rounds', type=int, default=80)
    p.add_argument('--onset', type=int, default=20)
    p.add_argument('--clients', type=int, default=20)
    p.add_argument('--malicious', type=int, default=4)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--fraction', type=float, default=0.25)
    p.add_argument('--test-samples', type=int, default=0)
    p.add_argument('--local-steps', type=int, default=2)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--sign-scale', type=float, default=5.0)
    p.add_argument('--ipm-epsilon', type=float, default=5.0)
    p.add_argument('--poison-fraction', type=float, default=0.5)
    p.add_argument('--backdoor-scale', type=float, default=5.0)
    p.add_argument('--target', type=int, default=0)
    p.add_argument('--rank', type=int, default=5)
    p.add_argument('--ridge', type=float, default=0.1)
    p.add_argument('--jacobian-cap', type=float, default=0.5)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.smoke:
        args.seeds, args.attacks, args.delays = [7], ['signflip'], [1, 2]
        args.rounds, args.onset, args.clients, args.malicious = 5, 1, 4, 1
        args.local_steps, args.batch, args.eval_every = 1, 16, 1
    run(args)
