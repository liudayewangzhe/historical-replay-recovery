"""Fixed, class-stratified train/validation split of official training data."""
import numpy as np
import torch


CIFAR10_MEAN = (.4914, .4822, .4465)
CIFAR10_STD = (.247, .243, .261)
CIFAR100_MEAN = (.5071, .4865, .4409)
CIFAR100_STD = (.2673, .2564, .2762)


def datasets(name, root):
    from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST
    cls = {'cifar10': CIFAR10, 'cifar100': CIFAR100, 'fashionmnist': FashionMNIST}[name]
    return cls(root, train=True, download=True), cls(root, train=False, download=True)


def stratified_indices(labels, seed, validation_fraction=.1, examples_per_class=0):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train = []
    validation = []
    # Iterating over the observed labels reproduces the original order for 10-class data.
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if examples_per_class:
            indices = indices[:examples_per_class]
        n_val = int(round(len(indices) * validation_fraction))
        validation.extend(indices[:n_val])
        train.extend(indices[n_val:])
    return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def tensors(dataset, indices, device):
    labels = np.asarray(dataset.targets, dtype=np.int64)
    images = np.asarray(dataset.data)
    if indices is not None:
        labels = labels[indices]
        images = images[indices]
    x = torch.as_tensor(images.copy())
    x = x.unsqueeze(1) if x.ndim == 3 else x.permute(0, 3, 1, 2)
    return x.float().div_(255).to(device), torch.as_tensor(labels.copy(), device=device)


def normalization(name):
    """Per-channel mean and std for colour datasets; None for grayscale data."""
    return {'cifar10': (CIFAR10_MEAN, CIFAR10_STD), 'cifar100': (CIFAR100_MEAN, CIFAR100_STD)}.get(name)


def normalize_cifar(x):
    mean = torch.tensor(CIFAR10_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=x.device).view(1, 3, 1, 1)
    x.sub_(mean).div_(std)


def normalize_for(name, x):
    stats = normalization(name)
    if stats is None:
        return
    mean = torch.tensor(stats[0], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(stats[1], device=x.device).view(1, 3, 1, 1)
    x.sub_(mean).div_(std)
