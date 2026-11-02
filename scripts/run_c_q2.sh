#!/bin/sh
# Server C queue 2: baselines on the remaining CIFAR-10 seeds, then on Fashion-MNIST (scale 3, delays 16 and 32).
cd /root/autodl-tmp/darg_revision
mkdir -p results/baselines_cifar10 results/baselines_fmnist
for s in 937 938 939 940; do
  /root/miniconda3/bin/python dr_recovery.py --checkpoint results/holdout_cifar10_s$s/checkpoint_seed$s.pt \
    --out results/baselines_cifar10/s$s --delays 16 --continuation 16 --eval-every 2 --scale 1 --threads 8 \
    --methods filter rollback P4 FedEraser Crab CrabStored exact \
    --continue-methods FedEraser Crab CrabStored exact > results/baselines_cifar10/s$s.log 2>&1
  echo "baselines cifar10 $s exit $?" >> results/c_queue.log
done
for s in 1772 1773 1774 1775 1776 1777 1778 1779 1780; do
  /root/miniconda3/bin/python dr_recovery.py --checkpoint results/holdout_fashionmnist_s$s/checkpoint_seed$s.pt \
    --out results/baselines_fmnist/s$s --delays 16 32 --continuation 16 --eval-every 2 --scale 3 --threads 8 \
    --methods filter rollback P4 FedEraser Crab CrabStored exact \
    --continue-methods FedEraser Crab CrabStored exact > results/baselines_fmnist/s$s.log 2>&1
  echo "baselines fmnist $s exit $?" >> results/c_queue.log
done
echo "Q2 DONE" >> results/c_queue.log
