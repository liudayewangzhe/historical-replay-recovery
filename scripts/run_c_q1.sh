#!/bin/sh
# Server C queue 1: FedEraser and Crab-style baselines on CIFAR-10 (delay 16), then partial participation 0.5.
cd /root/autodl-tmp/darg_revision
mkdir -p results/baselines_cifar10 results/partial_cifar10_r05
for s in 932 933 934 935 936; do
  /root/miniconda3/bin/python dr_recovery.py --checkpoint results/holdout_cifar10_s$s/checkpoint_seed$s.pt \
    --out results/baselines_cifar10/s$s --delays 16 --continuation 16 --eval-every 2 --scale 1 --threads 8 \
    --methods filter rollback P4 FedEraser Crab CrabStored exact \
    --continue-methods FedEraser Crab CrabStored exact > results/baselines_cifar10/s$s.log 2>&1
  echo "baselines cifar10 $s exit $?" >> results/c_queue.log
done
for s in 932 933 934 935 936 937 938 939 940; do
  /root/miniconda3/bin/python dr_partial.py --checkpoint results/holdout_cifar10_s$s/checkpoint_seed$s.pt \
    --out results/partial_cifar10_r05/s$s --rate 0.5 --delays 8 16 --continuation 16 --threads 8 \
    > results/partial_cifar10_r05/s$s.log 2>&1
  echo "partial0.5 cifar10 $s exit $?" >> results/c_queue.log
done
echo "Q1 DONE" >> results/c_queue.log
