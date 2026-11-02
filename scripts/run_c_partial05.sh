#!/bin/sh
# Server C: partial participation 0.5 on the given CIFAR-10 held-out seeds.
cd /root/autodl-tmp/darg_revision
mkdir -p results/partial_cifar10_r05
for s in "$@"; do
  /root/miniconda3/bin/python dr_partial.py --checkpoint results/holdout_cifar10_s$s/checkpoint_seed$s.pt \
    --out results/partial_cifar10_r05/s$s --rate 0.5 --delays 8 16 --continuation 16 --threads 8 \
    > results/partial_cifar10_r05/s$s.log 2>&1
  echo "partial0.5 cifar10 $s exit $?" >> results/c_queue.log
done
echo "partial0.5 DONE ($*)" >> results/c_queue.log
