#!/bin/sh
# Server C queue 3: partial participation 0.2 on CIFAR-10 held-out seeds.
cd /root/autodl-tmp/darg_revision
mkdir -p results/partial_cifar10_r02
for s in 932 933 934 935 936 937 938 939 940; do
  /root/miniconda3/bin/python dr_partial.py --checkpoint results/holdout_cifar10_s$s/checkpoint_seed$s.pt \
    --out results/partial_cifar10_r02/s$s --rate 0.2 --delays 8 16 --continuation 16 --threads 8 \
    > results/partial_cifar10_r02/s$s.log 2>&1
  echo "partial0.2 cifar10 $s exit $?" >> results/c_queue.log
done
echo "Q3 DONE" >> results/c_queue.log
