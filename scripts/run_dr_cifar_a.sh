#!/bin/sh
# Durable shared-patch backdoor (scale 1), CIFAR-10 validation-selected seeds 931-935.
cd /root/autodl-tmp/darg_revision
for s in 931 932 933 934 935; do
  /root/miniconda3/bin/python dr_recovery.py \
    --checkpoint results/holdout_cifar10_s$s/checkpoint_seed$s.pt \
    --out results/dr_main_cifar_a/s$s --delays 8 16 --continuation 16 --eval-every 2 \
    --poison-fraction 0.5 --scale 1 --neurotoxin 0 --threads 8 \
    > results/dr_main_cifar_a/s$s.log 2>&1
  echo "seed $s exit $?" >> results/dr_main_cifar_a/queue.log
done
echo DONE >> results/dr_main_cifar_a/queue.log
