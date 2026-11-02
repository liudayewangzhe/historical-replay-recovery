#!/bin/sh
# Longer post-detection continuation (64 honest rounds) for the Fashion-MNIST durable backdoor,
# to test whether the post-recovery advantage of historical replay is transient.
cd /root/autodl-tmp/darg_revision
for s in 1772 1773 1774 1775 1776 1777 1778 1779 1780; do
  /root/miniconda3/bin/python dr_recovery.py \
    --checkpoint results/holdout_fashionmnist_s$s/checkpoint_seed$s.pt \
    --out results/dr_long_fmnist_s3/s$s --delays 16 32 --continuation 64 --eval-every 8 \
    --poison-fraction 0.5 --scale 3 --neurotoxin 0 --threads 8 \
    --methods filter rollback S4 P4 V4 exact --continue-methods filter rollback S4 P4 V4 exact \
    > results/dr_long_fmnist_s3/s$s.log 2>&1
  echo "long fmnist seed $s exit $?" >> results/dr_long_fmnist_s3/queue.log
done
echo DONE >> results/dr_long_fmnist_s3/queue.log
