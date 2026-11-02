#!/bin/sh
# Neurotoxin-style robustness check on Fashion-MNIST held-out seeds with the corrected coordinate mask.
cd /root/autodl-tmp/darg_revision
for s in 1772 1773 1774 1775 1776 1777 1778 1779 1780; do
  /root/miniconda3/bin/python dr_recovery.py \
    --checkpoint results/holdout_fashionmnist_s$s/checkpoint_seed$s.pt \
    --out results/dr_neurotoxin_fmnist/s$s --delays 16 32 --continuation 16 --eval-every 4 \
    --poison-fraction 0.5 --scale 3 --neurotoxin 0.05 --threads 8 \
    --methods filter rollback S4 P4 V4 exact --continue-methods filter rollback S4 P4 V4 exact \
    > results/dr_neurotoxin_fmnist/s$s.log 2>&1
  echo "neurotoxin fmnist seed $s exit $?" >> results/dr_neurotoxin_fmnist/queue.log
done
echo DONE >> results/dr_neurotoxin_fmnist/queue.log
