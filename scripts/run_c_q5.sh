#!/bin/sh
# Server C queue 5: matched-cost study on Fashion-MNIST (scale 3, delays 16 and 32); small memory footprint.
cd /root/autodl-tmp/darg_revision
mkdir -p results/matched_fmnist
for s in 1772 1773 1774 1775 1776 1777 1778 1779 1780; do
  /root/miniconda3/bin/python dr_recovery.py --checkpoint results/holdout_fashionmnist_s$s/checkpoint_seed$s.pt \
    --out results/matched_fmnist/s$s --delays 16 32 --continuation 16 --eval-every 2 --scale 3 --threads 8 \
    --methods filter rollback S2 S4 S8 P2 P4 P8 FT1 FT5 exact \
    --continue-methods filter rollback S4 P4 FT1 FT5 exact --latency-matched > results/matched_fmnist/s$s.log 2>&1
  echo "matched fmnist $s exit $?" >> results/c_queue.log
done
echo "Q5 DONE" >> results/c_queue.log
