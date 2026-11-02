#!/bin/sh
# Server C: Fashion-MNIST robustness runs (small memory).
# (1) late onset estimate: recovery restarts k rounds after the true onset (delay 32);
# (2) two further replay-sample draws per seed (delays 16 and 32).
cd /root/autodl-tmp/darg_revision
mkdir -p results/late_onset_fmnist results/draws_fmnist
for s in 1772 1773 1774 1775 1776 1777 1778 1779 1780; do
  ck=results/holdout_fashionmnist_s$s/checkpoint_seed$s.pt
  for k in 2 4 8; do
    /root/miniconda3/bin/python dr_recovery.py --checkpoint $ck --out results/late_onset_fmnist/k$k/s$s \
      --delays 32 --continuation 16 --scale 3 --start-offset $k --threads 8 \
      --methods filter rollback S4 P4 exact --continue-methods rollback S4 P4 exact \
      > results/late_onset_fmnist/k${k}_s$s.log 2>&1
    echo "late k$k fmnist $s exit $?" >> results/c_queue.log
  done
  for r in 1 2; do
    /root/miniconda3/bin/python dr_recovery.py --checkpoint $ck --out results/draws_fmnist/r$r/s$s \
      --delays 16 32 --continuation 16 --scale 3 --replay-seed-offset $r --threads 8 \
      --methods filter rollback S4 P4 exact --continue-methods S4 P4 exact \
      > results/draws_fmnist/r${r}_s$s.log 2>&1
    echo "draws r$r fmnist $s exit $?" >> results/c_queue.log
  done
done
echo "fmnist extra DONE" >> results/c_queue.log
