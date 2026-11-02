#!/bin/sh
# Server C: FedRecover-style schedule sensitivity on Fashion-MNIST (delays 16 and 32), after the full FMNIST lane.
cd /root/autodl-tmp/darg_revision
while ! grep -q "full fmnist lane DONE" results/c_queue.log; do sleep 120; done
mkdir -p results/frsens_fmnist
for s in 1772 1773 1774 1775 1776 1777 1778 1779 1780; do
  /root/miniconda3/bin/python dr_recovery.py --checkpoint results/holdout_fashionmnist_s$s/checkpoint_seed$s.pt \
    --out results/frsens_fmnist/s$s --delays 16 32 --continuation 16 --scale 3 --threads 8 \
    --methods rollback FedRecover_w1_p20 FedRecover_w3_p5 FedRecover_w2_p1000 exact \
    --continue-methods FedRecover_w1_p20 FedRecover_w3_p5 FedRecover_w2_p1000 exact > results/frsens_fmnist/s$s.log 2>&1
  echo "frsens fmnist $s exit $?" >> results/c_queue.log
done
echo "frsens DONE" >> results/c_queue.log
