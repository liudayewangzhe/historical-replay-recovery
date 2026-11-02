#!/bin/sh
# Heterogeneity sensitivity: Fashion-MNIST with Dirichlet alpha 0.1, same backdoor (scale 3, not re-tuned).
# Usage: sh run_hetero.sh SEED [SEED ...]
cd /root/autodl-tmp/darg_revision
PY=/root/miniconda3/bin/python
Q=results/hetero_queue.log
METHODS="filter rollback S4 P4 exact"
mkdir -p results/hetero_fmnist_a01
for s in "$@"; do
  out="results/holdout_fashionmnist_a01_s$s"
  $PY -u train_holdout.py --dataset fashionmnist --data data --out "$out" --device cuda --seeds $s --clients 50 --malicious 10 --alpha 0.1 --rounds 400 --local-epochs 1 --batch 64 --lr .1 --validation-fraction .1 --eval-interval 5 --target-accuracy .80 --initialization random > "$out.log" 2>&1
  echo "trained fmnist a0.1 $s exit $?" >> $Q
  $PY dr_recovery.py --checkpoint $out/checkpoint_seed$s.pt --out results/hetero_fmnist_a01/s$s --delays 16 32 --continuation 16 --eval-every 4 --poison-fraction 0.5 --scale 3 --neurotoxin 0 --threads 8 --methods $METHODS --continue-methods $METHODS > results/hetero_fmnist_a01/s$s.log 2>&1
  echo "recovered fmnist a0.1 $s exit $?" >> $Q
done
echo DONE >> $Q
