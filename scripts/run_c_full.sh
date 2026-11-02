#!/bin/sh
# Server C: one merged run per seed with every compared method, delay 16 for CIFAR-10 and
# delays 16 and 32 for Fashion-MNIST. Zero-round methods also continue for delay-1 extra rounds
# (latency matching). Usage: sh run_c_full.sh <dataset> <seed> [<seed> ...]
cd /root/autodl-tmp/darg_revision
data=$1; shift
METHODS="filter rollback S2 S4 S8 P2 P4 P8 FedRecover FedEraser Crab CrabStored FT1 FT5 exact"
CONT="filter rollback S4 P4 FedRecover FedEraser Crab CrabStored FT1 FT5 exact"
for s in "$@"; do
  if [ "$data" = cifar10 ]; then
    ck=results/holdout_cifar10_s$s/checkpoint_seed$s.pt; out=results/full_cifar10; opts="--delays 16 --scale 1"
  else
    ck=results/holdout_fashionmnist_s$s/checkpoint_seed$s.pt; out=results/full_fmnist; opts="--delays 16 32 --scale 3"
  fi
  mkdir -p $out
  /root/miniconda3/bin/python dr_recovery.py --checkpoint $ck --out $out/s$s $opts --continuation 16 --eval-every 2 \
    --threads 8 --methods $METHODS --continue-methods $CONT --latency-matched > $out/s$s.log 2>&1
  echo "full $data $s exit $?" >> results/c_queue.log
done
echo "full $data lane DONE ($*)" >> results/c_queue.log
