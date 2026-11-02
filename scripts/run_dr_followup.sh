#!/bin/sh
# Follow-up queue: Fashion-MNIST durable backdoor (scale 3) on held-out validation seeds,
# then the newly trained CIFAR-10 and Fashion-MNIST seeds once their training queue has finished.
cd /root/autodl-tmp/darg_revision
PY=/root/miniconda3/bin/python
mkdir -p results/dr_followup
run_fmnist() {
  $PY dr_recovery.py --checkpoint results/holdout_fashionmnist_s$1/checkpoint_seed$1.pt \
    --out results/dr_main_fmnist_s3/s$1 --delays 8 16 32 64 --continuation 16 --eval-every 2 \
    --poison-fraction 0.5 --scale 3 --neurotoxin 0 --threads 8 > results/dr_main_fmnist_s3_s$1.log 2>&1
  echo "fmnist $1 exit $?" >> results/dr_followup/queue.log
}
run_cifar() {
  $PY dr_recovery.py --checkpoint results/holdout_cifar10_s$1/checkpoint_seed$1.pt \
    --out results/dr_main_cifar_a/s$1 --delays 8 16 --continuation 16 --eval-every 2 \
    --poison-fraction 0.5 --scale 1 --neurotoxin 0 --threads 8 > results/dr_main_cifar_a/s$1.log 2>&1
  echo "cifar $1 exit $?" >> results/dr_followup/queue.log
}
for s in 1772 1773 1774 1775; do run_fmnist $s; done
while ! grep -q DONE results/new_seeds_queue.log 2>/dev/null; do sleep 60; done
for s in 1776 1777 1778 1779 1780; do run_fmnist $s; done
while ! grep -q DONE results/dr_main_cifar_a/queue.log 2>/dev/null; do sleep 60; done
for s in 936 937 938 939 940; do run_cifar $s; done
echo DONE >> results/dr_followup/queue.log
