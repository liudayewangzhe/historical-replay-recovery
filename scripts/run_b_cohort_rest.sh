#!/bin/sh
# Server B: finish the CIFAR-10 replication cohort (942-945) after seed 941 completes. CIFAR-100 moved to server A.
cd /root/autodl-tmp/darg_revision
PY=/root/miniconda3/bin/python
Q=results/b_phase2_queue.log
METHODS="filter rollback S4 P4 FedRecover exact"
while pgrep -f "dr_recovery.py --checkpoint results/holdout_cifar10_s941" > /dev/null; do sleep 30; done
echo "recovered cifar10 941 (runner replaced)" >> $Q
for s in 942 943 944 945; do
  $PY dr_recovery.py --checkpoint results/holdout_cifar10_s$s/checkpoint_seed$s.pt --out results/cohort_cifar10/s$s --delays 16 32 --continuation 16 --eval-every 4 --poison-fraction 0.5 --scale 1 --neurotoxin 0 --threads 8 --methods $METHODS --continue-methods $METHODS > results/cohort_cifar10/s$s.log 2>&1
  echo "recovered cifar10 $s exit $?" >> $Q
done
echo DONE >> $Q
