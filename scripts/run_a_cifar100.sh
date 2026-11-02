#!/bin/sh
# Server A: CIFAR-100 third dataset. Train 950 (calibration) and 951-955, calibrate the backdoor scale on 950
# by a pre-registered rule, then recover 951-955 at delays 8 and 16 (60 GiB memory limit).
cd /root/autodl-tmp/darg_revision
PY=/root/miniconda3/bin/python
Q=results/a_cifar100_queue.log
METHODS="filter rollback S4 P4 FedRecover exact"
$PY -c "from torchvision.datasets import CIFAR100; CIFAR100('data', train=True, download=True); CIFAR100('data', train=False, download=True); print('cifar100 ok')" >> $Q 2>&1
for s in 950 951 952 953 954 955; do
  out="results/holdout_cifar100_s$s"
  $PY -u train_holdout.py --dataset cifar100 --data data --out "$out" --device cuda --seeds $s --clients 50 --malicious 10 --rounds 300 --local-epochs 1 --batch 64 --lr .03 --validation-fraction .1 --eval-interval 5 --target-accuracy .50 --initialization imagenet1k_gn > "$out.log" 2>&1
  echo "trained cifar100 $s exit $?" >> $Q
done
mkdir -p results/pilot_cifar100
for sc in 1 3; do
  $PY dr_pilot_durability.py --checkpoint results/holdout_cifar100_s950/checkpoint_seed950.pt --out results/pilot_cifar100/scale$sc --attack-rounds 32 --filter-rounds 32 --eval-every 4 --scale $sc --threads 8 > results/pilot_cifar100/scale$sc.log 2>&1
done
# Pre-registered rule: smallest scale whose ASR after 32 filtering rounds is at least 0.3; otherwise 3.
SCALE=$($PY -c "
import csv
for sc in (1, 3):
    rows = list(csv.DictReader(open(f'results/pilot_cifar100/scale{sc}/durability.csv')))
    if float(rows[-1]['asr']) >= 0.3:
        print(sc); break
else:
    print(3)")
echo "cifar100 calibrated scale $SCALE" >> $Q
mkdir -p results/cohort_cifar100
for s in 951 952 953 954 955; do
  $PY dr_recovery.py --checkpoint results/holdout_cifar100_s$s/checkpoint_seed$s.pt --out results/cohort_cifar100/s$s --delays 8 16 --continuation 16 --eval-every 4 --poison-fraction 0.5 --scale $SCALE --neurotoxin 0 --threads 8 --methods $METHODS --continue-methods $METHODS > results/cohort_cifar100/s$s.log 2>&1
  echo "recovered cifar100 $s exit $?" >> $Q
done
echo DONE >> $Q
