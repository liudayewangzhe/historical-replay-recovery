#!/bin/sh
# Server B, phase 2: shortened CIFAR-10 replication (941-945, delays 16/32) then CIFAR-100 (calibration 950, main 951-955).
cd /root/autodl-tmp/darg_revision
PY=/root/miniconda3/bin/python
Q=results/b_phase2_queue.log
METHODS="filter rollback S4 P4 FedRecover exact"
while pgrep -f "train_holdout.py --dataset cifar10" > /dev/null; do sleep 30; done
train() {  # dataset seed target rounds
  out="results/holdout_$1_s$2"
  $PY -u train_holdout.py --dataset $1 --data data --out "$out" --device cuda --seeds $2 --clients 50 --malicious 10 --rounds $4 --local-epochs 1 --batch 64 --lr .03 --validation-fraction .1 --eval-interval 5 --target-accuracy $3 --initialization imagenet1k_gn > "$out.log" 2>&1
  echo "trained $1 $2 exit $?" >> $Q
}
recover() {  # dataset seed scale outdir
  $PY dr_recovery.py --checkpoint results/holdout_$1_s$2/checkpoint_seed$2.pt --out results/$4/s$2 --delays 16 32 --continuation 16 --eval-every 4 --poison-fraction 0.5 --scale $3 --neurotoxin 0 --threads 8 --methods $METHODS --continue-methods $METHODS > results/$4/s$2.log 2>&1
  echo "recovered $1 $2 scale $3 exit $?" >> $Q
}
for s in 943 944 945; do train cifar10 $s .75 400; done
mkdir -p results/cohort_cifar10
for s in 941 942 943 944 945; do recover cifar10 $s 1 cohort_cifar10; done
# CIFAR-100
$PY -c "from torchvision.datasets import CIFAR100; CIFAR100('data', train=True, download=True); CIFAR100('data', train=False, download=True)" >> $Q 2>&1
for s in 950 951 952 953 954 955; do train cifar100 $s .50 300; done
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
for s in 951 952 953 954 955; do recover cifar100 $s $SCALE cohort_cifar100; done
echo DONE >> $Q
