#!/bin/sh
# Server C queue 4 (starts when queue 2 has left the CIFAR-10 baselines, to stay inside the memory limit):
# rerun of the OOM-killed baseline seed, then the matched-cost study on CIFAR-10 (delay 16):
# sampled/paired budgets, server fine-tuning, and latency-matched continuation of zero-round methods.
cd /root/autodl-tmp/darg_revision
while ! grep -q "baselines cifar10 940" results/c_queue.log; do sleep 120; done
mkdir -p results/baselines_cifar10 results/matched_cifar10
/root/miniconda3/bin/python dr_recovery.py --checkpoint results/holdout_cifar10_s932/checkpoint_seed932.pt \
  --out results/baselines_cifar10/s932 --delays 16 --continuation 16 --eval-every 2 --scale 1 --threads 8 \
  --methods filter rollback P4 FedEraser Crab CrabStored exact \
  --continue-methods FedEraser Crab CrabStored exact > results/baselines_cifar10/s932.log 2>&1
echo "baselines cifar10 932 rerun exit $?" >> results/c_queue.log
for s in 932 933 934 935 936 937 938 939 940; do
  /root/miniconda3/bin/python dr_recovery.py --checkpoint results/holdout_cifar10_s$s/checkpoint_seed$s.pt \
    --out results/matched_cifar10/s$s --delays 16 --continuation 16 --eval-every 2 --scale 1 --threads 8 \
    --methods filter rollback S2 S4 S8 P2 P4 P8 FT1 FT5 exact \
    --continue-methods filter rollback S4 P4 FT1 FT5 exact --latency-matched > results/matched_cifar10/s$s.log 2>&1
  echo "matched cifar10 $s exit $?" >> results/c_queue.log
done
echo "Q4 DONE" >> results/c_queue.log
