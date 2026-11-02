#!/bin/sh
# Train additional validation-selected checkpoints as held-out seeds (same protocol as seeds 931-935 and 1771-1775).
cd /root/autodl-tmp/darg_revision
PY=/root/miniconda3/bin/python
for seed in 936 937 938 939 940; do
  train="results/holdout_cifar10_s${seed}"
  $PY -u train_holdout.py --dataset cifar10 --data data --out "$train" --device cuda --seeds "$seed" --clients 50 --malicious 10 --rounds 400 --local-epochs 1 --batch 64 --lr .03 --validation-fraction .1 --eval-interval 5 --target-accuracy .75 --initialization imagenet1k_gn > "${train}.log" 2>&1
  $PY check_holdout.py "$train/checkpoint_seed${seed}.pt" --data data > "$train/check.log" 2>&1
  echo "cifar seed $seed done" >> results/new_seeds_queue.log
done
for seed in 1776 1777 1778 1779 1780; do
  train="results/holdout_fashionmnist_s${seed}"
  $PY -u train_holdout.py --dataset fashionmnist --data data --out "$train" --device cuda --seeds "$seed" --clients 50 --malicious 10 --rounds 400 --local-epochs 1 --batch 64 --lr .1 --validation-fraction .1 --eval-interval 5 --target-accuracy .85 --initialization random > "${train}.log" 2>&1
  $PY check_holdout.py "$train/checkpoint_seed${seed}.pt" --data data > "$train/check.log" 2>&1
  echo "fmnist seed $seed done" >> results/new_seeds_queue.log
done
echo DONE >> results/new_seeds_queue.log
