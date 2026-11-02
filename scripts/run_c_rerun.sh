#!/bin/sh
# Server C: rerun CIFAR-10 seeds that failed in the merged lanes, after lane B has finished.
cd /root/autodl-tmp/darg_revision
while ! grep -q "full cifar10 lane DONE (937" results/c_queue.log; do sleep 120; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in $(grep "full cifar10 .* exit [1-9]" results/c_queue.log | awk '{print $3}' | sort -u); do
  sh run_c_full.sh cifar10 $s
done
echo "rerun DONE" >> results/c_queue.log
