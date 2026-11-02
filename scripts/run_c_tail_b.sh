#!/bin/sh
# Server C, after lane B: rerun two failed CIFAR-10 seeds, then the other half of partial 0.5.
cd /root/autodl-tmp/darg_revision
while ! grep -q "full cifar10 lane DONE (937" results/c_queue.log; do sleep 120; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sh run_c_full.sh cifar10 937 938
sh run_c_partial05.sh 937 938 939 940
