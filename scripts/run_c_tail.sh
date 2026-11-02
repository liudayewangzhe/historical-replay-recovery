#!/bin/sh
# Server C, after lane A: rerun the CIFAR-10 seeds that failed (GPU OOM or stopped), then half of partial 0.5.
cd /root/autodl-tmp/darg_revision
while ! grep -q "full cifar10 lane DONE (932" results/c_queue.log; do sleep 120; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
sh run_c_full.sh cifar10 934 935
sh run_c_partial05.sh 932 933 934 935 936
