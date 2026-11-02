#!/bin/sh
# Server C: waits for a marker line in the queue log, then runs the given command.
cd /root/autodl-tmp/darg_revision
marker=$1; shift
while ! grep -q "$marker" results/c_queue.log; do sleep 120; done
"$@"
