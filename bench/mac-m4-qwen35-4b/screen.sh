#!/bin/bash
# Flag screening on 1x1024 and 4x1024, one server launch per config (the s0..s8 runs in results/).
set -u
cd "$(dirname "$0")"
run() { echo "=== $1 $(date +%T)"; timeout 1200 python3 run.py "$@" 2>&1 | grep -E "Mac busy|Error|error" ; }
run s0_base screen -- -fa on
run s1_ctxcp0 screen -- -fa on -ctxcp 0
run s2_ub1024 screen -- -fa on -ctxcp 0 -b 4096 -ub 1024
run s3_ub2048 screen -- -fa on -ctxcp 0 -b 4096 -ub 2048
run s4_ub4096 screen -- -fa on -ctxcp 0 -b 4096 -ub 4096
run s5_fa_off screen -- -fa off -ctxcp 0
run s6_kvu screen -- -fa on -ctxcp 0 -kvu
METAL_ENV="GGML_METAL_TENSOR_ENABLE=1" run s7_tensor screen -- -fa on -ctxcp 0
run s8_rep_ctxcp0 screen -- -fa on -ctxcp 0
echo done
