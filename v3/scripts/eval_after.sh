#!/bin/bash
# eval_after.sh <gpu> <exp-dir> [评测参数...]: 等当前评测进程结束后跑一次评测 (命令行里不含评测模块名, 不会挡住 watcher)
GPU=$1; D=$2; shift 2
cd /data/neko/cxip || exit 1
PY=/home/ubuntu/miniconda3/envs/cxip/bin/python
sleep $((RANDOM % 20))
while pgrep -f "^$PY -m v3[.]eval_exp" > /dev/null; do sleep 20; done
LOG=/data/neko/cxip_v3/logs/eval/after_$(basename "$D")_$(date +%H%M%S).log
CUDA_VISIBLE_DEVICES=$GPU $PY -m v3.eval_exp "$D" "$@" > "$LOG" 2>&1
grep '^{' "$LOG" | tail -n 1 | cut -c1-400
