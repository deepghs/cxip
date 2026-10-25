#!/bin/bash
# 自动评测: 每 60s 扫描 <exp-root>, 对有 DONE 且没有 eval/suite.json 的实验跑 v3.eval_exp (逐个, 与训练共用一张卡)。
# eval_watch.sh <exp-root> [gpu=4]
ROOT=$1; GPU=${2:-4}
cd /data/neko/cxip || exit 1
while true; do
  for d in "$ROOT"/*/; do
    d=${d%/}; n=$(basename "$d")
    [[ "$n" == _* || ! -f "$d/DONE" || -f "$d/eval/suite.json" || -f "$d/eval/FAILED" ]] && continue
    echo "$(date +%T) eval $n"
    mkdir -p "$d/eval"
    CUDA_VISIBLE_DEVICES=$GPU /home/ubuntu/miniconda3/envs/cxip/bin/python -m v3.eval_exp "$d" > "$d/eval/stdout.log" 2>&1 \
      || touch "$d/eval/FAILED"
    grep '^{' "$d/eval/stdout.log" | tail -n 1
  done
  sleep 60
done
