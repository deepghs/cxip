#!/bin/bash
# 自动评测: eval_watch2.sh <gpu> <exp-root> [more roots...]
# 每 60s 扫描各 root, 对有 DONE 且没有 eval/suite.json 的实验逐个跑 v3.eval_exp; 已有评测进程时先等它结束。
GPU=$1; shift
cd /data/neko/cxip || exit 1
while true; do
  for root in "$@"; do
    for d in "$root"/*/; do
      d=${d%/}; n=$(basename "$d")
      [[ "$n" == _* || ! -f "$d/DONE" || -f "$d/eval/suite.json" || -f "$d/eval/FAILED" ]] && continue
      while pgrep -f "python -m v3.eval_exp" > /dev/null; do sleep 30; done
      [[ -f "$d/eval/suite.json" ]] && continue
      echo "$(date +%T) eval $root/$n"
      mkdir -p "$d/eval"
      CUDA_VISIBLE_DEVICES=$GPU /home/ubuntu/miniconda3/envs/cxip/bin/python -m v3.eval_exp "$d" > "$d/eval/stdout.log" 2>&1 \
        || touch "$d/eval/FAILED"
      grep '^{' "$d/eval/stdout.log" | tail -n 1 | cut -c1-400
    done
  done
  sleep 60
done
