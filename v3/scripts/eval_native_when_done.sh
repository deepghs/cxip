#!/bin/bash
# eval_native_when_done.sh <gpu> <exp-dir>...: 按顺序等每个实验出现 DONE (FAILED 则跳过), 再跑 eval_native.sh
GPU=$1; shift
cd /data/neko/cxip || exit 1
for d in "$@"; do
  d=${d%/}
  while [[ ! -f "$d/DONE" ]]; do [[ -f "$d/FAILED" ]] && break; sleep 60; done
  [[ -f "$d/DONE" ]] && bash v3/scripts/eval_native.sh "$GPU" "$d"
done
