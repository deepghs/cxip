#!/bin/bash
# CSIP 整图原尺度评测: eval_native.sh <gpu> <exp-dir>... (等前一个评测进程结束再跑, 结果在 <exp>/eval_native/)
# 只评 test100_0918 + csip_eval_public (test100 有泄漏), 不跑 bias (与 crop 协议的 bias 相同输入尺度时无新信息)。
# 注意: 本脚本的命令行里不能出现评测模块名, 否则 eval_watch2.sh 的 pgrep 会把它当成评测进程而一直等待。
GPU=$1; shift
cd /data/neko/cxip || exit 1
PAT="^/home/ubuntu/miniconda3/envs/cxip/bin/python -m v3[.]eval_exp"
for d in "$@"; do
  d=${d%/}
  [[ -f "$d/eval_native/suite.json" ]] && continue
  sleep $((RANDOM % 20))  # 与 watcher 错开, 降低同时启动的概率
  while pgrep -f "$PAT" > /dev/null; do sleep 20; done
  echo "$(date +%T) native $d"
  mkdir -p "$d/eval_native"
  CUDA_VISIBLE_DEVICES=$GPU /home/ubuntu/miniconda3/envs/cxip/bin/python -m v3.eval_exp "$d" --protocol native_full \
    --no-bias --sets test100_0918 csip_eval_public > "$d/eval_native/stdout.log" 2>&1 || echo "FAILED $d"
  grep '^{' "$d/eval_native/stdout.log" | tail -n 1 | cut -c1-400
done
