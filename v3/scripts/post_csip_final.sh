#!/bin/bash
# CSIP final 结束后的流水线 (与队列里的 CCIP 消融并行):
#   1. 伪画师: 用 CSIP final 提取 CCIP 训练图的画风向量 (4 卡) 并聚类, 完成后把 l12b_pa 插进队列 (CCIP 消融段末尾)
#   2. 评测 (GPU 5, 与 watcher 串行): crop 套件 + native (EMA) + native (非 EMA)
# 注意: 本脚本命令行里不能出现评测模块名 (见 eval_native.sh)。
E=/data/neko/cxip_v3/exps/final/csip_teacher_v1
PY=/home/ubuntu/miniconda3/envs/cxip/bin/python
PAT="^$PY -m v3[.]eval_exp"
Q=/data/neko/cxip/v3/configs/queue_main.txt
cd /data/neko/cxip || exit 1
while [[ ! -f $E/DONE ]]; do [[ -f $E/FAILED ]] && { echo "final FAILED"; exit 1; }; sleep 60; done
echo "$(date +%T) final done"

(
  PYTHONPATH=. $PY -m v3.scripts.pseudo_artist extract --csip-exp $E --gpus 4,5,6,7 --bs 128 \
    && PYTHONPATH=. $PY -m v3.scripts.pseudo_artist cluster --k 8192 --gpu 4 \
    && $PY - "$Q" <<'EOF'
import sys
p = sys.argv[1]; L = open(p).read().split('\n')
line = ('ccip_p1/l12b_pa model.finetune=last_n model.n_last=12 loss.arc_negatives=batch '
        'data.index=/data/neko/cxip_v3/data/ccip-pseudo-artist-v1/train.parquet data.cross_key=pseudo_artist data.cross_frac=0.25')
if not any(l.startswith('ccip_p1/l12b_pa ') for l in L):
    j = max(k for k, l in enumerate(L) if l.startswith('ccip_p1/') and 'old_recipe' not in l)
    L.insert(j + 1, line)
    open(p, 'w').write('\n'.join(L))
    print('queued l12b_pa')
EOF
) > /data/neko/cxip_v3/logs/pseudo_artist.log 2>&1 &

run_eval() {  # run_eval <out-subdir> <args...>
  local sub=$1; shift
  [[ -f $E/$sub/suite.json ]] && return
  sleep $((RANDOM % 20))
  while pgrep -f "$PAT" > /dev/null; do sleep 20; done
  mkdir -p $E/$sub
  echo "$(date +%T) eval $sub"
  CUDA_VISIBLE_DEVICES=5 $PY -m v3.eval_exp $E "$@" > $E/$sub/stdout.log 2>&1 || echo "FAILED $sub"
  grep '^{' $E/$sub/stdout.log | tail -n 1 | cut -c1-400
}
run_eval eval
run_eval eval_native --protocol native_full --sets test100_0918 csip_eval_public test100 --no-bias
run_eval eval_native_last_raw --protocol native_full --sets test100_0918 csip_eval_public --no-bias --ckpt last_raw.pt
wait
echo "$(date +%T) post done"
