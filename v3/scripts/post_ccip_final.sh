#!/bin/bash
# CCIP final 结束后评测 (GPU 5): 默认协议 (cfg eval.res=512, 含 I_style) + 384 对照 (与 P1 消融可比)
# 注意: 命令行里不能出现评测模块名 (见 eval_native.sh)。
E=/data/neko/cxip_v3/exps/final/ccip_teacher_v1
cd /data/neko/cxip || exit 1
while [[ ! -f $E/DONE ]]; do [[ -f $E/FAILED ]] && { echo "final FAILED"; exit 1; }; sleep 60; done
echo "$(date +%T) final done"
bash v3/scripts/eval_after.sh 5 $E
bash v3/scripts/eval_after.sh 5 $E --res 384 --no-bias
echo "$(date +%T) post done"
