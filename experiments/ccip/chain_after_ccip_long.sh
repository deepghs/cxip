# CCIP_long 结束 -> 导出+评测 CCIP_long -> (等角色伪标签) -> CSIP v2.1 从 v2 微调 -> 导出+评测
set -e
source /data/neko/cxip_repro/env.sh
R=/data/neko/cxip_repro
S=$R/data/danbooru-artist-split-v2
cd $R
while tmux has-session -t cxip_train_ccip_long 2>/dev/null; do sleep 120; done
CKPT=$(ls -t $R/exps/ccip_dino_long/ckpts/*.safetensors | head -1)
echo "[$(date)] export $CKPT"
(cd $R/CxIP/export && python trans_ccip.py --ckpt $CKPT --out $R/exps/ccip_dino_long/hf_model)
echo "[$(date)] eval ccip_long"
CUDA_VISIBLE_DEVICES=4 python CxIP/experiments/common/pair_eval.py --model $R/exps/ccip_dino_long/hf_model \
  --data ccip_eval=folder:data/ccip/ccip_eval/eval test_ccip=folder:data/ccip/test_ccip ccip_test_v1=folder:data/ccip_test_v1 \
         zc50x15=folder:data/zerochan_test_c50x15 zc500x10=folder:data/zerochan_test_c500x10 \
  --out results/repro_ccip_long.json > results/logs/repro_ccip_long.log 2>&1 &
EVAL_PID=$!
while [ ! -s $S/train_char_labels.json ] || tmux has-session -t cxip_charlabels 2>/dev/null; do sleep 120; done
wait $EVAL_PID
echo "[$(date)] train csip v2.1 from v2"
(cd CxIP && neko_train --launch_cfg cfgs/launcher/launcher_4gpu.yaml --cfg experiments/csip_v2.1/first_attempts/csip2_1_fromv2_repro.py > ../logs/train_csip_fromv2.log 2>&1)
CKPT=$(ls -t $R/exps/csip_dino_v2.1_fromv2/ckpts/*.safetensors | head -1)
echo "[$(date)] export $CKPT"
(cd $R/CxIP/export && python trans_csip.py --ckpt $CKPT --out $R/exps/csip_dino_v2.1_fromv2/hf_model)
echo "[$(date)] eval csip"
T=test100v2=json:$S/test_split_100.json:data/danbooru-w640-ws-full-artist/train
CUDA_VISIBLE_DEVICES=4 python CxIP/experiments/common/pair_eval.py --model $R/exps/csip_dino_v2.1_fromv2/hf_model --data $T \
  --out results/repro_csip_fromv2_test100v2.json --save-feats results/feats/repro_csip_fromv2 > results/logs/repro_csip_fromv2_test100v2.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python CxIP/experiments/common/pair_eval.py --model $R/exps/csip_dino_v2.1_fromv2/hf_model --data csip_eval_public=folder:data/csip_eval \
  --out results/repro_csip_fromv2_pub.json > results/logs/repro_csip_fromv2_pub.log 2>&1 &
wait
echo "[$(date)] DONE"
