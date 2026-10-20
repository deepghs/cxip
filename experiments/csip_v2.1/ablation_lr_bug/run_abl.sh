# 消融: CSIP v2.1 微调在 300 步内是否崩溃. A: 只有 siglip; B: debias loss 权重 0.1. 各跑到 model-300 后停止并评测
source /data/neko/cxip_repro/env.sh
R=/data/neko/cxip_repro
S=$R/data/danbooru-artist-split-0918
T=test100_0918=json:$S/test_split_100_avail.json:data/danbooru-w640-ws-full-artist/train
for name in siglip_only w01; do
  E=$R/exps/abl_csip_0918_$name
  echo "[$(date)] train $name"
  cd $R/CxIP
  neko_train --launch_cfg cfgs/launcher/launcher_4gpu.yaml --cfg experiments/csip_v2.1/ablation_lr_bug/abl_csip_0918_$name.py > $R/logs/abl_$name.log 2>&1 &
  until [ -f $E/ckpts/model-300.safetensors ]; do sleep 60; done
  sleep 60
  pkill -f "abl_csip_0918_$name.py"
  sleep 30
  (cd export && python trans_csip.py --ckpt $E/ckpts/model-300.safetensors --out $E/hf_model_300)
  cd $R
  CUDA_VISIBLE_DEVICES=4 python CxIP/experiments/common/pair_eval.py --model $E/hf_model_300 --data $T --out results/abl_${name}_300_test100_0918.json > results/logs/abl_$name.log 2>&1
done
python CxIP/experiments/common/summarize.py results/abl_*.json results/diag_*.json
echo "[$(date)] DONE"
