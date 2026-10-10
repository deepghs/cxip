# 修正后的 debias 搜索 (GPU 7): siglip 对照 (同样 bf16+compile, 用来核对 bf16 与之前 fp32 结果是否一致) + grl_alpha 1/0.3/0.1, 各 300 步
export CUDA_VISIBLE_DEVICES=7
source /data/neko/cxip_repro/env.sh
R=/data/neko/cxip_repro
S=$R/data/danbooru-artist-split-0918
T=test100_0918=json:$S/test_split_100_avail.json:data/danbooru-w640-ws-full-artist/train
MODELS="dghstmp/AnimeCSIP-v2 dghstmp/AnimeCSIP-v2.1 $R/exps/abl_csip_0918_siglip_only/hf_model_300"
for name in siglip g1 g03 g01; do
  E=$R/exps/fix_csip_0918_$name
  echo "[$(date)] train $name"
  cd $R/CxIP
  neko_train --launch_cfg repro/cfgs/launcher_1gpu.yaml --cfg repro/cfgs/fix_csip_0918_$name.py > $R/logs/fix_$name.log 2>&1 &
  PID=$!
  until [ -f $E/ckpts/model-300.safetensors ]; do
    if ! kill -0 $PID 2>/dev/null; then echo "[$(date)] $name training died, see logs/fix_$name.log"; continue 2; fi
    sleep 60
  done
  sleep 60
  pkill -f "[f]ix_csip_0918_$name.py"
  sleep 30
  (cd transfs && python trans_csip.py --ckpt $E/ckpts/model-300.safetensors --out $E/hf_model_300)
  cd $R
  python CxIP/repro/eval/pair_eval.py --model $E/hf_model_300 --data $T --out results/fix_${name}_300_test100_0918.json > results/logs/fix_$name.log 2>&1
  MODELS="$MODELS $E/hf_model_300"
done
echo "[$(date)] bias eval"
(cd $R/CxIP && python repro/eval/bias_eval.py --models $MODELS --out $R/results/bias_fix.json > $R/logs/bias_fix.log 2>&1)
cd $R
python CxIP/repro/eval/summarize.py results/fix_*.json results/abl_siglip_only*.json results/diag_v2init*.json
tail -10 logs/bias_fix.log
echo "[$(date)] DONE"
