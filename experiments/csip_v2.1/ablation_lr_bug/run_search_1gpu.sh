# debias 超参搜索: 每组从 HF v2 训练 300 步, 测原始 test100 的 PR-AUC 和 content/char bias
export CUDA_VISIBLE_DEVICES=7  # 4-6 号卡被别人的 vLLM 占用, 只用 7 号卡
source /data/neko/cxip_repro/env.sh
R=/data/neko/cxip_repro
S=$R/data/danbooru-artist-split-0918
T=test100_0918=json:$S/test_split_100_avail.json:data/danbooru-w640-ws-full-artist/train
MODELS="dghstmp/AnimeCSIP dghstmp/AnimeCSIP-v2 dghstmp/AnimeCSIP-v2.1 $R/exps/abl_csip_0918_siglip_only/hf_model_300 $R/exps/abl_csip_0918_w01/hf_model_300"
for name in a01 a003 w001 a01pre; do
  E=$R/exps/abl_csip_0918_$name
  echo "[$(date)] train $name"
  cd $R/CxIP
  neko_train --launch_cfg cfgs/launcher/launcher_1gpu.yaml --cfg experiments/csip_v2.1/ablation_lr_bug/abl1g_csip_0918_$name.py > $R/logs/abl_$name.log 2>&1 &
  PID=$!
  until [ -f $E/ckpts/model-300.safetensors ]; do
    if ! kill -0 $PID 2>/dev/null; then echo "[$(date)] $name training died, see logs/abl_$name.log"; continue 2; fi
    sleep 60
  done
  sleep 60
  pkill -f "abl1g_csip_0918_$name.py"
  sleep 30
  (cd export && python trans_csip.py --ckpt $E/ckpts/model-300.safetensors --out $E/hf_model_300)
  cd $R
  python CxIP/experiments/common/pair_eval.py --model $E/hf_model_300 --data $T --out results/abl_${name}_300_test100_0918.json > results/logs/abl_$name.log 2>&1
  MODELS="$MODELS $E/hf_model_300"
done
echo "[$(date)] bias eval"
(cd $R/CxIP && python experiments/common/bias_eval.py --models $MODELS --out $R/results/bias_search.json > $R/logs/bias_search.log 2>&1)
cd $R
python CxIP/experiments/common/summarize.py results/abl_*.json results/diag_v2init*.json
tail -12 logs/bias_search.log
echo "[$(date)] DONE"
