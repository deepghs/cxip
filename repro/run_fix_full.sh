# grl_alpha 0.03 完整 1 epoch (GPU 7), 训完导出最终和中间 ckpt 并评测 PR-AUC + bias
export CUDA_VISIBLE_DEVICES=7
source /data/neko/cxip_repro/env.sh
R=/data/neko/cxip_repro; E=$R/exps/fix_csip_0918_g003_full
S=$R/data/danbooru-artist-split-0918
T=test100_0918=json:$S/test_split_100_avail.json:data/danbooru-w640-ws-full-artist/train
cd $R/CxIP
echo "[$(date)] train"
neko_train --launch_cfg repro/cfgs/launcher_1gpu.yaml --cfg repro/cfgs/fix_csip_0918_g003_full.py > $R/logs/fix_g003_full.log 2>&1 || echo "[$(date)] train exit $?"
MODELS="dghstmp/AnimeCSIP-v2 dghstmp/AnimeCSIP-v2.1"
for c in $(ls $E/ckpts/model-*.safetensors | sort -t- -k2 -n); do
  s=$(basename $c .safetensors | cut -d- -f2)
  (cd $R/CxIP/transfs && python trans_csip.py --ckpt $c --out $E/hf_model_$s)
  (cd $R && python CxIP/repro/eval/pair_eval.py --model $E/hf_model_$s --data $T --out results/fix_g003_full_${s}_test100_0918.json > results/logs/fix_g003_full_$s.log 2>&1)
  MODELS="$MODELS $E/hf_model_$s"
done
(cd $R && python CxIP/repro/eval/pair_eval.py --model $E/hf_model_$s --data csip_eval_public=folder:data/csip_eval --out results/fix_g003_full_${s}_pub.json > results/logs/fix_g003_full_${s}_pub.log 2>&1)
echo "[$(date)] bias eval"
python repro/eval/bias_eval.py --models $MODELS --out $R/results/bias_fix_full.json > $R/logs/bias_fix_full.log 2>&1
cd $R; python CxIP/repro/eval/summarize.py results/fix_g003_full_*.json results/diag_v2init*.json results/hf_csip*_pub.json
tail -10 logs/bias_fix_full.log
echo "[$(date)] DONE"
