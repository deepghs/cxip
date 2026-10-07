# 诊断 CSIP 微调崩溃: (a) v2 初始化权重经同一导出流程 -> 应接近 HF v2 的 0.79; (b) step-1000 ckpt
source /data/neko/cxip_repro/env.sh
R=/data/neko/cxip_repro
S=$R/data/danbooru-artist-split-0918
cd $R/CxIP/transfs
python trans_csip.py --ckpt $R/exps/init/csip_v2_init.safetensors --out $R/exps/init/csip_v2_init_hf
python trans_csip.py --ckpt $R/exps/csip_dino_v2.1_fromv2_0918/ckpts/model-1000.safetensors --out $R/exps/csip_dino_v2.1_fromv2_0918/hf_model_1000
cd $R
T=test100_0918=json:$S/test_split_100_avail.json:data/danbooru-w640-ws-full-artist/train
CUDA_VISIBLE_DEVICES=4 python CxIP/repro/eval/pair_eval.py --model $R/exps/init/csip_v2_init_hf --data $T --out results/diag_v2init_test100_0918.json > results/logs/diag_v2init.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python CxIP/repro/eval/pair_eval.py --model $R/exps/csip_dino_v2.1_fromv2_0918/hf_model_1000 --data $T --out results/diag_step1000_test100_0918.json > results/logs/diag_step1000.log 2>&1 &
wait
python CxIP/repro/eval/summarize.py results/diag_*.json
