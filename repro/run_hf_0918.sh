# HF CSIP v1/v2/v2.1 在恢复出的原始 test100 (2025-09-18 划分, 去掉 54 张缺图) 上评测
source /data/neko/cxip_repro/env.sh
cd /data/neko/cxip_repro
T=test100_0918=json:data/danbooru-artist-split-0918/test_split_100_avail.json:data/danbooru-w640-ws-full-artist/train
CUDA_VISIBLE_DEVICES=5 python CxIP/repro/eval/pair_eval.py --model dghstmp/AnimeCSIP --data $T --out results/hf_csip_v1_test100_0918.json > results/logs/hf_csip_v1_test100_0918.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python CxIP/repro/eval/pair_eval.py --model dghstmp/AnimeCSIP-v2 --data $T --out results/hf_csip_v2_test100_0918.json > results/logs/hf_csip_v2_test100_0918.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python CxIP/repro/eval/pair_eval.py --model dghstmp/AnimeCSIP-v2.1 --data $T --out results/hf_csip_v2.1_test100_0918.json > results/logs/hf_csip_v2.1_test100_0918.log 2>&1 &
wait
python CxIP/repro/eval/summarize.py results/hf_csip*_test100_0918.json
