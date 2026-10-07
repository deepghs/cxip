# CSIP v2.1 在恢复的原始划分 (2025-09, witha) 上重做: 可用图片筛选 -> CCIP 特征 -> 聚类伪标签 -> 从 HF v2 微调 1 epoch -> 导出 -> 评测
set -e
source /data/neko/cxip_repro/env.sh
R=/data/neko/cxip_repro
S=$R/data/danbooru-artist-split-0918
cd $R/CxIP
echo "[$(date)] available train images"
python - <<PY
import json
from pathlib import Path
root = Path("$R/data/danbooru-w640-ws-full-artist/train")
for name in ("train_split", "test_split"):
    d = json.load(open(f"$S/{name}.json"))
    d = {k: v for k, v in d.items() if (root / k).exists()}
    json.dump(d, open(f"$S/{name}_avail.json", "w"))
    print(name, len(d), "images", len(set(d.values())), "artists")
PY
echo "[$(date)] extract missing ccip features"
python repro/data/assemble_feats_v2.py need danbooru-artist-split-0918 train_split_avail.json
CUDA_VISIBLE_DEVICES=4 python repro/data/extract_ccip_feats.py --label-file $S/extra_labels.json --img-root $R/data/danbooru-w640-ws-full-artist/train --out $S/train_ccip_feats_extra --workers 8
echo "[$(date)] assemble features"
python repro/data/assemble_feats_v2.py build danbooru-artist-split-0918 train_split_avail.json
echo "[$(date)] clustering"
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=64 NUMBA_NUM_THREADS=64 python tools/ccip_cls.py --feat-npy $S/train_ccip_feats --output-dir $S --out-name ccip_clusters.json \
  --memmap-path $R/scratch/ccip_feat_matrix_0918.dat --umap-jobs 64 --core-dist-jobs 64 --umap-parallel 2>&1 | grep -v "Loading features\|Epochs completed"
python tools/clusters_to_char_labels.py --clusters $S/ccip_clusters.json --style-label-file $S/train_split_avail.json --output $S/train_char_labels.json
rm -f $R/scratch/ccip_feat_matrix_0918.dat
echo "[$(date)] train csip v2.1 from v2 (split 0918)"
neko_train --launch_cfg repro/cfgs/launcher_4gpu.yaml --cfg repro/cfgs/csip2_1_fromv2_0918.py > $R/logs/train_csip_fromv2_0918.log 2>&1
CKPT=$(ls -t $R/exps/csip_dino_v2.1_fromv2_0918/ckpts/*.safetensors | head -1)
echo "[$(date)] export $CKPT"
(cd transfs && python trans_csip.py --ckpt $CKPT --out $R/exps/csip_dino_v2.1_fromv2_0918/hf_model)
cd $R
M=$R/exps/csip_dino_v2.1_fromv2_0918/hf_model
echo "[$(date)] eval"
CUDA_VISIBLE_DEVICES=4 python CxIP/repro/eval/pair_eval.py --model $M --data test100_0918=json:$S/test_split_100_avail.json:data/danbooru-w640-ws-full-artist/train \
  --out results/repro_csip_fromv2_0918_test100_0918.json > results/logs/repro_csip_fromv2_0918_test100_0918.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python CxIP/repro/eval/pair_eval.py --model $M --data csip_eval_public=folder:data/csip_eval \
  --out results/repro_csip_fromv2_0918_pub.json > results/logs/repro_csip_fromv2_0918_pub.log 2>&1 &
wait
python CxIP/repro/eval/summarize.py results/hf_csip*_test100_0918.json results/hf_csip*_pub.json results/repro_csip_fromv2_0918_*.json
echo "[$(date)] DONE"
