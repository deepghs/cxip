# CUDA_VISIBLE_DEVICES=1 neko_eval_1gpu --cfg cfgs/py/eval/ccip.py
# CUDA_VISIBLE_DEVICES=1 neko_eval_1gpu --cfg cfgs/py/eval/csip.py

# CUDA_VISIBLE_DEVICES=1 neko_run --cfg cfgs/py/infer/ccip_save.py test_img_root="/nfs/ccip/test_ccip/" out_dir_name="feat_ccip"
# CUDA_VISIBLE_DEVICES=1 neko_run --cfg cfgs/py/infer/ccip_save.py test_img_root="/nfs/zerochan-character-w640-ws-m50-full/small_val_folder" out_dir_name="feat_ccip_zero"

# echo danbooru
# python scripts/eval/eval_ccip_zero.py --emb_dir feat_ccip --N 20
# python scripts/eval/eval_ccip_zero.py --emb_dir feat_ccip --N 40

# echo zerochan
# python scripts/eval/eval_ccip_zero.py --emb_dir feat_ccip_zero --N 20
# python scripts/eval/eval_ccip_zero.py --emb_dir feat_ccip_zero --N 40
# python scripts/eval/eval_ccip_zero.py --emb_dir feat_ccip_zero --N 100

# 循环ood_thr从-5到0，步长0.1
for ood_thr in $(seq -5.0 0.1 0.0); do
    echo "Running with ood_thr = $ood_thr"
    python scripts/eval/eval_ccip_zero.py --seen 40 --ood_thr $ood_thr
done

# ---- 以下为恢复时补充 ----
# CSIP v2.1 角色伪标签: 训练集 CCIP 特征 -> 聚类 -> train_char_labels.json
# CUDA_VISIBLE_DEVICES=1 neko_run --cfg cfgs/py/infer/ccip_save.py test_img_root="/nfs/danbooru-w640-ws-full-artist/train" out_dir_name="/nfs/danbooru-w640-ws-full-artist/train_ccip_feats"
# python scripts/char_labels/ccip_cls.py --feat-dir /nfs/danbooru-w640-ws-full-artist/train_ccip_feats --output-dir /nfs/danbooru-w640-ws-full-artist
# python scripts/char_labels/clusters_to_char_labels.py --clusters /nfs/danbooru-w640-ws-full-artist/ccip_clusters_xxx.json --style-label-file /nfs/danbooru-w640-ws-full-artist/train_split.json --output /nfs/danbooru-w640-ws-full-artist/train_char_labels.json

# CSIP 特征与偏差评估
# CUDA_VISIBLE_DEVICES=1 neko_run --cfg cfgs/py/infer/csip_save.py ckpt_path="exps/csip_dino_v2.1/ckpts/model-xxx.safetensors" out_dir_name="feat"
# python scripts/eval/eval_csip_zero.py
# python evaluate/content_bias.py --style-path feats/feat_csip_v2.1 --content-path /nfs/danbooru-wdtagger-v4-w640-ws-30k/train/probs.pkl
# python evaluate/char_bias.py --style-path feats/feat_csip_v2.1-char --char-path feats/feat_ccip_zero
