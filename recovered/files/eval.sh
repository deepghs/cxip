# CUDA_VISIBLE_DEVICES=1 neko_eval_1gpu --cfg cfgs/py/eval/ccip.py
# CUDA_VISIBLE_DEVICES=1 neko_eval_1gpu --cfg cfgs/py/eval/csip.py

# CUDA_VISIBLE_DEVICES=1 neko_run --cfg cfgs/py/infer/ccip_save.py test_img_root="/nfs/ccip/test_ccip/" out_dir_name="feat_ccip"
# CUDA_VISIBLE_DEVICES=1 neko_run --cfg cfgs/py/infer/ccip_save.py test_img_root="/nfs/zerochan-character-w640-ws-m50-full/small_val_folder" out_dir_name="feat_ccip_zero"

# echo danbooru
# python eval_ccip_zero.py --emb_dir feat_ccip --N 20
# python eval_ccip_zero.py --emb_dir feat_ccip --N 40

# echo zerochan
# python eval_ccip_zero.py --emb_dir feat_ccip_zero --N 20
# python eval_ccip_zero.py --emb_dir feat_ccip_zero --N 40
# python eval_ccip_zero.py --emb_dir feat_ccip_zero --N 100

# 循环ood_thr从-5到0，步长0.1
for ood_thr in $(seq -5.0 0.1 0.0); do
    echo "Running with ood_thr = $ood_thr"
    python eval_ccip_zero.py --seen 40 --ood_thr $ood_thr
done