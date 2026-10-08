# 等 4-7 号卡全部空闲 (显存 < 2GB) 后启动 debias 超参搜索; 不动别人的任务
while [ "$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1>=4 && $2<2000' | wc -l)" -lt 4 ]; do sleep 300; done
echo "[$(date)] GPUs 4-7 free, start search"
bash /data/neko/cxip_repro/CxIP/repro/run_search.sh
