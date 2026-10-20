# eraser 诊断: 冻结 backbone 只训 eraser 300 步 (GPU 7)
export CUDA_VISIBLE_DEVICES=7
source /data/neko/cxip_repro/env.sh
R=/data/neko/cxip_repro; E=$R/exps/diag_eraser_0918
cd $R/CxIP
neko_train --launch_cfg cfgs/launcher/launcher_1gpu.yaml --cfg experiments/csip_v2.1/search/diag_eraser_0918.py > $R/logs/diag_eraser.log 2>&1 &
PID=$!
until [ -f $E/ckpts/model-300.safetensors ]; do
  if ! kill -0 $PID 2>/dev/null; then echo "[$(date)] diag died"; exit 1; fi
  sleep 30
done
sleep 30; pkill -f "[d]iag_eraser_0918.py"; echo "[$(date)] DONE"
