#!/bin/bash
# 顺序跑消融队列: run_queue.sh <queue-file> <exp-root> [gpus=4,5,6,7] [config=v3/configs/ccip_p1_base.py]
# 队列文件每行 "<name> key=val ...", # 开头为注释; 运行中可追加行 (每轮重新读取)。
# 完成写 DONE, 失败写 FAILED (删掉标记即可重跑)。
Q=$1; ROOT=$2; GPUS=${3:-4,5,6,7}; CFG=${4:-v3/configs/ccip_p1_base.py}
NP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
PY=/home/ubuntu/miniconda3/envs/cxip/bin
cd /data/neko/cxip || exit 1
while true; do
  line=""
  while read -r name rest; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    if [[ ! -f "$ROOT/$name/DONE" && ! -f "$ROOT/$name/FAILED" ]]; then line="$name $rest"; break; fi
  done < "$Q"
  [[ -z "$line" ]] && { echo "$(date +%T) queue empty"; break; }
  read -r name rest <<< "$line"
  mkdir -p "$ROOT/$name"
  echo "$(date +%T) start $name: $rest"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$GPUS $PY/torchrun --nproc_per_node "$NP" --master_port 29620 -m v3.train_ccip \
    --config "$CFG" exp_dir="$ROOT/$name" $rest > "$ROOT/$name/stdout.log" 2>&1 \
    && touch "$ROOT/$name/DONE" || touch "$ROOT/$name/FAILED"
  echo "$(date +%T) end $name: $(ls "$ROOT/$name" | grep -E '^(DONE|FAILED)$')"
done
