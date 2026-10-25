#!/bin/bash
# 多配置消融队列: run_queue2.sh <queue-file> [gpus=4,5,6,7]
# 队列每行 "<exp-path> [cfg=<config.py>] key=val ...", exp-path 相对 /data/neko/cxip_v3/exps (如 csip_p2/base);
# 不写 cfg= 时用最近一行 "@cfg <config.py>" 指定的默认配置。# 开头为注释; 运行中可编辑 (每轮重新读取)。
# 完成写 DONE, 失败写 FAILED (删掉即重跑)。启动时先等其他 v3.train_* 训练进程结束;
# 没有标记但 stdout.log 以 "done" 结束的实验 (被旧队列接手前跑完的) 自动补 DONE。
Q=$1; GPUS=${2:-4,5,6,7}
EXPS=/data/neko/cxip_v3/exps
NP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
PY=/home/ubuntu/miniconda3/envs/cxip/bin
cd /data/neko/cxip || exit 1
while true; do
  while pgrep -f "torchrun.*-m v3.train_" > /dev/null; do sleep 30; done
  line=""; cfg=""
  while read -r name rest; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    if [[ "$name" == "@cfg" ]]; then cfg=$rest; continue; fi
    d="$EXPS/$name"
    if [[ ! -f "$d/DONE" && ! -f "$d/FAILED" && -f "$d/stdout.log" ]] && grep -qE '[0-9]{2}:[0-9]{2}:[0-9]{2} done$' "$d/stdout.log"; then
      touch "$d/DONE"
    fi
    if [[ ! -f "$d/DONE" && ! -f "$d/FAILED" ]]; then line="$name $rest"; break; fi
  done < "$Q"
  [[ -z "$line" ]] && { echo "$(date +%T) queue empty"; break; }
  read -r name rest <<< "$line"
  if [[ "$rest" == cfg=* ]]; then cfg=${rest%% *}; cfg=${cfg#cfg=}; rest=${rest#cfg=* }; [[ "$rest" == cfg=* ]] && rest=""; fi
  d="$EXPS/$name"; mkdir -p "$d"
  echo "$(date +%T) start $name [$cfg]: $rest"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$GPUS $PY/torchrun --nproc_per_node "$NP" --master_port 29620 -m v3.train_cxip \
    --config "$cfg" exp_dir="$d" $rest > "$d/stdout.log" 2>&1 && touch "$d/DONE" || touch "$d/FAILED"
  echo "$(date +%T) end $name: $(ls "$d" | grep -E '^(DONE|FAILED)$')"
done
