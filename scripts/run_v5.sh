#!/bin/bash
# Usage: scripts/run_v5.sh <run name> [extra main.py flags]
# The v4_minimal settings (DeepSeek xhigh, no video_qa, no hours) on the five main-table datasets, one after another; egor1 with --bilingual (as v4_minimal_bi).
cd /raid/wg25r/videojson_gym
run=$1
shift
for dataset in egolifeqa mm_week mm_day mm_month egor1; do
    extra=""
    if [ "$dataset" = egor1 ]; then
        extra="--bilingual"
    fi
    .venv/bin/python src/main.py --dataset $dataset --run-name $run --concurrency 75 \
        --model deepseek-v4.1-flash-1017 --base-url http://127.0.0.1:8400/v1 --reasoning-effort xhigh \
        --no-video-qa --no-hours $extra "$@" >> logs/${run}_${dataset}.log 2>&1
    echo "$dataset exit $?"
done
echo CHAIN_DONE
.venv/bin/python -c "1 / 0"
