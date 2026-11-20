#!/bin/bash
# Wall time of building our MM-Lifelong day index (the index-time run, the user, 2026-11-15): Qwen3-ASR-1.7B (1 shard, batch 64),
# then the one-minute captions (<per-server concurrency> per server; no reasoning effort, max_tokens 16384), then the minute index
# (build_data.py); all on the first <minutes> minutes of day. Each stage prints its start and end unix time.
# Usage: bash scripts/index_time_ours.sh <out dir under runs/index_time> <per-server concurrency> <model> <ports> <minutes> <asr gpu>
# (the user, 2026-11-15, last round: 16, Qwen/Qwen3.5-35B-A3B-FP8, 8301, 180 = 3 one-hour units of MAGIC, GPU 0)
cd /raid/wg25r/videojson_gym
OUT=runs/index_time/$1
[ ! -e $OUT ] || { echo "EXISTS $OUT"; exit 1; }
mkdir -p $OUT
LOG=logs/index_time_$1

check() {
    tail -n 3 $LOG.$1.log | grep -q ZeroDivisionError || { echo "STAGE_FAILED $1"; exit 1; }
}

echo "RUN_START $(date +%s.%N)"
echo "ASR_START $(date +%s.%N)"
CUDA_VISIBLE_DEVICES=$6 .venv/bin/python scripts/caption_old/asr_subsets.py 0 $5 --subsets day --shards 1 --out $OUT/asr >> $LOG.asr.log 2>&1
check asr
echo "ASR_END $(date +%s.%N)"
echo "CAPTION_START $(date +%s.%N)"
.venv/bin/python scripts/caption_old/caption_subset.py day $2 $5 --model $3 --ports $4 --asr $OUT/asr --out $OUT/captions >> $LOG.caption.log 2>&1
check caption
echo "CAPTION_END $(date +%s.%N)"
echo "INDEX_START $(date +%s.%N)"
.venv/bin/python scripts/caption_old/build_data.py --captions $OUT/captions --out $OUT/data --subsets day --limit $5 >> $LOG.index.log 2>&1
check index
echo "INDEX_END $(date +%s.%N)"
echo "RUN_END $(date +%s.%N)"
.venv/bin/python -c "1 / 0"
