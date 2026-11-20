"""Transcribe the audio of every one-minute clip of MM-Lifelong day, then month, then week with Qwen3-ASR-1.7B. Usage: asr_subsets.py <shard 0|1> [clip limit per subset]
Copied from /raid/wg25r/video_sql_trace2skill/src/asr_subsets.py; the index-time run (2026-11-15) adds --subsets, --shards and --out."""

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subset_clips import OLD, build_clips


parser = argparse.ArgumentParser()
parser.add_argument('shard', type=int)
parser.add_argument('limit', type=int, nargs='?')  # clips per subset
parser.add_argument('--subsets', default='day,month,week')
parser.add_argument('--shards', type=int, default=2)
parser.add_argument('--out', help='output root (default: the old repository runs/asr-qwen3-asr-1.7b)')
args = parser.parse_args()
MODEL = 'Qwen/Qwen3-ASR-1.7B-hf'
SUBSETS = tuple(args.subsets.split(','))
SHARD = args.shard
LIMIT = args.limit
# the old repository is the ROOT of the original script: its media and outputs live there
OUT = Path(args.out) if args.out else OLD / ('runs/asr-qwen3-asr-1.7b' if LIMIT is None else 'runs/asr-qwen3-asr-1.7b-debug')
BATCH = 64
# Fast speech reaches about 300 words a minute; the model can miss its stop token, so generation stops at twice that.
MAX_NEW_TOKENS = 600
SAMPLE_RATE = 16000
EOS = (151643, 151645)


def read_audio(subset, clip):
    parts = []
    for piece in clip['sources']:
        path = OLD / 'runs/media' / subset / Path(piece['path']).relative_to('/videos')
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-ss', f'{piece["video_start_s"]:.6f}', '-t', f'{piece["video_end_s"] - piece["video_start_s"]:.6f}', '-i', str(path), '-vn', '-ac', '1', '-ar', str(SAMPLE_RATE), '-f', 'f32le', 'pipe:1']
        result = subprocess.run(command, capture_output=True)
        if result.returncode:
            raise RuntimeError(f'ffmpeg failed: clip={clip["clip_id"]} path={path} stderr={result.stderr.decode(errors="replace")}')
        parts.append(np.frombuffer(result.stdout, dtype=np.float32))
    return np.concatenate(parts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(MODEL, cache_dir='/raid/huggingface/hub')
    processor.tokenizer.padding_side = 'left'
    model = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL, cache_dir='/raid/huggingface/hub', dtype=torch.bfloat16).to('cuda')
    if args.out:
        run = wandb.init(project='videojson_gym', name=f'asr-{OUT.name}-shard{SHARD}', config={'model': MODEL, 'shard': SHARD, 'shards': args.shards, 'batch': BATCH, 'max_new_tokens': MAX_NEW_TOKENS, 'subsets': SUBSETS, 'limit': LIMIT, 'out': str(OUT)}, dir=str(OUT))
    else:
        run = wandb.init(entity='3dsmile', project='video-sql-v2', name=f'asr-qwen3-asr-1.7b-shard{SHARD}', config={'model': MODEL, 'shard': SHARD, 'batch': BATCH, 'max_new_tokens': MAX_NEW_TOKENS, 'subsets': SUBSETS, 'limit': LIMIT}, dir=str(OUT))
    try:
        with concurrent.futures.ThreadPoolExecutor(32) as pool:
            for subset in SUBSETS:
                clips, _ = build_clips(subset)
                clip_ids = sorted(clips)[SHARD::args.shards]
                if LIMIT is not None:
                    clip_ids = clip_ids[:LIMIT]
                (OUT / subset).mkdir(exist_ok=True)
                todo = [clip_id for clip_id in clip_ids if not (OUT / subset / f'{clip_id}.json').exists()]
                stats = {'finished': len(clip_ids) - len(todo), 'hit_limit': 0, 'finished_this_launch': 0}
                print(f'ASR_START subset={subset} shard={SHARD} clips={len(clip_ids)} todo={len(todo)}', flush=True)
                started = time.time()
                batches = [todo[number:number + BATCH] for number in range(0, len(todo), BATCH)]
                pending = [pool.submit(read_audio, subset, clips[clip_id]) for clip_id in batches[0]] if batches else []
                for number, batch in enumerate(batches):
                    audios = [future.result() for future in pending]
                    if number + 1 < len(batches):
                        pending = [pool.submit(read_audio, subset, clips[clip_id]) for clip_id in batches[number + 1]]
                    inputs = processor.apply_transcription_request(audio=audios).to(model.device, model.dtype)
                    output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
                    generated = output[:, inputs['input_ids'].shape[1]:].tolist()
                    for clip_id, audio, tokens, mel_frames in zip(batch, audios, generated, inputs['input_features_mask'].sum(-1).tolist()):
                        ends = [position for position, token in enumerate(tokens) if token in EOS]
                        kept = tokens[:ends[0]] if ends else tokens
                        parsed = processor.decode(kept, return_format='parsed')
                        record = {
                            'clip': clips[clip_id],
                            'audio_seconds': len(audio) / SAMPLE_RATE,
                            'mel_frames': mel_frames,
                            'language': parsed['language'],
                            'transcription': parsed['transcription'],
                            'raw': processor.decode(kept),
                            'generated_tokens': len(kept),
                            'hit_limit': not ends,
                        }
                        (OUT / subset / f'{clip_id}.json').write_text(json.dumps(record, ensure_ascii=False, indent=1) + '\n')
                        stats['hit_limit'] += record['hit_limit']
                    stats['finished'] += len(batch)
                    stats['finished_this_launch'] += len(batch)
                    rate = stats['finished_this_launch'] / (time.time() - started)
                    remaining = len(clip_ids) - stats['finished']
                    run.log({f'{subset}/finished': stats['finished'], f'{subset}/remaining': remaining, f'{subset}/hit_limit_this_launch': stats['hit_limit'], f'{subset}/hit_limit_ratio_this_launch': stats['hit_limit'] / stats['finished_this_launch'], f'{subset}/clips_per_minute': rate * 60, f'{subset}/eta_seconds': remaining / rate})
                    print(f'ASR_BATCH subset={subset} shard={SHARD} finished={stats["finished"]}/{len(clip_ids)} hit_limit={stats["hit_limit"]} rate={rate * 60:.1f}/min eta_min={remaining / rate / 60:.1f}', flush=True)
                print(f'ASR_SUBSET_DONE subset={subset} shard={SHARD} finished={stats["finished"]} hit_limit_this_launch={stats["hit_limit"]}', flush=True)
    finally:
        run.finish()


if __name__ == '__main__':
    main()
    print('DONE', flush=True)
    1 / 0
