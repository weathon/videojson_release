"""Caption every one-minute clip of an MM-Lifelong subset with the generic caption schema and prompt on the local Qwen3.5-35B-A3B-FP8 vLLM server; week adds the EgoLife transcript (harness.load_week). Usage: caption_subset.py <day|month|week> <concurrency per server> [clip limit]"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import ClassVar

import av
import wandb
from jsonschema import Draft202012Validator, ValidationError as SchemaValidationError
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, BadRequestError, InternalServerError, LengthFinishReasonError, RateLimitError
from pydantic import RootModel, ValidationError, model_validator
from wandb.sdk.lib.runid import generate_id

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subset_clips import OLD, ROOT, build_clips
from harness import load_week


parser = argparse.ArgumentParser()
parser.add_argument('subset')
parser.add_argument('concurrency', type=int)  # per server
parser.add_argument('limit', type=int, nargs='?')
# the index-time run (2026-11-15): Qwen/Qwen3.8-27B on 8303,8304, the ASR of asr_subsets.py --out, captions into a new directory
parser.add_argument('--model', default='Qwen/Qwen3.5-35B-A3B-FP8')
parser.add_argument('--ports', default='8301')
parser.add_argument('--asr', help='directory of <subset>/<clip_id>.json ASR files (default: the old repository)')
parser.add_argument('--out', help='output root; captions go to <out>/<subset> (default: runs/captions/qwen3.5-35b-a3b-fp8)')
args = parser.parse_args()
SUBSET = args.subset
CONCURRENCY = args.concurrency
LIMIT = args.limit
OUT = (Path(args.out) if args.out else ROOT / ('runs/captions/qwen3.5-35b-a3b-fp8' if LIMIT is None else 'runs/captions/qwen3.5-35b-a3b-fp8-debug')) / SUBSET
ASR = (Path(args.asr) if args.asr else OLD / 'runs/asr-qwen3-asr-1.7b') / SUBSET
SCHEMA = json.loads((ROOT / 'prompts/egolife/vsi_caption_schema.json').read_text())
PROMPT = (ROOT / 'prompts/caption/caption_prompt.txt').read_text()
MODEL = args.model
PORTS = tuple(int(port) for port in args.ports.split(','))
IMAGE_SIZE = (1024, 1024)
ATTEMPTS = 5
MAX_RETRIES = 5
MAX_TOKENS = 16384  # output cap of every caption call, reasoning included (the user, 2026-11-04: 16k)
RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError, LengthFinishReasonError, ValidationError, SchemaValidationError, json.JSONDecodeError)


class Caption(RootModel[dict]):
    schema_document: ClassVar[dict] = SCHEMA

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        return cls.schema_document

    @model_validator(mode='after')
    def validate_document(self):
        Draft202012Validator(self.schema_document).validate(self.root)
        return self


def read_frames(clip):
    images = [None] * 60
    frames = [None] * 60
    pieces = clip['sources']
    owner = [max(index for index, piece in enumerate(pieces) if piece['clip_start_s'] <= second + 1e-6) for second in range(60)]
    for number, piece in enumerate(pieces):
        offsets = [second for second in range(60) if owner[second] == number]
        if not offsets:
            continue
        path = OLD / 'runs/media' / SUBSET / Path(piece['path']).relative_to('/videos')
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            fps = float(stream.average_rate)
            requested = {}
            for second in offsets:
                video_s = piece['video_start_s'] + second - piece['clip_start_s']
                requested[min(round(video_s * fps), stream.frames - 1)] = second
            container.seek(int(min(requested) / fps / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                video_s = float(frame.pts * stream.time_base)
                frame_index = round(video_s * fps)
                # A frame whose timestamp jitters past a requested index stands for that index: the first decoded frame at or after it.
                while requested and min(requested) <= frame_index:
                    second = requested.pop(min(requested))
                    buffer = io.BytesIO()
                    frame.to_image().resize(IMAGE_SIZE).save(buffer, format='JPEG', quality=95)
                    images[second] = buffer.getvalue()
                    frames[second] = {'clip_s': second, 'path': piece['path'], 'video_s': video_s, 'frame_index': frame_index, 'source_size': [frame.width, frame.height], 'size': list(IMAGE_SIZE)}
                if not requested:
                    break
            if requested:
                raise RuntimeError(f'Video did not decode the requested frames: path={path} missing={requested}')
    if any(image is None for image in images):
        raise RuntimeError(f'One-second sampling has uncovered positions: clip={clip["clip_id"]} frames={frames}')
    return images, frames


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config_path = OUT / 'config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        resume = 'must'
    else:
        config = {'subset': SUBSET, 'model': MODEL, 'ports': PORTS, 'image_size': IMAGE_SIZE, 'jpeg_quality': 95, 'attempts': ATTEMPTS, 'max_retries': MAX_RETRIES, 'asr': str(ASR), 'schema': 'prompts/egolife/vsi_caption_schema.json', 'prompt': 'prompts/caption/caption_prompt.txt', 'limit': LIMIT, 'wandb_id': generate_id()}
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        resume = 'never'
    clips = load_week()[0] if SUBSET == 'week' else build_clips(SUBSET)[0]
    clip_ids = sorted(clips, key=lambda clip_id: (clips[clip_id]['video'], clips[clip_id]['tick_start_s']))
    if LIMIT is not None:
        clip_ids = clip_ids[:LIMIT]
    run = wandb.init(project='videojson_gym', name=f'captions-{Path(args.out).name}-{SUBSET}' if args.out else f'captions-qwen3.5-35b-a3b-fp8-{SUBSET}' + ('-debug' if LIMIT is not None else ''), id=config['wandb_id'], resume=resume, config={**config, 'clips': len(clip_ids), 'concurrency_per_server_this_launch': CONCURRENCY, 'max_tokens': MAX_TOKENS}, dir=str(OUT))
    stats = {'finished': 0, 'failed': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'caption_chars': 0, 'finished_this_launch': 0}
    for clip_id in clip_ids:
        path = OUT / 'captions' / clip_id / 'caption.json'
        if path.exists():
            usage = json.loads((OUT / 'captions' / clip_id / 'usage.json').read_text())
            stats['finished'] += 1
            stats['prompt_tokens'] += usage['prompt_tokens']
            stats['completion_tokens'] += usage['completion_tokens']
            stats['caption_chars'] += len(path.read_text())
        elif (OUT / 'captions' / clip_id / 'failure.json').exists():
            stats['failed'] += 1
    started = time.time()
    semaphores = [asyncio.Semaphore(CONCURRENCY) for _ in PORTS]
    print(f'START subset={SUBSET} clips={len(clip_ids)} finished={stats["finished"]} failed={stats["failed"]} concurrency_per_server={CONCURRENCY}', flush=True)

    def log():
        remaining = len(clip_ids) - stats['finished'] - stats['failed']
        rate = stats['finished_this_launch'] / (time.time() - started)
        metrics = {
            'captions/finished': stats['finished'],
            'captions/failed': stats['failed'],
            'captions/remaining': remaining,
            'tokens/prompt': stats['prompt_tokens'],
            'tokens/completion': stats['completion_tokens'],
            'length/avg_completion_tokens': stats['completion_tokens'] / stats['finished'],
            'length/avg_prompt_tokens': stats['prompt_tokens'] / stats['finished'],
            'length/avg_caption_chars': stats['caption_chars'] / stats['finished'],
            'rate/captions_per_minute': rate * 60,
            'eta/seconds': remaining / rate,
            'eta/finish_unix': time.time() + remaining / rate,
        }
        run.log(metrics)
        return metrics

    async def one(clients, position, clip_id):
        output = OUT / 'captions' / clip_id
        if (output / 'caption.json').exists():
            return
        if (output / 'failure.json').exists():
            (output / 'failure.json').unlink()
            stats['failed'] -= 1
        server = position % len(PORTS)
        async with semaphores[server]:
            clip = clips[clip_id]
            transcript = json.loads((ASR / f'{clip_id}.json').read_text())
            images, frames = await asyncio.to_thread(read_frames, clip)
            prompt = f'One-minute segment. The 60 images are in order at offsets 0 through 59 seconds.\nRaw speech recognition transcript of this minute (detected language: {transcript["language"]}):\n{transcript["transcription"] or "none"}'
            if SUBSET == 'week':
                # the text of harness.caption_clip (the original week captions)
                prompt += f'\nEgoLife transcript of this minute, made by people, with speaker names and English translations; start_s and end_s are seconds of this minute and source names the recording it comes from:\n{json.dumps(clip["egolife_transcript"], ensure_ascii=False) if clip["egolife_transcript"] else "none"}'
                if clip['egolife_uncovered_s']:
                    prompt += f'\nThe EgoLife transcript does not cover these second ranges of this minute: {json.dumps(clip["egolife_uncovered_s"])}'
            output.mkdir(parents=True, exist_ok=True)
            (output / 'input.json').write_text(json.dumps({'clip': clip, 'frames': frames, 'transcript': transcript, 'prompt': prompt, 'port': PORTS[server]}, ensure_ascii=False, indent=2) + '\n')
            content = [{'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(image).decode()}} for image in images]
            content.append({'type': 'text', 'text': prompt})
            feedback = ''
            offset = len(list(output.glob('attempt-*.json')))
            for attempt in range(offset + 1, offset + ATTEMPTS + 1):
                messages = [{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': content}]
                if feedback:
                    messages.append({'role': 'user', 'content': f'The previous attempt failed with this exact error. Correct it and return the requested caption:\n{feedback}'})
                trace_path = output / f'attempt-{attempt:04d}.json'
                trace = {'model': MODEL, 'port': PORTS[server], 'feedback': feedback, 'started': time.time(), 'response': None, 'error': None}
                trace_path.write_text(json.dumps(trace, ensure_ascii=False) + '\n')
                print(f'CAPTION_START clip={clip_id} port={PORTS[server]} attempt={attempt}', flush=True)
                try:
                    response = await clients[server].chat.completions.with_raw_response.parse(model=MODEL, messages=messages, response_format=Caption, max_tokens=MAX_TOKENS)
                    trace['response'] = json.loads(response.text)
                    trace['seconds'] = time.time() - trace['started']
                    trace_path.write_text(json.dumps(trace, ensure_ascii=False) + '\n')
                    parsed = response.parse().choices[0].message.parsed
                    if parsed is None:
                        raise SchemaValidationError(f'Caption has no structured output: {trace["response"]["choices"][0]["message"]}')
                except (BadRequestError, *RETRYABLE) as error:
                    trace['error'] = {'type': type(error).__name__, 'message': str(error)}
                    trace_path.write_text(json.dumps(trace, ensure_ascii=False) + '\n')
                    print(f'CAPTION_ERROR clip={clip_id} attempt={attempt} error={error}', flush=True)
                    if isinstance(error, BadRequestError):
                        break
                    feedback = str(error)
                    await asyncio.sleep(2 ** (attempt - offset))
                    continue
                usage = trace['response']['usage']
                (output / 'usage.json').write_text(json.dumps(usage, indent=2) + '\n')
                (output / 'caption.json').write_text(json.dumps(parsed.root, ensure_ascii=False, indent=2) + '\n')
                stats['finished'] += 1
                stats['finished_this_launch'] += 1
                stats['prompt_tokens'] += usage['prompt_tokens']
                stats['completion_tokens'] += usage['completion_tokens']
                stats['caption_chars'] += len((output / 'caption.json').read_text())
                metrics = log()
                print(f'CAPTION_DONE clip={clip_id} port={PORTS[server]} seconds={trace["seconds"]:.1f} prompt_tokens={usage["prompt_tokens"]} completion_tokens={usage["completion_tokens"]} finished={stats["finished"]} rate={metrics["rate/captions_per_minute"]:.2f}/min eta_h={metrics["eta/seconds"] / 3600:.2f}', flush=True)
                return
            (output / 'failure.json').write_text(json.dumps({'error': trace['error'], 'attempts': attempt - offset}, ensure_ascii=False) + '\n')
            stats['failed'] += 1
            print(f'SAMPLE_SKIPPED clip={clip_id} error={trace["error"]}', flush=True)

    clients = [AsyncOpenAI(base_url=f'http://127.0.0.1:{port}/v1', api_key='EMPTY', timeout=None, max_retries=MAX_RETRIES) for port in PORTS]
    try:
        async with asyncio.TaskGroup() as group:
            for position, clip_id in enumerate(clip_ids):
                group.create_task(one(clients, position, clip_id))
        print(f'CAPTIONS_COMPLETE subset={SUBSET} finished={stats["finished"]} failed={stats["failed"]}', flush=True)
    finally:
        for client in clients:
            await client.close()
        run.finish()


if __name__ == '__main__':
    asyncio.run(main())
    print('DONE', flush=True)
    1 / 0
