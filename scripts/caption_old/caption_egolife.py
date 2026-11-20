"""Caption every one-minute clip of the EgoLife wearers A2 to A6 with the generic caption schema and prompt, the clip's ASR and the EgoLife transcript (the wearer's own first), exactly as the A1_JAKE week baseline captions (vsi/baseline.py), on port 8001 only with 50 concurrent requests, leaving port 8000 free for other experiments. A failed clip is retried on the next launch. Usage: caption_egolife.py <debug|formal>"""

import asyncio
import json
import os
import sys
from pathlib import Path

import wandb
from dotenv import load_dotenv
from openai import AsyncOpenAI
from wandb.sdk.lib.runid import generate_id

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness
from egolife_people import VIDEOS, load_person

MODE = sys.argv[1]
if MODE not in ('debug', 'formal'):
    raise ValueError('Expected debug or formal')
TESTING = MODE == 'debug'
PEOPLE = ('A2_ALICE', 'A3_TASHA', 'A4_LUCIA', 'A5_KATRINA', 'A6_SHURE')
PORTS = (8001,)
CONCURRENCY = 50
DEBUG_CLIPS = 2
OUT = harness.ROOT / ('runs/egolife/captions-debug' if TESTING else 'runs/egolife/captions')


async def main():
    load_dotenv(harness.ROOT / '.env')
    schema = json.loads(harness.INITIAL_SCHEMA.read_text())
    prompt = harness.INITIAL_PROMPT.read_text()
    clips = {}
    for person in PEOPLE:
        loaded, _ = load_person(person)
        ordered = sorted(loaded, key=lambda clip_id: (int(loaded[clip_id]['day'].removeprefix('DAY')), loaded[clip_id]['tick_start_s']))
        if TESTING:
            ordered = ordered[:DEBUG_CLIPS]
        for clip_id in ordered:
            clips[clip_id] = loaded[clip_id]
    OUT.mkdir(parents=True, exist_ok=True)
    config_path = OUT / 'config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        resume = 'must'
    else:
        config = {'model': harness.MODEL, 'ports': PORTS, 'concurrency_per_port': CONCURRENCY, 'schema': str(harness.INITIAL_SCHEMA), 'prompt': str(harness.INITIAL_PROMPT), 'people': PEOPLE, 'clips': len(clips), 'testing': TESTING, 'wandb_id': generate_id()}
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        resume = 'never'
    run = wandb.init(entity='3dsmile', project='video-sql-v2', name='egolife-captions' + ('-debug' if TESTING else ''), id=config['wandb_id'], resume=resume, config=config, dir=str(OUT))
    clients = [AsyncOpenAI(base_url=f'http://127.0.0.1:{port}/v1', api_key=os.environ['QWEN_HOST_API_KEY'], timeout=None, max_retries=harness.MAX_RETRIES) for port in PORTS]
    semaphores = [asyncio.Semaphore(CONCURRENCY) for _ in PORTS]
    stats = {'captioned': 0, 'failed': 0}
    try:

        async def caption(position, clip_id):
            clip = clips[clip_id]
            output = OUT / clip['person'] / clip_id
            if not (output / 'caption.json').exists():
                if (output / 'failure.json').exists():
                    (output / 'failure.json').unlink()
                server = position % len(PORTS)
                async with semaphores[server]:
                    await harness.caption_clip(clients[server], PORTS[server], clip, schema, prompt, output, VIDEOS)
            if (output / 'caption.json').exists():
                stats['captioned'] += 1
            else:
                stats['failed'] += 1
            run.log({'captions/finished': stats['captioned'], 'captions/failed': stats['failed'], 'captions/remaining': len(clips) - stats['captioned'] - stats['failed']})

        print(f'CAPTIONS_START clips={len(clips)}', flush=True)
        async with asyncio.TaskGroup() as group:
            for position, clip_id in enumerate(clips):
                group.create_task(caption(position, clip_id))
        failed = [clip_id for clip_id, clip in clips.items() if not (OUT / clip['person'] / clip_id / 'caption.json').exists()]
        (OUT / 'failed-clips.json').write_text(json.dumps(failed) + '\n')
        print(f'CAPTIONS_COMPLETE captioned={stats["captioned"]} failed={len(failed)}', flush=True)
    finally:
        for client in clients:
            await client.close()
        run.finish()


if __name__ == '__main__':
    asyncio.run(main())
