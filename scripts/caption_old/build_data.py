"""Write one JSON index per EgoLife wearer and per MM-Lifelong subset, from caption directories. Each file is a list of one-minute clips in time order. Without flags: the captions of the old repository (/raid/wg25r/video_sql_trace2skill), all six wearers, into data/. With --captions <dir> --out <dir>: the captions under <dir>/{week,day,month}/captions (caption_subset.py), A1_JAKE, day and month only, into <out>/egolife and <out>/mm_lifelong."""

import argparse
import json
import sys
from pathlib import Path

OLD = Path('/raid/wg25r/video_sql_trace2skill')
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(OLD / 'vsi'))
import harness
from egolife_people import load_person

parser = argparse.ArgumentParser()
parser.add_argument('--captions')
parser.add_argument('--out')
parser.add_argument('--subsets', default='week,day,month')  # with --captions; week is the A1_JAKE index (the index-time run, 2026-11-15: day only)
parser.add_argument('--limit', type=int)  # day and month: only the first N clips in time order (the index-time run: the first 3 hours of day)
args = parser.parse_args()
SUBSETS = args.subsets.split(',')
DATA = Path(args.out) if args.captions else Path(__file__).resolve().parents[2] / 'data'


def clock(seconds):
    return f'{int(seconds // 3600):02d}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d}'


for person in (('A1_JAKE',) if 'week' in SUBSETS else ()) if args.captions else ('A1_JAKE', 'A2_ALICE', 'A3_TASHA', 'A4_LUCIA', 'A5_KATRINA', 'A6_SHURE'):
    if person == 'A1_JAKE':
        clips, _ = harness.load_week()
        captions = Path(args.captions) / 'week/captions' if args.captions else OLD / 'runs/vsi/baseline/captions'
    else:
        clips, _ = load_person(person)
        captions = OLD / 'runs/egolife/captions' / person
    order = sorted(clips, key=lambda clip_id: (int(clips[clip_id]['day'].removeprefix('DAY')), clips[clip_id]['tick_start_s']))
    records = [
        {
            'clip_id': clip_id,
            'day': clips[clip_id]['day'],
            'start_time': clock(clips[clip_id]['wall_start_s']),
            'end_time': clock(clips[clip_id]['wall_end_s']),
            'tick_start_s': clips[clip_id]['tick_start_s'],
            'tick_end_s': clips[clip_id]['tick_end_s'],
            'wall_start_s': clips[clip_id]['wall_start_s'],
            'wall_end_s': clips[clip_id]['wall_end_s'],
            'caption': json.loads((captions / clip_id / 'caption.json').read_text()),
        }
        for clip_id in order
    ]
    (DATA / 'egolife').mkdir(parents=True, exist_ok=True)
    (DATA / 'egolife' / f'{person}.json').write_text(json.dumps(records, ensure_ascii=False) + '\n')
    print(f'WROTE {DATA}/egolife/{person}.json clips={len(records)}', flush=True)

for subset in [subset for subset in ('day', 'month') if subset in SUBSETS or not args.captions]:
    clips, _ = harness.build_clips(subset)
    order = sorted(clips, key=lambda clip_id: (int(clips[clip_id]['video']) if subset == 'month' else 0, clips[clip_id]['tick_start_s']))[:args.limit]
    captions =Path(args.captions) / subset / 'captions' if args.captions else OLD / 'runs/captions-generic-qwen3.8-flash-next' / subset / 'captions'
    records = [
        {
            'clip_id': clip_id,
            'video': clips[clip_id]['video'],
            'tick_start_s': clips[clip_id]['tick_start_s'],
            'tick_end_s': clips[clip_id]['tick_end_s'],
            'caption': json.loads((captions / clip_id / 'caption.json').read_text()),
        }
        for clip_id in order
    ]
    (DATA / 'mm_lifelong').mkdir(parents=True, exist_ok=True)
    (DATA / 'mm_lifelong' / f'{subset}.json').write_text(json.dumps(records, ensure_ascii=False) + '\n')
    print(f'WROTE {DATA}/mm_lifelong/{subset}.json clips={len(records)}', flush=True)

print('DONE', flush=True)
1 / 0
