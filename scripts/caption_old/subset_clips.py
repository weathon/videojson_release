"""Cut the MM-Lifelong day, month and week videos into one-minute clips and hardlink the media under runs/media/<subset>."""

import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OLD = Path('/raid/wg25r/video_sql_trace2skill')
MEDIA_MAPS = Path('/raid/wg25r/video_sql_v2/data/runtime')


def build_clips(subset):
    grouped = {}
    for source in json.loads((MEDIA_MAPS / subset / 'media-map.json').read_text())['sources']:
        video = Path(source['path']).parent.name if subset == 'week' else 'day' if subset == 'day' else source['source_video_id'].split(':')[1]
        grouped.setdefault(video, []).append(source)
    clips = {}
    timeline = {}
    for video, records in sorted(grouped.items()):
        records.sort(key=lambda row: row['timeline_start_s'])
        offset = records[0]['timeline_start_s']
        normalized = []
        for row in records:
            source_path = Path(row['path'])
            target = OLD / 'runs/media' / subset / video / source_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(source_path.resolve(), target)
            item = {'path': f'/videos/{video}/{source_path.name}', 'start': row['timeline_start_s'] - offset, 'end': row['timeline_end_s'] - offset}
            if subset == 'week':
                clock = source_path.stem[-8:]
                item['wall_start'] = int(clock[:2]) * 3600 + int(clock[2:4]) * 60 + int(clock[4:6]) + int(clock[6:8]) / 20
            normalized.append(item)
        timeline[video] = [{'path': row['path'], 'start': row['start'], 'end': row['end']} for row in normalized]
        duration = normalized[-1]['end']
        starts = list(range(0, math.floor(duration / 60) * 60, 60))
        if duration % 60:
            starts.append(duration - 60)
        for number, start in enumerate(starts):
            end = start + 60
            pieces = []
            for source in normalized:
                if source['end'] <= start or source['start'] >= end:
                    continue
                low = max(start, source['start'])
                high = min(end, source['end'])
                piece = {
                    'path': source['path'],
                    'clip_start_s': low - start,
                    'clip_end_s': high - start,
                    'video_start_s': low - source['start'],
                    'video_end_s': high - source['start'],
                }
                if subset == 'week':
                    piece['wall_start_s'] = source['wall_start'] + low - source['start']
                    piece['wall_end_s'] = source['wall_start'] + high - source['start']
                pieces.append(piece)
            clip_id = f'{video}_{number:04d}'
            clips[clip_id] = {'clip_id': clip_id, 'video': video, 'tick_start_s': start, 'tick_end_s': end, 'sources': pieces}
    return clips, timeline
