"""EgoLife footage of every wearer as one-minute clips, built the same way as the MM-Lifelong week clips of A1_JAKE (src/subset_clips.py, week branch): per day the files are concatenated in time order by their probed durations, the concatenation is cut into one-minute ticks (the last tick ends at the end of the day), and each piece keeps its wall-clock time from the file name. Paths are absolute files in the Hugging Face cache.

Usage: egolife_people.py <person>  (probe the durations and write runs/egolife/<person>/media-map.json)
"""

import concurrent.futures
import json
import math
import subprocess
import sys
from pathlib import Path

import srt

ROOT = Path(__file__).resolve().parents[1]
# Clip and timeline paths are '/videos/<person>/<DAYn>/<file>' under this snapshot, as the week clips are under runs/media/week.
VIDEOS = Path('/raid/huggingface/hub/datasets--lmms-lab--EgoLife/snapshots/143fb319be7aa5ae210c936bf4f0f3a86092afb0')
TRANSCRIPTS = VIDEOS / 'EgoLifeCap/Transcript'
PEOPLE = ('A1_JAKE', 'A2_ALICE', 'A3_TASHA', 'A4_LUCIA', 'A5_KATRINA', 'A6_SHURE')
OUT = ROOT / 'runs/egolife'
ASR = ROOT / 'runs/asr-qwen3-asr-1.7b/egolife'


def probe(path):
    result = subprocess.run(
        [
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            'format=duration:stream=duration',
            '-of',
            'json',
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f'ffprobe failed: path={path} stderr={result.stderr}')
    probed = json.loads(result.stdout)
    container = round(float(probed['format']['duration']), 2)
    video = round(float(probed['streams'][0]['duration']), 2)
    # The container duration, as the week media map; but 7 files carry seconds of audio past their last video frame, and there the footage ends with the video. Most files differ by a 0.04 s audio tail, which keeps the container duration.
    return video if container - video > 0.1 else container


def build_media_map(person):
    files = sorted(
        (VIDEOS / person).glob('DAY*/*.mp4'),
        key=lambda path: (int(path.parent.name.removeprefix('DAY')), path.name),
    )
    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        durations = list(pool.map(probe, files))
    sources = []
    offset = 0.0
    for path, duration in zip(files, durations):
        sources.append(
            {
                'path': '/videos/' + str(path.relative_to(VIDEOS)),
                'name': path.name,
                'day': path.parent.name,
                'duration_s': duration,
                'timeline_start_s': offset,
                'timeline_end_s': offset + duration,
            }
        )
        offset += duration
    (OUT / person).mkdir(parents=True, exist_ok=True)
    (OUT / person / 'media-map.json').write_text(json.dumps({'sources': sources}, indent=1) + '\n')
    print(f'MEDIA_MAP person={person} files={len(sources)} hours={offset / 3600:.1f}', flush=True)


def build_clips(person):
    """Clips and per-day timeline of one wearer, as src/subset_clips.build_clips('week')."""
    grouped = {}
    for source in json.loads((OUT / person / 'media-map.json').read_text())['sources']:
        grouped.setdefault(source['day'], []).append(source)
    clips = {}
    timeline = {}
    for day, records in sorted(grouped.items(), key=lambda item: int(item[0].removeprefix('DAY'))):
        records.sort(key=lambda row: row['timeline_start_s'])
        offset = records[0]['timeline_start_s']
        normalized = []
        for row in records:
            clock = Path(row['name']).stem[-8:]
            normalized.append(
                {
                    'path': row['path'],
                    'start': row['timeline_start_s'] - offset,
                    'end': row['timeline_end_s'] - offset,
                    'wall_start': int(clock[:2]) * 3600
                    + int(clock[2:4]) * 60
                    + int(clock[4:6])
                    + int(clock[6:8]) / 20,
                }
            )
        timeline[day] = normalized
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
                pieces.append(
                    {
                        'path': source['path'],
                        'clip_start_s': low - start,
                        'clip_end_s': high - start,
                        'video_start_s': low - source['start'],
                        'video_end_s': high - source['start'],
                        'wall_start_s': source['wall_start'] + low - source['start'],
                        'wall_end_s': source['wall_start'] + high - source['start'],
                    }
                )
            clip_id = f'{person}_{day}_{number:04d}'
            clips[clip_id] = {
                'clip_id': clip_id,
                'person': person,
                'video': day,
                'tick_start_s': start,
                'tick_end_s': end,
                'sources': pieces,
            }
    return clips, timeline


def load_person(person):
    """Clips of one wearer with their day, wall-clock bounds, ASR and EgoLife transcript, and the footage timeline per day, as vsi/harness.load_week for A1_JAKE. The wearer's own transcript is used where it exists; an hour without it uses the first other wearer, in A1 to A6 order, that has that hour."""
    clips, timeline = build_clips(person)
    order = [person] + [other for other in PEOPLE if other != person]
    hours = {other: set() for other in PEOPLE}
    segments = {}
    for other in PEOPLE:
        for path in sorted((TRANSCRIPTS / other).glob('DAY*/*.srt')):
            key = (other, path.parent.name, int(path.stem[-8:-6]))
            hours[other].add(key[1:])
            segments[key] = [
                {
                    'id': f'{path.stem}:{subtitle.index}',
                    'start': key[2] * 3600 + subtitle.start.total_seconds(),
                    'end': key[2] * 3600 + subtitle.end.total_seconds(),
                    'text': subtitle.content,
                }
                for subtitle in srt.parse(path.read_text())
            ]
    loaded = {}
    for clip_id, clip in clips.items():
        day = clip['video']
        spoken = {}
        uncovered = []
        for piece in clip['sources']:
            for hour in range(int(piece['wall_start_s'] // 3600), int(piece['wall_end_s'] // 3600) + 1):
                low_window = max(piece['wall_start_s'], hour * 3600)
                high_window = min(piece['wall_end_s'], (hour + 1) * 3600)
                if low_window >= high_window:
                    continue
                people = [other for other in order if (day, hour) in hours[other]]
                if not people:
                    uncovered.append(
                        [
                            round(piece['clip_start_s'] + low_window - piece['wall_start_s'], 2),
                            round(piece['clip_start_s'] + high_window - piece['wall_start_s'], 2),
                        ]
                    )
                    continue
                for segment in segments[(people[0], day, hour)]:
                    low = max(low_window, segment['start'])
                    high = min(high_window, segment['end'])
                    if low >= high:
                        continue
                    local_start = round(piece['clip_start_s'] + low - piece['wall_start_s'], 2)
                    local_end = round(piece['clip_start_s'] + high - piece['wall_start_s'], 2)
                    if segment['id'] in spoken:
                        spoken[segment['id']]['end_s'] = local_end
                    else:
                        spoken[segment['id']] = {
                            'start_s': local_start,
                            'end_s': local_end,
                            'text': segment['text'],
                            'source': segment['id'],
                        }
        asr = json.loads((ASR / person / f'{clip_id}.json').read_text())
        loaded[clip_id] = {
            **clip,
            'day': day,
            'wall_start_s': min(piece['wall_start_s'] for piece in clip['sources']),
            'wall_end_s': max(piece['wall_end_s'] for piece in clip['sources']),
            'asr': {'language': asr['language'], 'transcription': asr['transcription']},
            'egolife_transcript': sorted(spoken.values(), key=lambda segment: segment['start_s']),
            'egolife_uncovered_s': uncovered,
        }
    days = {
        day: [{'path': str(VIDEOS / Path(row['path']).relative_to('/videos')), 'start': row['start'], 'end': row['end'], 'wall_start': row['wall_start']} for row in rows]
        for day, rows in timeline.items()
    }
    return loaded, days


if __name__ == '__main__':
    build_media_map(sys.argv[1])
