"""Shared pieces of the Visual Self Improvement runs on EgoLife A1_JAKE week: the one-minute clips with ASR and EgoLife transcripts, clip captioning, the answering harness (the tool loop of joint/agent_worker.py run outside Docker, with exec in a container that has only the index) and the GPT-5 judge."""

import asyncio
import base64
import bisect
import io
import json
import math
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import ClassVar, Literal

import av
import srt
from jsonschema import Draft202012Validator, ValidationError as SchemaValidationError
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    LengthFinishReasonError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, RootModel, ValidationError, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subset_clips import OLD, ROOT, build_clips
sys.path.insert(0, str(OLD / 'training_free'))
from evaluate_mm_official import SYSTEM_PROMPT as RUBRIC

MODEL = 'Qwen/Qwen3.5-35B-A3B-FP8'
PORTS = (8301,)
QUESTIONS = OLD / 'results/vsi/questions_open.json'
ASR = OLD / 'runs/asr-qwen3-asr-1.7b/week'
MEDIA = OLD / 'runs/media/week'
TRANSCRIPTS = Path(
    '/raid/huggingface/hub/datasets--lmms-lab--EgoLife/snapshots/143fb319be7aa5ae210c936bf4f0f3a86092afb0/EgoLifeCap/Transcript'
)
PEOPLE = ('A1_JAKE', 'A2_ALICE', 'A3_TASHA', 'A4_LUCIA', 'A5_KATRINA', 'A6_SHURE')
INITIAL_SCHEMA = ROOT / 'prompts/egolife/vsi_caption_schema.json'
INITIAL_PROMPT = ROOT / 'prompts/caption/caption_prompt.txt'
IMAGE = 'video-sql-joint:cerebras-claude-0.2.160'
CAPTION_IMAGE_SIZE = (1024, 1024)
CAPTION_ATTEMPTS = 5
MAX_RETRIES = 5
CALL_LIMIT = 20
IMAGE_LIMIT = 650
INTERVAL_LIMIT_S = 1200
FPS_LIMIT = 0.5
JUDGE_MODEL = 'gpt-5'
JUDGE_ATTEMPTS = 3
RETRYABLE = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    LengthFinishReasonError,
    ValidationError,
    SchemaValidationError,
    json.JSONDecodeError,
)
VIDEO_QA_INSTRUCTIONS = 'Answer the visual question using only the supplied ordered frame sequence of the video. Return unknown when the visible evidence is insufficient. Cite supporting frames with zero-based evidence_frame_indices; answered requires at least one evidence frame.'
EXEC_QA_MODEL = 'qwen/qwen3.8-flash'
EXEC_QA_EXTRA_BODY = {'reasoning': {'enabled': False}}
EXEC_QA_INSTRUCTIONS = 'You get a piece of code, its output (stdout, stderr and exit code) and a question about that output. Answer the question using only the output, with the details the question asks for. If the code failed or the output does not contain what the question asks, say so and give the error.'

SCHEMA_HEADER = (
    'The JSON Schema of the caption object of every index record (version v0, also at /index/schemas/v0.json):\n'
    + json.dumps(json.loads(INITIAL_SCHEMA.read_text()), ensure_ascii=False, indent=2)
)

SKILL_BODY = """Answer the question from the video index at /index/index.json. The video is the first-person footage of Jake, who wears the camera, over several days; the questions are asked by Jake, so I, me and my refer to Jake. The user message gives the day and the time at which the question is asked. The index is a JSON list of chronological one-minute clips. Each record has clip_id, day (DAY1 to DAY7), start_time and end_time (wall-clock time of that day, HH:MM:SS), tick_start_s and tick_end_s (seconds of recorded footage within that day), s1_version and caption. /index/schemas/<s1_version>.json is the JSON Schema of the caption object of that version.

The caption has dense_caption (a detailed description of the minute in time order), events (what happens, one entry per occurrence, with start_s, end_s, type, who and description), unexpected (surprising or striking things with start_s and description), entities (people, animals and objects with name, type, description and state), places (name and description), actions (who and a hierarchy from the overall activity down to the current step), relations (subject, relation, object), transcription (the cleaned transcript of the speech of the minute), explanation (the purpose of the minute in the whole video), on_screen_text (text and where) and mood. Seconds inside a caption are 0 to 59 of that minute.

Use the exec tool to run ordinary Python or bash that loads the JSON and searches, filters and aggregates these fields. Be mindful about tool call output size and chunk the output when needed; the captions are very long. Use the video_qa tool to ask a vision model a question about an interval of the footage: give day as the day of the record and start_s and end_s as tick seconds of the record. You have 20 model calls, shared by your own turns and your video_qa calls; after that you must answer without tools.

Give the final answer as your reply without a tool call. If the index does not contain the information needed to answer, answer NG (not given)."""
SKILL = SCHEMA_HEADER + '\n\n' + SKILL_BODY

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'exec',
            'strict': True,
            'description': 'Run bash or python code in your working directory /work and return its stdout, stderr and exit code. The code is killed after timeout_seconds.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'language': {'type': 'string', 'enum': ['bash', 'python']},
                    'code': {'type': 'string'},
                    'timeout_seconds': {
                        'type': 'integer',
                        'minimum': 1,
                        'maximum': 600,
                    },
                },
                'required': ['language', 'code', 'timeout_seconds'],
                'additionalProperties': False,
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'video_qa',
            'strict': True,
            'description': f'Ask a visual question about an interval of the raw footage. day is the day of the index record (for example DAY5); start_s and end_s are recording tick seconds within that day, as tick_start_s and tick_end_s in the record. Frames are sampled at fps frames per second and answered by a separate vision model; the images do not enter your context. fps can be at most {FPS_LIMIT}. The interval can be at most {INTERVAL_LIMIT_S // 60} minutes and at most {IMAGE_LIMIT} frames. Returns status (answered or unknown), the answer and the tick times of the evidence frames. Each call uses one model call of your budget.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'day': {'type': 'string'},
                    'start_s': {'type': 'number'},
                    'end_s': {'type': 'number'},
                    'fps': {'type': 'number'},
                    'question': {'type': 'string'},
                },
                'required': ['day', 'start_s', 'end_s', 'fps', 'question'],
                'additionalProperties': False,
            },
        },
    },
]
EXEC_QA_TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'exec_qa',
            'strict': True,
            'description': 'Run bash or python code in your working directory /work and ask a question about its output. The code is killed after timeout_seconds. The code, its stdout, stderr and exit code, and the question go to a separate language model, which answers the question from the output; the raw output does not enter your context. Returns that answer.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'language': {'type': 'string', 'enum': ['bash', 'python']},
                    'code': {'type': 'string'},
                    'timeout_seconds': {
                        'type': 'integer',
                        'minimum': 1,
                        'maximum': 600,
                    },
                    'question': {'type': 'string'},
                },
                'required': ['language', 'code', 'timeout_seconds', 'question'],
                'additionalProperties': False,
            },
        },
    },
    TOOLS[1],
]
VALIDATORS = {
    tool['function']['name']: tool['function']['parameters'] for tool in TOOLS
}
SUBMIT_TOOL = {
    'type': 'function',
    'function': {
        'name': 'submit_answer',
        'strict': True,
        'description': 'Submit the letter of the chosen option as your final answer. This ends the question.',
        'parameters': {
            'type': 'object',
            'properties': {
                'answer': {'type': 'string', 'enum': ['A', 'B', 'C', 'D']},
            },
            'required': ['answer'],
            'additionalProperties': False,
        },
    },
}


class InvalidTrace(Exception):
    pass


class ClipAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    status: Literal['answered', 'unknown']
    answer: str
    evidence_frame_indices: list[int]


def load_week():
    """Return the week clips with their day, wall-clock bounds, ASR and EgoLife transcript, and the footage timeline per day label. A1_JAKE's transcript is used where it exists; an hour without it uses the first of A2 to A6 that has that hour."""
    clips, timeline = build_clips('week')
    hours = {person: set() for person in PEOPLE}
    segments = {}
    for person in PEOPLE:
        for path in sorted((TRANSCRIPTS / person).glob('DAY*/*.srt')):
            key = (person, path.parent.name, int(path.stem[-8:-6]))
            hours[person].add(key[1:])
            segments[key] = [
                {
                    'id': f'{path.stem}:{subtitle.index}',
                    'start': key[2] * 3600 + subtitle.start.total_seconds(),
                    'end': key[2] * 3600 + subtitle.end.total_seconds(),
                    'text': subtitle.content,
                }
                for subtitle in srt.parse(path.read_text())
            ]
    week = {}
    for clip_id, clip in clips.items():
        day = clip['video'].upper()
        spoken = {}
        uncovered = []
        for piece in clip['sources']:
            for hour in range(
                int(piece['wall_start_s'] // 3600), int(piece['wall_end_s'] // 3600) + 1
            ):
                low_window = max(piece['wall_start_s'], hour * 3600)
                high_window = min(piece['wall_end_s'], (hour + 1) * 3600)
                if low_window >= high_window:
                    continue
                people = [person for person in PEOPLE if (day, hour) in hours[person]]
                if not people:
                    uncovered.append(
                        [
                            round(
                                piece['clip_start_s']
                                + low_window
                                - piece['wall_start_s'],
                                2,
                            ),
                            round(
                                piece['clip_start_s']
                                + high_window
                                - piece['wall_start_s'],
                                2,
                            ),
                        ]
                    )
                    continue
                for segment in segments[(people[0], day, hour)]:
                    low = max(low_window, segment['start'])
                    high = min(high_window, segment['end'])
                    if low >= high:
                        continue
                    local_start = round(
                        piece['clip_start_s'] + low - piece['wall_start_s'], 2
                    )
                    local_end = round(
                        piece['clip_start_s'] + high - piece['wall_start_s'], 2
                    )
                    if segment['id'] in spoken:
                        spoken[segment['id']]['end_s'] = local_end
                    else:
                        spoken[segment['id']] = {
                            'start_s': local_start,
                            'end_s': local_end,
                            'text': segment['text'],
                            'source': segment['id'],
                        }
        asr = json.loads((ASR / f'{clip_id}.json').read_text())
        wall_start = min(piece['wall_start_s'] for piece in clip['sources'])
        wall_end = max(piece['wall_end_s'] for piece in clip['sources'])
        week[clip_id] = {
            **clip,
            'day': day,
            'wall_start_s': wall_start,
            'wall_end_s': wall_end,
            'asr': {'language': asr['language'], 'transcription': asr['transcription']},
            'egolife_transcript': sorted(
                spoken.values(), key=lambda segment: segment['start_s']
            ),
            'egolife_uncovered_s': uncovered,
        }
    days = {}
    for video, rows in timeline.items():
        days[video.upper()] = []
        for row in rows:
            stamp = Path(row['path']).stem[-8:]
            days[video.upper()].append(
                {
                    'path': str(MEDIA / Path(row['path']).relative_to('/videos')),
                    'start': row['start'],
                    'end': row['end'],
                    'wall_start': int(stamp[:2]) * 3600
                    + int(stamp[2:4]) * 60
                    + int(stamp[4:6])
                    + int(stamp[6:8]) / 20,
                }
            )
    return week, days


def load_questions():
    questions = json.loads(QUESTIONS.read_text())
    for item in questions:
        raw = item['query_time']['time']
        item['day'] = item['query_time']['date']
        item['wall_s'] = (
            int(raw[:2]) * 3600
            + int(raw[2:4]) * 60
            + int(raw[4:6])
            + int(raw[6:8]) / 20
        )
        item['time_text'] = f'{raw[:2]}:{raw[2:4]}:{raw[4:6]}'
        item['metadata'] = {
            'questioner': 'A1_JAKE',
            'questioner_name': 'Jake',
            'perspective': 'first-person narrator and camera wearer',
            'query_time_raw_DHHMMSSFF': item['day'].removeprefix('DAY') + raw,
            'query_day': int(item['day'].removeprefix('DAY')),
            'query_time_of_day': item['time_text'],
            'query_frame_index': int(raw[6:8]),
        }
    return sorted(
        questions,
        key=lambda item: (int(item['day'].removeprefix('DAY')), item['wall_s']),
    )


def visible(records, question):
    """Index records that end at or before the time the question is asked."""
    day = int(question['day'].removeprefix('DAY'))
    return [
        record
        for record in records
        if int(record['day'].removeprefix('DAY')) < day
        or (
            record['day'] == question['day']
            and record['wall_end_s'] <= question['wall_s']
        )
    ]


def index_record(clip, caption, version):
    start = clip['wall_start_s']
    end = clip['wall_end_s']
    return {
        'clip_id': clip['clip_id'],
        'day': clip['day'],
        'start_time': f'{int(start // 3600):02d}:{int(start % 3600 // 60):02d}:{int(start % 60):02d}',
        'end_time': f'{int(end // 3600):02d}:{int(end % 3600 // 60):02d}:{int(end % 60):02d}',
        'tick_start_s': clip['tick_start_s'],
        'tick_end_s': clip['tick_end_s'],
        'wall_start_s': clip['wall_start_s'],
        'wall_end_s': clip['wall_end_s'],
        's1_version': version,
        'caption': caption,
    }


def read_caption_frames(clip, media=MEDIA):
    images = [None] * 60
    frames = [None] * 60
    pieces = clip['sources']
    owner = [
        max(
            index
            for index, piece in enumerate(pieces)
            if piece['clip_start_s'] <= second + 1e-6
        )
        for second in range(60)
    ]
    for number, piece in enumerate(pieces):
        offsets = [second for second in range(60) if owner[second] == number]
        if not offsets:
            continue
        path = media / Path(piece['path']).relative_to('/videos')
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            fps = float(stream.average_rate)
            requested = {}
            for second in offsets:
                video_s = piece['video_start_s'] + second - piece['clip_start_s']
                requested[min(round(video_s * fps), stream.frames - 1)] = second
            container.seek(
                int(min(requested) / fps / stream.time_base),
                stream=stream,
                backward=True,
            )
            for frame in container.decode(stream):
                video_s = float(frame.pts * stream.time_base)
                frame_index = round(video_s * fps)
                # A frame whose timestamp jitters past a requested index stands for that index: the first decoded frame at or after it.
                while requested and min(requested) <= frame_index:
                    second = requested.pop(min(requested))
                    buffer = io.BytesIO()
                    frame.to_image().resize(CAPTION_IMAGE_SIZE).save(
                        buffer, format='JPEG', quality=95
                    )
                    images[second] = buffer.getvalue()
                    frames[second] = {
                        'clip_s': second,
                        'path': piece['path'],
                        'video_s': video_s,
                        'frame_index': frame_index,
                        'source_size': [frame.width, frame.height],
                        'size': list(CAPTION_IMAGE_SIZE),
                    }
                if not requested:
                    break
            if requested:
                raise RuntimeError(
                    f'Video did not decode the requested frames: path={path} missing={requested}'
                )
    if any(image is None for image in images):
        raise RuntimeError(
            f'One-second sampling has uncovered positions: clip={clip["clip_id"]} frames={frames}'
        )
    return images, frames


async def caption_clip(client, port, clip, schema, prompt, output, media=MEDIA):
    """Caption one clip with the given caption schema and prompt; returns the caption, or None after writing failure.json when every attempt failed."""

    class Caption(RootModel[dict]):
        schema_document: ClassVar[dict] = schema

        @classmethod
        def model_json_schema(cls, *args, **kwargs):
            return cls.schema_document

        @model_validator(mode='after')
        def validate_document(self):
            Draft202012Validator(self.schema_document).validate(self.root)
            return self

    images, frames = await asyncio.to_thread(read_caption_frames, clip, media)
    text = f'One-minute segment. The 60 images are in order at offsets 0 through 59 seconds.\nRaw speech recognition transcript of this minute (detected language: {clip["asr"]["language"]}):\n{clip["asr"]["transcription"] or "none"}\nEgoLife transcript of this minute, made by people, with speaker names and English translations; start_s and end_s are seconds of this minute and source names the recording it comes from:\n{json.dumps(clip["egolife_transcript"], ensure_ascii=False) if clip["egolife_transcript"] else "none"}'
    if clip['egolife_uncovered_s']:
        text += f'\nThe EgoLife transcript does not cover these second ranges of this minute: {json.dumps(clip["egolife_uncovered_s"])}'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'input.json').write_text(
        json.dumps(
            {'clip': clip, 'frames': frames, 'prompt': text, 'port': port},
            ensure_ascii=False,
            indent=2,
        )
        + '\n'
    )
    content = [
        {
            'type': 'image_url',
            'image_url': {
                'url': 'data:image/jpeg;base64,' + base64.b64encode(image).decode()
            },
        }
        for image in images
    ]
    content.append({'type': 'text', 'text': text})
    feedback = ''
    offset = len(list(output.glob('attempt-*.json')))
    for attempt in range(offset + 1, offset + CAPTION_ATTEMPTS + 1):
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': content},
        ]
        if feedback:
            messages.append(
                {
                    'role': 'user',
                    'content': f'The previous attempt failed with this exact error. Correct it and return the requested caption:\n{feedback}',
                }
            )
        trace_path = output / f'attempt-{attempt:04d}.json'
        trace = {
            'model': MODEL,
            'port': port,
            'feedback': feedback,
            'started': time.time(),
            'response': None,
            'error': None,
        }
        trace_path.write_text(json.dumps(trace, ensure_ascii=False) + '\n')
        print(
            f'CAPTION_START clip={clip["clip_id"]} port={port} attempt={attempt}',
            flush=True,
        )
        try:
            response = await client.chat.completions.with_raw_response.parse(
                model=MODEL, messages=messages, response_format=Caption
            )
            trace['response'] = json.loads(response.text)
            trace['seconds'] = time.time() - trace['started']
            trace_path.write_text(json.dumps(trace, ensure_ascii=False) + '\n')
            parsed = response.parse().choices[0].message.parsed
            if parsed is None:
                raise SchemaValidationError(
                    f'Caption has no structured output: {trace["response"]["choices"][0]["message"]}'
                )
        except (BadRequestError, *RETRYABLE) as error:
            trace['error'] = {'type': type(error).__name__, 'message': str(error)}
            trace_path.write_text(json.dumps(trace, ensure_ascii=False) + '\n')
            print(
                f'CAPTION_ERROR clip={clip["clip_id"]} attempt={attempt} error={error}',
                flush=True,
            )
            if isinstance(error, BadRequestError):
                break
            feedback = str(error)
            await asyncio.sleep(2 ** (attempt - offset))
            continue
        (output / 'usage.json').write_text(
            json.dumps(trace['response']['usage'], indent=2) + '\n'
        )
        (output / 'caption.json').write_text(
            json.dumps(parsed.root, ensure_ascii=False, indent=2) + '\n'
        )
        print(
            f'CAPTION_DONE clip={clip["clip_id"]} port={port} seconds={trace["seconds"]:.1f} prompt_tokens={trace["response"]["usage"]["prompt_tokens"]} completion_tokens={trace["response"]["usage"]["completion_tokens"]}',
            flush=True,
        )
        return parsed.root
    (output / 'failure.json').write_text(
        json.dumps(
            {'error': trace['error'], 'attempts': attempt - offset}, ensure_ascii=False
        )
        + '\n'
    )
    print(
        f'SAMPLE_SKIPPED clip={clip["clip_id"]} stage=caption error={trace["error"]}',
        flush=True,
    )
    return None


def read_qa_frames(sources, times):
    starts = [source['start'] for source in sources]
    by_path = {}
    for position, tick in enumerate(times):
        source = sources[bisect.bisect_right(starts, tick) - 1]
        if not source['start'] <= tick < source['end']:
            raise ValueError(f'There is no footage at tick {tick}')
        by_path.setdefault(source['path'], []).append(
            (position, tick - source['start'])
        )
    images = [None] * len(times)
    recipes = [None] * len(times)
    for path, wanted in by_path.items():
        with av.open(path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            requested = {}
            for position, local_s in wanted:
                requested.setdefault(
                    min(round(local_s * fps), stream.frames - 1), []
                ).append(position)
            container.seek(
                int(min(requested) / fps / stream.time_base),
                stream=stream,
                backward=True,
            )
            for frame in container.decode(stream):
                frame_index = round(float(frame.pts * stream.time_base) * fps)
                # A frame whose timestamp jitters past a requested index stands for that index: the first decoded frame at or after it.
                if not requested or min(requested) > frame_index:
                    continue
                image = frame.to_image()
                image.thumbnail((854, 480))
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=95)
                while requested and min(requested) <= frame_index:
                    for position in requested.pop(min(requested)):
                        images[position] = buffer.getvalue()
                        recipes[position] = {
                            'tick_s': times[position],
                            'path': path,
                            'video_s': float(frame.pts * stream.time_base),
                            'frame_index': frame_index,
                            'size': image.size,
                        }
                if not requested:
                    break
            if requested:
                raise RuntimeError(
                    f'Video did not decode the requested frames: path={path} missing={requested}'
                )
    return images, recipes


async def harness_qa(
    name,
    output,
    runtime,
    question,
    skill,
    records,
    schemas,
    client,
    port,
    video_client,
    days,
    testing,
    choices,
    context_limit=None,
    base_tools=TOOLS,
    agent_model=MODEL,
    agent_extra_body=None,
    choice_letters=('A', 'B', 'C', 'D'),
    exec_client=None,
):
    """Answer one question with the tool loop of joint/agent_worker.py. exec runs in a container that mounts only the visible index; exec_qa (with base_tools=EXEC_QA_TOOLS) runs the code the same way and answers the agent's question about the output with EXEC_QA_MODEL through exec_client; video_qa reads frames here and only up to the time the question is asked (all footage when question['day'] is None). With choices the answer is the option letter given through the submit_answer tool. The budget is CALL_LIMIT model calls, or with context_limit the tools stop once a model call has used context_limit tokens of context. Returns result.json, or None after writing error.json when the trace failed."""
    if (output / 'result.json').exists():
        return json.loads((output / 'result.json').read_text())
    if output.exists():
        shutil.rmtree(output)
    if runtime.exists():
        shutil.rmtree(runtime)
    output.mkdir(parents=True)
    (runtime / 'work').mkdir(parents=True)
    (runtime / 'index' / 'schemas').mkdir(parents=True)
    (runtime / 'index' / 'index.json').write_text(
        json.dumps(records, ensure_ascii=False) + '\n'
    )
    for version, schema in schemas.items():
        (runtime / 'index' / 'schemas' / f'{version}.json').write_text(
            json.dumps(schema, ensure_ascii=False, indent=2) + '\n'
        )
    (output / 'visible.json').write_text(
        json.dumps([record['clip_id'] for record in records]) + '\n'
    )
    (output / 'question.json').write_text(
        json.dumps(
            {
                'ID': question['ID'],
                'day': question['day'],
                'time': question['time_text'],
                'question': question['question'],
            },
            ensure_ascii=False,
            indent=2,
        )
        + '\n'
    )
    limits = {}
    for day, sources in days.items():
        if question['day'] is None:
            limits[day] = sources[-1]['end']
        elif int(day.removeprefix('DAY')) < int(question['day'].removeprefix('DAY')):
            limits[day] = sources[-1]['end']
        elif day == question['day']:
            limits[day] = max(
                [
                    (
                        source['end']
                        if question['wall_s']
                        >= source['wall_start'] + source['end'] - source['start']
                        else source['start'] + question['wall_s'] - source['wall_start']
                    )
                    for source in sources
                    if question['wall_s'] > source['wall_start']
                ]
                + [0]
            )
    container = f'vsi-{name}-{uuid.uuid4().hex[:8]}'
    started = await asyncio.create_subprocess_exec(
        'docker',
        'run',
        '-d',
        '--init',
        '--name',
        container,
        '--network',
        'none',
        '--user',
        f'{os.getuid()}:{os.getgid()}',
        '-e',
        'HOME=/work',
        '-w',
        '/work',
        '-v',
        f'{runtime / "work"}:/work',
        '-v',
        f'{runtime / "index"}:/index:ro',
        '--entrypoint',
        'sleep',
        IMAGE,
        'infinity',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await started.communicate()
    if started.returncode:
        raise RuntimeError(f'docker run failed: {stderr.decode()}')
    events = (output / 'events.jsonl').open('a', buffering=1)
    usage = {
        'agent_prompt_tokens': 0,
        'agent_completion_tokens': 0,
        'agent_cached_tokens': None,
        'agent_reasoning_tokens': None,
        'video_qa_prompt_tokens': 0,
        'video_qa_completion_tokens': 0,
        'exec_qa_prompt_tokens': 0,
        'exec_qa_completion_tokens': 0,
    }
    state = {'calls': 0, 'video_calls': 0, 'exec_qa_calls': 0, 'context': 0}
    if question['day'] is None:
        messages = [
            {'role': 'system', 'content': skill},
            {'role': 'user', 'content': f'Question: {question["question"]}'},
        ]
    else:
        messages = [
            {'role': 'system', 'content': skill},
            {
                'role': 'user',
                'content': f'Question time: {question["day"]} {question["time_text"]}\nQuestion: {question["question"]}',
            },
        ]

    async def complete(tools, tool_choice):
        state['calls'] += 1
        print(
            f'AGENT_CALL name={name} call={state["calls"]} messages={len(messages)} tools={tools is not None}',
            flush=True,
        )
        arguments = {'model': agent_model, 'messages': messages}
        if agent_extra_body is not None:
            arguments['extra_body'] = agent_extra_body
        if tools is not None:
            arguments['tools'] = tools
        if tool_choice is not None:
            arguments['tool_choice'] = tool_choice
        response = await client.chat.completions.create(**arguments)
        record = response.model_dump(mode='json')
        events.write(
            json.dumps(
                {
                    'type': 'response',
                    'call': state['calls'],
                    'time': time.time(),
                    'response': record,
                },
                ensure_ascii=False,
            )
            + '\n'
        )
        usage['agent_prompt_tokens'] += response.usage.prompt_tokens
        usage['agent_completion_tokens'] += response.usage.completion_tokens
        state['context'] = (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )
        if response.usage.prompt_tokens_details is not None:
            usage['agent_cached_tokens'] = (
                response.usage.prompt_tokens_details.cached_tokens
                if usage['agent_cached_tokens'] is None
                else usage['agent_cached_tokens']
                + response.usage.prompt_tokens_details.cached_tokens
            )
        if response.usage.completion_tokens_details is not None:
            usage['agent_reasoning_tokens'] = (
                response.usage.completion_tokens_details.reasoning_tokens
                if usage['agent_reasoning_tokens'] is None
                else usage['agent_reasoning_tokens']
                + response.usage.completion_tokens_details.reasoning_tokens
            )
        if response.choices[0].finish_reason not in ('stop', 'tool_calls'):
            raise InvalidTrace(
                f'Agent turn did not finish: {response.choices[0].finish_reason}'
            )
        message = response.choices[0].message.model_dump(exclude_none=True)
        messages.append(message)
        return message

    async def run_code(args):
        command = (
            ['bash', '-c', args['code']]
            if args['language'] == 'bash'
            else ['python', '-c', args['code']]
        )
        began = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            'docker',
            'exec',
            '-w',
            '/work',
            container,
            'timeout',
            '-s',
            'KILL',
            str(args['timeout_seconds']),
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        stdout = stdout.decode(errors='replace')
        stderr = stderr.decode(errors='replace')
        if process.returncode == 137:
            return f'{stdout}\n[stderr]\n{stderr}\n[killed after {args["timeout_seconds"]} seconds]'
        text = stdout
        if stderr:
            text += '\n[stderr]\n' + stderr
        print(
            f'EXEC_DONE name={name} language={args["language"]} exit={process.returncode} elapsed={time.monotonic() - began:.1f}',
            flush=True,
        )
        return text + f'\n[exit code {process.returncode}]'

    async def exec_qa(args, call_number):
        code_output = await run_code(args)
        prompt = json.dumps(
            {
                'language': args['language'],
                'code': args['code'],
                'output': code_output,
                'question': args['question'],
            },
            ensure_ascii=False,
        )
        trace_path = output / 'exec_qa' / f'call-{call_number:04d}.json'
        trace_path.parent.mkdir(exist_ok=True)
        trace = {
            'arguments': args,
            'model': EXEC_QA_MODEL,
            'instructions': EXEC_QA_INSTRUCTIONS,
            'output': code_output,
            'response': None,
        }
        trace_path.write_text(json.dumps(trace, ensure_ascii=False) + '\n')
        print(
            f'EXEC_QA_START name={name} call={call_number} output_chars={len(code_output)}',
            flush=True,
        )
        response = await exec_client.chat.completions.create(
            model=EXEC_QA_MODEL,
            messages=[
                {'role': 'system', 'content': EXEC_QA_INSTRUCTIONS},
                {'role': 'user', 'content': prompt},
            ],
            extra_body=EXEC_QA_EXTRA_BODY,
        )
        trace['response'] = response.model_dump(mode='json')
        trace_path.write_text(json.dumps(trace, ensure_ascii=False) + '\n')
        usage['exec_qa_prompt_tokens'] += response.usage.prompt_tokens
        usage['exec_qa_completion_tokens'] += response.usage.completion_tokens
        if response.choices[0].finish_reason == 'content_filter':
            print(f'EXEC_QA_FILTERED name={name} call={call_number}', flush=True)
            raise ValueError(
                'The exec_qa model stopped on its content filter and gave no answer; ask a narrower question or print less verbatim text'
            )
        if response.choices[0].finish_reason == 'error':
            print(f'EXEC_QA_PROVIDER_ERROR name={name} call={call_number}', flush=True)
            raise ValueError(
                'The exec_qa model provider failed while answering and gave no answer; call exec_qa again'
            )
        if response.choices[0].finish_reason != 'stop':
            raise InvalidTrace(
                f'exec_qa did not finish: {response.choices[0].finish_reason} trace={trace_path}'
            )
        print(f'EXEC_QA_DONE name={name} call={call_number}', flush=True)
        return json.dumps(
            {'answer': response.choices[0].message.content}, ensure_ascii=False
        )

    async def video_qa(args, call_number):
        start_s = args['start_s']
        end_s = args['end_s']
        fps = args['fps']
        if args['day'] not in days:
            raise ValueError(f'Unknown day {args["day"]}; days are {sorted(days)}')
        if args['day'] not in limits:
            raise ValueError(
                f'{args["day"]} is after the question time {question["day"]} {question["time_text"]}; its footage is not available'
            )
        if (
            not math.isfinite(start_s)
            or not math.isfinite(end_s)
            or start_s < 0
            or end_s <= start_s
        ):
            raise ValueError('The interval must be finite, non-negative and increasing')
        if end_s > limits[args['day']]:
            raise ValueError(
                f'Footage after the question time {question["day"]} {question["time_text"]} is not available; on {args["day"]} end_s can be at most tick {limits[args["day"]]:.2f}'
            )
        if end_s - start_s > INTERVAL_LIMIT_S:
            raise ValueError(
                f'video_qa rejects intervals longer than {INTERVAL_LIMIT_S // 60} minutes'
            )
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('fps must be positive')
        if fps > FPS_LIMIT:
            raise ValueError(f'fps can be at most {FPS_LIMIT}')
        count = math.ceil((end_s - start_s) * fps)
        if count > IMAGE_LIMIT:
            raise ValueError(
                f'The sampling gives {count} frames; at most {IMAGE_LIMIT} are allowed'
            )
        times = [start_s + index / fps for index in range(count)]
        images, recipes = await asyncio.to_thread(
            read_qa_frames, days[args['day']], times
        )
        prompt = json.dumps(
            {
                'question': args['question'],
                'day': args['day'],
                'start_s': start_s,
                'end_s': end_s,
                'fps': fps,
                'frame_times_s': times,
            },
            ensure_ascii=False,
        )
        content = [{'type': 'text', 'text': prompt}]
        content.extend(
            {
                'type': 'image_url',
                'image_url': {
                    'url': 'data:image/jpeg;base64,' + base64.b64encode(image).decode()
                },
            }
            for image in images
        )
        trace_path = output / 'video_qa' / f'call-{call_number:04d}.json'
        trace_path.parent.mkdir(exist_ok=True)
        trace = {
            'arguments': args,
            'model': MODEL,
            'instructions': VIDEO_QA_INSTRUCTIONS,
            'prompt': prompt,
            'frames': recipes,
            'response': None,
        }
        trace_path.write_text(json.dumps(trace, ensure_ascii=False) + '\n')
        print(
            f'VIDEO_QA_START name={name} call={call_number} day={args["day"]} start={start_s} end={end_s} frames={len(times)}',
            flush=True,
        )
        try:
            response = await video_client.chat.completions.parse(
                model=MODEL,
                messages=[
                    {'role': 'system', 'content': VIDEO_QA_INSTRUCTIONS},
                    {'role': 'user', 'content': content},
                ],
                response_format=ClipAnswer,
            )
        except LengthFinishReasonError as error:
            trace['response'] = error.completion.model_dump(mode='json', warnings=False)
            trace_path.write_text(json.dumps(trace, ensure_ascii=False) + '\n')
            usage['video_qa_prompt_tokens'] += error.completion.usage.prompt_tokens
            usage['video_qa_completion_tokens'] += error.completion.usage.completion_tokens
            print(f'VIDEO_QA_LENGTH name={name} call={call_number} frames={len(times)} prompt_tokens={error.completion.usage.prompt_tokens}', flush=True)
            raise ValueError(
                f'The vision model ran out of context with {len(times)} frames ({error.completion.usage.prompt_tokens} prompt tokens) and gave no answer; ask about a shorter interval or with a lower fps'
            )
        trace['response'] = response.model_dump(mode='json', warnings=False)
        trace_path.write_text(json.dumps(trace, ensure_ascii=False) + '\n')
        usage['video_qa_prompt_tokens'] += response.usage.prompt_tokens
        usage['video_qa_completion_tokens'] += response.usage.completion_tokens
        if response.choices[0].finish_reason != 'stop':
            raise InvalidTrace(
                f'video_qa did not finish: {response.choices[0].finish_reason} trace={trace_path}'
            )
        submission = response.choices[0].message.parsed
        if submission is None:
            raise InvalidTrace(
                f'video_qa returned no structured answer: trace={trace_path}'
            )
        print(
            f'VIDEO_QA_DONE name={name} call={call_number} status={submission.status}',
            flush=True,
        )
        if submission.status == 'answered' and not submission.evidence_frame_indices:
            raise ValueError(
                'The vision model answered without evidence frames; ask again'
            )
        if any(
            index < 0 or index >= len(times)
            for index in submission.evidence_frame_indices
        ):
            raise ValueError(
                'The vision model cited frames outside the sampled sequence; ask again'
            )
        return json.dumps(
            {
                'status': submission.status,
                'answer': submission.answer,
                'day': args['day'],
                'evidence_times_s': [
                    times[index] for index in submission.evidence_frame_indices
                ],
            },
            ensure_ascii=False,
        )

    answer = None
    exhausted = False
    submit_tool = json.loads(json.dumps(SUBMIT_TOOL))
    submit_tool['function']['parameters']['properties']['answer']['enum'] = list(choice_letters)
    tools = base_tools + [submit_tool] if choices else base_tools
    offered = {
        tool['function']['name']: tool['function']['parameters'] for tool in tools
    }
    if context_limit is None:
        within_budget = lambda: state['calls'] < CALL_LIMIT
        budget_text = f'Your budget of {CALL_LIMIT} model calls is used up'
    else:
        within_budget = lambda: state['context'] < context_limit
        budget_text = f'Your context has reached {context_limit} tokens'
    try:
        while within_budget():
            message = await complete(tools, None)
            if 'tool_calls' not in message:
                if choices:
                    messages.append(
                        {
                            'role': 'user',
                            'content': 'Give your final answer by calling submit_answer with the letter of the closest option.',
                        }
                    )
                    continue
                answer = message['content']
                break
            if context_limit is not None and not within_budget():
                for call in message['tool_calls']:
                    messages.append(
                        {
                            'role': 'tool',
                            'tool_call_id': call['id'],
                            'content': json.dumps(
                                {'error': f'{budget_text}; this tool call was not run.'}
                            ),
                        }
                    )
                (output / 'messages.json').write_text(
                    json.dumps(messages, ensure_ascii=False, indent=2) + '\n'
                )
                break
            for call in message['tool_calls']:
                events.write(
                    json.dumps(
                        {'type': 'tool_call', 'time': time.time(), 'call': call},
                        ensure_ascii=False,
                    )
                    + '\n'
                )
                try:
                    args = json.loads(call['function']['arguments'])
                    Draft202012Validator(offered[call['function']['name']]).validate(
                        args
                    )
                    invalid = None
                except (json.JSONDecodeError, SchemaValidationError, KeyError) as error:
                    invalid = json.dumps(
                        {
                            'error': f'Invalid tool call {call["function"]["name"]}: {error}'
                        },
                        ensure_ascii=False,
                    )
                if invalid is not None:
                    tool_output = invalid
                elif call['function']['name'] == 'submit_answer':
                    answer = args['answer']
                    tool_output = json.dumps({'status': 'submitted'})
                elif call['function']['name'] == 'exec':
                    tool_output = await run_code(args)
                elif call['function']['name'] == 'exec_qa':
                    state['exec_qa_calls'] += 1
                    try:
                        tool_output = await exec_qa(args, state['exec_qa_calls'])
                    except ValueError as error:
                        tool_output = json.dumps(
                            {'error': str(error)}, ensure_ascii=False
                        )
                elif context_limit is None and state['calls'] >= CALL_LIMIT:
                    tool_output = json.dumps(
                        {
                            'error': 'The model-call budget is used up; video_qa is unavailable.'
                        }
                    )
                else:
                    state['calls'] += 1
                    state['video_calls'] += 1
                    try:
                        tool_output = await video_qa(args, state['video_calls'])
                    except ValueError as error:
                        tool_output = json.dumps(
                            {'error': str(error)}, ensure_ascii=False
                        )
                events.write(
                    json.dumps(
                        {
                            'type': 'tool_result',
                            'time': time.time(),
                            'tool_call_id': call['id'],
                            'output': tool_output,
                        },
                        ensure_ascii=False,
                    )
                    + '\n'
                )
                messages.append(
                    {'role': 'tool', 'tool_call_id': call['id'], 'content': tool_output}
                )
                if answer is not None:
                    break
            (output / 'messages.json').write_text(
                json.dumps(messages, ensure_ascii=False, indent=2) + '\n'
            )
            if answer is not None:
                break
        if answer is None and choices:
            exhausted = True
            messages.append(
                {
                    'role': 'user',
                    'content': f'{budget_text} and the other tools are no longer available. Call submit_answer with the letter of the closest option now.',
                }
            )
            final = await complete(
                [submit_tool],
                {'type': 'function', 'function': {'name': 'submit_answer'}},
            )
            if 'tool_calls' not in final:
                raise InvalidTrace('The forced submit_answer turn has no tool call.')
            try:
                args = json.loads(final['tool_calls'][0]['function']['arguments'])
                Draft202012Validator(submit_tool['function']['parameters']).validate(
                    args
                )
            except (json.JSONDecodeError, SchemaValidationError) as error:
                raise InvalidTrace(f'The forced submit_answer call is invalid: {error}')
            answer = args['answer']
        elif answer is None:
            exhausted = True
            messages.append(
                {
                    'role': 'user',
                    'content': f'{budget_text} and the tools are no longer available. Give your final answer now.',
                }
            )
            answer = (await complete(None, None))['content']
        (output / 'messages.json').write_text(
            json.dumps(messages, ensure_ascii=False, indent=2) + '\n'
        )
        if answer is None or not answer.strip():
            raise InvalidTrace(
                'The final answer is empty; NG is the answer when the index contains no answer.'
            )
    except (APIError, LengthFinishReasonError, ValidationError, InvalidTrace) as error:
        (output / 'messages.json').write_text(
            json.dumps(messages, ensure_ascii=False, indent=2) + '\n'
        )
        (output / 'error.json').write_text(
            json.dumps(
                {
                    'type': type(error).__name__,
                    'error': str(error),
                    'calls': state['calls'],
                    'usage': usage,
                },
                ensure_ascii=False,
            )
            + '\n'
        )
        print(f'SAMPLE_FAILED name={name} error={error}', flush=True)
        if testing:
            raise
        return None
    finally:
        events.close()
        removed = await asyncio.create_subprocess_exec(
            'docker',
            'rm',
            '-f',
            container,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, removed_error = await removed.communicate()
        if removed.returncode:
            raise RuntimeError(
                f'Could not remove container {container}: {removed_error.decode()}'
            )
        shutil.rmtree(runtime / 'index')
    result = {
        'model': agent_model,
        'video_qa_model': MODEL,
        'output': {'answer': answer.strip()},
        'calls': state['calls'],
        'video_qa_calls': state['video_calls'],
        'exec_qa_calls': state['exec_qa_calls'],
        'budget_exhausted': exhausted,
        'last_context_tokens': state['context'],
        'usage': usage,
    }
    (output / 'result.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    )
    print(
        f'ANSWER_DONE name={name} calls={state["calls"]} video_qa_calls={state["video_calls"]} exhausted={exhausted}',
        flush=True,
    )
    return result


async def judge(client, question, answer, directory, testing):
    """Score one answer with GPT-5 and the evaluate_mm_official.py rubric, with the EgoLife metadata line. The output must end with Final Score; errors are fed back for up to JUDGE_ATTEMPTS attempts."""
    saved = directory / 'judgment.json'
    if saved.exists():
        return json.loads(saved.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    prompt = f'Question: {question["question"]}\nMetadata: {json.dumps(question["metadata"], ensure_ascii=False)}\nGroundtruth answer: {question["answer"]}\nCandidate answer: {answer}\nYour response: '
    feedback = ''
    offset = len(list(directory.glob('judge-*.json')))
    for attempt in range(offset + 1, offset + (1 if testing else JUDGE_ATTEMPTS) + 1):
        messages = [
            {'role': 'system', 'content': RUBRIC},
            {'role': 'user', 'content': prompt},
        ]
        if feedback:
            messages.append(
                {
                    'role': 'user',
                    'content': f'Correct this output error and follow the required scoring format:\n{feedback}',
                }
            )
        trace_path = directory / f'judge-{attempt:04d}.json'
        trace = {
            'request': {'model': JUDGE_MODEL, 'messages': messages},
            'response': None,
            'error': None,
        }
        trace_path.write_text(json.dumps(trace, ensure_ascii=False) + '\n')
        print(
            f'JUDGE_START id={question["ID"]} attempt={attempt} path={trace_path}',
            flush=True,
        )
        try:
            response = await client.chat.completions.create(
                model=JUDGE_MODEL, messages=messages
            )
            trace['response'] = response.model_dump(mode='json')
            text = response.choices[0].message.content
            if text is None:
                raise SchemaValidationError('Judge returned no text.')
            match = re.search(r'Final Score:\s*([0-5])\s*$', text, re.IGNORECASE)
            if match is None:
                raise SchemaValidationError(f'Invalid official judge output: {text}')
            score = int(match.group(1))
        except RETRYABLE as error:
            trace['error'] = {'type': type(error).__name__, 'message': str(error)}
            trace_path.write_text(json.dumps(trace, ensure_ascii=False) + '\n')
            print(
                f'JUDGE_ERROR id={question["ID"]} attempt={attempt} error={error}',
                flush=True,
            )
            if testing:
                raise
            feedback = str(error)
            await asyncio.sleep(2 ** (attempt - offset))
            continue
        trace_path.write_text(json.dumps(trace, ensure_ascii=False) + '\n')
        result = {
            'score': score,
            'mapped_score': 1.0 if score in (4, 5) else 0.5 if score == 3 else 0.0,
            'analysis': text,
            'trace': trace_path.name,
            'usage': trace['response']['usage'],
        }
        saved.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(f'JUDGE_DONE id={question["ID"]} score={score}', flush=True)
        return result
    (directory / 'judge-failure.json').write_text(
        json.dumps({'error': feedback, 'attempts': JUDGE_ATTEMPTS}, ensure_ascii=False)
        + '\n'
    )
    print(
        f'SAMPLE_SKIPPED id={question["ID"]} stage=judge error={feedback}', flush=True
    )
    return None
