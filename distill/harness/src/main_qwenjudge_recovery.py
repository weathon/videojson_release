"""The long-video question answering harness, one file for every dataset and every model; what differs between runs is set by flags.

An agent answers a question about a long video from a text index, without seeing the video. One container per question (image video-sql-joint:cerebras-claude-0.2.160, no network, or the docker network of --container-network) has the minute index as /data/minutes.json and the hour index as /data/hours.json, read-only. The tools are exec (bash or Python in the container, timeout_seconds 1 to 600 and killed after it, marked in the output; code without the docstring or # comment is run anyway, with a reminder before the result; an output of more than 36,000 characters is not returned, the agent is told to print less or use exec_qa, and the agent turn that asked for it is not counted toward the budget, one turn back per turn), exec_qa (the same code, and the qa model answers a question about its full output), video_qa (frames of an interval at fps from start_s, long side 1080, at most 0.5 fps, 20 minutes and 650 frames; the qa model gives a schema-constrained status, answer and evidence frames) and submit_answer. A budget of 30 model calls applies, counted as vsi: each agent call and each video_qa call counts, video_qa returns the vsi budget error once 30 are used, and once 30 are used the agent gets the vsi message and one forced turn that offers only submit_answer. The context limit is 250,000 tokens: two reminders, then the question ends with the answer A (choice datasets) or null (open datasets). The instructions are the vsi SKILL: the caption schema, then the skill text of the dataset.

The agent and the qa model are served with the OpenAI Chat Completions API on ports of this machine (--agent-ports, the questions alternating between them, and --qa-port); the agent is called with tool_choice required and gets every earlier turn's reasoning back. The main agent keeps its original reasoning and sampling settings. Both QA helpers explicitly disable thinking, make one request without SDK retries, and have a 600 second wall deadline; a helper timeout is returned as a tool error. A qa request over the context length of the server goes back to the agent as an error, and so do an exec_qa answer that runs to the end of the context and an exec_qa or video_qa reply that stops without an answer.

Datasets (--dataset):
egolifeqa: the EgoLifeQA A1_JAKE questions (500, Chinese question and options, four options), scored by letter match. Jake's minute and hour index (data/egolife/) and footage are used up to the question time; the skill text is prompts/egolife/vsi_skill_template_30.txt for Jake and four options.
memreason: the EgoMemReason questions (500, the public annotations, English, four to ten options, no answer key). Each question is asked by one EgoLife wearer, whose index and footage are used up to the question time; the skill text has the wearer's name and the question's number of options. submission.json in the leaderboard format is written into the run directory.
racklife: the RackLife v3 questions (168, dcworld/benchmark/questions_mc.json, English, four options), scored by letter match. Alex's minute and hour index of the v3 videos (data/racklife/run4/) and footage are used up to the question time; the skill text is prompts/racklife/skill_30.txt.
mm_week, mm_day, mm_month: the MM-Lifelong week test (200), day test (200) and month val (623) questions, open answer, no question time, the whole index and footage available. week is over the A1_JAKE index and footage (video_qa takes a day), day over the one day video and month over the 23 month videos (video_qa takes a video). The skill text is prompts/mm_lifelong/vsi_skill_<subset>_30.txt. The answers are judged with the official MM-Lifelong judge prompt and openai/gpt-5 through OpenRouter (the key in the .env variable of --judge-key-env; up to 3 attempts to get a 0-5 score; 4-5 count 1, 3 counts 0.5); the judge cost that OpenRouter reports goes to spent.jsonl.

--skill appends the text of one or more files to the instructions, in order. --no-video-qa takes video_qa away: the agent has exec, exec_qa and submit_answer only, and the skill text of the dataset is sent without its sentence about the video_qa tool and without the mention of video_qa calls in its budget sentence (a --skill file is sent as it is, so one without video_qa has to be given). --merge-commands puts the sentence of prompts/skill_merge_commands.txt after the first sentence of the Budget paragraph of the --skill text. --reasoning-effort sends reasoning_effort with every agent, exec_qa and video_qa request. answer.json has the input and output tokens of every agent, exec_qa and video_qa call. --split picks the month questions, val (623) or train (266). --trials N rolls an open question out again, each trial in q<id>/trial<k>/, until a trial is judged 1 or it has N trials. --skill-dir mounts a directory read-only at /skill in the container (the script library of a --skill text). --magic-index gives the agent MAGIC-Video's text memory exported to JSON by baselines/magic/export_index.py instead of our index: the directory (egolifeqa: its q<ID> subdirectory, filtered by the question time as MAGIC filters it) is mounted read-only at /data, and prompts/magic_index/<dataset>.txt replaces the caption schema and the first two paragraphs of the skill text. --bilingual (egolifeqa) gives the Chinese question followed by the English one, and every option as 'Chinese / English'. --comment-code appends prompts/comment_code.txt to the instructions: the agent is told to comment its exec and exec_qa code chunk by chunk and line by line. The run directory gets the instructions (instructions.txt) and the agent's tools (agent_tools.json). --open-answer (egolifeqa) keeps the options in the question but takes a short free-text answer (the skill text asks for one instead of the letter, submit_answer takes any text) and judges it as the open datasets are judged, with the question and its options as the question and the gold letter with its option text as the gold answer. --open-answer (racklife) gives the question without its options (the skill text loses its sentence about the four options), takes a short free-text answer and does not judge it here: the answers are judged afterwards by a workflow.

A run that ends with a reply without a tool call has no answer; that and a tool call that the Agents SDK does not accept rerun the whole question, up to 3 more times, and then the question fails alone (error.txt). A question also fails alone when its prompt is over the agent's context length or its forced last turn does not end the run. A request of the agent or inside a tool that still fails after the SDK's retries (connection, rate limit, 5xx) stops the question for this launch without a file, so a relaunch runs it again; a judge that gives no score fails the question alone and a relaunch only judges it. Any other error stops the run. Per question files go to runs/<dataset>/<run_name>/q<id>/ (answer.json, messages.json, tools.json, judge.json); accuracy, longest context, average latency per agent and qa model response, results per minute and ETA go to wandb and stdout. Finished questions are skipped on relaunch.

Usage: .venv/bin/python src/main.py --dataset <dataset> --run-name <name> --questions <number of questions, from the first> --concurrency <questions at a time, split evenly over the agent ports> --agent-model <name> --agent-ports <port> [<port> ...] --qa-model <name> --qa-port <port> [--api-key-env <variable in .env with the key of the model servers; without it the key EMPTY is sent>] [--temperature <t>] [--top-p <p>] [--top-k <k>] [--presence-penalty <x>] [--skill <file> [<file> ...]] [--no-video-qa] [--merge-commands] [--reasoning-effort <effort>] [--split <val|train>, mm_month only and required there] [--trials <n>] [--skill-dir <dir>] [--base-url <URL of an OpenAI-compatible API for the agent and qa requests; the ports then only group the questions>] [--anthropic-workspace-env <variable in .env with the anthropic-workspace-id header>] [--container-network <docker network of the question containers instead of none>] [--judge-key-env <variable in .env with the OpenRouter key of the judge, open datasets only and required there>] [--keys <key> [<key> ...]] [--force-decision] [--decision-dir <library of the decision model, mounted read-only at /decision>] [--service-tier <OpenAI service tier of the agent and qa requests, e.g. flex>] [--magic-index <directory of a dataset written by baselines/magic/export_index.py>] [--bilingual] [--comment-code] [--open-answer]"""

import argparse
import ast
import asyncio
import base64
import bisect
import contextvars
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import traceback
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
import wandb
from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunConfig,
    RunContextWrapper,
    RunHooks,
    Runner,
    ToolsToFinalOutputResult,
    function_tool,
    set_tracing_disabled,
)
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError, UserError
from agents.run_config import ModelInputData
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    DefaultAsyncHttpxClient,
    InternalServerError,
    LengthFinishReasonError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    required=True,
    choices=['egolifeqa', 'memreason', 'mm_week', 'mm_day', 'mm_month', 'video_index', 'racklife'],
)
parser.add_argument('--run-name', required=True)
parser.add_argument('--external-month-dir')
parser.add_argument('--questions', required=True, type=int)
parser.add_argument('--concurrency', required=True, type=int)
parser.add_argument('--agent-model', required=True)
parser.add_argument('--agent-ports', required=True, type=int, nargs='+')
parser.add_argument('--qa-model', required=True)
parser.add_argument('--qa-port', type=int)
# --qa-base-url: exec_qa and video_qa go to that URL (for example OpenRouter) with the key of the .env variable --qa-key-env, while the agent stays on --agent-ports; --qa-port is then not used
parser.add_argument('--qa-base-url')
parser.add_argument('--qa-key-env')
parser.add_argument('--qa-no-reasoning-effort', action='store_true')
parser.add_argument('--api-key-env')
parser.add_argument('--temperature', type=float)
parser.add_argument('--top-p', type=float)
parser.add_argument('--top-k', type=int)
parser.add_argument('--presence-penalty', type=float)
parser.add_argument('--skill', nargs='+')
parser.add_argument('--no-video-qa', action='store_true')
parser.add_argument('--merge-commands', action='store_true')
parser.add_argument('--reasoning-effort')
parser.add_argument('--split', choices=['val', 'train'])
parser.add_argument('--trials', type=int)
parser.add_argument('--skill-dir')
parser.add_argument('--base-url')
parser.add_argument('--anthropic-workspace-env')
parser.add_argument('--container-network')
parser.add_argument('--judge-key-env')
parser.add_argument('--judge-model', default='openai/gpt-5')
parser.add_argument('--judge-base-url', default='https://openrouter.ai/api/v1')
parser.add_argument('--judge-max-tokens', type=int)
parser.add_argument('--judge-reasoning-effort')
parser.add_argument('--judge-timeout', type=float, default=60)
parser.add_argument('--keys', nargs='+')
parser.add_argument('--force-decision', action='store_true')
parser.add_argument('--decision-dir')
parser.add_argument('--service-tier')
parser.add_argument('--magic-index')
parser.add_argument('--bilingual', action='store_true')
parser.add_argument('--comment-code', action='store_true')
parser.add_argument('--open-answer', action='store_true')
args = parser.parse_args()
if args.open_answer and args.dataset not in ('egolifeqa', 'racklife'):
    raise ValueError('--open-answer is for egolifeqa and racklife only')
# the judged questions: the open MM-Lifelong datasets, and EgoLifeQA with --open-answer; RackLife with --open-answer is judged afterwards by a workflow, not here
JUDGED = args.dataset not in ('egolifeqa', 'memreason', 'video_index', 'racklife') or (args.open_answer and args.dataset == 'egolifeqa')
if (args.judge_key_env is not None) != JUDGED:
    raise ValueError(
        '--judge-key-env is given for the judged questions only (the open MM-Lifelong datasets, egolifeqa with --open-answer), and they need it'
    )
if (args.split is not None) != (args.dataset == 'mm_month'):
    raise ValueError('--split is given for mm_month only, and mm_month needs it')
# a trial is judged, so only the open datasets have trials
if args.trials is not None and args.dataset in ('egolifeqa', 'memreason', 'video_index', 'racklife'):
    raise ValueError('--trials is for the open MM-Lifelong datasets only')

# through OpenRouter
judge_model = args.judge_model
# the budget: agent calls and video_qa calls together, counted as vsi
CALL_LIMIT = 30
CONTEXT_LIMIT = 250_000
# the 90th percentile of the exec output lengths of the runs without a limit (36,387 characters), rounded
OUTPUT_LIMIT = 36_000
# video_qa limits of vsi
FPS_LIMIT = 0.5
INTERVAL_LIMIT_S = 1200
IMAGE_LIMIT = 650
FRAME_DECODE_SEMAPHORE = asyncio.Semaphore(16)

if args.external_month_dir and args.dataset != 'mm_month':
    raise ValueError('--external-month-dir requires mm_month')
ROOT = Path(__file__).resolve().parents[1]
DATASET = args.dataset
# the datasets with options and a question time; the others are the open MM-Lifelong sets
WEARER = DATASET in ('egolifeqa', 'memreason', 'racklife')
CHOICE = WEARER or DATASET == 'video_index'
if DATASET == 'video_index' and (
    args.magic_index or args.container_network or args.force_decision or args.decision_dir
):
    raise ValueError('video_index uses its own index and isolated native tools')
MM = Path('/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443')
IMAGE = (json.loads((Path(args.external_month_dir) / 'sandbox-image.json').read_text())['image_id']
         if args.external_month_dir else 'video-sql-joint:cerebras-claude-0.2.160')
RUN = ROOT / 'runs' / DATASET / args.run_name
PROMPT = {path.stem: path.read_text() for path in (ROOT / 'prompts').glob('*.txt')}
TOOL_PROMPT = {path.stem: path.read_text() for path in (ROOT / 'prompts/tools').glob('*.txt')}
EGOLIFE_PROMPT = {path.stem: path.read_text() for path in (ROOT / 'prompts/egolife').glob('*.txt')}
EGOLIFE_TOOL_PROMPT = {path.stem: path.read_text() for path in (ROOT / 'prompts/egolife/tools').glob('*.txt')}
MM_TOOL_PROMPT = {path.stem: path.read_text() for path in (ROOT / 'prompts/mm_lifelong/tools').glob('*.txt')}
VIDEO_QA_ERRORS = json.loads(
    (
        ROOT / ('prompts/egolife/video_qa_errors.json' if WEARER else 'prompts/mm_lifelong/video_qa_errors.json')
    ).read_text()
)
# the sampling parameters whose flags are given
SAMPLING = {'max_tokens': 16384}
if args.temperature is not None:
    SAMPLING['temperature'] = args.temperature
if args.top_p is not None:
    SAMPLING['top_p'] = args.top_p
if args.presence_penalty is not None:
    SAMPLING['presence_penalty'] = args.presence_penalty
if args.top_k is not None or args.reasoning_effort is not None or args.service_tier is not None:
    SAMPLING['extra_body'] = {}
if args.top_k is not None:
    SAMPLING['extra_body']['top_k'] = args.top_k
# for every request to the agent and QA servers (agent, exec_qa, video_qa); the Qwen3.8-Flash-Next server takes xhigh (its default), medium or low
if args.reasoning_effort is not None:
    SAMPLING['extra_body']['reasoning_effort'] = args.reasoning_effort
# the OpenAI service tier (flex: half price) for every agent and qa request; the judge is not affected
if args.service_tier is not None:
    SAMPLING['extra_body']['service_tier'] = args.service_tier
QA_SAMPLING = copy.deepcopy(SAMPLING)
if args.qa_no_reasoning_effort:
    QA_SAMPLING.get('extra_body', {}).pop('reasoning_effort', None)
QA_SAMPLING.setdefault('extra_body', {}).setdefault('chat_template_kwargs', {})['enable_thinking'] = False
HELPER_TIMEOUT_SECONDS = 600
VIDEO_QA_SYSTEM = EGOLIFE_PROMPT['video_qa_system'] + (
    '\nReturn a JSON object with status ("answered" or "unknown"), answer (a string), '
    'and evidence_frame_indices (a list of integers). Set status to "answered" whenever '
    'the supplied frames support an answer, and to "unknown" only when they do not.'
)

QUESTIONS = []
if DATASET == 'egolifeqa':
    PEOPLE = ('A1_JAKE',)
    # every question of EgoLifeQA A1_JAKE as key, set, person, ID, day, HHMMSSFF stamp, type, Chinese question and options and gold letter
    for item in json.loads(Path('/raid/wg25r/egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json').read_text()):
        QUESTIONS.append(
            {
                'key': f'q{item["ID"]}',
                'set': 'egolifeqa',
                'person': 'A1_JAKE',
                'ID': item['ID'],
                'day': item['query_time']['date'],
                'stamp': item['query_time']['time'],
                'type': item['type'],
                # --bilingual: the Chinese question, a new line and the English one; every option 'Chinese / English'
                'question': f'{item["question_chinese"]}\n{item["question"]}'
                if args.bilingual
                else item['question_chinese'],
                'options': {
                    letter: f'{item[f"choice_{letter.lower()}_chinese"]} / {item[f"choice_{letter.lower()}"]}'
                    if args.bilingual
                    else item[f'choice_{letter.lower()}_chinese']
                    for letter in 'ABCD'
                },
                'gold': item['answer'],
                'port': args.agent_ports[len(QUESTIONS) % len(args.agent_ports)],
            }
        )
if DATASET == 'memreason':
    PEOPLE = ('A1_JAKE', 'A2_ALICE', 'A3_TASHA', 'A4_LUCIA', 'A5_KATRINA', 'A6_SHURE')
    # every question as key, person (the wearer who asks), example_id, day, HHMMSS00 stamp, type, question and options; the public annotations have no answer key
    for item in [
        json.loads(line)
        for line in Path(
            '/raid/huggingface/hub/datasets--Ted412--EgoMemReason/snapshots/7e581505b9dce0e85193a27ae689ff899d0bc507/annotations_public.jsonl'
        )
        .read_text()
        .splitlines()
    ]:
        day, clock = item['query_time'].split(', ')
        QUESTIONS.append(
            {
                'key': f'q{item["example_id"]}',
                'person': item['identity'],
                'example_id': item['example_id'],
                'day': day,
                'stamp': clock.replace(':', '') + '00',
                'type': item['query_type'],
                'question': item['question'],
                'options': item['options'],
                'port': args.agent_ports[len(QUESTIONS) % len(args.agent_ports)],
            }
        )
if DATASET == 'racklife':
    PEOPLE = ('ALEX',)
    # every RackLife v3 question (dcworld/benchmark/questions_mc.json, four options) as key, person, ID, day, HHMMSSFF stamp, type, question and options and gold letter
    for item in json.loads((ROOT / 'dcworld/benchmark/questions_mc.json').read_text()):
        QUESTIONS.append(
            {
                'key': f'q{item["ID"]}',
                'person': 'ALEX',
                'ID': item['ID'],
                'day': item['query_time']['date'],
                'stamp': item['query_time']['time'],
                'type': item['type'],
                'question': item['question'],
                'options': {letter: item[f'choice_{letter.lower()}'] for letter in 'ABCD'},
                'gold': item['answer'],
                'port': args.agent_ports[len(QUESTIONS) % len(args.agent_ports)],
            }
        )
if DATASET == 'video_index':
    # The controller alone reads gold labels; only two caption files enter the tools.
    VIDEO_INDEX = ROOT / 'data/video_index/v1_1/cafb5bddb933bfcb76a3f394d15249432c1ad4a1'
    annotations = (VIDEO_INDEX / 'items/test.jsonl').read_bytes()
    if hashlib.sha256(annotations).hexdigest() != 'c82cb3c52983bd7f61788513860e00b0ab6cbdbd311dee4beff223cb1ff461e7':
        raise ValueError('Pinned Video-Index annotations changed')
    for item in map(json.loads, annotations.decode().splitlines()):
        options = dict(zip('ABCDEFGHIJ', item['options']))
        if (
            not 2 <= len(item['options']) <= 10
            or item['answer'] not in options
            or item['answer'] != 'ABCDEFGHIJ'[item['answer_idx']]
        ):
            raise ValueError(f'{item["item_id"]}: invalid options or gold')
        QUESTIONS.append(
            {
                'key': 'q' + item['item_id'],
                'item_id': item['item_id'],
                'question': item['question'],
                'options': options,
                'gold': item['answer'],
                'duration_s': item['duration_s'],
                'video_path': str(VIDEO_INDEX / item['video']),
                'video_sha256': item['video_id'],
                'port': args.agent_ports[len(QUESTIONS) % len(args.agent_ports)],
            }
        )
if DATASET == 'racklife':
    # the index of the v3 videos (scripts/caption_racklife.py, scripts/hourly_index_racklife.py) and their footage
    MINUTES = {'ALEX': json.loads((ROOT / 'data/racklife/run4/ALEX.json').read_text())}
    HOURS = {'ALEX': json.loads((ROOT / 'data/racklife/run4/ALEX_hourly.json').read_text())}
    TIMELINES = {'ALEX': json.loads((ROOT / 'data/racklife/run4/timeline.json').read_text())}
elif WEARER:
    MINUTES = {person: json.loads((ROOT / f'data/egolife/{person}.json').read_text()) for person in PEOPLE}
    HOURS = {person: json.loads((ROOT / f'data/egolife/{person}_hourly.json').read_text()) for person in PEOPLE}
    # footage files of each day of each wearer
    TIMELINES = {person: json.loads((ROOT / f'data/egolife/timelines/{person}.json').read_text()) for person in PEOPLE}
# the whole index: the MM-Lifelong questions have no question time
if DATASET == 'mm_week':
    MOUNTS = {'data/egolife/A1_JAKE.json': '/data/minutes.json', 'data/egolife/A1_JAKE_hourly.json': '/data/hours.json'}
    # footage files of each day of A1_JAKE, the MM-Lifelong week videos
    TIMELINE = json.loads((ROOT / 'data/egolife/timelines/A1_JAKE.json').read_text())
if DATASET == 'mm_day':
    MOUNTS = {'data/mm_lifelong/day.json': '/data/minutes.json', 'data/mm_lifelong/day_hourly.json': '/data/hours.json'}
    # the day video is one file that starts at second 0; the index calls it "day"
    path = MM / 'videos/day/0.mp4'
    probe = json.loads(
        subprocess.run(
            [
                'ffprobe',
                '-v',
                'error',
                '-select_streams',
                'v:0',
                '-show_entries',
                'stream=nb_frames,r_frame_rate:format=duration',
                '-of',
                'json',
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    TIMELINE = {
        'day': [
            {
                'path': str(path),
                'start': 0.0,
                'end': float(probe['format']['duration']),
                'frames': int(probe['streams'][0]['nb_frames']),
                'fps': float(Fraction(probe['streams'][0]['r_frame_rate'])),
            }
        ]
    }
if DATASET == 'mm_month':
    if args.external_month_dir:
        external = Path(args.external_month_dir).resolve()
        MOUNTS = {str(external / 'minutes.json'): '/data/minutes.json', str(external / 'hours.json'): '/data/hours.json'}
        TIMELINE = json.loads((external / 'timeline.json').read_text())
    else:
        MOUNTS = {
            'data/mm_lifelong/month.json': '/data/minutes.json',
            'data/mm_lifelong/month_hourly.json': '/data/hours.json',
        }
        # each month video is one file that starts at second 0
        TIMELINE = {}
        for video in range(1, 24):
            path = MM / f'videos/month/{video}.mp4'
            probe = json.loads(
                subprocess.run(
                    [
                        'ffprobe',
                        '-v',
                        'error',
                        '-select_streams',
                        'v:0',
                        '-show_entries',
                        'stream=nb_frames,r_frame_rate:format=duration',
                        '-of',
                        'json',
                        str(path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            )
            TIMELINE[str(video)] = [
                {
                    'path': str(path),
                    'start': 0.0,
                    'end': float(probe['format']['duration']),
                    'frames': int(probe['streams'][0]['nb_frames']),
                    'fps': float(Fraction(probe['streams'][0]['r_frame_rate'])),
                }
            ]
if not CHOICE:
    SUBSET = DATASET.removeprefix('mm_')
    # every question of the test file of day and week, or of the --split file (val or train) of month, as key, index, question type, question and gold answer
    for item in json.loads(((Path(args.external_month_dir) if args.external_month_dir else MM) / SUBSET / (f'{args.split}.json' if SUBSET == 'month' else 'test.json')).read_text()):
        QUESTIONS.append(
            {
                'key': f'q{item["index"]}',
                'index': item['index'],
                'type': item['question_type'],
                'question': item['question'],
                'gold': item['answer'],
                'port': args.agent_ports[len(QUESTIONS) % len(args.agent_ports)],
            }
        )
# the first questions, as many as the command line says
QUESTIONS = QUESTIONS[: args.questions]
# --keys: only these questions
if args.keys is not None:
    QUESTIONS = [question for question in QUESTIONS if question['key'] in args.keys]

if DATASET == 'video_index':
    if not QUESTIONS:
        raise ValueError('No Video-Index keys selected')
    for question in QUESTIONS:
        index = ROOT / 'data/video_index/indices' / question['item_id']
        if not all((index / name).is_file() for name in ('minutes.json', 'hours.json')):
            raise ValueError(f'Caption index incomplete: {index}')
        provenance = json.loads((index / 'provenance.json').read_text())
        if (
            provenance['video_sha256'] != question['video_sha256']
            or sha256_file(question['video_path']) != question['video_sha256']
            or any(
                hashlib.sha256((index / (name + '.json')).read_bytes()).hexdigest() != provenance[name + '_sha256']
                for name in ('minutes', 'hours')
            )
        ):
            raise ValueError('Video or caption index changed since preprocessing')

# the cost holder of the judge that makes the request
COST = contextvars.ContextVar('COST')


async def add_cost(response):
    """Add the cost that OpenRouter reports in the usage of every successful judge response to the current cost holder."""
    if response.status_code == 200:
        await response.aread()
        COST.get()['cost'] += response.json()['usage']['cost']


load_dotenv(ROOT / '.env')
# a vLLM server without a key takes any key
API_KEY = os.environ[args.api_key_env] if args.api_key_env is not None else 'EMPTY'
# with --base-url every agent and qa request goes to that URL and the ports only group the questions for --concurrency; --anthropic-workspace-env names the .env variable with the workspace id that a key without a workspace needs
HEADERS = (
    {'anthropic-workspace-id': os.environ[args.anthropic_workspace_env]}
    if args.anthropic_workspace_env is not None
    else None
)
# no timeout and 5 retries, as the clients of vsi
agent_clients = {
    port: AsyncOpenAI(
        base_url=args.base_url if args.base_url is not None else f'http://127.0.0.1:{port}/v1',
        api_key=API_KEY,
        timeout=None,
        max_retries=5,
        default_headers=HEADERS,
    )
    for port in args.agent_ports
}
if args.qa_base_url is not None:
    qa_client = AsyncOpenAI(base_url=args.qa_base_url, api_key=os.environ[args.qa_key_env], timeout=None, max_retries=0)
else:
    qa_client = AsyncOpenAI(
        base_url=args.base_url if args.base_url is not None else f'http://127.0.0.1:{args.qa_port}/v1',
        api_key=API_KEY,
        timeout=None,
        max_retries=0,
        default_headers=HEADERS,
    )
JUDGE_CONFIG = {
    'model': judge_model,
    'base_url': args.judge_base_url,
    'max_tokens': args.judge_max_tokens,
    'reasoning_effort': args.judge_reasoning_effort,
    'timeout': args.judge_timeout,
    'judge_system_sha256': hashlib.sha256(PROMPT['judge_system'].encode()).hexdigest(),
    'judge_user_sha256': hashlib.sha256(PROMPT['judge_user'].encode()).hexdigest(),
    'score_mapping': '4-5:1;3:0.5;0-2:0',
    'parse_attempts': 3,
    'required_finish_reason': 'stop',
}
JUDGE_CONFIG_SHA256 = hashlib.sha256(json.dumps(JUDGE_CONFIG, sort_keys=True).encode()).hexdigest()
JUDGE_SAMPLING = {}
if args.judge_max_tokens is not None:
    JUDGE_SAMPLING['max_tokens'] = args.judge_max_tokens
if args.judge_reasoning_effort is not None:
    JUDGE_SAMPLING['extra_body'] = {'reasoning_effort': args.judge_reasoning_effort}
JUDGE_COST_REPORTED = args.judge_base_url.rstrip('/') == 'https://openrouter.ai/api/v1'
if JUDGED:
    judge_client = AsyncOpenAI(
        base_url=args.judge_base_url,
        api_key=os.environ[args.judge_key_env],
        http_client=DefaultAsyncHttpxClient(
            event_hooks={'response': [add_cost]} if JUDGE_COST_REPORTED else {}
        ),
    )
set_tracing_disabled(True)
# the vsi SKILL of vsi/baseline_choice.py with a budget of 30 calls: the caption schema, then the skill text of the dataset (for the choice datasets with {name} and {option_count} still to fill), then the text of --skill
INSTRUCTIONS = (
    'The JSON Schema of the caption object of every index record:\n'
    + json.dumps(
        json.loads(((Path(args.external_month_dir) / 'caption_schema.json') if args.external_month_dir else (ROOT / 'prompts/egolife/vsi_caption_schema.json')).read_text()), ensure_ascii=False, indent=2
    )
    + '\n\n'
    + (
        (ROOT / 'prompts/video_index/vsi_skill_30.txt').read_text()
        if DATASET == 'video_index'
        else (ROOT / 'prompts/racklife/skill_30.txt').read_text()
        if DATASET == 'racklife'
        else (
            EGOLIFE_PROMPT['vsi_skill_template_30']
            if CHOICE
            else ((Path(args.external_month_dir) / 'skill.txt') if args.external_month_dir else (ROOT / f'prompts/mm_lifelong/vsi_skill_{SUBSET}_30.txt')).read_text()
        )
    )
)
# --magic-index: the description of MAGIC's index replaces the caption schema and the first two paragraphs of the skill text, which describe our index
if args.magic_index is not None:
    INSTRUCTIONS = (
        (ROOT / f'prompts/magic_index/{DATASET}.txt').read_text()
        + '\n'
        + '\n\n'.join(
            (
                EGOLIFE_PROMPT['vsi_skill_template_30']
                if CHOICE
                else ((Path(args.external_month_dir) / 'skill.txt') if args.external_month_dir else (ROOT / f'prompts/mm_lifelong/vsi_skill_{SUBSET}_30.txt')).read_text()
            ).split('\n\n')[2:]
        )
    )
# without video_qa the skill text loses its sentence about the tool, and its budget sentence the mention of video_qa calls
if args.no_video_qa:
    INSTRUCTIONS = re.sub(
        r'Use the video_qa tool to ask a vision model a question about an interval of the footage: [^\n]*? as tick seconds of the record\. ',
        '',
        INSTRUCTIONS,
    )
    INSTRUCTIONS = INSTRUCTIONS.replace(', shared by your own turns and your video_qa calls', '')
if args.skill is not None:
    for skill in args.skill:
        INSTRUCTIONS += '\n' + Path(skill).read_text()
# a sentence after the first sentence of the skill's Budget paragraph; only the --skill text without video_qa has that sentence
if args.merge_commands:
    if 'Budget: a turn costs one call.' not in INSTRUCTIONS:
        raise ValueError('--merge-commands: the instructions have no "Budget: a turn costs one call."')
    INSTRUCTIONS = INSTRUCTIONS.replace(
        'Budget: a turn costs one call.', 'Budget: a turn costs one call. ' + PROMPT['skill_merge_commands'].strip()
    )
# --force-decision: the agent is told it must use the decision model; submit_answer refuses until it has
if args.force_decision:
    INSTRUCTIONS += '\n' + PROMPT['force_decision_instruction']
# --comment-code: the agent is told to comment its exec and exec_qa code chunk by chunk and line by line
if args.comment_code:
    INSTRUCTIONS += '\n' + PROMPT['comment_code']
# --open-answer: the options are still given, but the answer is a short free-text answer, as in the MM-Lifelong skill text
if args.open_answer:
    if 'Give the final answer by calling submit_answer with the letter of the option.' not in INSTRUCTIONS:
        raise ValueError(
            '--open-answer: the instructions have no "Give the final answer by calling submit_answer with the letter of the option."'
        )
    INSTRUCTIONS = INSTRUCTIONS.replace(
        'Give the final answer by calling submit_answer with the letter of the option.',
        'Give the final answer by calling submit_answer with a short, direct answer to the question.',
    )
    # RackLife with --open-answer gives no options, so the skill text does not mention them
    if DATASET == 'racklife':
        if 'Each question comes with four options. ' not in INSTRUCTIONS:
            raise ValueError('--open-answer: the racklife instructions have no "Each question comes with four options. "')
        INSTRUCTIONS = INSTRUCTIONS.replace('Each question comes with four options. ', '')


def instructions(context, agent):
    """The instructions of the question of this run: for the choice datasets with the asking wearer's name and the number of options (the word four for EgoLifeQA, the number for EgoMemReason)."""
    if DATASET == 'video_index':
        return INSTRUCTIONS.replace('{duration_s}', str(context.context['duration_s']))
    if not CHOICE:
        return INSTRUCTIONS
    return INSTRUCTIONS.replace('{name}', context.context['name']).replace(
        '{option_count}', 'four' if DATASET == 'egolifeqa' else str(len(context.context['letters']))
    )


class ClipAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    status: Literal['answered', 'unknown']
    answer: str
    evidence_frame_indices: list[int]


class ContextLimit(Exception):
    pass


class NoSubmission(Exception):
    """The run ended with a model reply that called no tool, so no answer was submitted."""


class JudgeFailed(Exception):
    """The judge gave no score in 3 attempts."""


class ProviderError(Exception):
    """A helper request returned a provider error (status 200 without choices, or finish_reason error)."""


class HelperTimeout(Exception):
    """The helper's single bounded request timed out."""


def write_helper_event(state, record):
    with state['helper_events_path'].open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def helper_response_metadata(response):
    usage = response.usage.model_dump(mode='json') if response.usage is not None else None
    details = (usage or {}).get('completion_tokens_details') or {}
    choices = response.choices or []
    return {
        'response_observed': True,
        'response_id': getattr(response, 'id', None),
        'usage': usage,
        'reasoning_tokens': details.get('reasoning_tokens'),
        'reasoning_chars': sum(
            max(len(getattr(choice.message, field, None) or '') for field in ('reasoning_content', 'reasoning'))
            for choice in choices
        ),
        'finish_reason': choices[0].finish_reason if choices else None,
    }


async def call_helper(state, tool, request, method, **kwargs):
    started = time.monotonic()
    record = {
        'record_type': 'helper_request',
        'event': 'start',
        'attempt_id': state['helper_attempt_id'],
        'request_id': uuid.uuid4().hex,
        'tool': tool,
        'model': args.qa_model,
        'effective_enable_thinking': QA_SAMPLING['extra_body']['chat_template_kwargs']['enable_thinking'],
        'sampling': copy.deepcopy(QA_SAMPLING),
        'timeout_s': HELPER_TIMEOUT_SECONDS,
        'time': time.time(),
        'request': request,
        'response_observed': False,
        'usage': None,
        'reasoning_tokens': None,
        'reasoning_chars': None,
    }
    state['tools'].append(record)
    write_helper_event(state, record)
    try:
        async with asyncio.timeout(HELPER_TIMEOUT_SECONDS):
            response = await method(
                model=args.qa_model,
                **QA_SAMPLING,
                extra_headers={'X-Request-Id': record['request_id']},
                **kwargs,
            )
        record.update(helper_response_metadata(response))
        if not response.choices or response.choices[0].finish_reason == 'error':
            raise ProviderError(response.model_dump_json())
        record['event'] = 'finish'
        return response
    except LengthFinishReasonError as error:
        record.update(helper_response_metadata(error.completion))
        record.update(event='error', error_type='length')
        raise
    except (APITimeoutError, TimeoutError) as error:
        record.update(event='error', error_type='helper_timeout', error=str(error))
        raise HelperTimeout(f'{tool} exceeded its {HELPER_TIMEOUT_SECONDS} second request deadline.') from error
    except asyncio.CancelledError:
        record.update(event='cancelled', error_type='CancelledError')
        raise
    except BaseException as error:
        record.update(event='error', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record.update(time=time.time(), elapsed_s=time.monotonic() - started)
        write_helper_event(state, record)


class BudgetExhausted(Exception):
    """Raised before a model call once CALL_LIMIT calls are used; carries the conversation so far."""

    def __init__(self, items):
        super().__init__()
        self.items = items


async def execute_code(state, language, code, timeout_seconds):
    if DATASET == 'video_index':
        from local_tools import run_tool

        return await run_tool(ROOT, state['index'], state['work'], args.skill_dir, language, code, timeout_seconds)
    process = await asyncio.create_subprocess_exec(
        'docker',
        'exec',
        '-i',
        state['container'],
        'timeout',
        '-s',
        'KILL',
        str(timeout_seconds),
        *(['python', '-'] if language == 'python' else ['bash', '-s']),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    return (await process.communicate(code.encode()))[0], process.returncode


@function_tool(
    name_override='exec', description_override=EGOLIFE_TOOL_PROMPT['exec_timeout_cap'], failure_error_function=None
)
async def exec_code(
    ctx: RunContextWrapper[dict], language: Literal['bash', 'python'], code: str, timeout_seconds: int
) -> str:
    # the time limit of vsi
    if not 1 <= timeout_seconds <= 600:
        return 'The code was not run: timeout_seconds must be 1 to 600.'
    # code without the docstring or # comment still runs, with a reminder before the output
    reminder = ''
    if language == 'python':
        try:
            tree = ast.parse(code)
        except SyntaxError as error:
            return ''.join(traceback.format_exception_only(error))
        if ast.get_docstring(tree) is None:
            reminder = PROMPT['missing_docstring_ran']
    else:
        # the first line after an optional #! line must be a # comment
        lines = [line.strip() for line in code.splitlines() if line.strip() and not line.startswith('#!')]
        if not lines or not lines[0].startswith('#'):
            reminder = EGOLIFE_PROMPT['missing_comment_ran']
    # killed after timeout_seconds as vsi, which marks it in the output
    raw_output, returncode = await execute_code(ctx.context, language, code, timeout_seconds)
    # bytes that are not UTF-8 (for example a multibyte character cut by head -c) become U+FFFD, as in vsi
    output = raw_output.decode(errors='replace')
    if returncode == 137:
        output += f'\n[killed after {timeout_seconds} seconds]'
    ctx.context['tools'].append(
        {
            'tool': 'exec',
            'language': language,
            'code': code,
            'timeout_seconds': timeout_seconds,
            'output': output,
            'reminder': bool(reminder),
        }
    )
    # an output over OUTPUT_LIMIT is not returned, and the agent turn that asked for it is not counted toward the budget (one turn back per turn)
    if len(output) > OUTPUT_LIMIT:
        if not ctx.context['turn_refunded']:
            ctx.context['calls'] -= 1
            ctx.context['turn_refunded'] = True
        return PROMPT['output_over_limit'].format(n=len(output), limit=OUTPUT_LIMIT)
    return reminder + output


@function_tool(
    name_override='exec_qa', description_override=EGOLIFE_TOOL_PROMPT['exec_qa_timeout'], failure_error_function=None
)
async def exec_qa(
    ctx: RunContextWrapper[dict], language: Literal['bash', 'python'], code: str, timeout_seconds: int, question: str
) -> str:
    # the time limit of vsi
    if not 1 <= timeout_seconds <= 600:
        return 'The code was not run: timeout_seconds must be 1 to 600.'
    # code without the docstring or # comment still runs, with a reminder before the answer
    reminder = ''
    if language == 'python':
        try:
            tree = ast.parse(code)
        except SyntaxError as error:
            return ''.join(traceback.format_exception_only(error))
        if ast.get_docstring(tree) is None:
            reminder = PROMPT['missing_docstring_ran']
    else:
        # the first line after an optional #! line must be a # comment
        lines = [line.strip() for line in code.splitlines() if line.strip() and not line.startswith('#!')]
        if not lines or not lines[0].startswith('#'):
            reminder = EGOLIFE_PROMPT['missing_comment_ran']
    # killed after timeout_seconds as vsi, which marks it in the output
    raw_output, returncode = await execute_code(ctx.context, language, code, timeout_seconds)
    # bytes that are not UTF-8 (for example a multibyte character cut by head -c) become U+FFFD, as in vsi
    output = raw_output.decode(errors='replace')
    if returncode == 137:
        output += f'\n[killed after {timeout_seconds} seconds]'
    started = time.time()
    try:
        response = await call_helper(
            ctx.context,
            'exec_qa',
            {'language': language, 'code': code, 'timeout_seconds': timeout_seconds, 'output': output, 'question': question},
            qa_client.chat.completions.create,
            messages=[
                {
                    'role': 'user',
                    'content': EGOLIFE_PROMPT['exec_qa'].format(
                        language=language, code=code, output=output, question=question
                    ),
                }
            ],
        )
    except HelperTimeout as error:
        return json.dumps({'error': 'helper_timeout', 'message': str(error)})
    except BadRequestError as error:
        if 'maximum context length' not in str(error):
            raise
        ctx.context['tools'].append(
            {
                'tool': 'exec_qa',
                'language': language,
                'code': code,
                'timeout_seconds': timeout_seconds,
                'output': output,
                'question': question,
                'error': str(error),
            }
        )
        return PROMPT['exec_qa_too_long'].format(n=len(output))
    ctx.context['qa_latencies'].append(time.time() - started)
    ctx.context['qa_output_tokens'].append(response.usage.completion_tokens)
    ctx.context['qa_input_tokens'].append(response.usage.prompt_tokens)
    # an answer that ran to the end of the model's context is no answer: the agent is told as for an output that is too long
    if response.choices[0].finish_reason == 'length':
        ctx.context['tools'].append(
            {
                'tool': 'exec_qa',
                'language': language,
                'code': code,
                'timeout_seconds': timeout_seconds,
                'output': output,
                'question': question,
                'error': 'length',
            }
        )
        return PROMPT['exec_qa_too_long'].format(n=len(output))
    if response.choices[0].finish_reason != 'stop':
        raise RuntimeError(f'exec_qa did not finish: {response.choices[0].finish_reason}')
    answer = response.choices[0].message.content
    # a reply that stops without content is no answer: the agent is told to ask again
    if answer is None:
        print(f'exec_qa returned no answer for an output of {len(output)} characters', flush=True)
        ctx.context['tools'].append(
            {
                'tool': 'exec_qa',
                'language': language,
                'code': code,
                'timeout_seconds': timeout_seconds,
                'output': output,
                'question': question,
                'error': 'no answer',
            }
        )
        return PROMPT['exec_qa_no_answer']
    ctx.context['tools'].append(
        {
            'tool': 'exec_qa',
            'language': language,
            'code': code,
            'timeout_seconds': timeout_seconds,
            'output': output,
            'question': question,
            'answer': answer,
            'reminder': bool(reminder),
        }
    )
    return reminder + answer


# the argument of video_qa that names the footage: a day of a wearer's footage (the choice datasets and the week), or the video of the day set or of the month
KEY = 'day' if WEARER or DATASET == 'mm_week' else 'video'


async def settle_frame_tasks(tasks):
    settled = asyncio.gather(*tasks, return_exceptions=True)
    interrupted = False
    while not settled.done():
        try:
            await asyncio.shield(settled)
        except asyncio.CancelledError:
            interrupted = True
    settled.result()
    return interrupted


async def read_video_frame(path, seek, question_limit):
    async with question_limit, FRAME_DECODE_SEMAPHORE:
        creation = asyncio.create_task(asyncio.create_subprocess_exec(
            'ffmpeg', '-v', 'error', '-threads', '1', '-filter_threads', '1',
            '-ss', f'{seek:.4f}', '-i', path, '-frames:v', '1',
            '-vf', 'scale=w=1080:h=1080:force_original_aspect_ratio=decrease',
            '-f', 'image2pipe', '-c:v', 'mjpeg', '-q:v', '2', '-threads', '1', '-',
            stdout=asyncio.subprocess.PIPE,
        ))
        communication = None
        try:
            process = await asyncio.shield(creation)
            communication = asyncio.create_task(process.communicate())
            image = (await asyncio.shield(communication))[0]
        except BaseException:
            await settle_frame_tasks([creation])
            if not creation.cancelled() and creation.exception() is None:
                process = creation.result()
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                await settle_frame_tasks([communication if communication is not None else process.communicate(), process.wait()])
            raise
        if not image:
            raise RuntimeError(f'ffmpeg returned no frame from {path} at {seek:.4f} s')
        return image


async def ask_frames(state, source, start_s, end_s, fps, question):
    """The video_qa of vsi over the footage of one day (source is the day; for the choice datasets only up to the question time) or of one video (source is "day" for the day set, the video number for the month): frames at fps from start_s, long side 1080, answered with a status, an answer and evidence frames. As in vsi, every video_qa call counts toward the CALL_LIMIT budget, and once it is used up video_qa is unavailable."""
    if state['calls'] >= CALL_LIMIT:
        return json.dumps({'error': 'The model-call budget is used up; video_qa is unavailable.'})
    state['calls'] += 1
    state['video_qa_calls'] += 1
    timeline = state['timeline'] if DATASET == 'video_index' else (TIMELINES[state['person']] if WEARER else TIMELINE)
    if source not in timeline:
        if KEY == 'day':
            return json.dumps({'error': VIDEO_QA_ERRORS['unknown_day'].format(day=source, days=sorted(timeline))})
        return json.dumps(
            {
                'error': VIDEO_QA_ERRORS['unknown_video'].format(
                    video=source,
                    videos=', '.join(timeline)
                    if DATASET == 'video_index'
                    else (', '.join(timeline) if args.external_month_dir else ('1 to 23' if DATASET == 'mm_month' else 'day')),
                )
            }
        )
    if WEARER and source not in state['limits']:
        return json.dumps(
            {'error': VIDEO_QA_ERRORS['after_question_day'].format(day=source, question_time=state['question_time'])}
        )
    rows = timeline[source]
    if not math.isfinite(start_s) or not math.isfinite(end_s) or start_s < 0 or end_s <= start_s:
        return json.dumps({'error': VIDEO_QA_ERRORS['bad_interval']})
    if WEARER and end_s > state['limits'][source]:
        return json.dumps(
            {
                'error': VIDEO_QA_ERRORS['after_question_time'].format(
                    question_time=state['question_time'], day=source, limit=state['limits'][source]
                )
            }
        )
    if not WEARER and end_s > rows[-1]['end']:
        return json.dumps(
            {
                'error': VIDEO_QA_ERRORS['past_end'].format(
                    source={
                        'mm_week': source,
                        'mm_day': 'The video',
                        'mm_month': f'Video {source}',
                        'video_index': 'The video',
                    }[DATASET],
                    limit=rows[-1]['end'],
                )
            }
        )
    if end_s - start_s > INTERVAL_LIMIT_S:
        return json.dumps({'error': VIDEO_QA_ERRORS['too_long']})
    if not math.isfinite(fps) or fps <= 0:
        return json.dumps({'error': VIDEO_QA_ERRORS['bad_fps']})
    if fps > FPS_LIMIT:
        return json.dumps({'error': VIDEO_QA_ERRORS['fps_too_high']})
    count = math.ceil((end_s - start_s) * fps)
    if count > IMAGE_LIMIT:
        return json.dumps({'error': VIDEO_QA_ERRORS['too_many_frames'].format(count=count)})
    times = [start_s + index / fps for index in range(count)]
    starts = [row['start'] for row in rows]
    jobs = []
    question_limit = state.setdefault('frame_decode_limit', asyncio.Semaphore(2))
    for t in times:
        row = rows[bisect.bisect_right(starts, t) - 1]
        frame = min(round((t - row['start'] + row.get('file_offset_s', 0)) * row['fps']), row['frames'] - 1)
        seek = max(frame - 0.5, 0) / row['fps']
        jobs.append(asyncio.create_task(read_video_frame(row['path'], seek, question_limit)))
    try:
        images = await asyncio.gather(*jobs)
    finally:
        for job in jobs:
            if not job.done() and not job.cancelling():
                job.cancel()
        if await settle_frame_tasks(jobs):
            raise asyncio.CancelledError
    content = [
        {
            'type': 'text',
            'text': json.dumps(
                {
                    'question': question,
                    KEY: source,
                    'start_s': start_s,
                    'end_s': end_s,
                    'fps': fps,
                    'frame_times_s': times,
                },
                ensure_ascii=False,
            ),
        }
    ]
    for image in images:
        content.append(
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(image).decode()}}
        )
    started = time.time()
    # the qa model gives the schema-constrained answer
    try:
        response = await call_helper(
            state,
            'video_qa',
            {
                KEY: source, 'start_s': start_s, 'end_s': end_s, 'fps': fps, 'times': times, 'question': question,
                'frame_sha256': [hashlib.sha256(image).hexdigest() for image in images],
            },
            qa_client.chat.completions.parse,
            messages=[
                {'role': 'system', 'content': VIDEO_QA_SYSTEM},
                {'role': 'user', 'content': content},
            ],
            response_format=ClipAnswer,
        )
    except HelperTimeout as error:
        return json.dumps({'error': 'helper_timeout', 'message': str(error)})
    except LengthFinishReasonError as error:
        state['video_qa_input_tokens'].append(error.completion.usage.prompt_tokens)
        state['video_qa_output_tokens'].append(error.completion.usage.completion_tokens)
        state['tools'].append(
            {
                'tool': 'video_qa',
                KEY: source,
                'start_s': start_s,
                'end_s': end_s,
                'fps': fps,
                'times': times,
                'question': question,
                'error': 'length',
                'raw_completion': error.completion.model_dump(mode='json'),
                'elapsed_s': time.time() - started,
            }
        )
        return json.dumps(
            {
                'error': VIDEO_QA_ERRORS['out_of_context'].format(
                    count=count, prompt_tokens=error.completion.usage.prompt_tokens
                )
            }
        )
    except BadRequestError as error:
        # the server reports frames over its context length with this message
        if 'maximum context length' not in str(error):
            raise
        state['tools'].append(
            {
                'tool': 'video_qa',
                KEY: source,
                'start_s': start_s,
                'end_s': end_s,
                'fps': fps,
                'times': times,
                'question': question,
                'error': str(error),
            }
        )
        return json.dumps({'error': VIDEO_QA_ERRORS['too_many_images'].format(count=count)})
    state['qa_latencies'].append(time.time() - started)
    state['video_qa_input_tokens'].append(response.usage.prompt_tokens)
    state['video_qa_output_tokens'].append(response.usage.completion_tokens)
    if response.choices[0].finish_reason != 'stop':
        raise RuntimeError(f'video_qa did not finish: {response.choices[0].finish_reason}')
    submission = response.choices[0].message.parsed
    # a reply that stops without the structured answer is no answer: the agent is told to ask again
    if submission is None:
        print(
            f'video_qa returned no structured answer for {KEY} {source} {start_s} to {end_s}: content {response.choices[0].message.content!r}',
            flush=True,
        )
        state['tools'].append(
            {
                'tool': 'video_qa',
                KEY: source,
                'start_s': start_s,
                'end_s': end_s,
                'fps': fps,
                'times': times,
                'question': question,
                'error': 'no structured answer',
            }
        )
        return json.dumps({'error': VIDEO_QA_ERRORS['no_answer']})
    state['tools'].append(
        {
            'tool': 'video_qa',
            KEY: source,
            'start_s': start_s,
            'end_s': end_s,
            'fps': fps,
            'times': times,
            'question': question,
            'answer': submission.model_dump(),
        }
    )
    if submission.status == 'answered' and not submission.evidence_frame_indices:
        return json.dumps({'error': VIDEO_QA_ERRORS['no_evidence']})
    if any(index < 0 or index >= count for index in submission.evidence_frame_indices):
        return json.dumps({'error': VIDEO_QA_ERRORS['bad_evidence']})
    return json.dumps(
        {
            'status': submission.status,
            'answer': submission.answer,
            KEY: source,
            'evidence_times_s': [times[index] for index in submission.evidence_frame_indices],
        },
        ensure_ascii=False,
    )


# the same tool under the argument name of the dataset's index records
@function_tool(
    name_override='video_qa', description_override=EGOLIFE_TOOL_PROMPT['video_qa'], failure_error_function=None
)
async def video_qa_by_day(
    ctx: RunContextWrapper[dict], day: str, start_s: float, end_s: float, fps: float, question: str
) -> str:
    return await ask_frames(ctx.context, day, start_s, end_s, fps, question)


VIDEO_TOOL_DESCRIPTION = MM_TOOL_PROMPT['video_qa_month' if DATASET == 'mm_month' else 'video_qa_day']
if DATASET == 'video_index':
    VIDEO_TOOL_DESCRIPTION = (ROOT / 'prompts/video_index/video_qa.txt').read_text()


@function_tool(name_override='video_qa', description_override=VIDEO_TOOL_DESCRIPTION, failure_error_function=None)
async def video_qa_by_video(
    ctx: RunContextWrapper[dict], video: str, start_s: float, end_s: float, fps: float, question: str
) -> str:
    return await ask_frames(ctx.context, video, start_s, end_s, fps, question)


# the choice datasets take an option letter, the open ones any answer
@function_tool(
    name_override='submit_answer',
    description_override=EGOLIFE_TOOL_PROMPT['submit_answer'],
    failure_error_function=None,
)
async def submit_letter(ctx: RunContextWrapper[dict], answer: str) -> str:
    # --force-decision: no answer before an exec has called the decision model, except in the forced last turn
    if (
        args.force_decision
        and not ctx.context['final_turn']
        and not any('alpha/decisions' in record['code'] for record in ctx.context['tools'] if 'code' in record)
    ):
        return PROMPT['force_decision_refused']
    if answer not in ctx.context['letters']:
        return EGOLIFE_PROMPT['invalid_option'].format(answer=answer, letters=', '.join(ctx.context['letters']))
    ctx.context['answer'] = answer
    return answer


@function_tool(
    name_override='submit_answer', description_override=TOOL_PROMPT['submit_answer'], failure_error_function=None
)
async def submit_text(ctx: RunContextWrapper[dict], answer: str) -> str:
    # --force-decision: no answer before an exec has called the decision model, except in the forced last turn
    if (
        args.force_decision
        and not ctx.context['final_turn']
        and not any('alpha/decisions' in record['code'] for record in ctx.context['tools'] if 'code' in record)
    ):
        return PROMPT['force_decision_refused']
    ctx.context['answer'] = answer
    return answer


submit_answer = submit_letter if CHOICE and not args.open_answer else submit_text


def stop_on_answer(context, tool_results):
    """End the run once submit_answer has accepted an answer."""
    if context.context['answer'] is None:
        return ToolsToFinalOutputResult(is_final_output=False)
    return ToolsToFinalOutputResult(is_final_output=True, final_output=context.context['answer'])


class ContextHooks(RunHooks):
    async def on_llm_start(self, context, agent, system_prompt, input_items):
        context.context['llm_started'] = time.time()
        # every agent call counts toward the vsi budget
        context.context['calls'] += 1
        context.context['turn_refunded'] = False

    async def on_llm_end(self, context, agent, response):
        state = context.context
        state['agent_latencies'].append(time.time() - state['llm_started'])
        # reasoning included
        state['agent_output_tokens'].append(response.usage.output_tokens)
        state['agent_input_tokens'].append(response.usage.input_tokens)
        state['context'] = response.usage.input_tokens + response.usage.output_tokens
        state['max_context'] = max(state['max_context'], state['context'])


def limit_context(data):
    """Before each model call: once CALL_LIMIT calls are used, stop for the forced submit_answer turn of vsi; once the context reaches CONTEXT_LIMIT, remind the model to submit; after two ignored reminders, stop."""
    state = data.context
    if state['calls'] >= CALL_LIMIT and not state['final_turn']:
        raise BudgetExhausted(data.model_data.input)
    if state['context'] < CONTEXT_LIMIT:
        return data.model_data
    if state['reminders'] == 2:
        raise ContextLimit()
    state['reminders'] += 1
    return ModelInputData(
        input=data.model_data.input + [{'role': 'user', 'content': PROMPT['context_limit']}],
        instructions=data.model_data.instructions,
    )


# one agent per agent port
agents = {
    port: Agent(
        name=f'{DATASET}_qa',
        instructions=instructions,
        # send every earlier turn's reasoning back to the model
        model=OpenAIChatCompletionsModel(
            model=args.agent_model,
            openai_client=agent_clients[port],
            should_replay_reasoning_content=lambda context: True,
        ),
        model_settings=ModelSettings(tool_choice='required', **SAMPLING),
        tools=[exec_code, exec_qa, submit_answer]
        if args.no_video_qa
        else [exec_code, exec_qa, video_qa_by_day if KEY == 'day' else video_qa_by_video, submit_answer],
        tool_use_behavior=stop_on_answer,
        reset_tool_choice=False,
    )
    for port in args.agent_ports
}
# the forced last turn of vsi once the budget is used up: only submit_answer, which must be called
final_agents = {
    port: agents[port].clone(
        tools=[submit_answer],
        model_settings=ModelSettings(tool_choice='submit_answer', **SAMPLING),
    )
    for port in args.agent_ports
}


async def rollout(question, out):
    out.mkdir(parents=True, exist_ok=True)
    state = {
        'helper_events_path': out / 'helper-events.jsonl',
        'helper_attempt_id': uuid.uuid4().hex,
        'answer': None,
        'context': 0,
        'max_context': 0,
        'reminders': 0,
        'tools': [],
        'agent_latencies': [],
        'qa_latencies': [],
        'agent_output_tokens': [],
        'qa_output_tokens': [],
        'agent_input_tokens': [],
        'qa_input_tokens': [],
        'video_qa_input_tokens': [],
        'video_qa_output_tokens': [],
        'calls': 0,
        'video_qa_calls': 0,
        'turn_refunded': False,
        'final_turn': False,
    }
    if DATASET == 'video_index':
        mounts = []
        state['duration_s'] = question['duration_s']
        state['letters'] = list(question['options'])
        # Bind anonymous copies so /proc mount metadata cannot reveal the source item ID.
        capsule = RUN / '.sandbox' / uuid.uuid4().hex
        state['index'], state['work'] = capsule / 'index', capsule / 'work'
        state['index'].mkdir(parents=True)
        (state['work'] / 'tmp').mkdir(parents=True)
        source_index = ROOT / 'data/video_index/indices' / question['item_id']
        for filename in ('minutes.json', 'hours.json'):
            shutil.copy2(source_index / filename, state['index'] / filename)
        with (out / 'sandbox-attempts.jsonl').open('a') as ledger:
            ledger.write(json.dumps({'capsule': str(capsule), 'created_at_epoch': time.time()}) + '\n')
        probe = json.loads(
            subprocess.run(
                [
                    'ffprobe',
                    '-v',
                    'error',
                    '-select_streams',
                    'v:0',
                    '-show_entries',
                    'stream=nb_frames,r_frame_rate:format=duration',
                    '-of',
                    'json',
                    question['video_path'],
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        state['timeline'] = {
            'video': [
                {
                    'path': question['video_path'],
                    'start': 0.0,
                    'end': question['duration_s'],
                    'frames': int(probe['streams'][0]['nb_frames']),
                    'fps': float(Fraction(probe['streams'][0]['r_frame_rate'])),
                }
            ]
        }
        prompt = (
            (ROOT / 'prompts/video_index/question.txt')
            .read_text()
            .format(
                duration_s=question['duration_s'],
                question=question['question'],
                options='\n'.join(f'{letter}. {option}' for letter, option in question['options'].items()),
            )
        )
    elif WEARER:
        person = question['person']
        day = question['day']
        query_day = int(day.removeprefix('DAY'))
        # the stamp is HHMMSS followed by two digits within the second
        stamp = question['stamp']
        clock = f'{stamp[0:2]}:{stamp[2:4]}:{stamp[4:6]}'
        query_wall = int(stamp[0:2]) * 3600 + int(stamp[2:4]) * 60 + int(stamp[4:6])
        if args.magic_index is None:
            # index entries that end by the question time
            visible_minutes = [
                i
                for i, clip in enumerate(MINUTES[person])
                if (int(clip['day'].removeprefix('DAY')), clip['wall_end_s']) <= (query_day, query_wall)
            ]
            visible_hours = [
                i
                for i, hour in enumerate(HOURS[person])
                if (int(hour['end_day'].removeprefix('DAY')), hour['wall_end_s']) <= (query_day, query_wall)
            ]
            index = out / 'index'
            if index.exists():
                shutil.rmtree(index)
            index.mkdir()
            (index / 'minutes.json').write_text(
                json.dumps([MINUTES[person][i] for i in visible_minutes], ensure_ascii=False)
            )
            (index / 'hours.json').write_text(json.dumps([HOURS[person][i] for i in visible_hours], ensure_ascii=False))
        else:
            # MAGIC's memory that the question sees at its question time, filtered by baselines/magic/export_index.py as MAGIC filters it
            index = Path(args.magic_index).resolve() / question['key']
        # footage seconds of each day that are recorded by the question time
        limits = {}
        for footage_day, rows in TIMELINES[person].items():
            if int(footage_day.removeprefix('DAY')) < query_day:
                limits[footage_day] = rows[-1]['end']
            elif int(footage_day.removeprefix('DAY')) == query_day and rows[0]['wall_start'] < query_wall:
                limits[footage_day] = max(
                    min(row['end'], row['start'] + query_wall - row['wall_start'])
                    for row in rows
                    if row['wall_start'] < query_wall
                )
        mounts = ['-v', f'{index}:/data:ro']
        state['person'] = person
        state['name'] = 'Alex' if DATASET == 'racklife' else person.split('_')[1].title()
        state['limits'] = limits
        state['question_time'] = f'{day} {clock}'
        state['letters'] = list(question['options'])
        prompt = EGOLIFE_PROMPT['question'].format(
            name=state['name'],
            day=day,
            time=clock,
            question=question['question'],
            options='\n'.join(f'{letter}. {option}' for letter, option in question['options'].items()),
        )
        # RackLife with --open-answer: the question without its options
        if DATASET == 'racklife' and args.open_answer:
            prompt = f'Wearer: {state["name"]}\nQuestion time: {day} {clock}\nQuestion: {question["question"]}'
    else:
        mounts = [argument for source, target in MOUNTS.items() for argument in ('-v', f'{ROOT / source}:{target}:ro')]
        if args.magic_index is not None:
            mounts = ['-v', f'{Path(args.magic_index).resolve()}:/data:ro']
        # the user message of vsi for a question without a question time
        prompt = f'Question: {question["question"]}'
    # the script library of the skill
    if args.skill_dir is not None:
        mounts += ['-v', f'{Path(args.skill_dir).resolve()}:/skill:ro']
    # the library of the decision model
    if args.decision_dir is not None:
        mounts += ['-v', f'{Path(args.decision_dir).resolve()}:/decision:ro']
    if DATASET != 'video_index':
        container = f'video-json-{DATASET}-{args.run_name}-{question["key"]}'
        # a container left by an interrupted launch has the same name
        await (
            await asyncio.create_subprocess_exec(
                'docker', 'rm', '-f', container, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
            )
        ).wait()
        process = await asyncio.create_subprocess_exec(
            # --container-network: a docker network instead of none (the internal network of the decision proxy)
            'docker',
            'run',
            '-d',
            '--name',
            container,
            '--network',
            args.container_network if args.container_network is not None else 'none',
            '-w',
            '/work',
            *mounts,
            '--entrypoint',
            'sleep',
            IMAGE,
            'infinity',
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        stderr = (await process.communicate())[1]
        if process.returncode:
            raise RuntimeError(f'docker run {container} failed: {stderr.decode()}')
        state['container'] = container
    started = time.time()
    try:
        result = await Runner.run(
            agents[question['port']],
            prompt,
            context=state,
            max_turns=10**9,
            hooks=ContextHooks(),
            run_config=RunConfig(call_model_input_filter=limit_context),
        )
        # a reply without a tool call ends the run without an answer
        if state['answer'] is None:
            (out / 'messages.json').write_text(json.dumps(result.to_input_list(), ensure_ascii=False, indent=1))
            raise NoSubmission(f'final output {result.final_output!r}')
        answer = result.final_output
        status = 'submitted'
        (out / 'messages.json').write_text(json.dumps(result.to_input_list(), ensure_ascii=False, indent=1))
    except BudgetExhausted as error:
        # the vsi message and a turn that can only call submit_answer
        state['final_turn'] = True
        result = await Runner.run(
            final_agents[question['port']],
            error.items
            + [
                {
                    'role': 'user',
                    'content': f'Your budget of {CALL_LIMIT} model calls is used up and the other tools are no longer available. Call submit_answer with {"the letter of the closest option" if CHOICE and not args.open_answer else "your final answer"} now.',
                }
            ],
            context=state,
            # one turn: an invalid letter here fails the question (MaxTurnsExceeded), as vsi's InvalidTrace
            max_turns=1,
            hooks=ContextHooks(),
            run_config=RunConfig(call_model_input_filter=limit_context),
        )
        if state['answer'] is None:
            (out / 'messages.json').write_text(json.dumps(result.to_input_list(), ensure_ascii=False, indent=1))
            raise NoSubmission(f'final output {result.final_output!r}')
        answer = result.final_output
        status = 'budget_exhausted'
        (out / 'messages.json').write_text(json.dumps(result.to_input_list(), ensure_ascii=False, indent=1))
    except ContextLimit:
        answer = 'A' if CHOICE and not args.open_answer else 'null'
        status = 'context_limit'
    finally:
        (out / 'tools.json').write_text(json.dumps(state['tools'], ensure_ascii=False, indent=1))
        if DATASET != 'video_index':
            await (
                await asyncio.create_subprocess_exec('docker', 'rm', '-f', container, stdout=asyncio.subprocess.DEVNULL)
            ).wait()
        if WEARER and args.magic_index is None:
            shutil.rmtree(index)
    record = {
        **question,
        'answer': answer,
        'status': status,
        'reminders': state['reminders'],
        'max_context': state['max_context'],
        'calls': state['calls'],
        'video_qa_calls': state['video_qa_calls'],
        'seconds': time.time() - started,
        'answerer_source_sha256': sha256_file(__file__),
        'generation_max_tokens': 16384,
        'answerer_segment': 'helper-v2-20261102',
        'agent_latencies': state['agent_latencies'],
        'qa_latencies': state['qa_latencies'],
        'agent_output_tokens': state['agent_output_tokens'],
        'qa_output_tokens': state['qa_output_tokens'],
        'agent_input_tokens': state['agent_input_tokens'],
        'qa_input_tokens': state['qa_input_tokens'],
        'video_qa_input_tokens': state['video_qa_input_tokens'],
        'video_qa_output_tokens': state['video_qa_output_tokens'],
    }
    if WEARER:
        if args.magic_index is None:
            record['visible_minutes'] = len(visible_minutes)
            record['visible_hours'] = len(visible_hours)
        record['footage_limits'] = limits
    if DATASET in ('egolifeqa', 'video_index', 'racklife') and not args.open_answer:
        record['correct'] = answer == question['gold']
    (out / 'answer.json').write_text(json.dumps(record, ensure_ascii=False, indent=1))


async def judge(question, out):
    """Official MM-Lifelong eval_acc.py scoring without temperature: up to 3 attempts to get a 0-5 score, mapped to 1 (4-5), 0.5 (3) or 0."""
    answer = json.loads((out / 'answer.json').read_text())['answer']
    # --open-answer (egolifeqa): the judge gets the question with its options, and the gold letter with its option text
    if CHOICE:
        judged_question = (
            question['question']
            + '\n'
            + '\n'.join(f'{letter}. {option}' for letter, option in question['options'].items())
        )
        gold = f'{question["gold"]}. {question["options"][question["gold"]]}'
    else:
        judged_question = question['question']
        gold = question['gold']
    spent = {'cost': 0.0}
    usages = []
    COST.set(spent)
    try:
        for attempt in range(3):
            response = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {'role': 'system', 'content': PROMPT['judge_system']},
                    {
                        'role': 'user',
                        'content': PROMPT['judge_user'].format(question=judged_question, answer=gold, pred=answer),
                    },
                ],
                timeout=args.judge_timeout,
                **JUDGE_SAMPLING,
            )
            usages.append(response.usage.model_dump() if response.usage is not None else None)
            text = response.choices[0].message.content
            if response.choices[0].finish_reason != 'stop':
                print(f'{question["key"]} judge attempt {attempt + 1}: incomplete response, '
                      f'finish_reason={response.choices[0].finish_reason}', flush=True)
                continue
            # an empty judge message gives no score, as an output without a score does: it is one of the 3 attempts
            if text is None:
                print(
                    f'{question["key"]} judge attempt {attempt + 1}: empty message, finish_reason={response.choices[0].finish_reason}',
                    flush=True,
                )
                continue
            text = text.strip()
            # official parse_score: "Final Score: n", otherwise the first standalone 0-5
            match = re.search(r'Final Score:\s*([0-5])', text, re.IGNORECASE)
            if match is None:
                match = re.search(r'\b([0-5])\b', text)
            if match is not None:
                score = int(match.group(1))
                (out / 'judge.json.tmp').write_text(
                    json.dumps(
                        {
                            'gpt_score': score,
                            'acc': 1.0 if score >= 4 else 0.5 if score == 3 else 0.0,
                            'gpt_raw': text,
                            'attempts': attempt + 1,
                            'cost': spent['cost'],
                            'cost_reported': JUDGE_COST_REPORTED,
                            'cost_note': 'API-reported USD only; unreported local GPU cost is not estimated.',
                            'judge_model': judge_model,
                            'judge_config': JUDGE_CONFIG,
                            'judge_config_sha256': JUDGE_CONFIG_SHA256,
                            'answer_sha256': sha256_file(out / 'answer.json'),
                            'request_usage': usages,
                            'finish_reason': response.choices[0].finish_reason,
                        },
                        ensure_ascii=False,
                        indent=1,
                    )
                )
                (out / 'judge.json.tmp').replace(out / 'judge.json')
                return
        raise JudgeFailed(f'judge gave no score in 3 attempts, last output: {text}')
    finally:
        with open(RUN / 'spent.jsonl', 'a') as record:
            record.write(json.dumps({'key': question['key'], 'cost': spent['cost']}) + '\n')


async def run_once(question, out, semaphore):
    """One rollout of the question into out and, for an open question, its judge score; False when the question failed or paused in this launch."""
    async with semaphore:
        if (out / DONE_FILE).exists():
            return True
        # invalid tool-call JSON or a reply without a tool call reruns the whole question, up to 3 more times
        for attempt in range(4):
            if (out / 'answer.json').exists():
                break
            try:
                await rollout(question, out)
                break
            except (ModelBehaviorError, NoSubmission):
                print(f'RETRY {question["key"]} attempt {attempt + 1} of 4\n{traceback.format_exc()}', flush=True)
                if attempt == 3:
                    (out / 'error.txt').write_text(traceback.format_exc())
                    print(f'FAILED {question["key"]}', flush=True)
                    return False
            # failures of this question only: a prompt over the agent's context length, a forced last turn that does not end the run; any other error stops the run
            except (BadRequestError, MaxTurnsExceeded) as error:
                # the server reports a prompt over the context length with this message
                if not (isinstance(error, MaxTurnsExceeded) or 'maximum context length' in str(error)):
                    raise
                (out / 'error.txt').write_text(traceback.format_exc())
                print(f'FAILED {question["key"]}\n{traceback.format_exc()}', flush=True)
                return False
            # a request of the agent or inside a tool that still fails after the SDK's retries (connection, rate limit, 5xx) or exec_qa's provider-error retries stops this question for this launch; no answer.json or error.txt is written, so a relaunch runs it again
            except (APIConnectionError, RateLimitError, InternalServerError, UserError) as error:
                if isinstance(error, UserError) and not isinstance(
                    error.__cause__, (APIConnectionError, RateLimitError, InternalServerError, ProviderError)
                ):
                    raise
                print(f'PAUSED {question["key"]}\n{traceback.format_exc()}', flush=True)
                return False
        if JUDGED:
            try:
                await judge(question, out)
            # a judge without a score in 3 attempts, or whose request still times out, is rate limited or gets a 5xx after the SDK's retries, fails this question only; its answer.json stays, so a relaunch only judges it
            except (JudgeFailed, APIConnectionError, RateLimitError, InternalServerError):
                (out / 'error.txt').write_text(traceback.format_exc())
                print(f'FAILED {question["key"]}\n{traceback.format_exc()}', flush=True)
                return False
    answer = json.loads((out / 'answer.json').read_text())
    answers = [json.loads(path.read_text()) for path in RUN.glob(f'{TRIAL_GLOB}/answer.json')]
    done = len(list(RUN.glob(f'{TRIAL_GLOB}/{DONE_FILE}')))
    rate = (done - DONE_BEFORE) / ((time.time() - STARTED) / 60)
    stats = {
        'max_context': max(answer['max_context'] for answer in answers),
        'agent_latency_s': float(np.mean([latency for answer in answers for latency in answer['agent_latencies']])),
        'qa_latency_s': float(np.mean([latency for answer in answers for latency in answer['qa_latencies']])),
        'results_per_min': rate,
    }
    # with --trials the number of trials still to come is not known
    if args.trials is None:
        stats['eta_min'] = (len(QUESTIONS) - done) / rate
    if DATASET in ('egolifeqa', 'racklife') and not args.open_answer:
        print(
            f'{question["key"]} {answer["status"]} answer={answer["answer"]} gold={answer["gold"]} max_context={answer["max_context"]} seconds={answer["seconds"]:.0f} {answer["person"]} {answer["day"]} {answer["stamp"]} {answer["type"]}',
            flush=True,
        )
        stats['answered'] = len(answers)
        stats['accuracy'] = float(np.mean([answer['correct'] for answer in answers]))
    if DATASET == 'racklife' and args.open_answer:
        print(
            f'{question["key"]} {answer["status"]} answer={answer["answer"]!r} max_context={answer["max_context"]} seconds={answer["seconds"]:.0f} {answer["day"]} {answer["stamp"]} {answer["type"]}',
            flush=True,
        )
        stats['answered'] = len(answers)
    if DATASET == 'video_index':
        print(f'{question["key"]} {answer["status"]} answer={answer["answer"]} gold={answer["gold"]}', flush=True)
        stats['answered'] = len(answers)
        stats['accuracy'] = float(np.mean([answer['correct'] for answer in answers]))
    if DATASET == 'memreason':
        print(
            f'{question["key"]} {answer["status"]} answer={answer["answer"]} max_context={answer["max_context"]} seconds={answer["seconds"]:.0f} {answer["person"]} {answer["day"]} {answer["stamp"]} {answer["type"]}',
            flush=True,
        )
        stats['answered'] = len(answers)
        # the leaderboard format
        (RUN / 'submission.json').write_text(
            json.dumps(
                [
                    {'example_id': answer['example_id'], 'predicted_answer': answer['answer']}
                    for answer in sorted(answers, key=lambda answer: answer['example_id'])
                ],
                indent=1,
            )
        )
    if JUDGED:
        score = json.loads((out / 'judge.json').read_text())
        print(
            f'{question["key"]} {answer["status"]} score={score["gpt_score"]} max_context={answer["max_context"]} seconds={answer["seconds"]:.0f} answer={answer["answer"]!r} gold={answer["gold"]!r}',
            flush=True,
        )
        stats['judged'] = done
        stats['acc'] = float(
            np.mean([json.loads(path.read_text())['acc'] for path in RUN.glob(f'{TRIAL_GLOB}/judge.json')])
        )
        stats['cost_usd'] = sum(json.loads(line)['cost'] for line in (RUN / 'spent.jsonl').read_text().splitlines())
    wandb.log(stats)
    print(json.dumps(stats), flush=True)
    return True


async def solve(question, semaphore):
    """With --trials an open question is rolled out again, each trial in q<id>/trial<k>/, until a trial is judged 1 or it has --trials trials; a trial that fails or pauses ends the question for this launch."""
    if args.trials is None:
        await run_once(question, RUN / question['key'], semaphore)
        return
    for trial in range(1, args.trials + 1):
        out = RUN / question['key'] / f'trial{trial}'
        if not await run_once(question, out, semaphore):
            return
        if json.loads((out / 'judge.json').read_text())['acc'] == 1.0:
            return


async def main():
    await asyncio.gather(*[solve(question, semaphores[question['port']]) for question in QUESTIONS])


RUN.mkdir(parents=True, exist_ok=True)
if (
    DATASET == 'video_index'
    and (RUN / 'args.json').exists()
    and json.loads((RUN / 'args.json').read_text()) != vars(args)
):
    raise ValueError('Video-Index resume configuration changed; use a new run name')
(RUN / 'args.json').write_text(json.dumps(vars(args), indent=1))
# the instructions (choice datasets: with {name} and {option_count} unfilled) and the tools of the agent, as the model sees them, for turning the rollouts into a messages dataset
(RUN / 'instructions.txt').write_text(INSTRUCTIONS)
(RUN / 'agent_tools.json').write_text(
    json.dumps(
        [
            {
                'type': 'function',
                'function': {'name': tool.name, 'description': tool.description, 'parameters': tool.params_json_schema},
            }
            for tool in agents[args.agent_ports[0]].tools
        ],
        ensure_ascii=False,
        indent=1,
    )
)
wandb.init(
    project='video_json',
    name=f'{DATASET}-{args.run_name}',
    config={
        **vars(args),
        'judge_model': judge_model,
        'call_limit': CALL_LIMIT,
        'context_limit': CONTEXT_LIMIT,
        'output_limit': OUTPUT_LIMIT,
    },
)
# --concurrency is split evenly over the agent ports, each port with its own limit
if args.concurrency % len(args.agent_ports):
    raise ValueError('--concurrency must split evenly over the agent ports')
semaphores = {port: asyncio.Semaphore(args.concurrency // len(args.agent_ports)) for port in args.agent_ports}
STARTED = time.time()
# the file that marks a question as finished: the answer of a choice question, the judge score of an open one
DONE_FILE = 'judge.json' if JUDGED else 'answer.json'
# the directories of the rollouts: one per question, or one per trial with --trials
TRIAL_GLOB = '*' if args.trials is None else '*/trial*'
DONE_BEFORE = len(list(RUN.glob(f'{TRIAL_GLOB}/{DONE_FILE}')))
asyncio.run(main())
wandb.finish()

1 / 0
