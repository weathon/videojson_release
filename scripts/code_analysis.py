"""Code and trace analysis of a harness run: DeepSeek classifies every exec / exec_qa code into search, read_interval or other and lists the index keys it reads or searches; thread length, Bash vs Python and exec vs exec_qa come straight from the traces. Usage: code_analysis.py <concurrency> [code limit]"""

import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import wandb
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, LengthFinishReasonError, RateLimitError
from pydantic import BaseModel, ConfigDict, ValidationError

ROOT = Path(__file__).resolve().parents[1]
RUNS = [("egolifeqa", "v4_minimal"), ("mm_week", "v4_minimal"), ("mm_day", "v4_minimal"), ("mm_month", "v4_minimal"), ("egor1", "v4_minimal_bi")]
CONCURRENCY = int(sys.argv[1])
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
OUT = ROOT / "runs/code_analysis" / ("v4_minimal_wholejson" if LIMIT is None else "v4_minimal_wholejson_debug")
MODEL = "deepseek-v4.1-flash-1017"
ATTEMPTS = 5  # per code, with the exact error fed back (the user, 2026-11-05)
MAX_TOKENS = 16384  # output cap of every labeling call; generations that repeat keys until the 1M context is full are retried (the user, 2026-11-05)
RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError, LengthFinishReasonError, ValidationError, json.JSONDecodeError, RuntimeError)
RECORD_KEYS = ["clip_id", "day", "start_time", "end_time", "tick_start_s", "tick_end_s", "wall_start_s", "wall_end_s", "video"]
CAPTION_KEYS = ["dense_caption", "events", "unexpected", "entities", "places", "actions", "relations", "transcription", "explanation", "on_screen_text", "mood"]
Key = Literal[tuple(RECORD_KEYS + [f"caption.{k}" for k in CAPTION_KEYS] + ["whole_json"])]  # whole_json: the code stringifies a whole record or caption (the user, 2026-11-05)
SYSTEM = (ROOT / "prompts/code_analysis_system.txt").read_text()


class Label(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Literal["search", "read_interval", "other"]
    keys: list[Key]


codes = []
for dataset, run in RUNS:
    for path in sorted((ROOT / "runs" / dataset / run).glob("*/messages.json")):
        number = 0
        for item in json.load(open(path)):
            if item.get("type") == "function_call" and item["name"] in ("exec", "exec_qa"):
                arguments = json.loads(item["arguments"])
                codes.append({"dataset": dataset, "question": path.parent.name, "number": number, "tool": item["name"], "language": arguments["language"], "code": arguments["code"]})
                number += 1
todo = codes[:LIMIT] if LIMIT is not None else codes


async def main():
    run = wandb.init(project="videojson_gym", name="code-analysis-v4_minimal" + ("-debug" if LIMIT is not None else ""), config={"model": MODEL, "runs": RUNS, "codes": len(todo), "concurrency": CONCURRENCY})
    client = AsyncOpenAI(base_url="http://127.0.0.1:8400/v1", api_key="EMPTY", timeout=None, max_retries=5)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    done = {"n": sum((OUT / "labels" / c["dataset"] / c["question"] / f"{c['number']}.json").exists() for c in todo)}
    started = time.time()
    print(f"START codes={len(todo)} done={done['n']}", flush=True)

    async def one(c):
        path = OUT / "labels" / c["dataset"] / c["question"] / f"{c['number']}.json"
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if c["language"] not in ("python", "bash"):
            # the language field is not python or bash (mostly the agent put the code into it; also "python3" and "python\n"); main.py ran the code field with bash -s (the user, 2026-11-05: its own class, not classified)
            path.write_text(json.dumps({**c, "category": "malformed", "keys": [], "usage": None}, ensure_ascii=False, indent=1))
            done["n"] += 1
            return
        async with semaphore:
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Records: {'MM-Lifelong' if c['dataset'] in ('mm_day', 'mm_month') else 'EgoLife'}\nLanguage field: {c['language']}\nCode:\n{c['code']}"}]
            for attempt in range(1, ATTEMPTS + 1):
                try:
                    response = await client.chat.completions.parse(model=MODEL, messages=messages, response_format=Label, max_tokens=MAX_TOKENS)
                    parsed = response.choices[0].message.parsed
                    if parsed is None:
                        raise RuntimeError(f"no structured output: {response.choices[0]}")
                    break
                except RETRYABLE as error:
                    print(f"LABEL_ERROR {c['dataset']} {c['question']} {c['number']} attempt={attempt} error={error}", flush=True)
                    if LIMIT is not None:
                        raise
                    if attempt == ATTEMPTS:
                        path.with_name(f"{c['number']}.failure.json").write_text(json.dumps({**c, "error": str(error)}, ensure_ascii=False, indent=1))
                        print(f"SAMPLE_SKIPPED {c['dataset']} {c['question']} {c['number']}", flush=True)
                        return
                    messages = messages[:2] + [{"role": "user", "content": f"The previous attempt failed with this exact error. Correct it and return the requested classification:\n{error}"}]
                    await asyncio.sleep(2 ** attempt)
            path.write_text(json.dumps({**c, **parsed.model_dump(), "usage": response.usage.model_dump()}, ensure_ascii=False, indent=1))
            done["n"] += 1
            run.log({"labeled": done["n"], "remaining": len(todo) - done["n"]})
            if done["n"] % 200 == 0:
                print(f"LABELED {done['n']}/{len(todo)} elapsed_min={(time.time() - started) / 60:.1f}", flush=True)

    async with asyncio.TaskGroup() as group:
        for c in todo:
            group.create_task(one(c))
    print(f"LABELS_COMPLETE {done['n']}/{len(todo)}", flush=True)

    summary = {}
    for dataset, run_name in RUNS + [("all", None)]:
        paths = [OUT / "labels" / c["dataset"] / c["question"] / f"{c['number']}.json" for c in todo if dataset in ("all", c["dataset"])]
        labels = [json.load(open(p)) for p in paths if p.exists()]
        traced = [c for c in todo if dataset in ("all", c["dataset"])]
        questions = [(d, p) for d, r in RUNS if dataset in ("all", d) for p in sorted((ROOT / "runs" / d / r).glob("*/messages.json"))]
        calls = [sum(item.get("type") == "function_call" for item in json.load(open(p))) for _, p in questions]
        tokens = [json.load(open(p.parent / "answer.json"))["tokens"] for _, p in questions]
        inputs = [sum(t["input"] for t in ts) for ts in tokens]
        outputs = [sum(t["output"] for t in ts) for ts in tokens]
        n = len(labels)
        summary[dataset] = {
            "questions": len(questions),
            "codes": n,
            "codes_failed": len(paths) - n,
            "tool_calls_per_question": {"mean": statistics.mean(calls), "median": statistics.median(calls)},
            "input_tokens_per_question": {"mean": statistics.mean(inputs), "median": statistics.median(inputs)},
            "output_tokens_per_question": {"mean": statistics.mean(outputs), "median": statistics.median(outputs)},
            "category": {k: v / n for k, v in Counter(l["category"] for l in labels).most_common()},
            "keys": {k: v / sum(l["category"] != "malformed" for l in labels) for k, v in Counter(k for l in labels for k in set(l["keys"])).most_common()},  # share of the classified (not malformed) codes
            "language": {k: v / len(traced) for k, v in Counter(c["language"] if c["language"] in ("python", "bash") else "malformed" for c in traced).most_common()},
            "tool": {k: v / len(traced) for k, v in Counter(c["tool"] for c in traced).most_common()},
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    run.summary.update({"summary": summary})
    run.finish()
    print(json.dumps(summary["all"], indent=1), flush=True)


asyncio.run(main())
print("DONE", flush=True)
1 / 0
