"""Answer long-video questions from a caption index: an agent runs code in a per-question sandbox container (HARNESS_SPEC.md)."""

import argparse
import base64
import json
import math
import os
import re
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dotenv
import jieba
import openai
import wandb
from rank_bm25 import BM25Okapi
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, RunConfig, RunContextWrapper, RunHooks, Runner
from agents import ToolsToFinalOutputResult, function_tool, set_tracing_disabled
from agents.exceptions import AgentsException, ModelBehaviorError

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "prompts"
MML = Path("/raid/huggingface/hub/datasets--MM-Lifelong--MM-Lifelong/snapshots/248aa82039a574e63a2e524746a7cd8f32330443")
parser = argparse.ArgumentParser()
for flag in ["--dataset", "--run-name", "--keys", "--model", "--base-url", "--reasoning-effort", "--skill", "--skill-dir"]:
    parser.add_argument(flag)
for flag in ["--decision-dir", "--container-network", "--magic-index", "--temperature", "--top-p", "--top-k", "--presence-penalty"]:
    parser.add_argument(flag)
parser.add_argument("--data-dir", default=str(ROOT / "data"))  # the index set: data/ (the original captions) or the output of scripts/caption_old/build_data.py --out
parser.add_argument("--split", default="val")
parser.add_argument("--concurrency", type=int, required=True)
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--max-tokens", type=int)  # output cap of every model call (agent, exec_qa, video_qa); local models only, so a generation that loops ends (the user, 2026-11-02: 16384, as MAGIC)
for flag in ["--no-video-qa", "--no-exec-qa", "--no-hours", "--dense-caption-only", "--bilingual", "--context-budget", "--bm25", "--bm25-calls"]:
    parser.add_argument(flag, action="store_true")
args = parser.parse_args()
dotenv.load_dotenv(ROOT / ".env")
set_tracing_disabled(True)
choice = args.dataset in ("egolifeqa", "egor1", "memreason")
subset = args.dataset.removeprefix("mm_")
videos = subset in ("day", "month")
run_dir = ROOT / "runs" / args.dataset / args.run_name

# 1. load the dataset
if args.dataset in ("egolifeqa", "egor1"):
    both = lambda row, key, sep: row[key + "_chinese"] + (sep + row[key] if args.bilingual else "")
    if args.dataset == "egolifeqa":
        rows = [("", row) for row in json.load(open("/raid/wg25r/egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json"))]
    else:
        # Ego-R1 Bench A1_JAKE, both splits (the IDs of the two overlap, so the key carries the split)
        egor1 = Path("/raid/huggingface/hub/datasets--Ego-R1--Ego-R1-Bench/snapshots/c0e07328be08cc6014807fb1bebb511f6c5c6cdd")
        rows = [(f"{split}_", row) for split in ("manual", "gemini") for row in json.load(open(egor1 / f"{split}-benchmark/A1_JAKE.json"))]
    questions = [
        {
            "key": f"{prefix}q{row['ID']}",
            "question": both(row, "question", "\n"),
            "gold": row["answer"],
            "person": "A1_JAKE",
            "options": {x: both(row, f"choice_{x.lower()}", " / ") for x in "ABCD"},
            "day": row["query_time"]["date"],
            "time": ":".join(row["query_time"]["time"].zfill(8)[i : i + 2] for i in (0, 2, 4)),
        }
        for prefix, row in rows
    ]
elif args.dataset == "memreason":
    path = "/raid/huggingface/hub/datasets--Ted412--EgoMemReason/snapshots/7e581505b9dce0e85193a27ae689ff899d0bc507/annotations_public.jsonl"
    questions = [
        {
            "key": f"q{row['example_id']}",
            "question": row["question"],
            "gold": None,
            "person": row["identity"],
            "options": row["options"],
            "day": row["query_time"].split(", ")[0],
            "time": row["query_time"].split(", ")[1],
        }
        for row in map(json.loads, open(path))
    ]
else:
    source = MML / {"week": "week/test.json", "day": "day/test.json", "month": f"month/{args.split}.json"}[subset]
    questions = [{"key": f"q{r['index']}", "question": r["question"], "gold": r["answer"], "person": "A1_JAKE"} for r in json.load(open(source))]
if args.keys:
    keys = set(open(args.keys).read().split())
    questions = [q for q in questions if q["key"] in keys]


def cut(text, start, end, new=""):
    """Replace the span from start through end (None: the end of the line) by new."""
    begin = text.index(start)
    stop = text.index("\n", begin) if end is None else text.index(end, begin) + len(end)
    return text[:begin] + new + text[stop:]


schema = json.load(open(P / "egolife/vsi_caption_schema.json"))
description = (P / ("egolife/vsi_skill_template_30.txt" if choice else f"mm_lifelong/vsi_skill_{subset}_30.txt")).read_text()
if args.no_video_qa:
    description = cut(description, "Use the video_qa tool", "of the record. ")
    description = cut(description, ", shared by your own turns", "video_qa calls")
if args.no_hours:
    description = cut(description, " /data/hours.json is a JSON list", None)
if args.context_budget:  # no call budget; the answer is forced once the context is over 80% of the window (the user, 2026-11-13)
    description = cut(description, " You have 30 model calls", "submit_answer only.")
if args.bm25:  # baseline: no code, only BM25 search with a query and top_k, at most 5 searches (the user, 2026-11-15)
    description = cut(description, " at /data/minutes.json", "minutes.json")
    description = cut(description, "Use the exec tool", None, (P / "bm25/instructions.txt").read_text().rstrip("\n"))
if args.bm25_calls:  # --bm25 with the budget of the main experiment (30 model calls, 250,000-token context) instead of 5 searches (the user, 2026-11-15)
    description = cut(description, " You can call search at most 5 times", "submit_answer only.", " You have 30 model calls; after that you must answer with submit_answer only.")
if args.dense_caption_only:
    schema["properties"] = {"dense_caption": schema["properties"]["dense_caption"]}
    schema["required"] = ["dense_caption"]
    description = cut(description, ", events (what happens, ", " and mood.", ".")
    if not args.no_hours:
        description = cut(description, ", dense_caption, trace,", "highlights.", " and dense_caption.")
instructions = "The JSON Schema of the caption object of every index record:\n" + json.dumps(schema, indent=2) + "\n\n" + description
if args.magic_index:
    magic = (P / f"magic_index/{args.dataset}.txt").read_text().rstrip("\n")
    instructions = magic + "\n\n" + "\n\n".join(description.split("\n\n")[2:])
if args.skill:
    instructions += "\n" + Path(args.skill).read_text()

# 2. the tools
client = openai.OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=None, max_retries=5)
judge_client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY_PERSONAL"])
sampling = {k: float(getattr(args, k)) for k in ("temperature", "top_p", "presence_penalty") if getattr(args, k)}
if args.max_tokens:
    sampling["max_tokens"] = args.max_tokens
extra_body = {"top_k": int(float(args.top_k))} if args.top_k else None
effort = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}


def sh(*command, stdin=b"", check=False):
    return subprocess.run(command, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check).stdout


def qa(state, content):
    """One call of the QA model (exec_qa, video_qa)."""
    messages = [{"role": "user", "content": content}]
    r = client.chat.completions.create(model=args.model, messages=messages, extra_body=extra_body, **sampling, **effort)
    state["tokens"].append({"kind": "qa", "input": r.usage.prompt_tokens, "cached": 0, "output": r.usage.completion_tokens})
    if r.choices[0].finish_reason == "length":
        # the QA model reached --max-tokens: the question is run again (run_question) (the user, 2026-11-02)
        raise ModelBehaviorError(f"QA model output reached max_tokens {args.max_tokens}")
    return r.choices[0].message.content


def run_code(state, language, code, timeout_seconds):
    interpreter = ["python", "-"] if language == "python" else ["bash", "-s"]
    command = ["docker", "exec", "-i", state["container"], "timeout", "-s", "KILL", str(timeout_seconds), *interpreter]
    return sh(*command, stdin=code.encode()).decode(errors="replace")


exec_description = (P / "egolife/tools/exec_timeout_cap.txt").read_text()
over_limit = (P / "output_over_limit.txt").read_text()
if args.no_exec_qa:  # no exec_qa tool: an over-limit output only tells the agent to print less (the user, 2026-11-13)
    exec_description = cut(exec_description, ", or use exec_qa", ".", ".")
    over_limit = cut(over_limit, ", or use exec_qa", "full output.", ".")
if args.context_budget:
    exec_description = cut(exec_description, " Such a call does not count", "model calls.")
    over_limit = cut(over_limit, " This call does not count", "model calls.")


@function_tool(name_override="exec", description_override=exec_description, failure_error_function=None)
def exec_tool(ctx: RunContextWrapper, language: str, code: str, timeout_seconds: int) -> str:
    output = run_code(ctx.context, language, code, timeout_seconds)
    return output if len(output) <= 36000 else over_limit.format(n=len(output), limit=36000)


@function_tool(description_override=(P / "egolife/tools/exec_qa_timeout.txt").read_text(), failure_error_function=None)
def exec_qa(ctx: RunContextWrapper, language: str, code: str, timeout_seconds: int, question: str) -> str:
    output = run_code(ctx.context, language, code, timeout_seconds)
    prompt = (P / "egolife/exec_qa.txt").read_text().format(language=language, code=code, output=output, question=question)
    try:
        return qa(ctx.context, prompt)
    except openai.BadRequestError as e:  # over the QA model's context length: tell the agent (the user)
        if "maximum context length" not in str(e):
            raise
        print(ctx.context["q"]["key"], "exec_qa output over the context length:", len(output), "characters", flush=True)
        return "The output is too long for the QA model's context; make the output shorter"


def video_qa(state, source, start_s, end_s, fps, question):
    """Frames of a day (EgoLife timeline) or video (MM-Lifelong file) at fps and the question, in one QA model call."""
    state["calls"] += 1
    if fps > 0.5 or end_s - start_s > 1200 or (end_s - start_s) * fps > 650:
        return "fps can be at most 0.5, the interval at most 20 minutes and at most 650 frames"
    if choice and (int(source[3:]) > int(state["q"]["day"][3:]) or end_s > state["limit"].get(source, math.inf)):
        return "Footage after the question time is not available"
    content = [{"type": "text", "text": (P / "egolife/video_qa_system.txt").read_text() + "\nQuestion: " + question}]
    for t in [start_s + i / fps for i in range(math.ceil((end_s - start_s) * fps))]:
        files = [f for f in state["footage"][source] if f["start"] <= t < f["end"]]
        if not files:  # t is outside the day's timeline: skip it (the user)
            continue
        file = files[0]
        jpeg = sh("ffmpeg", "-v", "error", "-ss", str(t - file["start"]), "-i", file["path"], "-frames:v", "1", "-vf", "scale='min(1080,iw)':-2", "-f", "image2pipe", "-c:v", "mjpeg", "-", check=True)
        if not jpeg:  # t is past the file's last frame: use the last frame (the user)
            jpeg = sh("ffmpeg", "-v", "error", "-sseof", "-1", "-i", file["path"], "-frames:v", "1", "-vf", "reverse,scale='min(1080,iw)':-2", "-f", "image2pipe", "-c:v", "mjpeg", "-", check=True)
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}})
    if len(content) == 1:
        return "There is no footage in this interval"
    try:
        return qa(state, content)
    except openai.BadRequestError as e:  # the server's request body limit (between 32 and 48 MiB): tell the agent (the user)
        if "length limit exceeded" not in str(e):
            raise
        print(state["q"]["key"], "video_qa request too large:", len(content) - 1, "frames", flush=True)
        return "The request is too large for the QA model; use fewer frames"


def video_qa_day(ctx: RunContextWrapper, day: str, start_s: float, end_s: float, fps: float, question: str) -> str:
    return video_qa(ctx.context, day, start_s, end_s, fps, question)


def video_qa_video(ctx: RunContextWrapper, video: str, start_s: float, end_s: float, fps: float, question: str) -> str:
    return video_qa(ctx.context, video, start_s, end_s, fps, question)


video_description = (P / (f"mm_lifelong/tools/video_qa_{subset}.txt" if videos else "egolife/tools/video_qa.txt")).read_text()
video_tool = function_tool(video_qa_video if videos else video_qa_day, name_override="video_qa", description_override=video_description, failure_error_function=None)


@function_tool(description_override=(P / ("egolife/tools/submit_answer.txt" if choice else "tools/submit_answer.txt")).read_text(), failure_error_function=None)
def submit_answer(ctx: RunContextWrapper, answer: str) -> str:
    if choice and answer not in ctx.context["q"]["options"]:
        return (P / "egolife/invalid_option.txt").read_text().format(answer=answer, letters=", ".join(ctx.context["q"]["options"]))
    ctx.context["answer"] = answer
    return "Answer submitted."


def bm25_tokens(text):
    """Lowercased jieba tokens that contain a letter, a digit or a CJK character."""
    return [w for w in jieba.lcut(text.lower()) if re.search(r"\w", w)]


bm25_docs = {}  # --bm25: index file stem -> {clip_id: tokens of the whole record}, tokenized once
bm25_cache = {}  # --bm25: (stem, clip_ids before the question time) -> BM25Okapi over those records
if args.bm25:
    for stem in {f"egolife/{q['person']}" if choice or subset == "week" else f"mm_lifelong/{subset}" for q in questions}:
        bm25_docs[stem] = {r["clip_id"]: bm25_tokens(json.dumps(r, ensure_ascii=False)) for r in json.load(open(f"{args.data_dir}/{stem}.json"))}


search_description = (P / "bm25/search.txt").read_text()
search_over_limit = (P / "bm25/over_limit.txt").read_text()
if args.bm25_calls:  # the wording of the main experiment's exec tool
    search_description = cut(search_description, "your 5 searches", ".", "your budget of model calls.")
    search_over_limit = cut(search_over_limit, "your 5 searches", ".", "your budget of model calls.")


@function_tool(name_override="search", description_override=search_description, failure_error_function=None)
def search_tool(ctx: RunContextWrapper, query: str, top_k: int) -> str:
    state = ctx.context
    with state["search_lock"]:  # the SDK runs the tool calls of one turn in parallel threads; one search at a time keeps the count exact
        if state["searches"] >= 5 and not args.bm25_calls:  # parallel calls past the 5th search
            return "You have used your 5 searches and the search tool is no longer available."
        scores = state["bm25"].get_scores(bm25_tokens(query))
        output = json.dumps([state["records"][i] for i in scores.argsort()[::-1][:top_k]], ensure_ascii=False)
        if len(output) > 36000:
            return search_over_limit.format(n=len(output), limit=36000)
        state["searches"] += 1
        return output


class Stop(AgentsException):
    """Raised before an agent call once 30 model calls are used or the context is over 250,000 tokens (--context-budget: once the context is over 80% of the 1,048,576-token window; --bm25: once 5 searches are used)."""


class Hooks(RunHooks):
    async def on_llm_end(self, context, agent, response):
        u = response.usage
        context.context["tokens"].append({"kind": "agent", "input": u.input_tokens, "output": u.output_tokens, "first": context.context["calls"] == 1})
        context.context["context"] = u.input_tokens + u.output_tokens


def before_call(data):
    state = data.context
    if args.bm25 and not args.bm25_calls:
        stop = state["searches"] >= 5
        status = "searches_used"
    elif args.context_budget:
        stop = state["context"] > 0.8 * 1048576  # the server's max_model_len
        status = "context_forced"
    else:
        stop = state["calls"] >= 30 or state["context"] > 250000
        status = "budget_exhausted" if state["calls"] >= 30 else "context_limit"
    if not state["forced"] and stop:
        state["status"] = status
        state["items"] = list(data.model_data.input)
        raise Stop()
    state["calls"] += 1
    return data.model_data


settings = lambda tool_choice: ModelSettings(tool_choice=tool_choice, extra_body=extra_body, extra_args=effort or None, **sampling)
answered = lambda ctx, results: ToolsToFinalOutputResult(is_final_output=ctx.context["answer"] is not None)
tools = [exec_tool] + ([] if args.no_exec_qa else [exec_qa]) + ([] if args.no_video_qa else [video_tool]) + [submit_answer]
if args.bm25:
    tools = [search_tool, submit_answer]
docker_slots = threading.Semaphore(2)  # the host's systemd times out creating container scopes under load


# 3. run the agent (one question per thread; Runner.run_sync gives each run its own event loop, so its own async client)
def solve(q, qdir, tokens):
    stem = f"egolife/{q['person']}" if choice or subset == "week" else f"mm_lifelong/{subset}"
    minutes = json.load(open(f"{args.data_dir}/{stem}.json"))
    hours = [] if args.no_hours else json.load(open(f"{args.data_dir}/{stem}_hourly.json"))  # --no-hours: hours.json is not mounted, so it is not read
    footage = json.load(open(ROOT / f"data/egolife/timelines/{q['person']}.json")) if not videos else {
        name: [{"path": f"{MML}/videos/{subset}/{0 if subset == 'day' else name}.mp4", "start": 0, "end": math.inf}]
        for name in (["day"] if subset == "day" else map(str, range(1, 24)))
    }
    state = {"q": q, "footage": footage, "limit": {}, "calls": 0, "context": 0, "forced": False, "answer": None, "status": "submitted", "tokens": tokens, "searches": 0, "search_lock": threading.Lock()}
    if choice:
        until = (int(q["day"][3:]), sum(int(x) * s for x, s in zip(q["time"].split(":"), (3600, 60, 1))))
        minutes = [r for r in minutes if (int(r["day"][3:]), r["wall_end_s"]) <= until]
        hours = [r for r in hours if (int(r["end_day"][3:]), r["wall_end_s"]) <= until]
        today = [f for f in footage[q["day"]] if f["wall_start"] <= until[1]]
        state["limit"] = {q["day"]: max(min(f["start"] + until[1] - f["wall_start"], f["end"]) for f in today)}
    if args.dense_caption_only:
        minutes = [{**r, "caption": {"dense_caption": r["caption"]["dense_caption"]}} for r in minutes]
        hours = [{k: v for k, v in r.items() if k not in ("trace", "event_chains", "highlights")} for r in hours]
    (qdir / "index").mkdir(parents=True, exist_ok=True)
    (qdir / "index/minutes.json").write_text(json.dumps(minutes))
    (qdir / "index/hours.json").write_text(json.dumps(hours))
    if args.bm25:
        key = (stem, tuple(r["clip_id"] for r in minutes))
        if key not in bm25_cache:
            bm25_cache[key] = BM25Okapi([bm25_docs[stem][r["clip_id"]] for r in minutes])
        state["bm25"] = bm25_cache[key]
        state["records"] = minutes

    mounts = [f"{qdir}/index/minutes.json:/data/minutes.json"] + ([] if args.no_hours else [f"{qdir}/index/hours.json:/data/hours.json"])
    if args.magic_index:
        mounts = [f"{Path(args.magic_index).resolve() / (q['key'] if args.dataset == 'egolifeqa' else '')}:/data"]
    mounts += [f"{Path(d).resolve()}:{inside}" for d, inside in [(args.skill_dir, "/skill"), (args.decision_dir, "/decision")] if d]
    state["container"] = f"vj-{args.dataset}-{args.run_name}-{qdir.relative_to(run_dir)}".replace("/", "-").replace("_", "-")
    command = ["docker", "run", "-d", "--name", state["container"], f"--network={args.container_network or 'none'}", *[f"-v={m}:ro" for m in mounts], "--entrypoint", "sleep", "videojson-sandbox:1", "infinity"]
    if not args.bm25:  # --bm25: no code is run, so no container
        with docker_slots:
            for attempt in range(6):  # a systemd scope timeout (exit 125) is retried up to 5 times, 10 s apart (the user)
                sh("docker", "rm", "-f", state["container"])
                started = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if started.returncode == 0:
                    break
                print(q["key"], "docker run failed, attempt", attempt, started.stdout.decode(errors="replace").strip(), flush=True)
                assert attempt < 5 and b"cgroup" in started.stdout, started.stdout
                time.sleep(10)

    name = q["person"].split("_")[1].title()
    count = "four" if args.dataset in ("egolifeqa", "egor1") else str(len(q["options"])) if choice else ""
    async_client = openai.AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=None, max_retries=5)
    model = OpenAIChatCompletionsModel(model=args.model, openai_client=async_client, should_replay_reasoning_content=lambda c: True)
    agent = Agent(name="agent", instructions=instructions.replace("{name}", name).replace("{option_count}", count), tools=tools, model=model, model_settings=settings("required"), tool_use_behavior=answered, reset_tool_choice=False)
    if choice:
        options = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
        message = (P / "egolife/question.txt").read_text().format(name=name, day=q["day"], time=q["time"], question=q["question"], options=options)
    else:
        message = f"Question: {q['question']}"
    config = RunConfig(call_model_input_filter=before_call)
    try:
        state["messages"] = Runner.run_sync(agent, [{"role": "user", "content": message}], context=state, max_turns=10**9 if args.context_budget or (args.bm25 and not args.bm25_calls) else 1000, hooks=Hooks(), run_config=config).to_input_list()
    except Stop:
        state["messages"] = state["items"]
        if state["status"] in ("budget_exhausted", "context_forced", "searches_used"):
            state["forced"] = True
            final = "the letter of the closest option now." if choice else "your final answer now."
            if state["status"] == "searches_used":
                lead = "You have used your 5 searches and the search tool is no longer available"
            else:
                reason = "Your context is nearly full" if state["status"] == "context_forced" else "Your budget of 30 model calls is used up"
                lead = reason + " and the other tools are no longer available"
            note = {"role": "user", "content": lead + ". Call submit_answer with " + final}
            forced = agent.clone(tools=[submit_answer], model_settings=settings("submit_answer"))
            state["messages"] = Runner.run_sync(forced, state["items"] + [note], context=state, max_turns=1, hooks=Hooks(), run_config=config).to_input_list()
    finally:
        if not args.bm25:
            sh("docker", "rm", "-f", state["container"])
        for file in (qdir / "index").iterdir():
            file.unlink()
    return state


def run_question(q, qdir):
    tokens = []  # the calls of every run of the question count toward its cost
    for attempt in range(3):  # no tool call or invalid tool arguments: run the question again
        try:
            state = solve(q, qdir, tokens)
        except ModelBehaviorError as error:
            print(q["key"], "invalid tool arguments:", error, flush=True)
            continue
        if state["answer"] is not None or state["status"] == "context_limit":
            break
    if state["answer"] is None and state["status"] == "submitted":
        state["status"] = "no_answer"
    # cost assumes a perfect prefix cache: an agent call's cached input is the previous call's input + output of the same run
    calls = [t for t in state["tokens"] if t["kind"] == "agent"]
    for before, call in zip([None] + calls, calls):
        call["cached"] = 0 if call["first"] else before["input"] + before["output"]
    correct = state["answer"] == q["gold"] if q["gold"] and choice else None
    (qdir / "answer.json").write_text(json.dumps({"answer": state["answer"], "status": state["status"], "correct": correct, "tokens": state["tokens"]}, indent=1))
    (qdir / "messages.json").write_text(json.dumps(state["messages"], indent=1, ensure_ascii=False))
    if choice:
        print(q["key"], state["answer"], correct, flush=True)
        wandb.log({"correct": correct})
        return correct
    judge = {"score": 0, "acc": 0.0}
    if state["answer"]:
        user = (P / "judge_user.txt").read_text().format(question=q["question"], answer=q["gold"], pred=state["answer"])
        messages = [{"role": "system", "content": (P / "judge_system.txt").read_text()}, {"role": "user", "content": user}]
        r = judge_client.chat.completions.create(model="openai/gpt-5", messages=messages)
        score = int(re.search(r"Final Score:\s*([0-5])", r.choices[0].message.content, re.I).group(1))
        judge = {"score": score, "acc": 1.0 if score >= 4 else 0.5 if score == 3 else 0.0, "raw": r.choices[0].message.content}
    (qdir / "judge.json").write_text(json.dumps(judge, indent=1))
    print(q["key"], "score", judge["score"], flush=True)
    wandb.log({"acc": judge["acc"]})
    return judge["acc"]


def worker(q):
    try:
        for trial in range(1, args.trials + 1):
            qdir = run_dir / q["key"] / (f"trial{trial}" if args.trials > 1 else "")
            saved = qdir / ("answer.json" if choice else "judge.json")
            acc = json.load(open(saved))["correct" if choice else "acc"] if saved.exists() else run_question(q, qdir)
            if acc == 1:
                return
    except BaseException:
        traceback.print_exc()
        os._exit(1)  # an error stops the whole run


run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "instructions.txt").write_text(instructions)
wandb.init(project="videojson_gym", name=f"{args.dataset}-{args.run_name}", config=vars(args))
with ThreadPoolExecutor(args.concurrency) as pool:
    list(pool.map(worker, questions))
if args.dataset == "memreason":
    answers = [{"example_id": int(p.parent.name[1:]), "predicted_answer": json.load(open(p))["answer"]} for p in run_dir.glob("q*/answer.json")]
    (run_dir / "submission.json").write_text(json.dumps(sorted(answers, key=lambda a: a["example_id"]), indent=1))
wandb.finish()
print("DONE", flush=True)
1 / 0
