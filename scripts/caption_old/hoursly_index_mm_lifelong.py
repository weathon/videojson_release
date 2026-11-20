# %%
import os
import json
subsets = ["day", "month"]
data = {}
for subset in subsets:
    with open(f"../data/mm_lifelong/{subset}.json", "r") as f:
        data[subset] = json.load(f)

# %%
from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    video: str = Field(
        description="The video in which this time falls, same as video in the input index"
    )
    time: float = Field(
        description="Time of arrival at this place, in seconds, in the same format as tick_start_s in the input index"
    )
    place: str = Field(
        description="Name of the place, or a short description if it has no name"
    )


class ChainEvent(BaseModel):
    description: str = Field(description="Description of the event")
    video: str = Field(
        description="The video in which start_time and end_time fall, same as video in the input index"
    )
    start_time: float = Field(
        description="Start time of the event in the video, in seconds, in the same format as tick_start_s in the input index"
    )
    end_time: float = Field(
        description="End time of the event in the video, in seconds, in the same format as tick_end_s in the input index"
    )


class HourCaption(BaseModel):
    dense_caption: str = Field(
        description="Overall description of what happens in these 60 minutes. Leave out specific details, but cover the complete, coherent narrative of the period"
    )
    trace: list[TraceStep] = Field(
        description="Location changes during these 60 minutes in chronological order, one entry per change of place"
    )
    event_chains: list[list[ChainEvent]] = Field(
        description="Main event chains in these 60 minutes. The outer list holds all chains, and each inner list holds the events of one chain in chronological order"
    )
    highlights: str = Field(
        description="Key conversations or content worth highlighting, such as purchases, discussion outcomes, or a finished project"
    )

# %%
index_prompt = f"""
You are a video captioning assistant. You will receive the per-minute captions of one 60-minute segment of a video, in chronological order. Each entry is a JSON object describing one 60-second clip, with its video, tick_start_s and tick_end_s (seconds of recorded footage within that video). The segment may span more than one video; the seconds start again from 0 in each video.

Your task is to merge them into a single caption for the whole 60 minutes.

Guidelines:
- Write the dense caption as one continuous story of the 60 minutes. Do not go through the clips minute by minute.
- The same person, object, or place may be described differently in different clips. Treat them as the same thing and refer to them consistently.
- An event chain is a sequence of related events that follow one another, such as going to a store, picking items, and paying. Unrelated events belong to different chains, even if they happen at the same time.
- Take all times from the tick_start_s and tick_end_s of the input clips, together with the video of those clips.
- Use only information that appears in the input. Do not guess or add anything.

Output a single JSON object that follows this schema, with no other text: (covering the whole 60 minutes)
{json.dumps(HourCaption.model_json_schema(), indent=2)}
"""

# %%
{subset: len(clips) // 60 for subset, clips in data.items()}

# %%
for subset, clips in data.items():
    for i in range(1, len(clips)):
        if clips[i]["video"] == clips[i-1]["video"]:
            assert clips[i]["tick_start_s"] > clips[i-1]["tick_start_s"]

# %%
import openai
import pydantic
from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], max_retries=10086)

def get_hourly_caption(slice):
    messages = [
        {
            "role": "system",
            "content": index_prompt
        },
        {
            "role": "user",
            "content": json.dumps(slice)
        }
    ]
    for attempt in range(10086):
        try:
            response = client.responses.parse(
                model="openai/gpt-6-luna",
                input=messages,
                text_format=HourCaption
            )
            hourly_caption = json.loads(response.output_text)
            print(f"Finished {slice[0]['video']} {slice[0]['tick_start_s']}")
            break
        except:
            pass
    hourly_caption["start_video"] = slice[0]["video"]
    hourly_caption["tick_start_s"] = slice[0]["tick_start_s"]
    hourly_caption["end_video"] = slice[-1]["video"]
    hourly_caption["tick_end_s"] = slice[-1]["tick_end_s"]
    hourly_caption["cross_video_boundary"] = slice[0]["video"] != slice[-1]["video"]
    return hourly_caption

# %% run
from concurrent.futures import ThreadPoolExecutor, as_completed

results = {}
order = {}
jobs = []
for subset, clips in data.items():
    out_path = f"../data/mm_lifelong/{subset}_hourly.json"
    results[subset] = []
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            results[subset] = json.load(f)
    done = {(hour["start_video"], hour["tick_start_s"]) for hour in results[subset]}
    for start in range(0, len(clips), 60):
        slice = clips[start:start + 60]
        order[(subset, slice[0]["video"], slice[0]["tick_start_s"])] = start
        if (slice[0]["video"], slice[0]["tick_start_s"]) not in done:
            jobs.append((subset, slice))
print(f"{len(jobs)} hours to caption", flush=True)

with ThreadPoolExecutor(len(jobs)) as pool:
    futures = {pool.submit(get_hourly_caption, slice): (subset, slice) for subset, slice in jobs}
    for future in as_completed(futures):
        subset, slice = futures[future]
        try:
            hourly_caption = future.result()
        except (openai.APIError, json.JSONDecodeError, pydantic.ValidationError) as e:
            print(f"FAILED {subset} {slice[0]['video']} {slice[0]['tick_start_s']}: {type(e).__name__}: {e}", flush=True)
            continue
        results[subset].append(hourly_caption)
        results[subset].sort(key=lambda hour: order[(subset, hour["start_video"], hour["tick_start_s"])])
        with open(f"../data/mm_lifelong/{subset}_hourly.json", "w") as f:
            json.dump(results[subset], f, ensure_ascii=False, indent=2)
        print(f"DONE {subset} {slice[0]['video']} {slice[0]['tick_start_s']} ({len(results[subset])} saved)", flush=True)

5/0