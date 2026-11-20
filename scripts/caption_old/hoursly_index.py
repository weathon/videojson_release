# %%
import os
import json
people = ["A1_JAKE", "A2_ALICE", "A3_TASHA", "A4_LUCIA", "A5_KATRINA", "A6_SHURE"]
data = {}
for person in people:
    with open(f"../data/egolife/{person}.json", "r") as f:
        data[person] = json.load(f)

# %%
from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    time: str = Field(
        description="Time of arrival at this place, in the same format as start_time in the input index"
    )
    place: str = Field(
        description="Name of the place, or a short description if it has no name"
    )


class ChainEvent(BaseModel):
    description: str = Field(description="Description of the event")
    start_time: str = Field(
        description="Start time of the event in the video, in the same format as start_time in the input index"
    )
    end_time: str = Field(
        description="End time of the event in the video, in the same format as end_time in the input index"
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
import json

index_prompt = f"""
You are a video captioning assistant. You will receive the per-minute captions of one 60-minute segment of a video, in chronological order. Each entry is a JSON object describing one 60-second clip, with its start_time and end_time.

Your task is to merge them into a single caption for the whole 60 minutes.

Guidelines:
- Write the dense caption as one continuous story of the 60 minutes. Do not go through the clips minute by minute.
- The same person, object, or place may be described differently in different clips. Treat them as the same thing and refer to them consistently.
- An event chain is a sequence of related events that follow one another, such as going to a store, picking items, and paying. Unrelated events belong to different chains, even if they happen at the same time.
- Take all times from the start_time and end_time of the input clips.
- Use only information that appears in the input. Do not guess or add anything.

Output a single JSON object that follows this schema, with no other text: (covering the whole 60 minutes)
{json.dumps(HourCaption.model_json_schema(), indent=2)}
"""

# %%
{person: len(clips) // 60 for person, clips in data.items()}

# %%
import datetime
for person, clips in data.items():
    for i in range(1, len(clips)):
        if clips[i]["day"] == clips[i-1]["day"]:
            assert datetime.datetime.strptime(clips[i]["start_time"], "%H:%M:%S") > datetime.datetime.strptime(clips[i-1]["start_time"], "%H:%M:%S")

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
    for attempt in range(int(1e10)):
        try:
            response = client.responses.parse(
                model="openai/gpt-6-luna",
                input=messages,
                text_format=HourCaption
            )
            hourly_caption = json.loads(response.output_text)
            print(f"Finished {slice[0]['day']} {slice[0]['start_time']}")
            break
        except:
            pass
    hourly_caption["start_time"] = slice[0]["start_time"]
    hourly_caption["start_day"] = slice[0]["day"]
    hourly_caption["end_time"] = slice[-1]["end_time"]
    hourly_caption["end_day"] = slice[-1]["day"]
    hourly_caption["wall_start_s"] = slice[0]["wall_start_s"]
    hourly_caption["wall_end_s"] = slice[-1]["wall_end_s"]
    hourly_caption["cross_day_boundary"] = slice[0]["day"] != slice[-1]["day"]
    return hourly_caption

# %% run
from concurrent.futures import ThreadPoolExecutor, as_completed

results = {}
jobs = []
for person, clips in data.items():
    out_path = f"../data/egolife/{person}_hourly.json"
    results[person] = []
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            results[person] = json.load(f)
    done = {(hour["start_day"], hour["start_time"]) for hour in results[person]}
    for start in range(0, len(clips), 60):
        slice = clips[start:start + 60]
        if (slice[0]["day"], slice[0]["start_time"]) not in done:
            jobs.append((person, slice))
print(f"{len(jobs)} hours to caption", flush=True)

with ThreadPoolExecutor(len(jobs)) as pool:
    futures = {pool.submit(get_hourly_caption, slice): (person, slice) for person, slice in jobs}
    for future in as_completed(futures):
        person, slice = futures[future]
        try:
            hourly_caption = future.result()
        except (openai.APIError, json.JSONDecodeError, pydantic.ValidationError) as e:
            print(f"FAILED {person} {slice[0]['day']} {slice[0]['start_time']}: {type(e).__name__}: {e}", flush=True)
            continue
        results[person].append(hourly_caption)
        results[person].sort(key=lambda hour: (hour["start_day"], hour["start_time"]))
        with open(f"../data/egolife/{person}_hourly.json", "w") as f:
            json.dump(results[person], f, ensure_ascii=False, indent=2)
        print(f"DONE {person} {slice[0]['day']} {slice[0]['start_time']} ({len(results[person])} saved)", flush=True)




5/0