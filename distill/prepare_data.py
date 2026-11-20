#!/usr/bin/env python3
"""Export a frozen, within-livestream agent SFT pilot without executing tools."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


SEED = 20261030
PADDING_SECONDS = 300
MAX_SEQUENCE_TOKENS = 16384
MAX_TRAIN_EPISODES = 64
EXPECTED_IDS = set(range(407)) - {115, 132, 169, 218, 229, 259, 264}
QUARANTINED = {135}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json")
ARTIFACT_FILES = ("answer.json", "judge.json", "messages.json", "tools.json")


class DataError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(value).hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_json(path):
    return json.loads(Path(path).read_bytes())


def rank(namespace, value):
    return digest(f"{SEED}:{namespace}:{value}".encode())


def require(condition, message):
    if not condition:
        raise DataError(message)


def make_split(questions, eval_count=20, dev_count=12):
    ids = [q["index"] for q in questions]
    require(len(ids) == len(set(ids)), "duplicate question index")
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    intervals = []
    for q in questions:
        require(q.get("total_intervals"), f"missing total_intervals: q{q['index']}")
        for span in q["total_intervals"]:
            require(isinstance(span, list) and len(span) == 2, "invalid interval pair")
            start, end = span
            require(all(type(x) in (int, float) and math.isfinite(x) for x in span), "nonfinite interval")
            require(0 <= start < end, "invalid interval bounds")
            intervals.append((max(0, start - PADDING_SECONDS), end + PADDING_SECONDS, q["index"]))
    owner, furthest = None, -1
    for start, end, i in sorted(intervals):
        if start <= furthest:
            union(owner, i)
        else:
            owner = i
        if start > furthest or end > furthest:
            furthest, owner = end, i
    components = {}
    for i in sorted(ids):
        components.setdefault(find(i), []).append(i)
    groups = []
    for members in components.values():
        gid = "g-" + digest(compact(members).encode())[:16]
        groups.append({"group_id": gid, "question_ids": members})
    groups.sort(key=lambda g: rank("group", g["group_id"]))
    n_train = round(len(groups) * .70)
    n_dev = round(len(groups) * .15)
    pools = {"train": [], "dev": [], "eval": []}
    mapping = {}
    for n, group in enumerate(groups):
        side = "train" if n < n_train else "dev" if n < n_train + n_dev else "eval"
        group["split"] = side
        for i in group["question_ids"]:
            mapping[str(i)] = group["group_id"]
            if i not in QUARANTINED:
                pools[side].append(i)
    for side, pool in pools.items():
        pool.sort(key=lambda i: rank(f"{side}-question", i))
    require(len(pools["dev"]) >= dev_count, "fixed dev pool is too small; do not replace using scores")
    require(len(pools["eval"]) >= eval_count, "fixed eval pool is too small; do not replace using scores")
    return {
        "schema_version": 1, "seed": SEED, "padding_seconds_each_side": PADDING_SECONDS,
        "allocation": "hash-order components; round(70%) train, round(15%) dev, remainder eval",
        "scope": "same-livestream shared-corpus exploratory holdout; retrieved observations may overlap",
        "groups": groups, "group_id_by_question": mapping,
        "train_question_ids": pools["train"], "dev_question_ids": pools["dev"][:dev_count],
        "eval_question_ids": pools["eval"][:eval_count],
        "unused_heldout_question_ids": pools["dev"][dev_count:] + pools["eval"][eval_count:],
        "quarantined_question_ids": sorted(set(ids) & QUARANTINED),
    }


def validate_arguments(name, arguments, tools):
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    require(name in schemas, f"unknown tool: {name}")
    require(isinstance(arguments, dict), "tool arguments must be an object")
    schema = schemas[name]
    require(set(schema.get("required", [])) <= set(arguments), f"missing arguments: {name}")
    if schema.get("additionalProperties") is False:
        require(set(arguments) <= set(schema["properties"]), f"extra arguments: {name}")
    kinds = {"string": lambda v: isinstance(v, str), "integer": lambda v: type(v) is int,
             "number": lambda v: type(v) in (int, float) and math.isfinite(v)}
    for key, value in arguments.items():
        spec = schema.get("properties", {}).get(key, {})
        if spec.get("type") in kinds:
            require(kinds[spec["type"]](value), f"invalid argument type: {name}.{key}")
        if "enum" in spec:
            require(value in spec["enum"], f"invalid enum: {name}.{key}")


def response_groups(messages, instructions, tools):
    """Return causal prompt/next-turn pairs, preserving parallel calls and observations."""
    require(isinstance(messages, list) and messages, "empty messages")
    history = [{"role": "system", "content": instructions}]
    samples, seen_ids, seen_calls, pending = [], set(), set(), {}
    i = 0
    terminated = False
    while i < len(messages):
        item = messages[i]
        kind = item.get("type")
        if item.get("role") == "user" and kind in (None, "message"):
            require(not pending and not terminated, "user message interrupts a tool group")
            require(isinstance(item.get("content"), str), "non-text user content")
            history.append({"role": "user", "content": item["content"]})
            i += 1
            continue
        if kind == "function_call_output":
            call_id = item.get("call_id")
            require(call_id in pending, "orphan or duplicate tool output")
            require(isinstance(item.get("output"), str), "non-text tool output")
            history.append({"role": "tool", "tool_call_id": call_id,
                            "name": pending.pop(call_id), "content": item["output"]})
            i += 1
            continue
        require(kind in ("reasoning", "function_call"), f"unsupported message type: {kind}")
        require(not pending and not terminated, "assistant precedes pending outputs or follows submit")
        rid = item.get("provider_data", {}).get("response_id")
        require(isinstance(rid, str) and rid, "missing response_id")
        require(rid not in seen_ids, "noncontiguous response_id")
        seen_ids.add(rid)
        start, reasoning, calls = i, [], []
        while i < len(messages):
            item = messages[i]
            if item.get("type") not in ("reasoning", "function_call"):
                break
            if item.get("provider_data", {}).get("response_id") != rid:
                break
            if item["type"] == "reasoning":
                require(not calls, "reasoning follows a call within one response")
                for part in item.get("content", []):
                    require(part.get("type") == "reasoning_text" and isinstance(part.get("text"), str),
                            "unsupported reasoning content")
                    reasoning.append(part["text"])
            else:
                name, call_id = item.get("name"), item.get("call_id")
                require(isinstance(call_id, str) and call_id and call_id not in seen_calls,
                        "invalid/duplicate call_id")
                seen_calls.add(call_id)
                arguments = json.loads(item["arguments"])
                validate_arguments(name, arguments, tools)
                pending[call_id] = name
                calls.append({"id": call_id, "type": "function",
                              "function": {"name": name, "arguments": arguments}})
            i += 1
        require(calls, "reasoning-only response is not an action target")
        assistant = {"role": "assistant", "content": "", "reasoning_content": "\n".join(reasoning),
                     "tool_calls": calls}
        samples.append({"prompt": copy.deepcopy(history), "completion": [assistant], "tools": tools,
                        "response_id": rid, "source_item_start": start, "source_item_end": i})
        history.append(assistant)
        terminated = any(c["function"]["name"] == "submit_answer" for c in calls)
    require(not pending, "missing tool outputs at end of episode")
    require(terminated, "episode has no submit_answer")
    return samples


def encode_sample(sample, tokenizer, max_tokens=MAX_SEQUENCE_TOKENS):
    last_user = max(i for i, message in enumerate(sample["prompt"]) if message["role"] == "user")
    require(not any(m.get("reasoning_content", "").strip() for m in sample["prompt"][:last_user]),
            "native template would omit historical reasoning before latest user message")
    kwargs = {"tools": sample["tools"], "tokenize": False, "enable_thinking": True}
    prefix = tokenizer.apply_chat_template(sample["prompt"], add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(sample["prompt"] + sample["completion"],
                                         add_generation_prompt=False, **kwargs)
    require(full.startswith(prefix), "student template is not prefix-preserving for this target")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    ids = tokenizer.encode(full, add_special_tokens=False)
    require(ids[:len(prefix_ids)] == prefix_ids, "token boundary changes across prefix/target")
    require(len(ids) > len(prefix_ids), "empty target")
    if len(ids) > max_tokens:
        return None, len(ids)
    labels = [-100] * len(prefix_ids) + ids[len(prefix_ids):]
    require(any(v != -100 for v in labels[1:]), "no shifted target labels")
    result = {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids),
              "sequence_sha256": digest(compact(ids).encode()),
              "prefix_tokens": len(prefix_ids), "sequence_tokens": len(ids),
              "prefix_rendered_sha256": digest(prefix.encode()), "full_rendered_sha256": digest(full.encode())}
    return result, len(ids)


def load_episode(question, run, require_high_score):
    directory = Path(run) / f"q{question['index']}"
    missing = [name for name in ARTIFACT_FILES if not (directory / name).is_file()]
    if missing:
        return None, ["missing_artifact:" + name for name in missing]
    blobs = {name: (directory / name).read_bytes() for name in ARTIFACT_FILES}
    values = {name: json.loads(data) for name, data in blobs.items()}
    a, g = values["answer.json"], values["judge.json"]
    require((a["index"], a["question"], a["gold"]) ==
            (question["index"], question["question"], question["answer"]), "question identity mismatch")
    messages = values["messages.json"]
    require(messages and messages[0].get("role") == "user" and
            messages[0].get("content") == "Question: " + question["question"],
            "trace question identity mismatch")
    hashes = {name: digest(data) for name, data in blobs.items()}
    require(g["answer_sha256"] == hashes["answer.json"], "judge refers to a different answer")
    require(type(g["gpt_score"]) is int and 0 <= g["gpt_score"] <= 5, "invalid teacher score")
    reasons = []
    if require_high_score and g["gpt_score"] < 4:
        reasons.append("teacher_score_below_4")
    if a["status"] != "submitted":
        reasons.append("non_submitted:" + a["status"])
    if any(t.get("error") for t in values["tools.json"]):
        reasons.append("tool_error")
    if g.get("finish_reason") != "stop":
        reasons.append("judge_not_finished")
    if reasons:
        return None, reasons
    return {"messages": values["messages.json"], "artifact_sha256": hashes,
            "directory": str(directory), "score": g["gpt_score"]}, []


def jsonl_bytes(rows):
    return b"".join((compact(row) + "\n").encode() for row in rows)


def export_dataset(args, tokenizer):
    output = Path(args.output)
    require(not output.exists(), "output directory already exists; refusing to overwrite")
    question_bytes = Path(args.questions).read_bytes()
    questions = json.loads(question_bytes)
    require({q["index"] for q in questions} == EXPECTED_IDS and len(questions) == 400,
            "expected the original 400 real question IDs")
    by_id = {q["index"]: q for q in questions}
    contract_bytes = Path(args.contract).read_bytes()
    contract = json.loads(contract_bytes)
    instructions_bytes = (Path(args.teacher_run) / "instructions.txt").read_bytes()
    require(digest(instructions_bytes) == contract["instructions_sha256"], "instructions hash mismatch")
    require(digest(question_bytes) == contract["plan"]["questions_sha256"], "question file hash mismatch")
    instructions, tools = instructions_bytes.decode(), contract["agent_tools"]
    tokenizer_hashes = {name: digest((Path(args.tokenizer) / name).read_bytes()) for name in TOKENIZER_FILES}
    tokenizer_info = {"file_sha256": tokenizer_hashes,
                      "fingerprint": digest(compact(tokenizer_hashes).encode()),
                      "model": contract["plan"]["student_model"],
                      "revision": contract["plan"].get("student_revision"),
                      "class": type(tokenizer).__name__}
    split = make_split(questions)
    split["questions_sha256"] = digest(question_bytes)
    rows, selected, exclusions, lengths = {}, {}, {}, {}
    for side in ("train", "dev"):
        rows[side], selected[side], exclusions[side], lengths[side] = [], [], {}, {}
        for i in split[f"{side}_question_ids"]:
            key = f"q{i}"
            if side == "train" and len(selected[side]) == MAX_TRAIN_EPISODES:
                exclusions[side][key] = ["not_considered_after_episode_limit"]
                continue
            episode, reasons = load_episode(by_id[i], args.teacher_run, require_high_score=side == "train")
            if reasons:
                exclusions[side][key] = reasons
                continue
            try:
                turns = response_groups(episode["messages"], instructions, tools)
                encoded = [encode_sample(turn, tokenizer) for turn in turns]
            except (DataError, json.JSONDecodeError) as exc:
                exclusions[side][key] = ["unsupported_episode:" + str(exc)]
                continue
            lengths[side][key] = max(n for _, n in encoded)
            if any(value is None for value, _ in encoded):
                exclusions[side][key] = ["whole_episode_exceeds_16384"]
                continue
            for number, (turn, (tokens, _)) in enumerate(zip(turns, encoded)):
                canonical = {name: turn[name] for name in ("prompt", "completion", "tools")}
                row = {"row_id": f"{key}:{number}", "episode_id": key, "question_index": i,
                       "group_id": split["group_id_by_question"][str(i)], **tokens, **canonical,
                       "provenance": {"artifact_sha256": episode["artifact_sha256"],
                                      "source_directory": episode["directory"],
                                      "response_id": turn["response_id"],
                                      "source_item_start": turn["source_item_start"],
                                      "source_item_end": turn["source_item_end"],
                                      "tokenizer_fingerprint": tokenizer_info["fingerprint"],
                                      "instructions_sha256": digest(instructions_bytes),
                                      "tool_schema_sha256": digest(compact(tools).encode())}}
                rows[side].append(row)
            for name, old_hash in episode["artifact_sha256"].items():
                require(digest((Path(episode["directory"]) / name).read_bytes()) == old_hash,
                        "source artifact changed during export")
            selected[side].append(key)
    require(rows["train"], "no eligible train episodes")
    require(Path(args.questions).read_bytes() == question_bytes and
            Path(args.contract).read_bytes() == contract_bytes, "source contract changed during export")
    require((Path(args.teacher_run) / "instructions.txt").read_bytes() == instructions_bytes,
            "instructions changed during export")
    require(all(digest((Path(args.tokenizer) / name).read_bytes()) == old_hash
                for name, old_hash in tokenizer_hashes.items()), "tokenizer changed during export")
    split_bytes = (json.dumps(split, ensure_ascii=False, indent=2) + "\n").encode()
    train_bytes, dev_bytes = jsonl_bytes(rows["train"]), jsonl_bytes(rows["dev"])
    manifest = {
        "schema_version": 1, "scope": split["scope"], "seed": SEED,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS, "max_train_episodes": MAX_TRAIN_EPISODES,
        "whole_episode_filter": True, "truncation": "none", "loss": "current assistant response group only",
        "reasoning": "preserved; native student template; enable_thinking=True; historical reasoning replay",
        "tokenizer": tokenizer_info, "questions_sha256": digest(question_bytes),
        "contract_sha256": digest(contract_bytes), "instructions_sha256": digest(instructions_bytes),
        "tool_schema_sha256": digest(compact(tools).encode()), "split_sha256": digest(split_bytes),
        "train_jsonl_sha256": digest(train_bytes), "dev_jsonl_sha256": digest(dev_bytes),
        "selected_train_episode_ids": selected["train"], "selected_dev_loss_episode_ids": selected["dev"],
        "fixed_dev_question_ids": split["dev_question_ids"], "fixed_eval_question_ids": split["eval_question_ids"],
        "dev_note": "fixed dev questions are not replaced; offline loss may cover only eligible teacher episodes",
        "train_rows": len(rows["train"]), "dev_rows": len(rows["dev"]),
        "excluded": exclusions, "episode_max_sequence_tokens": lengths,
        "quality_note": "machine-filtered teacher trajectories; not independently verified gold demonstrations",
        "quarantine": {"q135": "teacher evidence/final-answer attribution inconsistency"},
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("split.json", split_bytes), ("train.jsonl", train_bytes), ("dev.jsonl", dev_bytes)):
        (output / name).write_bytes(data)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("questions", "teacher-run", "contract", "tokenizer", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    manifest = export_dataset(args, tokenizer)
    print(json.dumps({k: manifest[k] for k in ("train_rows", "dev_rows", "selected_train_episode_ids",
                                               "fixed_dev_question_ids", "fixed_eval_question_ids")}, indent=2))


if __name__ == "__main__":
    main()
