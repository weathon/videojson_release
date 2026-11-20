"""Score and cost per question of one harness run (runs/<dataset>/<run name>), at the prices of the paper: US$0.15 per million uncached input tokens, US$0.003 per million cached input tokens and US$0.60 per million output tokens. The judge is not counted. Usage: cost.py <run directory> [keys file]"""

import json
import sys
from pathlib import Path

RUN = Path(sys.argv[1])
keys = set(open(sys.argv[2]).read().split()) if len(sys.argv) > 2 else None
scores = []
cents = []
for path in sorted(RUN.glob("*/answer.json")):
    if keys is not None and path.parent.name not in keys:
        continue
    answer = json.load(open(path))
    judge = path.parent / "judge.json"  # open questions have judge.json (acc 0, 0.5 or 1); choice questions have correct in answer.json
    scores.append(json.load(open(judge))["acc"] if judge.exists() else answer["correct"])
    dollars = 0
    for call in answer["tokens"]:
        dollars += ((call["input"] - call["cached"]) * 0.15 + call["cached"] * 0.003 + call["output"] * 0.60) / 1e6
    cents.append(dollars * 100)
print(f"questions {len(scores)} score {100 * sum(scores) / len(scores):.2f} cost per question {sum(cents) / len(cents):.4f} US cents")
print("DONE", flush=True)
1 / 0
