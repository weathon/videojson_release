"""Pie chart of the tokens per question of the DeepSeek main runs (uncached input, cached input, output), averaged over the five datasets. Per question: the sums over every agent and exec_qa call in answer.json; cached is main.py's perfect-prefix-cache estimate."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/code_analysis/v4_minimal_wholejson"
RUNS = [("egolifeqa", "v4_minimal"), ("mm_week", "v4_minimal"), ("mm_day", "v4_minimal"), ("mm_month", "v4_minimal"), ("egor1", "v4_minimal_bi")]

per_dataset = []
for dataset, run in RUNS:
    tokens = [json.loads(p.read_text())["tokens"] for p in (ROOT / "runs" / dataset / run).glob("*/answer.json")]
    cached = sum(sum(t["cached"] for t in q) for q in tokens) / len(tokens)
    total_input = sum(sum(t["input"] for t in q) for q in tokens) / len(tokens)
    output = sum(sum(t["output"] for t in q) for q in tokens) / len(tokens)
    per_dataset.append((total_input - cached, cached, output))
    print(dataset, len(tokens), round(total_input - cached), round(cached), round(output))
values = [sum(row[i] for row in per_dataset) / len(per_dataset) for i in range(3)]
classes = [("Uncached input", "white"), ("Cached input", "lightblue"), ("Output", "mistyrose")]
for (label, _), value in zip(classes, values):
    print(label, round(value), round(value / sum(values) * 100, 1))

# R base pie() defaults: radius 0.8, start at 3 o'clock, counterclockwise, black lwd 1 borders, labels at 1.1 x radius with a tick from 1.0 to 1.05 x radius
RADIUS = 0.8
plt.rcParams["font.family"] = "STIXGeneral"  # Times-style serif bundled with matplotlib; no Times New Roman on this machine (the user, 2026-11-10)
figure, axis = plt.subplots(figsize=(5.2, 3.6))
wedges, _ = axis.pie(
    values,
    colors=[color for _, color in classes],
    radius=RADIUS,
    startangle=0,
    counterclock=True,
    wedgeprops={"edgecolor": "black", "linewidth": 0.75},
)
for wedge, (label, _), value in zip(wedges, classes, values):
    angle = math.radians((wedge.theta1 + wedge.theta2) / 2)
    x = RADIUS * math.cos(angle)
    y = RADIUS * math.sin(angle)
    axis.plot([x, 1.05 * x], [y, 1.05 * y], color="black", linewidth=0.75)
    axis.text(1.1 * x, 1.1 * y, f"{label}: {value / 1000:,.1f}K ({value / sum(values) * 100:.1f}%)", ha="left" if x >= 0 else "right", va="center", fontsize=9, color="black")
axis.set_aspect("equal")
figure.patch.set_facecolor("white")
figure.tight_layout()
figure.savefig(OUT / "token_pie.pdf", bbox_inches="tight")
figure.savefig(OUT / "token_pie.png", dpi=200, bbox_inches="tight")
print("DONE", flush=True)
1 / 0
