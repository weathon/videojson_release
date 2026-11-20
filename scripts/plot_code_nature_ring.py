"""Ring chart of the code nature (search / read_interval / other / malformed), ring chart of the code language (python / bash / malformed), and pie chart of the tokens per question (uncached input, cached input, output), each averaged over the five datasets; or all three side by side. Code nature and language come from runs/code_analysis/v4_minimal_wholejson/summary.json; tokens are the sums over every agent and exec_qa call in answer.json of the DeepSeek main runs, cached being main.py's perfect-prefix-cache estimate. Usage: plot_code_nature_ring.py <category|language|tokens|all>"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/code_analysis/v4_minimal_wholejson"
summary = json.loads((OUT / "summary.json").read_text())
datasets = ["egolifeqa", "mm_week", "mm_day", "mm_month", "egor1"]
RUNS = {"egolifeqa": "v4_minimal", "mm_week": "v4_minimal", "mm_day": "v4_minimal", "mm_month": "v4_minimal", "egor1": "v4_minimal_bi"}
FIELD = sys.argv[1]
codes = sum(summary[d]["codes"] for d in datasets)
panels = {}
for field, classes in [
    ("category", [("search", "Regex or search"), ("read_interval", "Read a specific interval"), ("other", "Other"), ("malformed", "Malformed call")]),
    ("language", [("python", "Python"), ("bash", "Bash"), ("malformed", "Malformed call")]),
]:
    shares = [sum(summary[d][field][key] for d in datasets) / len(datasets) for key, _ in classes]
    panels[field] = {
        "labels": [f"{label} ({share * 100:.1f}%)" for (_, label), share in zip(classes, shares)],
        "values": shares,
        "ring": True,
        "center": f"{codes:,}\ncode calls",
        "file": "code_nature_ring" if field == "category" else "code_language_ring",
    }
per_dataset = []
for dataset in datasets:
    tokens = [json.loads(p.read_text())["tokens"] for p in (ROOT / "runs" / dataset / RUNS[dataset]).glob("*/answer.json")]
    cached = sum(sum(t["cached"] for t in q) for q in tokens) / len(tokens)
    total_input = sum(sum(t["input"] for t in q) for q in tokens) / len(tokens)
    output = sum(sum(t["output"] for t in q) for q in tokens) / len(tokens)
    per_dataset.append((total_input - cached, cached, output))
values = [sum(row[i] for row in per_dataset) / len(per_dataset) for i in range(3)]
panels["tokens"] = {
    "labels": [f"{label}: {value / 1000:,.1f}K ({value / sum(values) * 100:.1f}%)" for label, value in zip(["Uncached input", "Cached input", "Output"], values)],
    "values": values,
    "ring": False,
    "center": None,
    "file": "token_pie",
}
for name, panel in panels.items():
    print(name, panel["labels"])

# R base pie() defaults: radius 0.8, start at 3 o'clock, counterclockwise, black lwd 1 borders, labels at 1.1 x radius with a tick from 1.0 to 1.05 x radius
RADIUS = 0.8
plt.rcParams["font.family"] = "STIXGeneral"  # Times-style serif bundled with matplotlib; no Times New Roman on this machine (the user, 2026-11-10)
R_COLORS = ["white", "lightblue", "mistyrose", "lightcyan"]  # R pie() default col


def draw(axis, panel):
    wedges, _ = axis.pie(
        panel["values"],
        colors=R_COLORS[: len(panel["values"])],
        radius=RADIUS,
        startangle=0,
        counterclock=True,
        wedgeprops={"width": 0.3, "edgecolor": "black", "linewidth": 0.75} if panel["ring"] else {"edgecolor": "black", "linewidth": 0.75},
    )
    for wedge, label in zip(wedges, panel["labels"]):
        angle = math.radians((wedge.theta1 + wedge.theta2) / 2)
        x = RADIUS * math.cos(angle)
        y = RADIUS * math.sin(angle)
        axis.plot([x, 1.05 * x], [y, 1.05 * y], color="black", linewidth=0.75)
        axis.text(1.1 * x, 1.1 * y, label, ha="left" if x >= 0 else "right", va="center", fontsize=9, color="black")
    if panel["center"]:
        axis.text(0, 0, panel["center"], ha="center", va="center", fontsize=9, color="black")
    axis.set_xlim(-0.9, 0.9)
    axis.set_ylim(-0.9, 0.9)
    axis.set_aspect("equal")


names = ["category", "language", "tokens"] if FIELD == "all" else [FIELD]
# every panel is a square axes of SIDE inches, so equal aspect never shrinks it; the panels are then placed side by side by the extent of each panel including its labels, GAP inches apart
SIDE = 1.6
GAP = 0.15
figure = plt.figure(figsize=(4 * len(names), 2.4))
axes = [figure.add_axes([(0.5 + 4 * i) / (4 * len(names)), 0.4 / 2.4, SIDE / (4 * len(names)), SIDE / 2.4]) for i in range(len(names))]
for axis, name in zip(axes, names):
    draw(axis, panels[name])
figure.patch.set_facecolor("white")
figure.canvas.draw()
renderer = figure.canvas.get_renderer()
boxes = [(axis.get_tightbbox(renderer), axis.get_window_extent(renderer)) for axis in axes]
dpi = figure.dpi
width = sum(tight.width for tight, _ in boxes) / dpi + GAP * (len(boxes) - 1)
height = max(tight.height for tight, _ in boxes) / dpi
figure.set_size_inches(width, height)
left = 0
for axis, (tight, window) in zip(axes, boxes):
    x0 = left + (window.x0 - tight.x0) / dpi
    y0 = (height - tight.height / dpi) / 2 + (window.y0 - tight.y0) / dpi
    axis.set_position([x0 / width, y0 / height, SIDE / width, SIDE / height])
    left += tight.width / dpi + GAP
file = "code_analysis_pies" if FIELD == "all" else panels[FIELD]["file"]
figure.savefig(OUT / f"{file}.pdf", bbox_inches="tight")
figure.savefig(OUT / f"{file}.png", dpi=200, bbox_inches="tight")
print("DONE", flush=True)
1 / 0
