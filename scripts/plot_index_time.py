"""Horizontal segmented bar of the index wall time (runs/index_time/summary.json): one bar for VideoJSON, one for MAGIC-Video, one segment per stage, in the colors of the paper's Figure 1 (the user, 2026-11-15)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / 'runs/index_time/summary.json').read_text())

# Figure 1 palette: grey process boxes, light-blue episodes, yellow/gold semantic triplets, green entities, pink visual clips,
# purple JSON panel; each fill with its darker edge
COLORS = {
    'ASR (Qwen3-ASR-1.7B)': ('#d9d9d9', '#666666'),
    'ASR (Whisper large-v3-turbo)': ('#d9d9d9', '#666666'),
    'ASR rewrite': ('#bfbfbf', '#666666'),
    'VLM captioner (minute JSON captions)': ('#dae8fc', '#6c8ebf'),
    'VLM captioner (30 s captions)': ('#dae8fc', '#6c8ebf'),
    'Align & merge': ('#f5f5f5', '#666666'),
    'Multi-granularity aggregation': ('#a9c4eb', '#6c8ebf'),
    'Named entity recognition (OpenIE)': ('#d5e8d4', '#82b366'),
    'Semantic extraction': ('#fff2cc', '#d6b656'),
    'Semantic consolidation': ('#ffd966', '#d6b656'),
    'VLM embedder (visual embeddings)': ('#e6d0de', '#996185'),
    'Other (model loading, file writes)': ('#ffffff', '#999999'),
    'Graph building': ('#fff9e6', '#d6b656'),
    'Topic chains': ('#e1d5e7', '#9673a6'),
    'Event chains': ('#c3abd0', '#9673a6'),
    'JSON index': ('#e1d5e7', '#9673a6'),
}

plt.rcParams['font.family'] = 'sans-serif'
figure, axis = plt.subplots(figsize=(11, 4.5))
XMAX = summary['magic']['total_s'] / 60 * 1.09
INCH = 0.42  # inches of figure height per y unit, so label rows and bars keep their size whatever the number of rows
axis.set_xlim(0, XMAX)
axis.set_ylim(0, 4.5 / INCH)
renderer = figure.canvas.get_renderer()
BAR, STEP, PAD, SLANT = 0.6, 0.38, 0.6, 2.5  # bar height; height of one label row; minutes kept free around a label; horizontal run of a leader line (min)
bars = []
y = 0.0
for name, run in [('MAGIC-Video', summary['magic']), ('VideoJSON (Ours)', summary['ours'])]:  # bottom up; each bar sits above the labels of the one below
    bars.append((name, run, y))
    left = 0
    centers = []
    for stage, seconds in run['stages_s'].items():
        fill, edge = COLORS[stage]
        axis.barh(y, seconds / 60, left=left, height=BAR, color=fill, edgecolor=edge, linewidth=1)
        centers.append((stage, left + seconds / 120))
        left += seconds / 60
    # labels from the right end, each beside its segment's center (right of it if it fits, else left of it) on the lowest row
    # where it overlaps no label and its slanted leader line, from the center to the label's near end, crosses no label
    boxes = []  # (row, start, end) in minutes of every placed label
    lines = []  # (center, anchor, label height) of every drawn leader line
    top = 0
    for index in reversed(range(len(centers))):
        stage, center = centers[index]
        # a label left of its center must stay right of the next segment to the left, whose line is drawn after it
        floor = centers[index - 1][1] + PAD if index else 0
        text = axis.text(center, 0, stage, ha='left', va='bottom', fontsize=8.5)
        corners = axis.transData.inverted().transform(text.get_window_extent(renderer))
        width = corners[1][0] - corners[0][0]
        row = 0
        while True:
            assert row < 30, stage
            label_y = y + BAR / 2 + 0.25 + row * STEP
            placed = None
            for start, anchor in ((center + SLANT, center + SLANT), (center - SLANT - width, center - SLANT)):
                if start < floor:  # a label may run past the axis into the right margin (texts are not clipped)
                    continue
                if any(r == row and a < start + width + PAD / 2 and start - PAD / 2 < b for r, a, b in boxes):
                    continue
                # the line from (center, bar top) to (anchor, label_y) must pass beside every lower label (checked at its mid height)
                line_x = lambda height: center + (anchor - center) * (height - y - BAR / 2) / (label_y - y - BAR / 2)
                if any(r < row and a - PAD / 2 <= line_x(y + BAR / 2 + 0.25 + r * STEP + 0.1) <= b + PAD / 2 for r, a, b in boxes):
                    continue
                # and no line already drawn to a higher label may pass through this label
                if any(ly > label_y and start - PAD / 2 <= cx + (ax - cx) * (label_y + 0.1 - y - BAR / 2) / (ly - y - BAR / 2) <= start + width + PAD / 2 for cx, ax, ly in lines):
                    continue
                placed = (start, anchor)
                break
            if placed:
                break
            row += 1
        start, anchor = placed
        boxes.append((row, start, start + width))
        lines.append((center, anchor, label_y))
        top = max(top, row + 1)
        text.set_position((start, label_y))
        axis.plot([center, anchor], [y + BAR / 2, label_y], color='#666666', linewidth=0.6)
    rows = range(top)
    axis.text(left + 1.5, y, f'{run["total_s"] / 60:.1f} min', va='center', fontsize=11, fontweight='bold')
    y += BAR / 2 + 0.25 + len(rows) * STEP + 0.55
top = y - 0.55 + 0.05  # just above the highest label row
axis.set_ylim(-BAR / 2 - 0.1, top)
figure.set_size_inches(11, (top + BAR / 2 + 0.1) * INCH + 0.7)
axis.set_yticks([y for _, _, y in bars], [name for name, _, _ in bars], fontsize=12, fontweight='bold')
axis.set_xlabel('Wall-clock time to index the first 3 hours of MM-Lifelong day (min)', fontsize=11)
for side in ('top', 'right'):
    axis.spines[side].set_visible(False)
figure.tight_layout()
for suffix in ('pdf', 'png'):
    figure.savefig(ROOT / f'runs/index_time/index_time_bar.{suffix}', dpi=200, bbox_inches='tight')
print('DONE', flush=True)
1 / 0
