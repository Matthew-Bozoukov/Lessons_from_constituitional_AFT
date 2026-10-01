# ABOUTME: Plot verified full-Lite arm scores from a saved, revision-pinned JSON input.
# ABOUTME: Writes standalone PNG and SVG charts without accessing inference resources.
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

parser = argparse.ArgumentParser()
parser.add_argument('data', type=Path)
args = parser.parse_args()
arms = json.loads(args.data.read_text())
assert len(arms) == 2 and all(a['total'] == 300 for a in arms)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'svg.fonttype': 'none', 'axes.titleweight': 'bold'})
fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=180)
fig.subplots_adjust(left=.13, right=.95, bottom=.25, top=.82)
fig.suptitle('SWE-bench Lite', x=.13, y=.96, ha='left', fontsize=22, weight='bold')
fig.text(.13, .89, 'Seed 0 · mini v5 · 300 tasks per arm', color='#52616b', fontsize=12)
bars = ax.bar([0, 1], [a['rate'] for a in arms], width=.5,
              color=['#315c80', '#278d83'], zorder=3)
ax.set_ylim(0, 100)
ax.set_xlim(-.65, 1.65)
ax.set_yticks(range(0, 101, 20))
ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
ax.set_ylabel('Tasks resolved')
ax.set_xticks([0, 1], ['Control\n(no DA)', 'DA-15'])
ax.tick_params(axis='both', length=0, pad=9)
ax.set_axisbelow(True)
ax.grid(axis='y', color='#dde3e8', linewidth=.8)
for side in ['top', 'right', 'left']:
    ax.spines[side].set_visible(False)
ax.spines['bottom'].set_color('#a5b1ba')
for bar, arm in zip(bars, arms):
    x = bar.get_x() + bar.get_width()/2
    ax.text(x, arm['rate']+3, f"{arm['rate']:.2f}%", ha='center', va='bottom',
            fontsize=19, weight='bold', color='#172b3a')
    ax.text(x, arm['rate']-7, f"{arm['resolved']} / {arm['total']}",
            ha='center', va='center', color='white', fontsize=12)
fig.text(.13, .12, '+6 tasks for DA-15  ·  +2.00 percentage points', fontsize=12, weight='bold')
fig.text(.13, .064, 'One run per checkpoint; descriptive comparison, not a causal estimate.', fontsize=9, color='#52616b')
fig.text(.13, .027, 'Local HTTPBin fixture; two Requests baseline passes included in each arm.', fontsize=9, color='#52616b')
for extension in ['png', 'svg']:
    out = args.data.with_suffix('.'+extension)
    fig.savefig(out, facecolor='white', metadata={'Description': json.dumps(arms)} if extension == 'svg' else None)
    print(out.resolve())
