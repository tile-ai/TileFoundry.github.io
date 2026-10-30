"""TileOPs operators at SOTA by the standard of the last nightly, week by week.

`tileops/sota_ops_weekly.csv` comes from `output/tilefoundry-blog/data` in the
TileOPs repository (main @ 2c54caec, nightly of 2026-10-28, collected 2026-10-29).
An op is at SOTA when every benchmark row is at or above 0.99x the fastest
baseline. This script only draws the CSV.

The chart takes the nightly of Oct 28, against all 48 baselines of that night, as
the standard: 205 of its 206 ops are at SOTA. `final_other` and `final_foundry` were
added from the same data's `op_daily_status_all.csv` (last run of each night): each
of the 205 ops counts from the first night it was at SOTA; an op a foundry PR worked
on counts only from its first night at SOTA after that PR, in `final_foundry`. The
four July weeks without a usable nightly are interpolated linearly.

Colours are those of the H200 Prefill chart (figure 14): blue is TileFoundry, orange
the ops that reached SOTA without it. The y axis runs from zero to 205.
"""
import argparse
import csv
import datetime as dt
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
DATA = HERE / 'tileops' / 'sota_ops_weekly.csv'
PREVIEWS = HERE / 'previews'

# Figure 14's colours, read from its PNG.
FOUNDRY, OTHER = '#4C72B0', '#DD8452'
TITLE, NOTE, GRID, SPINE = '#273442', '#5C6775', '#DCE1E7', '#8995A4'
# 12 x 6 in at 200 dpi is 2400 px wide, the width of the other PNGs; point sizes
# match the 1200 canvas's heading / body / label / small (28 / 24 / 20 / 18 units).
SIZE, DPI = (12, 6), 200
HEADING, BODY, LABEL, SMALL = 20.2, 17.3, 14.4, 13


def use_sans():
    """Helvetica, or what fontconfig substitutes for it (Nimbus Sans on Linux)."""
    names = []
    for style in ('Regular', 'Bold'):
        try:
            path = subprocess.run(['fc-match', '-f', '%{file}', f'Helvetica:style={style}'],
                                  capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            continue
        if path:
            font_manager.fontManager.addfont(path)
            names.append(font_manager.FontProperties(fname=path).get_name())
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = names[:1] + ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['svg.fonttype'] = 'none'


def draw():
    """Ops at SOTA by Oct 28's standard, each from the week it got there."""
    with open(DATA, newline='') as f:
        rows = list(csv.DictReader(f))
    weeks = [dt.date.fromisoformat(r['week']) for r in rows]
    foundry = [int(r['final_foundry']) for r in rows]
    other = [int(r['final_other']) for r in rows]

    fig, ax = plt.subplots(figsize=SIZE, dpi=DPI)
    fig.subplots_adjust(left=.08, right=.86, top=.75, bottom=.17)
    width = 4.4   # days, as wide as figure 14's bars are against their gaps
    ax.bar(weeks, foundry, width, color=FOUNDRY, zorder=4)
    ax.bar(weeks, other, width, bottom=foundry, color=OTHER, zorder=4)

    ax.set_ylim(0, 205)
    ax.set_yticks(range(0, 201, 50))
    ax.set_xlim(dt.date(2026, 6, 11), dt.date(2026, 11, 2))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(7, 8, 9, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.grid(axis='y', color=GRID, linewidth=1, zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(1.2)
    ax.tick_params(colors=SPINE, labelcolor=TITLE, labelsize=LABEL, length=5)
    ax.set_ylabel('Operators at SOTA', fontsize=BODY, color=TITLE)

    # Between the last night before the first PR and the first night after it.
    first = dt.datetime(2026, 8, 15, 12)
    for t, f, o in zip(weeks, foundry, other):
        if t.isoformat() in ('2026-08-12', '2026-10-28'):
            ax.text(t, f + o + 3, str(f + o), ha='center', va='bottom', fontsize=BODY, fontweight='bold',
                    color=FOUNDRY if f else TITLE, zorder=5)
    ax.axvline(first, color=TITLE, linewidth=1.4, zorder=4)
    ax.text(first + dt.timedelta(days=1.5), 196, 'first TileFoundry PR, Aug 14', fontsize=LABEL,
            style='italic', color=TITLE, va='center')
    ax.text(weeks[-1] + dt.timedelta(days=4), foundry[-1] / 2, f'{foundry[-1]} with\nTileFoundry',
            fontsize=LABEL, fontweight='bold', color=FOUNDRY, va='center')

    fig.text(.08, .93, f'Operators at SOTA on Oct 28: {foundry[-1] + other[-1]}, '
             f'{foundry[-1]} of them reached it with TileFoundry', fontsize=HEADING, fontweight='bold', color=TITLE)
    fig.legend(handles=[Patch(color=FOUNDRY, label='reached SOTA after a TileFoundry PR'),
                        Patch(color=OTHER, label='reached SOTA without one')],
               loc='upper left', bbox_to_anchor=(.07, .89), ncol=2, frameon=False, fontsize=LABEL,
               handlelength=2.2, handleheight=1.1, columnspacing=2.4)
    fig.text(.08, .015, 'H200 nightly. At SOTA: every benchmark row ≥ 0.99× the fastest baseline measured that '
             'night (48 on Oct 28).\nEach op counts from the week it first reached SOTA; one a TileFoundry PR worked on, '
             'from its first week at SOTA after that PR.', fontsize=SMALL, color=NOTE, linespacing=1.5)
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=draw.__doc__)
    parser.add_argument('--out', type=Path, default=PREVIEWS, help='output directory')
    parser.add_argument('--svg-only', action='store_true', help='skip the PNG export')
    args = parser.parse_args()
    use_sans()
    fig = draw()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / 'tileops-sota-ops.svg')
    if not args.svg_only:
        fig.savefig(args.out / 'tileops-sota-ops.png', dpi=DPI)
    print(args.out / 'tileops-sota-ops.svg')
