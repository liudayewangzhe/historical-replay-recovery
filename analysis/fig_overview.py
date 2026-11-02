"""Overview schematic of the recovery problem, the compared methods and the two-stage evaluation.

Writes manuscript/figures/fig_overview.pdf (vector; text stays live).
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
INK, GREY, PALE = '#1a1a1a', '#8c8c8c', '#f2f2f2'
RED, RED_PALE = '#b2182b', '#f9e0e0'
ACCENT, ACCENT_PALE = '#0b5394', '#dde9f6'

plt.rcParams.update({'font.size': 7, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans',
                     'mathtext.fontset': 'dejavusans'})


def box(ax, x, y, w, h, title, lines=(), face=PALE, edge=GREY, bold=False, size=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.06',
                                fc=face, ec=edge, lw=.8))
    ax.text(x + .08, y + h - .09, title, ha='left', va='top', fontsize=size,
            fontweight='bold' if bold else 'normal', color=INK)
    for i, line in enumerate(lines):
        ax.text(x + .08, y + h - .31 - .2 * i, line, ha='left', va='top', fontsize=6.2, color=INK)


def height(lines):
    return .34 + .2 * len(lines)


def arrow(ax, a, b, color=INK, lw=.9, style='-|>', ls='-'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=7, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def main():
    T = 4.75  # top of the drawing area
    fig, ax = plt.subplots(figsize=(7.1, 3.7))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(-.15, T + .1)
    ax.axis('off')

    # Left: attacked training and what the server stores.
    ax.text(.1, T, 'Attacked training', fontweight='bold', va='top', fontsize=7.5)
    y = T - .95
    ax.plot([.25, 3.0], [y, y], color=GREY, lw=.8)
    xs = [.55 + .3 * k for k in range(9)]
    ax.add_patch(FancyBboxPatch((xs[0], y - .17), xs[-1] - xs[0], .34, boxstyle='round,pad=0,rounding_size=0.05',
                                fc=RED_PALE, ec='none'))
    for k in range(len(xs) - 1):
        arrow(ax, (xs[k] + .04, y), (xs[k + 1] - .04, y), color=RED, lw=.8)
    ax.plot(xs[0], y, 'o', ms=5.5, mfc='white', mec=INK, mew=.9)
    ax.plot(xs[-1], y, 'o', ms=5.5, mfc=RED, mec=INK, mew=.9)
    ax.text(xs[0], y + .26, 'onset $w_a$', ha='center', fontsize=6.3)
    ax.text(xs[-1] + .1, y + .26, 'detection $w_{a+\\Delta}$', ha='right', fontsize=6.3)
    ax.text((xs[0] + xs[-1]) / 2, y - .32, '$\\Delta$ rounds with poisoned uploads', ha='center', va='top',
            fontsize=6.2, color=RED)
    store = ['global models $w_{a+j}$, $j<\\Delta$', 'retained uploads $u_{m,a+j}$', 'their round means $\\bar u_j$',
             'flagged set $\\mathcal{M}$ at detection']
    sy = 1.0
    box(ax, .25, sy, 2.75, height(store), 'Server storage in the window', store)
    arrow(ax, (1.62, y - .55), (1.62, sy + height(store) + .02), color=GREY, lw=.7)

    # Middle: recovery options.
    ax.text(3.45, T, 'Recovery at detection', fontweight='bold', va='top', fontsize=7.5)
    options = [
        ('Filter, server fine-tuning', ['keep $w_{a+\\Delta}$, drop $\\mathcal{M}$', 'optionally fine-tune on server data',
                                        '0 client calls'], PALE, GREY),
        ('Rollback', ['restart from $w_a$', '0 client calls'], PALE, GREY),
        ('Sampled replay $S_s$ / paired $P_s$',
         ['$s$ clients recompute per round', '$P_s$ adds $\\bar u_j$ minus their stored mean',
          '$(\\Delta-1)\\,s$ client calls'], ACCENT_PALE, ACCENT),
        ('History-based recovery', ['FedRecover-, FedEraser-, Crab-style', 'predict or calibrate from storage',
                                    'calls and rounds vary'], ACCENT_PALE, ACCENT),
        ('Exact replay', ['all retained clients recompute', '$(\\Delta-1)\\,|\\mathcal{R}|$ client calls'], PALE, GREY),
    ]
    heights = [height(o[1]) for o in options]
    top, gap, x0, w = T - .2, .08, 3.45, 3.05
    branch_x = 3.2
    feed = sy + height(store) / 2
    arrow(ax, (3.0, feed), (branch_x, feed), color=INK, lw=.9, style='-')
    bottom = top - sum(heights) - (len(options) - 1) * gap
    ax.plot([branch_x, branch_x], [top - heights[0] / 2, bottom + heights[-1] / 2], color=INK, lw=.9)
    yc = []
    for (title, lines, face, edge), h in zip(options, heights):
        box(ax, x0, top - h, w, h, title, lines, face=face, edge=edge, bold=edge == ACCENT)
        yc.append(top - h / 2)
        arrow(ax, (branch_x, top - h / 2), (x0, top - h / 2))
        top -= h + gap

    # Right: two-stage evaluation and the three matchings.
    ax.text(7.0, T, 'Two-stage evaluation', fontweight='bold', va='top', fontsize=7.5)
    merge = 6.78
    ax.plot([merge, merge], [yc[0], yc[-1]], color=INK, lw=.9)
    for c in yc:
        ax.plot([x0 + w, merge], [c, c], color=INK, lw=.9)
    stage1 = ['clean accuracy and attack success', 'disagreement with exact replay', 'client calls, rounds, storage']
    stage2 = ['accuracy and attack success', 'training-, latency- and call-matched', 'paired tests over held-out seeds']
    h1, h2 = height(stage1), height(stage2)
    s1 = T - .2
    box(ax, 7.0, s1 - h1, 3.1, h1, 'Stage 1: recovered model', stage1, face=ACCENT_PALE, edge=ACCENT, bold=True)
    arrow(ax, (merge, s1 - h1 / 2), (7.0, s1 - h1 / 2))
    s2 = sy
    arrow(ax, (8.55, s1 - h1), (8.55, s2 + h2))
    ax.text(8.62, (s1 - h1 + s2 + h2) / 2, '$C$ honest rounds,\nidentical for every method;\nzero-round methods keep\ntraining while others replay',
            fontsize=6.2, va='center')
    box(ax, 7.0, s2, 3.1, h2, 'Stage 2: after continuation', stage2, face=ACCENT_PALE, edge=ACCENT, bold=True)
    fig.savefig(ROOT / 'manuscript' / 'figures' / 'fig_overview.pdf', bbox_inches='tight', pad_inches=.02)


if __name__ == '__main__':
    main()
