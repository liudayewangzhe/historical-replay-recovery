"""Overview schematic of the recovery problem, the compared methods and the two-stage evaluation.

Drawn at the final width of the manuscript text (5.4 in), in inch coordinates, so that the font
sizes below are the printed sizes. Writes manuscript/figures/fig_overview.pdf (vector; text stays live).
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
HEAD, TITLE, BODY = 8.5, 7.5, 7.0     # printed font sizes (pt)
LINE = .135                           # line pitch (in)
PAD = .06

plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})


def height(lines):
    return .25 + LINE * len(lines) + .04


def box(ax, x, y, w, lines, title, face=PALE, edge=GREY, bold=False):
    """Draw a box whose top edge is at y; return its height."""
    h = height(lines)
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle='round,pad=0,rounding_size=0.04',
                                fc=face, ec=edge, lw=.7))
    ax.text(x + PAD, y - .05, title, ha='left', va='top', fontsize=TITLE,
            fontweight='bold' if bold else 'normal', color=INK)
    for i, line in enumerate(lines):
        ax.text(x + PAD, y - .24 - LINE * i, line, ha='left', va='top', fontsize=BODY, color=INK)
    return h


def arrow(ax, a, b, color=INK, lw=.8, style='-|>'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=6, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


def main():
    W, H = 5.4, 3.62
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    top = H - .05
    xl, wl = .02, 1.45        # left column
    xm, wm = 1.66, 1.78       # middle column
    xr, wr = 3.63, 1.75       # right column

    # Left: attacked training and what the server stores.
    ax.text(xl, top, 'Attacked training', fontweight='bold', va='top', fontsize=HEAD)
    y = top - .62
    xs = [xl + .12 + .16 * k for k in range(9)]
    ax.add_patch(FancyBboxPatch((xs[0], y - .09), xs[-1] - xs[0], .18, boxstyle='round,pad=0,rounding_size=0.03',
                                fc=RED_PALE, ec='none'))
    for k in range(len(xs) - 1):
        arrow(ax, (xs[k] + .02, y), (xs[k + 1] - .02, y), color=RED, lw=.7)
    ax.plot(xs[0], y, 'o', ms=4.5, mfc='white', mec=INK, mew=.8)
    ax.plot(xs[-1], y, 'o', ms=4.5, mfc=RED, mec=INK, mew=.8)
    ax.text(xs[0] - .06, y + .15, 'onset $w_a$', ha='left', fontsize=BODY)
    ax.text(xs[-1] + .06, y + .15, 'detection $w_{a+\\Delta}$', ha='right', fontsize=BODY)
    ax.text((xs[0] + xs[-1]) / 2, y - .15, '$\\Delta$ poisoned rounds', ha='center', va='top', fontsize=BODY,
            color=RED)
    store = ['global models $w_{a+j}$', 'retained uploads $u_{m,a+j}$', 'round means $\\bar u_j$',
             'flagged set $\\mathcal{M}$']
    sy = top - 1.25
    hs = box(ax, xl, sy, wl, store, 'Server storage')
    arrow(ax, (xl + wl / 2, y - .32), (xl + wl / 2, sy + .02), color=GREY, lw=.7)

    # Middle: recovery options.
    ax.text(xm, top, 'Recovery at detection', fontweight='bold', va='top', fontsize=HEAD)
    options = [
        ('Filter or server fine-tuning', ['keep $w_{a+\\Delta}$, drop $\\mathcal{M}$', '0 client calls'], PALE, GREY),
        ('Rollback', ['restart from $w_a$', '0 client calls'], PALE, GREY),
        ('Sampled / paired replay', ['$S_s$, $P_s$: $s$ clients per round',
                                     '$P_s$ adds stored round means',
                                     '$(\\Delta-1)\\,s$ client calls'], ACCENT_PALE, ACCENT),
        ('History-based recovery', ['FedRecover/FedEraser/Crab-style', 'predict or calibrate from storage'],
         ACCENT_PALE, ACCENT),
        ('Exact replay', ['all retained clients recompute', '$(\\Delta-1)\\,|\\mathcal{R}|$ client calls'], PALE, GREY),
    ]
    gap = .06
    yb = top - .27
    centres = []
    for title, lines, face, edge in options:
        h = box(ax, xm, yb, wm, lines, title, face=face, edge=edge, bold=edge == ACCENT)
        centres.append(yb - h / 2)
        yb -= h + gap
    bx = xm - .1
    feed = sy - hs / 2
    ax.plot([xl + wl, bx], [feed, feed], color=INK, lw=.8)
    ax.plot([bx, bx], [centres[0], centres[-1]], color=INK, lw=.8)
    for c in centres:
        arrow(ax, (bx, c), (xm, c))

    # Right: two-stage evaluation.
    ax.text(xr, top, 'Two-stage evaluation', fontweight='bold', va='top', fontsize=HEAD)
    mx = xr - .1
    ax.plot([mx, mx], [centres[0], centres[-1]], color=INK, lw=.8)
    for c in centres:
        ax.plot([xm + wm, mx], [c, c], color=INK, lw=.8)
    stage1 = ['accuracy and attack success', 'disagreement with exact replay', 'calls, rounds, storage']
    stage2 = ['accuracy and attack success', 'three matchings (Sec. 3)', 'paired tests over seeds']
    s1 = top - .27
    h1 = box(ax, xr, s1, wr, stage1, 'Stage 1: recovered model', face=ACCENT_PALE, edge=ACCENT, bold=True)
    arrow(ax, (mx, s1 - h1 / 2), (xr, s1 - h1 / 2))
    h2 = height(stage2)
    s2 = yb + gap + h2  # bottom aligned with the last recovery box
    arrow(ax, (xr + .25, s1 - h1), (xr + .25, s2))
    ax.text(xr + .33, (s1 - h1 + s2) / 2, '$C$ honest rounds,\nsame for every\nmethod', fontsize=BODY,
            va='center')
    box(ax, xr, s2, wr, stage2, 'Stage 2: after continuation', face=ACCENT_PALE, edge=ACCENT, bold=True)
    fig.savefig(ROOT / 'manuscript' / 'figures' / 'fig_overview.pdf')


if __name__ == '__main__':
    main()
