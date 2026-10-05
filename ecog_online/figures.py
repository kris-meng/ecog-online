"""Figures for the online work, drawn only from results/online/*.json.

    python -m ecog_online.figures     -> results/online/tradeoff.svg, open_vs_rest.svg
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import OUT

# reference palette, light mode: the first three categorical slots validate for every pair
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#bdbcb6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def frontier(x, y, better):
    """Best y reachable at or below each delay x (running best over sorted x)."""
    o = np.argsort(x)
    xs, ys = np.asarray(x)[o], np.asarray(y)[o]
    best, out_x, out_y = None, [], []
    for a, b in zip(xs, ys):
        if best is None or better(b, best):
            best = b
            out_x.append(a)
            out_y.append(b)
    return out_x, out_y


def tradeoff(r):
    rows = [w for w in r["sweep"]["rows"] if w["latency_s"] is not None]
    lat = np.array([w["latency_s"] for w in rows])
    cor = np.array([w["correct_rate"] for w in rows])
    fp = np.array([w["false_per_min"] for w in rows])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.scatter(lat, cor, s=10, color=MUTED, alpha=0.6, linewidths=0, label="every setting")
    for budget, col in ((1.0, BLUE), (2.0, ORANGE), (4.0, AQUA)):
        m = fp <= budget
        fx, fy = frontier(lat[m], cor[m], lambda b, best: b > best)
        a1.step(fx, fy, where="post", color=col, lw=2, label=f"best with ≤ {budget:g} false / min")
    a1.set(xlabel="median delay after movement onset, fist & peace (s)",
           ylabel="gestures detected with the right label", ylim=(0, 1),
           title="Detection rate against delay")
    a1.legend(loc="lower right", fontsize=9)
    a2.scatter(lat, fp, s=10, color=MUTED, alpha=0.6, linewidths=0, label="every setting")
    for floor, col in ((0.5, BLUE), (0.6, ORANGE), (0.7, AQUA)):
        m = cor >= floor
        fx, fy = frontier(lat[m], fp[m], lambda b, best: b < best)
        a2.step(fx, fy, where="post", color=col, lw=2, label=f"fewest with ≥ {floor:.0%} detected")
    a2.set(xlabel="median delay after movement onset, fist & peace (s)",
           ylabel="false detections per minute of rest", title="False alarms against delay")
    a2.legend(loc="upper right", fontsize=9)
    fig.suptitle(f"Event stage swept on pooled out-of-fold output, {r['sweep']['model'].replace('linear_', '')} "
                 f"({r['sweep']['n_settings']} settings, {r['sweep']['rest_min']:.1f} min of rest)",
                 color=INK2, fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "tradeoff.svg")
    plt.close(fig)


def open_vs_rest(r):
    o = r["open_vs_rest"]
    t = np.array(o["t_s"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax in (a1, a2):
        ax.axvspan(0, 2.0, color=GRID, alpha=0.5, lw=0)
        ax.axvline(0, color=INK2, lw=0.8)
    a1.text(0.05, 0.98, "cue on screen", transform=a1.get_xaxis_transform(), color=INK2, fontsize=8, va="top")
    for name, col in (("open", BLUE), ("fist", ORANGE), ("peace", AQUA)):
        m, se = np.array(o["hg_grid_mean"][name]["mean"]), np.array(o["hg_grid_mean"][name]["sem"])
        a1.fill_between(t, m - se, m + se, color=col, alpha=0.18, lw=0)
        a1.plot(t, m, color=col, lw=2, label=f"{name} (n={o['n_trials'][name]})")
    a1.set(xlabel="time from cue (s)", ylabel="high-gamma, grid mean (z)",
           title="High-gamma power, cue-locked")
    a1.legend(loc="upper right", fontsize=9)
    for name, col in (("open", BLUE), ("fist", ORANGE), ("peace", AQUA)):
        a2.plot(t, o["decoding"][name]["acc"], color=col, lw=2, label=f"{name} vs its own rest")
    a2.plot(t, o["decoding"]["open"]["null_95th"], color=BLUE, lw=1, ls="--",
            label="open: 95th percentile of shuffled labels")
    a2.axhline(0.5, color=INK2, lw=0.8)
    a2.set(xlabel="time from cue (s)", ylabel="decoding accuracy (5-fold, by trial)", ylim=(0.3, 1.02),
           title=f"Gesture vs rest at {o['rest_at_s']} s, at each moment")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "open_vs_rest.svg")
    plt.close(fig)


def main():
    r = json.loads((OUT / "phase2b.json").read_text())
    tradeoff(r)
    open_vs_rest(r)
    print(f"wrote {OUT / 'tradeoff.svg'}, {OUT / 'open_vs_rest.svg'}")


if __name__ == "__main__":
    main()
