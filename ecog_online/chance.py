"""Chance baselines for the 90-trial tables.

Step-level balanced accuracy has an exact chance level (0.25 for 4 classes, 0.333 for 3).
The user-side measures do not, so they are measured: the CNN's own out-of-fold output is
shifted in time by SHIFTS_S, which keeps its statistics (how often it moves, how confident
it is) but breaks its alignment with the movements, then scored with the same rules and
tuning as the real output. The finger outputs are shifted the same way.

    python -m ecog_online.chance     -> results/online/chance.json
"""

from __future__ import annotations

import json

import numpy as np

from . import allcv
from .common import OUT, Progress, balanced_accuracy, regression_scores
from .features import Config
from .protocol import load, sample_labels

# --- config ---------------------------------------------------------------------------
SOURCE = "cnn_A0.5_ensemble"
SHIFTS_S = [60, 120, 180]


def main():
    rec = load(str(OUT.parents[2] / "ECoG_Handpose.mat"))
    lab, info, _ = sample_labels(rec)
    _, ends = allcv.features(rec, 0, Config())
    d = np.load(allcv.OUT_DIR / f"{SOURCE}.npz")
    steps, P, F, G = d["steps"], d["P"], d["F"], d["G"]
    y = lab[ends[steps]]
    fold_of = np.searchsorted([b for _, b in allcv.blocks(rec)], ends[steps], side="right")
    prog = Progress(len(SHIFTS_S), "chance")
    rows = []
    for sh in SHIFTS_S:
        k = int(sh / 0.1)
        Ps, Fs = np.roll(P, k, axis=0), np.roll(F, k, axis=0)
        r, r2 = regression_scores(G, Fs)
        g = allcv.gated_metrics(Ps, steps, fold_of, ends, info, lab)
        rows.append({"shift_s": sh, "bal_acc": balanced_accuracy(y, Ps.argmax(1), range(4)),
                     "events_strict": allcv.events_xfold(Ps, steps, fold_of, ends, info, lab),
                     "settle": allcv.settle_metrics(Ps, steps, ends, info, lab),
                     "gated_cap_1.0": g["cap_1.0"], "fingers_mean_r": float(r.mean()), "fingers_mean_r2": float(r2.mean())})
        prog.step(f"shift {sh} s")

    def m(f):
        v = [f(x) for x in rows]
        return [float(np.mean(v)), float(np.std(v))]

    summary = {
        "bal_acc_shifted": m(lambda x: x["bal_acc"]),
        "strict": m(lambda x: x["events_strict"]["correct_rate"]),
        "settled": m(lambda x: x["settle"]["settled_rate"]),
        "settle_median_s": m(lambda x: x["settle"]["median_time_to_settle_s"] or np.nan),
        "false_holds_per_min": m(lambda x: x["settle"]["false_holds_per_min"]),
        "gated_first_call": m(lambda x: x["gated_cap_1.0"]["first_call_right"]),
        "gated_final_call": m(lambda x: x["gated_cap_1.0"]["final_call_right"]),
        "gated_switches": m(lambda x: x["gated_cap_1.0"]["mean_switches_per_detected"] or 0.0),
        "gated_false": m(lambda x: x["gated_cap_1.0"]["false_activations"]),
        "fingers_mean_r": m(lambda x: x["fingers_mean_r"]),
        "fingers_mean_r2": m(lambda x: x["fingers_mean_r2"]),
        "exact": {"4class_bal_acc": 0.25, "3class_bal_acc": 1 / 3, "triggered_3class_acc": 1 / 3},
    }
    (OUT / "chance.json").write_text(json.dumps({"source": SOURCE, "shifts_s": SHIFTS_S, "summary": summary,
                                                 "per_shift": rows}, indent=1, default=float) + "\n")
    for k, v in summary.items():
        print(f"{k:22s} {v}")


if __name__ == "__main__":
    main()
