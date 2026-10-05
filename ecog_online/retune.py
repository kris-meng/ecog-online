"""Re-tune the logistic regression's event stage the way the CNN's is tuned.

Per fold: fit the same model on the training steps minus the last 20 % block (purged),
predict that block, choose smoothing / threshold / N under the caps (<= 2 false per
minute, <= 0.8 s median delay; closest setting if none qualifies, and counted), then apply
the choice to the outer test predictions saved by linear.py. Sample-level accuracies are
unchanged; only the event metrics are recomputed, so they compare like for like with the CNN.

    python -m ecog_online.retune     -> results/online/retune_linear.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import (DELAY_CAP, FP_CAP, OUT, Progress, ShrinkLDA, events_on, fold_features, make_logreg,
                     summarise_events, tune_capped, val_split, windows)
from .linear import FEATURES
from .protocol import fold_masks, load, sample_labels

# --- config ---------------------------------------------------------------------------
RUNS = {                       # saved run -> (window steps, model, front-end preset)
    "linear_logreg_glove": (10, "logreg", "default"),
    "linear_logreg_glove_w15": (15, "logreg", "default"),
    "linear_lda_glove": (10, "lda", "default"),
    "linear_lda_glove_w15": (15, "lda", "default"),
    "linear_logreg_glove_lmp_w15": (15, "logreg", "lmp"),
}
FOLDS = [0, 1, 2, 3, 4]
LOGREG_C = 0.01


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "ECoG_Handpose.mat"))
    ap.add_argument("--out", default=str(OUT / "retune_linear.json"))
    args = ap.parse_args()
    rec = load(args.data)
    lab, info, _ = sample_labels(rec)
    prog = Progress(len(RUNS) * len(FOLDS), "retune_linear")
    out = {"config": {"fp_cap": FP_CAP, "delay_cap_s": DELAY_CAP, "val_frac": 0.2, "logreg_C": LOGREG_C}}
    for name, (window, model, preset) in RUNS.items():
        oof = np.load(OUT / "oof" / f"{name}.npz")
        pos = {s: i for i, s in enumerate(oof["steps"])}
        folds, parts_all = [], []
        for fold in FOLDS:
            feats, ends = fold_features(rec, fold, FEATURES[preset])
            X, valid = windows(feats, window)
            tr, te = fold_masks(ends, rec, fold)
            tr, te = np.flatnonzero(tr & valid), np.flatnonzero(te & valid)
            fit, val = val_split(tr, ends)
            m = (make_logreg(LOGREG_C) if model == "logreg" else ShrinkLDA()).fit(X[fit], lab[ends[fit]])
            ev = tune_capped(m.predict_proba(X[val]), val, ends, info, lab)
            P = oof["P"][[pos[s] for s in te]]
            _, parts = events_on(P, te, ends, info, lab, ev["smoother"], ev["param"], ev["thr"], ev["n"])
            s = summarise_events(parts)
            folds.append({"fold": fold, "tuned": ev, "events": s})
            parts_all += parts
            prog.step(f"{name} fold {fold}: {ev['smoother']} {ev['param']} thr {ev['thr']} n {ev['n']} "
                      f"feasible {ev['feasible']} -> test correct {s['correct_rate']:.2f}, "
                      f"false/min {s['false_per_min']:.2f}, delay {s['latency_s_median_fist_peace_glove_onset']}")
        pooled = summarise_events(parts_all)
        out[name] = {"window_steps": window, "model": model, "features": preset, "pooled_events": pooled,
                     "n_folds_feasible": int(sum(f["tuned"]["feasible"] for f in folds)),
                     "per_fold_correct_mean_sd": [float(np.mean([f["events"]["correct_rate"] for f in folds])),
                                                  float(np.std([f["events"]["correct_rate"] for f in folds]))],
                     "folds": folds}
        print(f"{name}: correct {pooled['correct_rate']:.3f}, false/min {pooled['false_per_min']:.2f} "
              f"({pooled['n_false']} in {pooled['rest_min']:.1f} min), delay "
              f"{pooled['latency_s_median_fist_peace_glove_onset']}, cap met in "
              f"{out[name]['n_folds_feasible']}/{len(FOLDS)} folds", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
