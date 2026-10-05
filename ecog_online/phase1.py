"""Phase 1 report: the protocol and the front end on the full recording, before any model.

Checks the fold layout and class balance per fold, the label construction, the cost of
fitting the front end on a full training fold, and that the features respond where they
should: movement-locked high-gamma rises for fist and peace, measured on test steps of a
fold whose front end never saw them.

    python -m ecog_online.phase1 --data ../ECoG_Handpose.mat   -> results/online/phase1.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .features import Config, fit
from .protocol import CLASSES, FS, PURGE_S, fold_blocks, fold_masks, load, sample_labels

ROOT = Path(__file__).resolve().parents[1]
FOLD = 0
LOCK_S = (-1.0, 2.0)          # window around movement onset for the response check


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT.parent / "ECoG_Handpose.mat"))
    ap.add_argument("--out", default=str(ROOT / "results" / "online" / "phase1.json"))
    args = ap.parse_args()
    t0 = time.time()

    rec = load(args.data)
    lab, info, open_rule = sample_labels(rec)
    r = {"purge_s": PURGE_S, "open_hand_rule": open_rule,
         "label_sources": {s: sum(i["source"] == s for i in info) for s in ("glove", "cue")},
         "gesture_duration_s_quantiles_5_50_95": {
             CLASSES[c]: np.percentile([(i["end"] - i["onset"]) / FS for i in info if i["class"] == c],
                                       [5, 50, 95]).round(2).tolist() for c in (1, 2, 3)}}

    blocks = fold_blocks(rec)
    steps = np.arange(int(2.0 * FS) + 119, rec.ecog.shape[1], 120)     # same grid as the front end
    r["folds"] = []
    for k, (a, b) in enumerate(blocks):
        tr, te = fold_masks(steps, rec, k)
        r["folds"].append({
            "block_s": [round(a / FS, 2), round(b / FS, 2)],
            "test_steps_per_class": {CLASSES[c]: int((lab[steps[te]] == c).sum()) for c in range(4)},
            "train_steps": int(tr.sum()), "purged_steps": int((~tr & ~te & (steps < rec.cut)).sum()),
        })
    r["final_test_steps_per_class"] = {CLASSES[c]: int((lab[steps[steps >= rec.cut]] == c).sum()) for c in range(4)}
    print(json.dumps({k: v for k, v in r.items()}, indent=1))

    # fit the front end on fold 0's training samples, full recording
    tr_samples = np.zeros(rec.ecog.shape[1], bool)
    tr_steps, _ = fold_masks(np.arange(rec.ecog.shape[1]), rec, FOLD)
    tr_samples[tr_steps] = True
    t1 = time.time()
    p, feats, ends = fit(rec.ecog, tr_samples, Config())
    r["fit_full_fold_s"] = round(time.time() - t1, 1)
    r["n_steps"] = int(len(ends))
    r["bad_channels_fold0"] = [b + 1 for b in p.bad]
    _, te = fold_masks(ends, rec, FOLD)
    print(f"front end fitted on fold {FOLD} training samples in {r['fit_full_fold_s']} s, "
          f"{len(ends)} steps, bad {r['bad_channels_fold0']}")

    # response check on fold-0 test steps only: mean z-scored power over the grid,
    # locked to each movement onset inside the test block
    a, b = blocks[FOLD]
    lo, hi = int(LOCK_S[0] / 0.1), int(LOCK_S[1] / 0.1)
    resp = {}
    for c in (1, 2, 3):
        rows = []
        for i in info:
            if i["class"] == c and a <= i["onset"] < b:
                j = int(np.searchsorted(ends, i["onset"]))
                if j + lo >= 0 and j + hi <= len(ends):
                    rows.append(feats[j + lo: j + hi].mean(axis=(2, 3)))    # steps x bands
        m = np.mean(rows, axis=0)
        resp[CLASSES[c]] = {"n_trials": len(rows),
                            "beta_13_30": m[:, 0].round(3).tolist(),
                            "high_gamma_mean_of_7": m[:, 1:].mean(1).round(3).tolist()}
        pre, post = m[: -lo, 1:].mean(), m[-lo + 2: -lo + 10, 1:].mean()
        print(f"{CLASSES[c]:5s} ({len(rows)} test trials): high-gamma z pre {pre:+.2f} -> "
              f"0.2-1.0 s after onset {post:+.2f}; beta pre {m[:-lo, 0].mean():+.2f} -> "
              f"{m[-lo + 2: -lo + 10, 0].mean():+.2f}")
    r["onset_locked_response_fold0_test"] = {"t_s": (np.arange(lo, hi) * 0.1).round(1).tolist(), **resp}

    r["runtime_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(r, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
