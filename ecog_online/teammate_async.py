"""The submitted offline pipeline (ecog/final.py headline), run as a sliding decoder.

Their features are cue-locked: log power in six 30 Hz high-gamma bands over 0.2-0.7,
0.5-1.0 and 1.0-2.0 s after the cue, into shrinkage LDA. To slide it, every 100 ms step
is treated as "cue + 2.0 s": the three windows become [now-1.8, now-1.3], [now-1.5, now-1.0]
and [now-1.0, now], each the mean of the causal front end's per-step log power with their
six bands. Three measurements on the same contiguous folds:

  a  triggered   one decision per trial at cue + 2.0 s, 3 classes: their own setting with
                 causal filters instead of filtfilt
  b  as trained  that 3-class model applied at every step; it has no rest class
  c  async       the same features and LDA retrained on every step with a rest class, event
                 stage tuned like the other online decoders (last 20 % block, capped)

The history reaches 1.8 s back, so the purge gap is 2.0 s here.

    python -m ecog_online.teammate_async     -> results/online/teammate_async.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import (OUT, Progress, ShrinkLDA, balanced_accuracy, confusion, events_on,
                     fold_features, merged, near_transition, summarise_events, tune_capped,
                     val_split)
from .features import Config
from .protocol import FS, N_SEARCH_TRIALS, fold_masks, load, sample_labels

# --- config ---------------------------------------------------------------------------
BANDS = ((60, 90), (90, 120), (120, 150), (150, 180), (180, 210), (210, 240))   # their HIGH_GAMMA_6
CFG = Config(bands=BANDS)
WINDOWS_S = [(0.2, 0.7), (0.5, 1.0), (1.0, 2.0)]                                # after the cue
DECISION_S = 2.0                     # a sliding decision at t stands for cue = t - 2.0 s
PURGE_S = 2.0
FOLDS = [0, 1, 2, 3, 4]


def their_features(feats):
    """(T, 6, 10, 6) per-step log power -> (T, 3*6*60): their three windows ending at each step."""
    T = len(feats)
    f = feats.reshape(T, -1).astype(np.float64)
    cs = np.vstack([np.zeros((1, f.shape[1])), np.cumsum(f, axis=0)])
    out = []
    for a, b in WINDOWS_S:
        # steps whose 100 ms span lies in [cue + a, cue + b) with cue = now - 2.0 s
        lo = int(round((a - DECISION_S) / 0.1)) + 1          # relative index of the first step
        hi = int(round((b - DECISION_S) / 0.1))              # relative index of the last step
        i = np.arange(T)
        s, e = np.clip(i + lo, 0, T), np.clip(i + hi + 1, 0, T)
        out.append((cs[e] - cs[s]) / np.maximum(e - s, 1)[:, None])
    valid = np.arange(T) >= int(round(DECISION_S / 0.1)) - 2
    return np.concatenate(out, axis=1), valid


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "ECoG_Handpose.mat"))
    ap.add_argument("--out", default=str(OUT / "teammate_async.json"))
    args = ap.parse_args()
    rec = load(args.data)
    lab, info, _ = sample_labels(rec)
    prog = Progress(len(FOLDS) * 4, "teammate_async")
    A = {"correct": 0, "n": 0, "per_fold": []}
    oof = {"steps": [], "Pb": [], "Pc": [], "Sc": [], "y": []}
    parts_all, tuned = [], []
    for fold in FOLDS:
        feats, ends = fold_features(rec, fold, CFG)
        prog.step(f"front end fold {fold}")
        X, valid = their_features(feats)
        tr, te = fold_masks(ends, rec, fold, purge_s=PURGE_S)
        tr, te = np.flatnonzero(tr & valid), np.flatnonzero(te & valid)
        y = lab[ends]

        # (a) triggered: one sample per development trial at cue + 2.0 s
        cue_step = np.array([int(np.searchsorted(ends, s + int(DECISION_S * FS))) for _, s, _ in rec.cues[:N_SEARCH_TRIALS]])
        cls = np.array([c for c, _, _ in rec.cues[:N_SEARCH_TRIALS]])
        in_tr, in_te = np.isin(cue_step, tr), np.isin(cue_step, te)
        m3 = ShrinkLDA().fit(X[cue_step[in_tr]], cls[in_tr])
        pa = m3.classes_[m3.decision_function(X[cue_step[in_te]]).argmax(1)]
        A["correct"] += int((pa == cls[in_te]).sum())
        A["n"] += int(in_te.sum())
        A["per_fold"].append(float(np.mean(pa == cls[in_te])))
        prog.step(f"fold {fold} triggered: {np.mean(pa == cls[in_te]):.3f} on {in_te.sum()} trials")

        # (b) the same 3-class model at every test step
        Pb = np.zeros((len(te), 4))
        Pb[:, 1:] = m3.predict_proba(X[te])
        prog.step(f"fold {fold} as-trained sliding")

        # (c) retrained on every step with rest; event stage tuned on the last 20 % block
        fit, val = val_split(tr, ends, purge_s=PURGE_S)
        ev = tune_capped(ShrinkLDA().fit(X[fit], y[fit]).predict_proba(X[val]), val, ends, info, lab)
        Pc = ShrinkLDA().fit(X[tr], y[tr]).predict_proba(X[te])
        Sc, parts = events_on(Pc, te, ends, info, lab, ev["smoother"], ev["param"], ev["thr"], ev["n"])
        parts_all += parts
        tuned.append(ev)
        prog.step(f"fold {fold} async: bal-acc {balanced_accuracy(y[te], Sc.argmax(1), range(4)):.3f}")
        for k, v in (("steps", te), ("Pb", Pb), ("Pc", Pc), ("Sc", Sc), ("y", y[te])):
            oof[k].append(v)

    o = {k: np.concatenate(v) for k, v in oof.items()}
    near = near_transition(lab, ends)[o["steps"]]
    yb, pb, pc = o["y"], o["Pb"].argmax(1), o["Sc"].argmax(1)

    def block(p):
        return {"bal_acc": balanced_accuracy(yb, p, range(4)),
                "bal_acc_no_transitions": balanced_accuracy(yb[~near], p[~near], range(4)),
                "bal_acc_merged_3class": balanced_accuracy(merged(yb), merged(p), range(3)),
                "confusion": confusion(yb, p).tolist()}

    res = {"config": {"bands": BANDS, "windows_after_cue_s": WINDOWS_S, "decision_s_after_cue": DECISION_S,
                      "purge_s": PURGE_S, "classifier": "shrinkage LDA (Ledoit-Wolf)"},
           "a_triggered_3class": {"accuracy": A["correct"] / A["n"], "n_trials": A["n"],
                                  "per_fold_mean_sd": [float(np.mean(A["per_fold"])), float(np.std(A["per_fold"]))]},
           "b_as_trained_sliding": block(pb),
           "c_async_with_rest": {**block(pc), "events": summarise_events(parts_all),
                                 "n_folds_cap_met": int(sum(t["feasible"] for t in tuned)), "tuned": tuned}}
    np.savez(OUT / "oof" / "teammate_async.npz", **o)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    e = res["c_async_with_rest"]["events"]
    print(f"(a) triggered 3-class, causal: {res['a_triggered_3class']['accuracy']:.3f} on {A['n']} trials")
    print(f"(b) as trained, sliding, no rest class: bal-acc {res['b_as_trained_sliding']['bal_acc']:.3f}")
    print(f"(c) async with rest: bal-acc {res['c_async_with_rest']['bal_acc']:.3f}, no-transitions "
          f"{res['c_async_with_rest']['bal_acc_no_transitions']:.3f}, merged {res['c_async_with_rest']['bal_acc_merged_3class']:.3f}, "
          f"events correct {e['correct_rate']:.2f}, false/min {e['false_per_min']:.2f}, "
          f"delay {e['latency_s_median_fist_peace_glove_onset']}, cap met {res['c_async_with_rest']['n_folds_cap_met']}/5")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
