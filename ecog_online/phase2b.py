"""Phase 2b: how much of the linear decoders' error is lag, labels, or the event stage.

  transitions  balanced accuracy on all steps and with steps within +/-0.25 s of any label
               change left out, for every saved out-of-fold run.
  phases       accuracy of the 1.5 s logistic regression by phase of the movement: rest,
               first 0.5 s after onset, hold, release-and-settle (labelled as the gesture).
  sweep        the event stage over a grid of smoothing, threshold and consecutive steps,
               on the pooled out-of-fold probabilities of the same model: detection rate and
               false detections per minute against delay. A description of the trade-off;
               an operating point used for a reported number must still be tuned inside CV.
  open_vs_rest high-gamma over time, cue-locked, for open hand against the same trials' own
               pre-cue rest, and open-vs-rest decodability at each moment (trial-wise CV,
               with a label-shuffle null), fist and peace alongside for scale.

    python -m ecog_online.phase2b     -> results/online/phase2b.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import (OUT, Progress, balanced_accuracy, detect, event_metrics, fold_features,
                     merged, runs, smooth, summarise_events)
from .protocol import FS, N_DEV_TRIALS, glove_events, load, sample_labels

# --- config ---------------------------------------------------------------------------
TRANSITION_S = 0.25
MAIN = "linear_logreg_glove"
RUNS = ["linear_lda_glove", "linear_logreg_glove", "linear_lda_cue", "linear_logreg_cue",
        "linear_lda_glove_w3", "linear_logreg_glove_w3", "linear_lda_glove_w5",
        "linear_logreg_glove_w5", "linear_lda_glove_w15", "linear_logreg_glove_w15"]
SWEEP_SMOOTHERS = [("none", 0.0)] + [("ema", a) for a in (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9)] \
                  + [("hmm", p) for p in (0.8, 0.9, 0.95, 0.98)]
SWEEP_THR = [round(x, 2) for x in np.arange(0.3, 0.91, 0.05)]
SWEEP_N = [1, 2, 3, 4, 5]
LOCK = (-1.0, 3.5)            # seconds around the cue for the time course
REST_AT = -0.8                # the pre-cue moment each trial's own rest sample is taken from
SMOOTH_STEPS = 3              # centred average for the time course (descriptive, not causal)
N_PERM = 100
SEED = 0


def transitions(rec, lab, ends):
    change = np.flatnonzero(np.diff(lab)) + 1
    near = np.zeros(len(ends), bool)
    w = int(TRANSITION_S * FS)
    idx = np.searchsorted(change, ends)
    for k in (idx - 1, idx):
        ok = (k >= 0) & (k < len(change))
        near[ok] |= np.abs(ends[ok] - change[k[ok]]) <= w
    out = {}
    for name in RUNS:
        o = np.load(OUT / "oof" / f"{name}.npz")
        keep = ~near[o["steps"]]
        y, raw, sm = o["y"], o["P"].argmax(1), o["S"].argmax(1)
        out[name] = {
            "all": {"raw": balanced_accuracy(y, raw, range(4)), "smoothed": balanced_accuracy(y, sm, range(4)),
                    "merged_3class": balanced_accuracy(merged(y), merged(sm), range(3))},
            "no_transitions": {"raw": balanced_accuracy(y[keep], raw[keep], range(4)),
                               "smoothed": balanced_accuracy(y[keep], sm[keep], range(4)),
                               "merged_3class": balanced_accuracy(merged(y[keep]), merged(sm[keep]), range(3))},
            "fraction_excluded": float(1 - keep.mean()),
        }
    return out, near


def phases(rec, info, ends):
    """Per-step phase for the main run, and recall by phase and class."""
    ev = glove_events(rec)
    phase = np.full(len(ends), "rest", dtype=object)
    for i, (g, (_, rel, settled)) in enumerate(zip(info, ev)):
        on, end = g["onset"], g["end"]
        inside = (ends >= on) & (ends < end)
        phase[inside] = "hold"
        phase[inside & (ends < on + int(0.5 * FS))] = "onset_0_0.5s"
        if g["source"] == "glove" and rel is not None:
            phase[inside & (ends >= rel)] = "release_settle"
        elif g["source"] == "cue":
            cue_end = rec.cues[i][2]
            phase[inside & (ends >= cue_end)] = "release_settle"
    o = np.load(OUT / "oof" / f"{MAIN}.npz")
    ph, y, sm = phase[o["steps"]], o["y"], o["S"].argmax(1)
    names = ["rest", "fist", "peace", "open"]
    out = {}
    for p in ("rest", "onset_0_0.5s", "hold", "release_settle"):
        m = ph == p
        row = {"n_steps": int(m.sum()), "accuracy": float(np.mean(sm[m] == y[m])) if m.any() else None}
        for c in range(1, 4):
            mc = m & (y == c)
            if mc.any():
                row[names[c]] = {"n": int(mc.sum()), "correct": float(np.mean(sm[mc] == c)),
                                 "called_rest": float(np.mean(sm[mc] == 0))}
        out[p] = row
    return out


def sweep(rec, info, lab, ends, log):
    o = np.load(OUT / "oof" / f"{MAIN}.npz")
    steps, P = o["steps"], o["P"]
    pos = {s: i for i, s in enumerate(steps)}
    rs = [np.array([pos[s] for s in r]) for r in runs(steps)]
    grid = list(itertools.product(SWEEP_SMOOTHERS, SWEEP_THR, SWEEP_N))
    prog = Progress(len(SWEEP_SMOOTHERS), "phase2b sweep")
    rows = []
    for kind, param in SWEEP_SMOOTHERS:
        S = np.empty_like(P)
        for i in rs:
            S[i] = smooth(P[i], kind, param)
        for thr, n in itertools.product(SWEEP_THR, SWEEP_N):
            parts = [event_metrics(detect(S[i], thr, n), steps[i], ends, info, lab[ends[steps[i]]]) for i in rs]
            s = summarise_events(parts)
            rows.append({"smoother": kind, "param": param, "thr": thr, "n": n,
                         "correct_rate": s["correct_rate"], "detected_rate": s["detected_rate"],
                         "correct_rate_fist_peace": s["correct_rate_fist_peace"],
                         "false_per_min": s["false_per_min"], "n_false": s["n_false"],
                         "latency_s": s["latency_s_median_fist_peace_glove_onset"],
                         "latency_all_s": s["latency_s_median"]})
        prog.step(f"{kind} {param}")
    rest_min = summarise_events([event_metrics([], steps[i], ends, info, lab[ends[steps[i]]]) for i in rs])["rest_min"]
    return {"model": MAIN, "n_settings": len(grid), "rest_min": rest_min, "rows": rows}


def open_vs_rest(rec, info, log):
    """Cue-locked time course on development trials, front end fitted on development data."""
    feats, ends = fold_features(rec, "final")
    hg = feats[:, 1:8].mean(axis=1)                                   # steps x 10 x 6
    flat = feats.reshape(len(feats), -1)
    lo, hi = int(round(LOCK[0] / 0.1)), int(round(LOCK[1] / 0.1))
    t = np.arange(lo, hi + 1) * 0.1
    dev = [(c, s) for c, s, _ in rec.cues[:N_DEV_TRIALS]]
    j0 = np.array([int(np.searchsorted(ends, s)) for _, s in dev])
    cls = np.array([c for c, _ in dev])
    r_at = int(round(REST_AT / 0.1))
    h = SMOOTH_STEPS // 2
    seg = lambda F, j, k: F[j + k - h: j + k + h + 1].mean(0)
    out = {"t_s": t.round(2).tolist(), "rest_at_s": REST_AT, "n_trials": {}, "hg_grid_mean": {}, "decoding": {}}
    rng = np.random.default_rng(SEED)
    prog = Progress(3, "phase2b open-vs-rest")
    for c, name in ((3, "open"), (1, "fist"), (2, "peace")):
        js = j0[cls == c]
        out["n_trials"][name] = int(len(js))
        curves = np.array([[hg[j + k].mean() for k in range(lo, hi + 1)] for j in js])
        m, se = curves.mean(0), curves.std(0, ddof=1) / np.sqrt(len(js))
        out["hg_grid_mean"][name] = {"mean": m.round(4).tolist(), "sem": se.round(4).tolist()}
        Xr = np.stack([seg(flat, j, r_at) for j in js])
        groups = np.r_[np.arange(len(js)), np.arange(len(js))]
        y = np.r_[np.zeros(len(js)), np.ones(len(js))]
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = list(cv.split(np.zeros(2 * len(js)), y, groups))

        def score(X, yy):
            acc = []
            for tr, te in splits:
                mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500))
                acc.append(np.mean(mdl.fit(X[tr], yy[tr]).predict(X[te]) == yy[te]))
            return float(np.mean(acc))

        acc, null95 = [], []
        for k in range(lo, hi + 1):
            X = np.r_[np.stack([seg(flat, j, k) for j in js]), Xr]
            acc.append(score(X, y))
            if c == 3:
                # shuffle which sample of each trial is called rest: keeps the pairing
                perms = []
                for _ in range(N_PERM):
                    flip = rng.random(len(js)) < 0.5
                    yp = y.copy()                      # gesture samples first (0), rest second (1)
                    yp[: len(js)][flip] = 1
                    yp[len(js):][flip] = 0
                    perms.append(score(X, yp))
                null95.append(float(np.percentile(perms, 95)))
        out["decoding"][name] = {"acc": np.round(acc, 4).tolist()}
        if c == 3:
            out["decoding"][name]["null_95th"] = np.round(null95, 4).tolist()
        prog.step(f"{name}: peak acc {max(acc):.2f}")
    # where in the trial is open hand separable: mean decoding in each phase window
    a = np.array(out["decoding"]["open"]["acc"])
    win = lambda x0, x1: float(a[(t >= x0) & (t <= x1)].mean())
    out["open_phase_means"] = {"pre_cue_-0.5_0": win(-0.5, -0.05), "onset_0.2_0.8": win(0.2, 0.8),
                               "hold_1.0_2.0": win(1.0, 2.0), "after_cue_end_2.2_3.0": win(2.2, 3.0)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "ECoG_Handpose.mat"))
    ap.add_argument("--out", default=str(OUT / "phase2b.json"))
    args = ap.parse_args()
    rec = load(args.data)
    lab, info, _ = sample_labels(rec)
    _, ends = fold_features(rec, 0)
    r = {"config": {"transition_s": TRANSITION_S, "main_run": MAIN, "lock_s": LOCK, "rest_at_s": REST_AT,
                    "smooth_steps": SMOOTH_STEPS, "n_perm": N_PERM, "seed": SEED}}
    r["transitions"], _ = transitions(rec, lab, ends)
    for k, v in r["transitions"].items():
        print(f"{k:26s} all {v['all']['smoothed']:.3f} -> without +/-{TRANSITION_S}s {v['no_transitions']['smoothed']:.3f} "
              f"(merged {v['all']['merged_3class']:.3f} -> {v['no_transitions']['merged_3class']:.3f}; "
              f"{100 * v['fraction_excluded']:.0f}% of steps excluded)", flush=True)
    r["phases"] = phases(rec, info, ends)
    print(json.dumps(r["phases"], indent=1), flush=True)
    r["sweep"] = sweep(rec, info, lab, ends, print)
    r["open_vs_rest"] = open_vs_rest(rec, info, print)
    print(f"open-vs-rest decoding by phase: {r['open_vs_rest']['open_phase_means']}", flush=True)
    Path(args.out).write_text(json.dumps(r, indent=1, default=float) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
