"""Why the CNN fires fewer correct events, and whether post-processing fixes it. CV only.

Works on saved out-of-fold probabilities; nothing is retrained. For each development fold, everything below is fitted on the OTHER four
folds' out-of-fold predictions and applied to this fold:

  block      the event stage as before: tuned on the fold's own last 20 % block (from the
             saved run's results, for reference)
  xfold      event stage tuned on the other folds' predictions (~4x more data)
  xfold+T    the same after temperature scaling, temperature fitted on the other folds
  ensemble   CNN seeds averaged before any of this

    python -m ecog_online.eventfix     -> results/online/eventfix.json
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize_scalar

from .common import OUT, Progress, events_on, fold_features, summarise_events, tune_capped
from .protocol import fold_blocks, load, sample_labels

# --- config ---------------------------------------------------------------------------
MODELS = {
    "logreg_1.0s": ["linear_logreg_glove"],
    "logreg_1.5s": ["linear_logreg_glove_w15"],
    "lda_1.5s": ["linear_lda_glove_w15"],
    "cnn_A0.5_seed0": ["cnn_A0.5_A0.5_gesture_head_seed0"],
    "cnn_A0.5_ensemble": [f"cnn_A0.5_A0.5_gesture_head_seed{s}" for s in range(3)],
    "cnn_A1_ensemble": [f"cnn_A1_A1_gesture_head_seed{s}" for s in range(3)],
    "cnn_C_ensemble": [f"cnn_C_C_gesture_head_seed{s}" for s in range(3)],
}


def load_probs(names):
    """Average the probabilities of several saved runs over their common steps."""
    runs = [np.load(OUT / "oof" / f"{n}.npz") for n in names]
    steps = runs[0]["steps"]
    for r in runs[1:]:
        assert np.array_equal(r["steps"], steps)
    return steps, np.mean([r["P"] for r in runs], axis=0)


def fit_temperature(P, y):
    L = np.log(np.clip(P, 1e-9, 1))

    def nll(logT):
        Z = L / np.exp(logT)
        Z -= Z.max(1, keepdims=True)
        Q = np.exp(Z)
        Q /= Q.sum(1, keepdims=True)
        return -np.mean(np.log(Q[np.arange(len(y)), y] + 1e-12))

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def scale(P, T):
    Z = np.log(np.clip(P, 1e-9, 1)) / T
    Z -= Z.max(1, keepdims=True)
    Q = np.exp(Z)
    return Q / Q.sum(1, keepdims=True)


def main():
    rec = load(str(OUT.parents[2] / "ECoG_Handpose.mat"))
    lab, info, _ = sample_labels(rec)
    _, ends = fold_features(rec, 0)
    blocks = fold_blocks(rec)
    fold_of = lambda st: np.searchsorted([b for _, b in blocks], ends[st], side="right")
    prog = Progress(len(MODELS) * 5 * 2, "eventfix")
    out = {"config": {"tuning": "cross-fold on saved out-of-fold probabilities", "models": MODELS}}
    for name, runs in MODELS.items():
        steps, P = load_probs(runs)
        y = lab[ends[steps]]
        f = fold_of(steps)
        res = {}
        for mode in ("xfold", "xfold+T"):
            parts_all, temps, tuned = [], [], []
            for k in range(5):
                te, tr = f == k, f != k
                Ptr, Pte = P[tr], P[te]
                if mode == "xfold+T":
                    T = fit_temperature(Ptr, y[tr])
                    temps.append(T)
                    Ptr, Pte = scale(Ptr, T), scale(Pte, T)
                ev = tune_capped(Ptr, steps[tr], ends, info, lab)
                tuned.append(ev)
                _, parts = events_on(Pte, steps[te], ends, info, lab, ev["smoother"], ev["param"], ev["thr"], ev["n"])
                parts_all += parts
                prog.step(f"{name} {mode} fold {k}: {ev['smoother']} {ev['param']} thr {ev['thr']} n {ev['n']}")
            s = summarise_events(parts_all)
            res[mode] = {"events": s, "temperatures": temps, "tuned": tuned,
                         "n_folds_cap_met": int(sum(t["feasible"] for t in tuned))}
        pg_rest = float(np.median(P[y == 0][:, 1:].max(1)))
        res["median_max_gesture_prob_during_rest"] = pg_rest
        out[name] = res
        a, b = res["xfold"]["events"], res["xfold+T"]["events"]
        print(f"{name:18s} xfold: correct {a['correct_rate']:.2f} false/min {a['false_per_min']:.2f} delay "
              f"{a['latency_s_median_fist_peace_glove_onset']:.2f} | +T: correct {b['correct_rate']:.2f} false/min "
              f"{b['false_per_min']:.2f} delay {b['latency_s_median_fist_peace_glove_onset']:.2f} | "
              f"P(gesture) at rest {pg_rest:.2f}", flush=True)
    (OUT / "eventfix.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {OUT / 'eventfix.json'}")


if __name__ == "__main__":
    main()
