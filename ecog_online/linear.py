"""Phase 2: linear baselines on the causal feature stream, scored like a live decoder.

  classify  4-class (rest, fist, peace, open) from the last 1.0 s of features, every 100 ms.
            Shrinkage LDA and L2 logistic regression, trained on glove-based or cue-based
            labels and always scored against the glove-based ones. Output smoothing (none,
            EMA, forward HMM) and the event rule (threshold, N steps) are tuned on contiguous
            inner folds of each training fold, under a false-detection budget.
  regress   ridge from the same windows to the five glove channels (0-1 on training data),
            with the brain-to-glove lag (0-300 ms) and alpha chosen on inner folds.

Cross-validated on the first 72 trials, where the configurations were chosen;
allcv.py re-runs the chosen ones over all 90 trials.

    python -m ecog_online.linear --task classify     -> results/online/linear_classify.json
    python -m ecog_online.linear --task regress      -> results/online/linear_regress.json
    progress: results/online/logs/linear_<task>.log
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .common import (OUT, WINDOW, Progress, ShrinkLDA, balanced_accuracy, confusion, detect,
                     event_metrics, fold_features, inner_splits, make_logreg, merged,
                     regression_scores, ridge_path, runs, smooth, summarise_events, windows)
from .features import Config
from .protocol import CLASSES, FS, fold_masks, load, sample_labels

# --- config ---------------------------------------------------------------------------
FOLDS = [0, 1, 2, 3, 4]
MODELS = ["lda", "logreg"]
LABELS = ["glove", "cue"]
LOGREG_C = 0.01
N_INNER = 4
SMOOTHERS = [("none", 0.0), ("ema", 0.5), ("ema", 0.7), ("ema", 0.85),
             ("hmm", 0.9), ("hmm", 0.95), ("hmm", 0.98)]
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]
N_STEPS = [1, 2, 3, 5]
FP_BUDGET = 2.0                # false detections per minute of rest allowed while tuning
LAGS_MS = [0, 50, 100, 150, 200, 250, 300]
ALPHAS = [1e1, 1e2, 1e3, 1e4, 1e5]
_HG = ((55, 85), (85, 115), (115, 145), (155, 185), (185, 215), (215, 245), (255, 285))
FEATURES = {                   # front-end presets; "default" is the spec's band list
    "default": Config(),
    "all_extra": Config(bands=((8, 13), (13, 30), (30, 48)) + _HG, lmp=True),
    "lmp": Config(lmp=True),
    "mu_lowgamma": Config(bands=((8, 13), (13, 30), (30, 48)) + _HG),
}


def cue_labels(rec):
    lab = np.zeros(rec.ecog.shape[1], int)
    for c, s, e in rec.cues:
        lab[s:e] = c
    return lab


def fit_predict(model, X, y, tr, te):
    m = ShrinkLDA() if model == "lda" else make_logreg(LOGREG_C)
    m.fit(X[tr], y[tr])
    P = m.predict_proba(X[te])
    return P


def run_events(P, steps, ends, truth, lab, kind, param, thr, n):
    """Smooth and detect on each contiguous run of `steps`; returns (smoothed, event parts)."""
    S = np.empty_like(P)
    parts = []
    pos = {s: i for i, s in enumerate(steps)}
    for r in runs(steps):
        i = np.array([pos[s] for s in r])
        S[i] = smooth(P[i], kind, param)
        parts.append(event_metrics(detect(S[i], thr, n), r, ends, truth, lab[ends[r]]))
    return S, parts


def tune(P, steps, ends, truth, lab):
    """Pick smoother, threshold and N on inner out-of-fold probabilities."""
    best, best_key = None, None
    smoothed = {}
    for kind, param in SMOOTHERS:
        S = np.empty_like(P)
        pos = {s: i for i, s in enumerate(steps)}
        rs = [np.array([pos[s] for s in r]) for r in runs(steps)]
        for i in rs:
            S[i] = smooth(P[i], kind, param)
        smoothed[(kind, param)] = (S, rs)
    for (kind, param), thr, n in itertools.product(SMOOTHERS, THRESHOLDS, N_STEPS):
        S, rs = smoothed[(kind, param)]
        parts = [event_metrics(detect(S[i], thr, n), steps[i], ends, truth, lab[ends[steps[i]]]) for i in rs]
        s = summarise_events(parts)
        ok = s["false_per_min"] <= FP_BUDGET
        key = (ok, s["correct_rate"] if ok else -s["false_per_min"], -s["false_per_min"])
        if best_key is None or key > best_key:
            best_key, best = key, {"smoother": kind, "param": param, "thr": thr, "n": n,
                                   "inner": s}
    return best


def classify(rec, folds, models, label_sets, log_name, window=WINDOW, tag="", cfg=Config()):
    lab, info, _ = sample_labels(rec)
    cue = cue_labels(rec)
    truth = [i for i in info]
    n_units = len(folds) * len(models) * len(label_sets) * (N_INNER + 1)
    prog = Progress(n_units + len(folds), log_name)
    results = {f"{m}/{l}": {"folds": [], "oof": {}} for m in models for l in label_sets}
    for fold in folds:
        feats, ends = fold_features(rec, fold, cfg)
        prog.step(f"front end fold {fold}")
        X, valid = windows(feats, window)
        y_true = lab[ends]
        tr, te = fold_masks(ends, rec, fold)
        tr, te = np.flatnonzero(tr & valid), np.flatnonzero(te & valid)
        for model, labels in itertools.product(models, label_sets):
            y_fit = y_true if labels == "glove" else cue[ends]
            key = f"{model}/{labels}"
            # inner out-of-fold probabilities for tuning the event stage
            P_in = np.zeros((len(tr), 4))
            pos = {s: i for i, s in enumerate(tr)}
            for j, (itr, iva) in enumerate(inner_splits(tr, ends, N_INNER)):
                P_in[[pos[s] for s in iva]] = fit_predict(model, X, y_fit, itr, iva)
                prog.step(f"fold {fold} {key} inner {j + 1}/{N_INNER}")
            ev_cfg = tune(P_in, tr, ends, truth, lab)
            P = fit_predict(model, X, y_fit, tr, te)
            prog.step(f"fold {fold} {key} outer")
            S, parts = run_events(P, te, ends, truth, lab, ev_cfg["smoother"], ev_cfg["param"], ev_cfg["thr"], ev_cfg["n"])
            yt = y_true[te]
            raw, sm = P.argmax(1), S.argmax(1)
            ev = summarise_events(parts)
            results[key]["folds"].append({
                "fold": fold, "tuned": ev_cfg,
                "bal_acc_raw": balanced_accuracy(yt, raw, range(4)),
                "bal_acc_smoothed": balanced_accuracy(yt, sm, range(4)),
                "bal_acc_merged_3class": balanced_accuracy(merged(yt), merged(sm), range(3)),
                "events": ev, "event_parts": parts,
            })
            o = results[key]["oof"]
            for name, arr in (("steps", te), ("P", P), ("S", S), ("y", yt)):
                o.setdefault(name, []).append(arr)
            print(f"    {key} fold {fold}: bal-acc raw {results[key]['folds'][-1]['bal_acc_raw']:.3f} "
                  f"smoothed {results[key]['folds'][-1]['bal_acc_smoothed']:.3f}, events correct "
                  f"{ev['correct_rate']:.2f}, false/min {ev['false_per_min']:.2f} "
                  f"({ev_cfg['smoother']} {ev_cfg['param']}, thr {ev_cfg['thr']}, n {ev_cfg['n']})", flush=True)
    out = {}
    for key, r in results.items():
        o = {k: np.concatenate(v) for k, v in r["oof"].items()}
        fl = r["folds"]
        agg = lambda k: [float(np.mean([f[k] for f in fl])), float(np.std([f[k] for f in fl]))]
        evs = summarise_events([p for f in fl for p in f["event_parts"]])
        out[key] = {
            "pooled": {"bal_acc_raw": balanced_accuracy(o["y"], o["P"].argmax(1), range(4)),
                       "bal_acc_smoothed": balanced_accuracy(o["y"], o["S"].argmax(1), range(4)),
                       "bal_acc_merged_3class": balanced_accuracy(merged(o["y"]), merged(o["S"].argmax(1)), range(3)),
                       "confusion_raw": confusion(o["y"], o["P"].argmax(1)).tolist(),
                       "confusion_smoothed": confusion(o["y"], o["S"].argmax(1)).tolist(),
                       "events": evs},
            "per_fold_mean_sd": {k: agg(k) for k in ("bal_acc_raw", "bal_acc_smoothed", "bal_acc_merged_3class")},
            "per_fold_events_correct_mean_sd": [float(np.mean([f["events"]["correct_rate"] for f in fl])),
                                                float(np.std([f["events"]["correct_rate"] for f in fl]))],
            "per_fold_false_per_min_mean_sd": [float(np.mean([f["events"]["false_per_min"] for f in fl])),
                                               float(np.std([f["events"]["false_per_min"] for f in fl]))],
            "folds": [{k: v for k, v in f.items() if k != "event_parts"} for f in fl],
        }
        np.savez(OUT / "oof" / f"linear_{key.replace('/', '_')}{tag}.npz",
                 **o)
    return out


def regress(rec, folds, log_name, window=WINDOW, tag="", cfg=Config()):
    lab, _, _ = sample_labels(rec)
    prog = Progress(len(folds) * (N_INNER + 2), log_name)
    lags = [int(l * FS / 1000) for l in LAGS_MS]
    fold_out, oof = [], {"steps": [], "Y": [], "Yhat": []}
    for fold in folds:
        feats, ends = fold_features(rec, fold, cfg)
        prog.step(f"front end fold {fold}")
        X, valid = windows(feats, window)
        tr, te = fold_masks(ends, rec, fold)
        tr, te = np.flatnonzero(tr & valid), np.flatnonzero(te & valid)
        # glove target: mean over the 100 ms ending at (step end + lag), 0-1 on training rows
        n = rec.glove.shape[1]
        g_all = np.stack([np.stack([rec.glove[:, min(max(e + L - 119, 0), n - 1): min(e + L + 1, n)].mean(1)
                                    for e in ends]) for L in lags])          # lags x T x 5
        lo, hi = g_all[0, tr].min(0), g_all[0, tr].max(0)
        g_all = (g_all - lo) / (hi - lo)
        # inner: score every (lag, alpha) by mean Pearson r over fingers
        score = np.zeros((len(lags), len(ALPHAS)))
        for j, (itr, iva) in enumerate(inner_splits(tr, ends, N_INNER)):
            Ycat = np.concatenate([g_all[i, itr] for i in range(len(lags))], axis=1)
            pred = ridge_path(X[itr], Ycat, X[iva], ALPHAS)                  # alphas x n x (lags*5)
            for li in range(len(lags)):
                for ai in range(len(ALPHAS)):
                    r, _ = regression_scores(g_all[li, iva], pred[ai][:, li * 5:(li + 1) * 5])
                    score[li, ai] += r.mean() / N_INNER
            prog.step(f"fold {fold} ridge inner {j + 1}/{N_INNER}")
        li, ai = np.unravel_index(np.argmax(score), score.shape)
        Ycat = np.concatenate([g_all[i, tr] for i in range(len(lags))], axis=1)
        pred = ridge_path(X[tr], Ycat, X[te], ALPHAS)
        prog.step(f"fold {fold} ridge outer")
        Yhat = pred[ai][:, li * 5:(li + 1) * 5]
        Y = g_all[li, te]
        r, r2 = regression_scores(Y, Yhat)
        lag_curve = {LAGS_MS[k]: float(regression_scores(g_all[k, te], pred[ai][:, k * 5:(k + 1) * 5])[0].mean())
                     for k in range(len(lags))}
        fold_out.append({"fold": fold, "lag_ms": LAGS_MS[li], "alpha": ALPHAS[ai],
                         "inner_score": score.round(4).tolist(),
                         "r": r.round(4).tolist(), "r2": r2.round(4).tolist(),
                         "mean_r": float(r.mean()), "mean_r2": float(r2.mean()),
                         "outer_mean_r_by_lag_at_chosen_alpha": lag_curve})
        for k, v in (("steps", te), ("Y", Y), ("Yhat", Yhat)):
            oof[k].append(v)
        print(f"    fold {fold}: lag {LAGS_MS[li]} ms, alpha {ALPHAS[ai]:g}, r {np.round(r, 2).tolist()} "
              f"mean {r.mean():.3f}, R2 mean {r2.mean():.3f}", flush=True)
    o = {k: np.concatenate(v) for k, v in oof.items()}
    r, r2 = regression_scores(o["Y"], o["Yhat"])
    fingers = ["thumb", "index", "middle", "ring", "little"]
    np.savez(OUT / "oof" / f"linear_ridge{tag}.npz", **o)
    return {
        "pooled": {"r": dict(zip(fingers, r.round(4).tolist())), "r2": dict(zip(fingers, r2.round(4).tolist())),
                   "mean_r": float(r.mean()), "mean_r2": float(r2.mean())},
        "per_fold_mean_sd": {"mean_r": [float(np.mean([f["mean_r"] for f in fold_out])), float(np.std([f["mean_r"] for f in fold_out]))],
                             "mean_r2": [float(np.mean([f["mean_r2"] for f in fold_out])), float(np.std([f["mean_r2"] for f in fold_out]))],
                             **{f"r_{n}": [float(np.mean([f["r"][i] for f in fold_out])), float(np.std([f["r"][i] for f in fold_out]))]
                                for i, n in enumerate(fingers)}},
        "folds": fold_out,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "ECoG_Handpose.mat"))
    ap.add_argument("--task", choices=["classify", "regress"], required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=FOLDS)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--labels", nargs="*", default=LABELS)
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--features", choices=list(FEATURES), default="default")
    ap.add_argument("--window", type=int, default=WINDOW, help="steps of 100 ms per decision (default 10 = 1.0 s)")
    args = ap.parse_args()
    (OUT / "oof").mkdir(parents=True, exist_ok=True)
    folds = args.folds

    rec = load(args.data)
    name = f"linear_{args.task}{args.tag}"
    cfg = FEATURES[args.features]
    config = {"folds": folds, "window_s": args.window / 10, "step_s": 0.1,
              "features": args.features, "bands": cfg.bands, "lmp": cfg.lmp}
    if args.task == "classify":
        config.update(models=args.models, labels=args.labels, logreg_C=LOGREG_C, n_inner=N_INNER,
                      smoothers=SMOOTHERS, thresholds=THRESHOLDS, n_steps=N_STEPS, fp_budget_per_min=FP_BUDGET)
        res = classify(rec, folds, args.models, args.labels, name, args.window, args.tag, cfg)
    else:
        config.update(lags_ms=LAGS_MS, alphas=ALPHAS, n_inner=N_INNER)
        res = regress(rec, folds, name, args.window, args.tag, cfg)
    out = OUT / f"{name}.json"
    out.write_text(json.dumps({"config": config, "results": res}, indent=1, default=float) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
