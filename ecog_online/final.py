"""Phase 4: the held-out test, run once on the frozen pipelines.

Everything is fitted on the 72 development trials (0-338.2 s): the front end, the models,
early stopping and the event stage, which is tuned on the development part's own last
20 % block under the same caps as in cross-validation. The last 18 trials (338.2-422.5 s)
are then decoded once, in recording order, which is the order a calibrated system would
meet them.

  logreg_1.0s   logistic regression, 1.0 s of history, default bands
  logreg_1.5s   the same, 1.5 s
  cnn_A0.5      CNN + GRU, multi-task (lambda 0.5), 1.5 s, 5 seeds
  teammate      the submitted offline design: (a) triggered at cue + 2.0 s, (b) its 3-class
                model sliding, (c) retrained with a rest class and sliding

    python -m ecog_online.final --i-mean-it     -> results/online/final.json
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import cnn, teammate_async as tm
from .common import (OUT, Progress, ShrinkLDA, balanced_accuracy, confusion, events_on, fold_features,
                     make_logreg, merged, near_transition, regression_scores, summarise_events,
                     tune_capped, val_split, windows)
from .protocol import FS, N_DEV_TRIALS, final_masks, load, sample_labels

# --- config ---------------------------------------------------------------------------
LOGREG_C = 0.01
CNN_SEEDS = [0, 1, 2, 3, 4]
CNN_LAMBDA = 0.5


def metrics(P, te, ends, lab, info, ev):
    y = lab[ends[te]]
    S, parts = events_on(P, te, ends, info, lab, ev["smoother"], ev["param"], ev["thr"], ev["n"])
    near = near_transition(lab, ends)[te]
    sm = S.argmax(1)
    e = summarise_events(parts)
    return {"bal_acc_raw": balanced_accuracy(y, P.argmax(1), range(4)),
            "bal_acc_smoothed": balanced_accuracy(y, sm, range(4)),
            "bal_acc_no_transitions": balanced_accuracy(y[~near], sm[~near], range(4)),
            "bal_acc_merged_3class": balanced_accuracy(merged(y), merged(sm), range(3)),
            "confusion_smoothed": confusion(y, sm).tolist(),
            "events": e, "tuned": ev}, S


def linear(rec, lab, info, window, log):
    feats, ends = fold_features(rec, "final")
    X, valid = windows(feats, window)
    tr, te = final_masks(ends, rec)
    tr, te = np.flatnonzero(tr & valid), np.flatnonzero(te & valid)
    fit, val = val_split(tr, ends)
    ev = tune_capped(make_logreg(LOGREG_C).fit(X[fit], lab[ends[fit]]).predict_proba(X[val]), val, ends, info, lab)
    P = make_logreg(LOGREG_C).fit(X[tr], lab[ends[tr]]).predict_proba(X[te])
    m, S = metrics(P, te, ends, lab, info, ev)
    np.savez(OUT / "final" / f"logreg_w{window}.npz", steps=te, P=P, S=S, y=lab[ends[te]])
    return m


def teammate(rec, lab, info):
    feats, ends = fold_features(rec, "final", tm.CFG)
    X, valid = tm.their_features(feats)
    tr, te = final_masks(ends, rec, purge_s=tm.PURGE_S)
    tr, te = np.flatnonzero(tr & valid), np.flatnonzero(te & valid)
    cue_step = np.array([int(np.searchsorted(ends, s + int(tm.DECISION_S * FS))) for _, s, _ in rec.cues])
    cls = np.array([c for c, _, _ in rec.cues])
    dev, test = np.isin(cue_step, tr), np.arange(len(cls)) >= N_DEV_TRIALS
    m3 = ShrinkLDA().fit(X[cue_step[dev]], cls[dev])
    pa = m3.classes_[m3.decision_function(X[cue_step[test]]).argmax(1)]
    Pb = np.zeros((len(te), 4))
    Pb[:, 1:] = m3.predict_proba(X[te])
    y = lab[ends[te]]
    fit, val = val_split(tr, ends, purge_s=tm.PURGE_S)
    ev = tune_capped(ShrinkLDA().fit(X[fit], lab[ends[fit]]).predict_proba(X[val]), val, ends, info, lab)
    Pc = ShrinkLDA().fit(X[tr], lab[ends[tr]]).predict_proba(X[te])
    mc, _ = metrics(Pc, te, ends, lab, info, ev)
    return {"a_triggered_3class": {"accuracy": float(np.mean(pa == cls[test])), "n_trials": int(test.sum()),
                                   "confusion": confusion(cls[test] - 1, pa - 1, 3).tolist()},
            "b_as_trained_sliding": {"bal_acc": balanced_accuracy(y, Pb.argmax(1), range(4)),
                                     "confusion": confusion(y, Pb.argmax(1)).tolist()},
            "c_async_with_rest": mc}


def cnn_final(rec, lab, info, prog):
    d = cnn.fold_data(rec, "final")
    fit, val = val_split(d["tr"], d["ends"])
    dd = {k: d[k] for k in ("feats", "y", "g")}
    jobs = [{"key": ("A0.5", "final", s, "outer"), "seed": s, "lambda": CNN_LAMBDA, "data": dd, "fit": fit,
             "val": val, "pred": {"val": val, "te": d["te"]}, "max_epochs": 0} for s in CNN_SEEDS]
    per_seed = []
    with ProcessPoolExecutor(max_workers=len(CNN_SEEDS)) as ex:
        for r in ex.map(cnn.train_one, jobs):
            prog.step(f"cnn_A0.5 seed {r['key'][2]}: {r['epochs']} epochs (best {r['best_epoch']})")
            Pv, _ = r["out"]["val"]
            Pt, Ft = r["out"]["te"]
            ev = tune_capped(Pv, val, d["ends"], info, lab)
            m, S = metrics(Pt, d["te"], d["ends"], lab, info, ev)
            rr, r2 = regression_scores(d["g"][d["te"]], Ft)
            m.update(seed=r["key"][2], epochs=r["epochs"], r=rr.round(4).tolist(), r2=r2.round(4).tolist(),
                     mean_r=float(rr.mean()), mean_r2=float(r2.mean()))
            per_seed.append(m)
            np.savez(OUT / "final" / f"cnn_A0.5_seed{r['key'][2]}.npz", steps=d["te"], P=Pt, S=S,
                     y=d["y"][d["te"]], F=Ft, G=d["g"][d["te"]])
    keys = ["bal_acc_raw", "bal_acc_smoothed", "bal_acc_no_transitions", "bal_acc_merged_3class", "mean_r", "mean_r2"]
    summ = {k: [float(np.mean([m[k] for m in per_seed])), float(np.std([m[k] for m in per_seed]))] for k in keys}
    for k in ("correct_rate", "false_per_min", "latency_s_median_fist_peace_glove_onset"):
        v = [m["events"][k] for m in per_seed if m["events"][k] is not None]
        summ[f"events_{k}"] = [float(np.mean(v)), float(np.std(v))] if v else None
    summ["events_n_false"] = [m["events"]["n_false"] for m in per_seed]
    return {"mean_sd_over_seeds": summ, "seeds": per_seed}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "ECoG_Handpose.mat"))
    ap.add_argument("--i-mean-it", action="store_true", help="required: this scores the held-out test set")
    args = ap.parse_args()
    if not args.i_mean_it:
        raise SystemExit("This scores the held-out 20 %. Pass --i-mean-it to run it.")
    out_json = OUT / "final.json"
    if out_json.exists():
        raise SystemExit(f"{out_json} exists: the held-out test has already been run once.")
    (OUT / "final").mkdir(parents=True, exist_ok=True)
    print("*** FINAL: scoring the held-out test set (last 18 trials) ***", flush=True)

    rec = load(args.data)
    lab, info, _ = sample_labels(rec)
    prog = Progress(3 + len(CNN_SEEDS), "final")
    r = {"config": {"train": "72 development trials, 0-338.2 s", "test": "18 trials, 338.2-422.5 s",
                    "test_classes": {"fist": 4, "peace": 6, "open": 8}, "logreg_C": LOGREG_C,
                    "cnn_seeds": CNN_SEEDS, "cnn_lambda": CNN_LAMBDA,
                    "event_tuning": "last 20 % of development, <= 2 false/min and <= 0.8 s delay"}}
    r["logreg_1.0s"] = linear(rec, lab, info, 10, print)
    prog.step("logreg 1.0 s")
    r["logreg_1.5s"] = linear(rec, lab, info, 15, print)
    prog.step("logreg 1.5 s")
    r["teammate"] = teammate(rec, lab, info)
    prog.step("teammate design")
    r["cnn_A0.5"] = cnn_final(rec, lab, info, prog)
    out_json.write_text(json.dumps(r, indent=1, default=float) + "\n")

    def line(name, m):
        e = m["events"]
        return (f"{name:14s} bal {m['bal_acc_smoothed']:.3f} | no-trans {m['bal_acc_no_transitions']:.3f} | merged "
                f"{m['bal_acc_merged_3class']:.3f} | events {e['correct_rate']:.2f} correct, {e['n_false']} false in "
                f"{e['rest_min']:.2f} min, delay {e['latency_s_median_fist_peace_glove_onset']}")
    print(line("logreg 1.0 s", r["logreg_1.0s"]))
    print(line("logreg 1.5 s", r["logreg_1.5s"]))
    print(line("teammate (c)", r["teammate"]["c_async_with_rest"]))
    print(f"teammate (a) triggered: {r['teammate']['a_triggered_3class']['accuracy']:.3f} on 18 trials; "
          f"(b) sliding as trained: bal {r['teammate']['b_as_trained_sliding']['bal_acc']:.3f}")
    s = r["cnn_A0.5"]["mean_sd_over_seeds"]
    print("cnn A0.5       " + ", ".join(f"{k} {v[0]:.3f}±{v[1]:.3f}" for k, v in s.items()
                                       if isinstance(v, list) and len(v) == 2 and isinstance(v[0], float)))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
