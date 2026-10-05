"""Cross-validation over all 90 trials, the way the paper reports, adapted to a decoder
that runs every 100 ms.

The paper (and the submitted offline pipeline) cross-validate over the whole recording with
shuffled trials. Shuffling is fine for one decision per trial, but neighbouring 100 ms
windows overlap, so here the 90 trials are cut into 6 contiguous blocks of 15 (in rest
gaps), training steps within 1.5 s of a test block are dropped, and every step is predicted
once by models that never saw its block. The triggered offline model is also scored with
the paper's own protocol (shuffled 10-fold, repeated), where that protocol is valid.

Every configuration is frozen at what the 72-trial development CV chose; nothing is tuned
on these results. The event stage and the gated decision rule are tuned per block on the
OTHER blocks' out-of-fold output, as in eventfix.py and gated.py.

    python -m ecog_online.allcv     -> results/online/allcv.json, results/online/allcv/*.npz
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from . import cnn, gated as gt, settle as st, teammate_async as tm
from .common import (CACHE, OUT, Progress, ShrinkLDA, balanced_accuracy, cfg_key, confusion, events_on,
                     make_logreg, merged, near_transition, regression_scores, ridge_path,
                     summarise_events, tune_capped, val_split, windows)
from .features import Config, fit
from .protocol import FS, START_S, load, sample_labels

# --- config (frozen from the development CV) -------------------------------------------
BLOCK_TRIALS = 15
N_BLOCKS = 6
PURGE_S = 1.5
LOGREG_C = 0.01
RIDGE_ALPHA = 1e4                  # chosen in every development fold
CNN_SEEDS = [0, 1, 2]
CNN_LAMBDA = 0.5
GATED_CAPS = [1.0, 2.0]
PAPER_REPEATS = 10
FINGERS = ["thumb", "index", "middle", "ring", "little"]
OUT_DIR = OUT / "allcv"


# --- folds ----------------------------------------------------------------------------

def blocks(rec):
    edges = [int(START_S * FS)]
    for k in range(1, N_BLOCKS):
        i = k * BLOCK_TRIALS
        edges.append((rec.cues[i - 1][2] + rec.cues[i][1]) // 2)
    edges.append(rec.ecog.shape[1])
    return list(zip(edges[:-1], edges[1:]))


def masks(step_end, rec, k, purge_s=PURGE_S):
    b = blocks(rec)
    a, z = b[k]
    p = int(purge_s * FS)
    test = (step_end >= a) & (step_end < z)
    near = (step_end >= a - p) & (step_end < z + p)
    return (step_end >= b[0][0]) & ~near, test


def features(rec, k, cfg, purge_s=PURGE_S):
    """Front end fitted on block k's training samples ("all" = every sample, label-free)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{cfg_key(cfg)}_all{k}.npz"
    if path.exists():
        d = np.load(path)
        return d["feats"], d["ends"]
    samples = np.arange(rec.ecog.shape[1])
    tr = np.ones(len(samples), bool) if k == "fit" else masks(samples, rec, k, purge_s)[0]
    _, f, e = fit(rec.ecog, tr, cfg)
    np.savez(path, feats=f.astype(np.float32), ends=e)
    return f.astype(np.float32), e


def glove_targets(rec, ends):
    return np.stack([rec.glove[:, max(e - 119, 0): e + 1].mean(1) for e in ends]).astype(np.float32)


# --- scoring --------------------------------------------------------------------------

def step_metrics(y, P, near):
    p = P.argmax(1)
    return {"bal_acc": balanced_accuracy(y, p, range(4)),
            "bal_acc_no_transitions": balanced_accuracy(y[~near], p[~near], range(4)),
            "bal_acc_merged_3class": balanced_accuracy(merged(y), merged(p), range(3)),
            "confusion": confusion(y, p).tolist()}


def per_block(y, P, fold_of):
    return [balanced_accuracy(y[fold_of == k], P[fold_of == k].argmax(1), range(4)) for k in range(N_BLOCKS)]


def events_xfold(P, steps, fold_of, ends, info, lab):
    parts_all, tuned = [], []
    for k in range(N_BLOCKS):
        te, tr = fold_of == k, fold_of != k
        ev = tune_capped(P[tr], steps[tr], ends, info, lab)
        _, parts = events_on(P[te], steps[te], ends, info, lab, ev["smoother"], ev["param"], ev["thr"], ev["n"])
        parts_all += parts
        tuned.append(ev)
    s = summarise_events(parts_all)
    s["n_blocks_cap_met_in_tuning"] = int(sum(t["feasible"] for t in tuned))
    return s


def settle_metrics(P, steps, ends, info, lab):
    s = st.summary(st.settle(P.argmax(1), steps, ends, info))
    nf, rest_min = st.false_holds(P.argmax(1), steps, ends, lab)
    s.update(false_holds=nf, false_holds_per_min=nf / rest_min)
    return s


def gated_metrics(P, steps, fold_of, ends, info, lab):
    score, G = 1 - P[:, 0], gt.renorm(P)
    grids = [gt.grid_eval(score[fold_of != k], G[fold_of != k], steps[fold_of != k], ends, info, lab)
             for k in range(N_BLOCKS)]
    out = {}
    for cap in GATED_CAPS:
        per_all, false_all, rest_all, ok = [], 0, 0.0, 0
        for k in range(N_BLOCKS):
            p = gt.choose(grids[k], cap)
            ok += p["feasible"]
            per, false, rest_min = gt.evaluate(score[fold_of == k], G[fold_of == k], steps[fold_of == k], ends, info, lab, p)
            per_all += per
            false_all += false
            rest_all += rest_min
        s = gt.summarise(per_all, false_all, rest_all)
        s["n_blocks_cap_met_in_tuning"] = int(ok)
        out[f"cap_{cap}"] = s
    return out


def save_cnn(cdata, cnn_out):
    """Per-seed and seed-averaged CNN gesture probabilities and finger outputs, all 90 trials."""
    steps = np.concatenate([cdata[k][0] for k in range(N_BLOCKS)])
    G = np.concatenate([cdata[k][1][cdata[k][0]] for k in range(N_BLOCKS)])
    order = np.argsort(steps)
    Ps, Fs = [], []
    for s in CNN_SEEDS:
        P = np.concatenate([cnn_out[(k, s)][0] for k in range(N_BLOCKS)])[order]
        F_ = np.concatenate([cnn_out[(k, s)][1] for k in range(N_BLOCKS)])[order]
        np.savez(OUT_DIR / f"cnn_A0.5_seed{s}.npz", steps=steps[order], P=P, F=F_, G=G[order])
        Ps.append(P)
        Fs.append(F_)
    np.savez(OUT_DIR / "cnn_A0.5_ensemble.npz", steps=steps[order], P=np.mean(Ps, 0), F=np.mean(Fs, 0), G=G[order])


# --- main -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "ECoG_Handpose.mat"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only-cnn", action="store_true", help="train and save the CNN outputs only (for the demo)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec = load(args.data)
    lab, info, _ = sample_labels(rec)
    cfg = Config()
    n_units = N_BLOCKS * 5 + len(CNN_SEEDS) * N_BLOCKS + N_BLOCKS + 1 + 5
    prog = Progress(n_units, "allcv")

    # front ends
    F, E = {}, None
    for k in range(N_BLOCKS):
        F[k], E = features(rec, k, cfg)
        prog.step(f"front end block {k}")
    ends = E
    near = near_transition(lab, ends)
    y_all = lab[ends]
    g_raw = glove_targets(rec, ends)
    res = {"config": {"blocks": f"{N_BLOCKS} contiguous blocks of {BLOCK_TRIALS} trials", "purge_s": PURGE_S,
                      "frozen": {"logreg_C": LOGREG_C, "ridge": {"alpha": RIDGE_ALPHA, "lag_ms": 0},
                                 "cnn": {"lambda": CNN_LAMBDA, "seeds": CNN_SEEDS, "window_steps": cnn.WINDOW}},
                      "block_edges_s": [[round(a / FS, 2), round(b / FS, 2)] for a, b in blocks(rec)]}}

    # linear models and ridge
    lin = {"logreg_1.0s": (10, "logreg"), "logreg_1.5s": (15, "logreg"), "lda_1.5s": (15, "lda")}
    oof = {name: {"steps": [], "P": []} for name in lin}
    ridge = {"steps": [], "Y": [], "Yhat": []}
    for k in range(N_BLOCKS):
        for name, (w, model) in lin.items():
            X, valid = windows(F[k], w)
            tr, te = masks(ends, rec, k)
            tr, te = np.flatnonzero(tr & valid), np.flatnonzero(te & valid)
            m = (make_logreg(LOGREG_C) if model == "logreg" else ShrinkLDA()).fit(X[tr], y_all[tr])
            oof[name]["steps"].append(te)
            oof[name]["P"].append(m.predict_proba(X[te]))
            prog.step(f"block {k} {name}")
            if name == "logreg_1.5s":
                lo, hi = g_raw[tr].min(0), g_raw[tr].max(0)
                G = (g_raw - lo) / (hi - lo)
                ridge["steps"].append(te)
                ridge["Y"].append(G[te])
                ridge["Yhat"].append(ridge_path(X[tr], G[tr], X[te], [RIDGE_ALPHA])[0])
                prog.step(f"block {k} ridge 1.5 s")

    # CNN A0.5
    jobs, cdata = [], {}
    for k in range(N_BLOCKS):
        tr, te = masks(ends, rec, k)
        valid = np.arange(len(ends)) >= cnn.WINDOW - 1
        tr, te = np.flatnonzero(tr & valid), np.flatnonzero(te & valid)
        lo, hi = g_raw[tr].min(0), g_raw[tr].max(0)
        g = ((g_raw - lo) / (hi - lo)).astype(np.float32)
        fit_, val = val_split(tr, ends)
        cdata[k] = (te, g)
        dd = {"feats": F[k], "y": y_all, "g": g}
        for s in CNN_SEEDS:
            jobs.append({"key": ("A0.5", k, s, "outer"), "seed": s, "lambda": CNN_LAMBDA, "data": dd, "fit": fit_,
                         "val": val, "pred": {"te": te}, "max_epochs": 0})
    cnn_out = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fu in as_completed([ex.submit(cnn.train_one, j) for j in jobs]):
            r = fu.result()
            cnn_out[r["key"][1:3]] = r["out"]["te"]
            prog.step(f"CNN block {r['key'][1]} seed {r['key'][2]}: {r['epochs']} epochs")
    save_cnn(cdata, cnn_out)
    if args.only_cnn:
        return

    # the offline design: blocked, triggered and sliding
    T = {"steps": [], "Pb": [], "Pc": []}
    trig_correct, trig_n = 0, 0
    cls_all = np.array([c for c, _, _ in rec.cues])
    for k in range(N_BLOCKS):
        f, e = features(rec, k, tm.CFG, purge_s=tm.PURGE_S)
        X, valid = tm.their_features(f)
        tr, te = masks(e, rec, k, purge_s=tm.PURGE_S)
        tr, te = np.flatnonzero(tr & valid), np.flatnonzero(te & valid)
        cue_step = np.array([int(np.searchsorted(e, s + int(tm.DECISION_S * FS))) for _, s, _ in rec.cues])
        itr, ite = np.isin(cue_step, tr), np.isin(cue_step, te)
        m3 = ShrinkLDA().fit(X[cue_step[itr]], cls_all[itr])
        trig_correct += int((m3.classes_[m3.decision_function(X[cue_step[ite]]).argmax(1)] == cls_all[ite]).sum())
        trig_n += int(ite.sum())
        Pb = np.zeros((len(te), 4))
        Pb[:, 1:] = m3.predict_proba(X[te])
        T["steps"].append(te)
        T["Pb"].append(Pb)
        T["Pc"].append(ShrinkLDA().fit(X[tr], y_all[tr]).predict_proba(X[te]))
        prog.step(f"offline design block {k}")
    # the paper's protocol for the triggered model: shuffled 10-fold over all 90 trials,
    # front end fitted on the whole recording without labels
    f, e = features(rec, "fit", tm.CFG)
    X, _ = tm.their_features(f)
    Xt = X[np.array([int(np.searchsorted(e, s + int(tm.DECISION_S * FS))) for _, s, _ in rec.cues])]
    reps = []
    for r_ in range(PAPER_REPEATS):
        acc = [np.mean(ShrinkLDA().fit(Xt[a], cls_all[a]).decision_function(Xt[b]).argmax(1) + 1 == cls_all[b])
               for a, b in StratifiedKFold(10, shuffle=True, random_state=r_).split(Xt, cls_all)]
        reps.append(float(np.mean(acc)))
    prog.step("offline design, paper protocol")

    # assemble and score
    def cat(d):
        o = {k_: np.concatenate(v) for k_, v in d.items()}
        order = np.argsort(o["steps"])
        return {k_: v[order] for k_, v in o.items()}

    models = {name: cat(o) for name, o in oof.items()}
    cnn_P = {s: [] for s in CNN_SEEDS}
    cnn_F = {s: [] for s in CNN_SEEDS}
    cnn_steps, cnn_G = [], []
    for k in range(N_BLOCKS):
        te, g = cdata[k]
        cnn_steps.append(te)
        cnn_G.append(g[te])
        for s in CNN_SEEDS:
            cnn_P[s].append(cnn_out[(k, s)][0])
            cnn_F[s].append(cnn_out[(k, s)][1])
    cs = np.concatenate(cnn_steps)
    order = np.argsort(cs)
    cnn_seed = {s: {"steps": cs[order], "P": np.concatenate(cnn_P[s])[order], "F": np.concatenate(cnn_F[s])[order]}
                for s in CNN_SEEDS}
    cnn_G = np.concatenate(cnn_G)[order]
    models["cnn_A0.5_seedmean"] = {"steps": cs[order], "P": np.mean([cnn_seed[s]["P"] for s in CNN_SEEDS], axis=0)}
    T = cat(T)
    models["offline_design_async"] = {"steps": T["steps"], "P": T["Pc"]}
    fold_of = lambda steps: np.searchsorted([b for _, b in blocks(rec)], ends[steps], side="right")

    out = {}
    for name, m in models.items():
        steps, P = m["steps"], m["P"]
        y = y_all[steps]
        fo = fold_of(steps)
        r = step_metrics(y, P, near[steps])
        r["per_block_bal_acc"] = per_block(y, P, fo)
        r["events_strict"] = events_xfold(P, steps, fo, ends, info, lab)
        r["settle"] = settle_metrics(P, steps, ends, info, lab)
        prog.step(f"scored {name} (steps, events, settle)")
        if name in ("logreg_1.0s", "logreg_1.5s", "cnn_A0.5_seedmean"):
            r["user_view_gated"] = gated_metrics(P, steps, fo, ends, info, lab)
        out[name] = r
        np.savez(OUT_DIR / f"{name}.npz", steps=steps, P=P, y=y)
    # CNN per seed, sample level, for the spread
    out["cnn_A0.5_per_seed"] = {str(s): step_metrics(y_all[cnn_seed[s]["steps"]], cnn_seed[s]["P"], near[cnn_seed[s]["steps"]])
                                for s in CNN_SEEDS}
    out["offline_design_sliding_as_trained"] = step_metrics(y_all[T["steps"]], T["Pb"], near[T["steps"]])
    out["offline_design_triggered_blocked"] = {"accuracy": trig_correct / trig_n, "n_trials": trig_n}
    out["offline_design_triggered_paper_protocol"] = {"mean": float(np.mean(reps)), "sd_over_repeats": float(np.std(reps)),
                                                      "repeats": PAPER_REPEATS, "folds": 10}
    # fingers
    rd = cat(ridge)
    rr, r2 = regression_scores(rd["Y"], rd["Yhat"])
    out["fingers_ridge_1.5s"] = {"r": dict(zip(FINGERS, rr.round(4).tolist())), "r2": dict(zip(FINGERS, r2.round(4).tolist())),
                                 "mean_r": float(rr.mean()), "mean_r2": float(r2.mean())}
    cr = [regression_scores(cnn_G, cnn_seed[s]["F"]) for s in CNN_SEEDS]
    out["fingers_cnn_A0.5"] = {"mean_r": [float(np.mean([a.mean() for a, _ in cr])), float(np.std([a.mean() for a, _ in cr]))],
                               "mean_r2": [float(np.mean([b.mean() for _, b in cr])), float(np.std([b.mean() for _, b in cr]))],
                               "r_per_finger_seedmean": dict(zip(FINGERS, np.mean([a for a, _ in cr], axis=0).round(4).tolist()))}
    np.savez(OUT_DIR / "ridge_1.5s.npz", **rd)
    res["results"] = out
    (OUT / "allcv.json").write_text(json.dumps(res, indent=1, default=float) + "\n")

    print("\n=== 90-trial blocked CV ===")
    for name in models:
        r = out[name]
        e, s = r["events_strict"], r["settle"]
        print(f"{name:22s} bal {r['bal_acc']:.3f} (blocks {np.mean(r['per_block_bal_acc']):.3f}±{np.std(r['per_block_bal_acc']):.3f}) "
              f"no-trans {r['bal_acc_no_transitions']:.3f} merged {r['bal_acc_merged_3class']:.3f} | strict events "
              f"{e['correct_rate']:.2f}, {e['n_false']} false in {e['rest_min']:.1f} min | settled {s['settled_rate']:.2f} "
              f"in {s['median_time_to_settle_s']:.2f} s, false holds {s['false_holds_per_min']:.1f}/min")
        if "user_view_gated" in r:
            for cap, g in r["user_view_gated"].items():
                print(f"{'':22s}   gated {cap}: first call {g['first_call_right']:.2f} (fist {g['first_call_right_fist']:.2f} "
                      f"peace {g['first_call_right_peace']:.2f} open {g['first_call_right_open']:.2f}) final {g['final_call_right']:.2f} "
                      f"switches {g['mean_switches_per_detected']:.2f} false {g['false_activations']} ({g['false_per_min']:.1f}/min) "
                      f"delay {g['median_delay_first_call_right_fist_peace_s']}")
    print(f"offline design: triggered blocked {out['offline_design_triggered_blocked']['accuracy']:.3f}, paper protocol "
          f"{out['offline_design_triggered_paper_protocol']['mean']:.3f}±{out['offline_design_triggered_paper_protocol']['sd_over_repeats']:.3f}, "
          f"sliding as trained {out['offline_design_sliding_as_trained']['bal_acc']:.3f}")
    print(f"fingers: ridge r {out['fingers_ridge_1.5s']['mean_r']:.3f} R2 {out['fingers_ridge_1.5s']['mean_r2']:.3f}; "
          f"CNN r {out['fingers_cnn_A0.5']['mean_r'][0]:.3f} R2 {out['fingers_cnn_A0.5']['mean_r2'][0]:.3f}")
    print(f"wrote {OUT / 'allcv.json'}")


if __name__ == "__main__":
    main()
