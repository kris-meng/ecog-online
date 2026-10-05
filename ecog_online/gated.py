"""Two-stage decoding: a rest-vs-movement gate decides WHETHER, a gesture model decides WHICH.

The gate is tuned for few false activations during rest first. The gesture model only
speaks while the gate is open, so it no longer competes with rest, and suppressing rest
does not swallow open hand. Scored from the user's side, per true movement:

  first call     the label the system commits to first, and its delay from movement onset;
                 a wrong first call is a wrong action even if corrected later
  final call     the label when the gate closes
  switches       label changes while the gate is open (a twitchy hand)
  missed, re-activations (gate opening again within one movement)
  false activations per minute of rest (and as counts: CV holds ~2.1 min of rest)

Pipelines, all on the same contiguous folds, 1.5 s of history:
  A  logreg 4-class alone: gate = 1 - P(rest), gesture = its gesture probabilities
  B  gate logreg (movement vs rest)  +  gesture logreg trained on movement steps only
  C  gate logreg  +  CNN A0.5 (3-seed mean) gesture probabilities
  D  CNN A0.5 alone

Decision parameters are tuned per fold on the OTHER folds' out-of-fold output, under a
false-activation cap. First 72 trials only.

    python -m ecog_online.gated     -> results/online/gated.json
"""

from __future__ import annotations

import itertools
import json

import numpy as np

from .common import OUT, Progress, fold_features, make_logreg, windows
from .protocol import FS, fold_blocks, fold_masks, load, sample_labels

# --- config ---------------------------------------------------------------------------
WINDOW = 15
LOGREG_C = 0.01
FA_CAPS = [0.5, 1.0, 2.0]          # false activations per minute of rest; 1.0 is the headline
HEADLINE_CAP = 1.0
TOL_S = 0.5                         # a first call this long before onset still belongs to the movement
GRID = {
    "alpha": [0.0, 0.5, 0.7, 0.85],         # EMA on the gate score
    "on": [0.5, 0.6, 0.7, 0.8, 0.9],        # gate opens after n_on steps at or above this
    "n_on": [1, 2, 3, 5],
    "hyst": [0.1, 0.2],                     # gate closes below on - hyst ...
    "n_off": [2, 5],                        # ... for this many steps
    "commit": [0, 2, 4],                    # first call after this many more steps of evidence
    "label": ["cumulative", "instant"],     # label = argmax of mean since gate-on, or of this step
}
CNN_RUNS = [f"cnn_A0.5_A0.5_gesture_head_seed{s}" for s in range(3)]


# --- out-of-fold scores ---------------------------------------------------------------

def stage_models(rec, lab, prog):
    """Out-of-fold gate probability and 3-class gesture probabilities from new logregs."""
    steps_all, gate, gest = [], [], []
    for k in range(5):
        feats, ends = fold_features(rec, k)
        X, valid = windows(feats, WINDOW)
        tr, te = fold_masks(ends, rec, k)
        tr, te = np.flatnonzero(tr & valid), np.flatnonzero(te & valid)
        y = lab[ends]
        g = make_logreg(LOGREG_C).fit(X[tr], (y[tr] > 0).astype(int))
        mv = tr[y[tr] > 0]
        c = make_logreg(LOGREG_C).fit(X[mv], y[mv])
        steps_all.append(te)
        gate.append(g.predict_proba(X[te])[:, 1])
        gest.append(c.predict_proba(X[te]))
        prog.step(f"fold {k}: gate and gesture logregs")
    return np.concatenate(steps_all), np.concatenate(gate), np.concatenate(gest)


def renorm(P):
    G = P[:, 1:]
    return G / np.clip(G.sum(1, keepdims=True), 1e-9, None)


# --- the decision rule ----------------------------------------------------------------

def decide(score, G, p):
    """Gate + labels on one contiguous run. Returns activations:
    [{"on": i, "off": j, "first": (i_call, cls), "labels": [cls per step from first call]}]"""
    n = len(score)
    a = p["alpha"]
    s = np.empty(n)
    acc = score[0]
    for i in range(n):
        acc = a * acc + (1 - a) * score[i] if i else score[0]
        s[i] = acc
    off_thr = p["on"] - p["hyst"]
    acts, i, above, below, cur = [], 0, 0, 0, None
    for i in range(n):
        if cur is None:
            above = above + 1 if s[i] >= p["on"] else 0
            if above >= p["n_on"]:
                cur = {"on": i, "off": None, "first": None, "labels": [], "sum": np.zeros(G.shape[1]), "k": 0}
                below = 0
        else:
            below = below + 1 if s[i] < off_thr else 0
            if below >= p["n_off"]:
                cur["off"] = i
                acts.append(cur)
                cur, above = None, 0
                continue
        if cur is not None:
            cur["sum"] += G[i]
            cur["k"] += 1
            if cur["first"] is None and cur["k"] > p["commit"]:
                cur["first"] = (i, int(np.argmax(cur["sum"])) + 1)
            if cur["first"] is not None:
                lab_now = int(np.argmax(cur["sum"] if p["label"] == "cumulative" else G[i])) + 1
                cur["labels"].append(lab_now)
    if cur is not None:
        cur["off"] = n - 1
        acts.append(cur)
    return [x for x in acts if x["first"] is not None]


def score_user(acts_by_run, steps, ends, gestures, lab):
    """User-side metrics for activations on runs of `steps` (indices into `ends`)."""
    t_end = ends
    calls = []                         # (time of first call, first, final, switches)
    for run, acts in acts_by_run:
        for a in acts:
            labs = a["labels"]
            calls.append({"t": t_end[run[a["first"][0]]], "first": a["first"][1], "final": labs[-1] if labs else a["first"][1],
                          "switches": int(np.sum(np.diff(labs) != 0)) if len(labs) > 1 else 0})
    lo, hi = ends[steps[0]], ends[steps[-1]]
    tol = int(TOL_S * FS)
    per, used = [], set()
    for g in gestures:
        if not (lo <= g["onset"] <= hi):
            continue
        hits = [j for j, c in enumerate(calls) if g["onset"] - tol <= c["t"] <= g["end"]]
        used |= set(hits)
        if not hits:
            per.append({"class": g["class"], "detected": False})
            continue
        c = calls[hits[0]]
        per.append({"class": g["class"], "detected": True, "first_ok": c["first"] == g["class"],
                    "final_ok": c["final"] == g["class"], "switches": c["switches"],
                    "reactivations": len(hits) - 1, "delay_s": (c["t"] - g["onset"]) / FS,
                    "glove_onset": g["source"] == "glove"})
    false = len(calls) - len(used)
    rest_min = float((lab[ends[steps]] == 0).sum()) * 0.1 / 60
    return per, false, rest_min


def summarise(per, false, rest_min):
    det = [p for p in per if p["detected"]]
    n = len(per)
    out = {"n_gestures": n, "detected": len(det) / n if n else None,
           "first_call_right": sum(p["first_ok"] for p in det) / n if n else None,
           "final_call_right": sum(p["final_ok"] for p in det) / n if n else None,
           "mean_switches_per_detected": float(np.mean([p["switches"] for p in det])) if det else None,
           "share_detected_with_any_switch": float(np.mean([p["switches"] > 0 for p in det])) if det else None,
           "reactivations": int(sum(p["reactivations"] for p in det)),
           "false_activations": int(false), "rest_min": round(rest_min, 3),
           "false_per_min": false / rest_min if rest_min else None}
    fp = [p for p in det if p["first_ok"] and p["class"] in (1, 2) and p["glove_onset"]]
    out["median_delay_first_call_right_fist_peace_s"] = float(np.median([p["delay_s"] for p in fp])) if fp else None
    for c, name in ((1, "fist"), (2, "peace"), (3, "open")):
        pc = [p for p in per if p["class"] == c]
        out[f"first_call_right_{name}"] = sum(p.get("first_ok", False) for p in pc) / len(pc) if pc else None
        out[f"detected_{name}"] = sum(p["detected"] for p in pc) / len(pc) if pc else None
    return out


def evaluate(score, G, steps, ends, gestures, lab, params):
    runs = np.split(np.arange(len(steps)), np.flatnonzero(np.diff(steps) != 1) + 1)
    acts = [(steps[r], decide(score[r], G[r], params)) for r in runs]
    return score_user(acts, steps, ends, gestures, lab)


def grid_eval(score, G, steps, ends, gestures, lab):
    """Every setting in GRID on these steps: [(params, summary)]."""
    res = []
    for vals in itertools.product(*GRID.values()):
        p = dict(zip(GRID, vals))
        res.append((p, summarise(*evaluate(score, G, steps, ends, gestures, lab, p))))
    return res


def choose(res, cap):
    """Most first calls right with false activations <= cap per minute of rest;
    ties: fewer switches, then shorter delay. If no setting meets the cap, fewest false."""
    def key(item):
        s = item[1]
        ok = s["false_per_min"] <= cap
        return (ok, s["first_call_right"] if ok else -s["false_per_min"],
                -(s["mean_switches_per_detected"] or 0), -(s["median_delay_first_call_right_fist_peace_s"] or 9))
    p, s = max(res, key=key)
    return {**p, "feasible": s["false_per_min"] <= cap, "tune": s}


def main():
    rec = load(str(OUT.parents[2] / "ECoG_Handpose.mat"))
    lab, info, _ = sample_labels(rec)
    _, ends = fold_features(rec, 0)
    blocks = fold_blocks(rec)
    n_grid = int(np.prod([len(v) for v in GRID.values()]))
    prog = Progress(5 + 4 * 5, "gated")

    st_new, gate_new, gest_new = stage_models(rec, lab, prog)
    lr = np.load(OUT / "oof" / "linear_logreg_glove_w15.npz")
    cnn = [np.load(OUT / "oof" / f"{r}.npz") for r in CNN_RUNS]
    Pc = np.mean([c["P"] for c in cnn], axis=0)
    common = np.intersect1d(np.intersect1d(st_new, lr["steps"]), cnn[0]["steps"])
    pick = lambda st, A: A[np.searchsorted(st, common)] if np.all(np.diff(st) > 0) else A[[int(np.flatnonzero(st == s)[0]) for s in common]]
    gate_new, gest_new = pick(st_new, gate_new), pick(st_new, gest_new)
    Plr, Pcn = pick(lr["steps"], lr["P"]), pick(cnn[0]["steps"], Pc)
    pipelines = {
        "A logreg 4-class alone": (1 - Plr[:, 0], renorm(Plr)),
        "B gate logreg + gesture logreg": (gate_new, gest_new),
        "C gate logreg + CNN A0.5 gestures": (gate_new, renorm(Pcn)),
        "D CNN A0.5 alone": (1 - Pcn[:, 0], renorm(Pcn)),
    }
    fold_of = np.searchsorted([b for _, b in blocks], ends[common], side="right")
    out = {"config": {"window_steps": WINDOW, "fa_caps_per_min": FA_CAPS, "headline_cap": HEADLINE_CAP,
                      "grid": GRID, "grid_size": n_grid, "tuning": "per fold on the other folds' out-of-fold output"}}
    for name, (score, G) in pipelines.items():
        out[name] = {}
        grids = []
        for k in range(5):
            tr = fold_of != k
            grids.append(grid_eval(score[tr], G[tr], common[tr], ends, info, lab))
            prog.step(f"{name}: grid on folds other than {k}")
        for cap in FA_CAPS:
            per_all, false_all, rest_all, tuned = [], 0, 0.0, []
            for k in range(5):
                te = fold_of == k
                p = choose(grids[k], cap)
                per, false, rest_min = evaluate(score[te], G[te], common[te], ends, info, lab, p)
                per_all += per
                false_all += false
                rest_all += rest_min
                tuned.append({**{k_: v for k_, v in p.items() if k_ != "tune"},
                              "tune_first_call_right": p["tune"]["first_call_right"]})
            s = summarise(per_all, false_all, rest_all)
            s["n_folds_cap_met_in_tuning"] = int(sum(t["feasible"] for t in tuned))
            out[name][f"cap_{cap}"] = {"summary": s, "tuned": tuned}
            print(f"{name:36s} cap {cap:>3}/min | first call right {s['first_call_right']:.2f} "
                  f"(fist {s['first_call_right_fist']:.2f} peace {s['first_call_right_peace']:.2f} open {s['first_call_right_open']:.2f}) "
                  f"| final {s['final_call_right']:.2f} | detected {s['detected']:.2f} | switches/detected "
                  f"{s['mean_switches_per_detected']:.2f} | false {s['false_activations']} in {s['rest_min']:.1f} min "
                  f"({s['false_per_min']:.1f}/min) | delay {s['median_delay_first_call_right_fist_peace_s']}", flush=True)
    (OUT / "gated.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {OUT / 'gated.json'}")


if __name__ == "__main__":
    main()
