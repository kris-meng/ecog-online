"""Time to settle: when does the decoder's output become the right gesture and stay there?

The strict event score counts a gesture only if the first event fired for it has the
right label. This measure is lenient about the start: a gesture counts as settled right
if, within the movement, the decoder's most likely class becomes the true gesture and
holds for HOLD consecutive steps. Time to settle runs from movement onset to the start of
that run. Nothing is tuned: it reads each model's raw per-step argmax, so every model is
judged the same way. It summarises saved predictions only and fits nothing.

    python -m ecog_online.settle     -> results/online/settle.json
"""

from __future__ import annotations

import json

import numpy as np

from .common import OUT, fold_features
from .protocol import FS, load, sample_labels

# --- config ---------------------------------------------------------------------------
HOLD = 5                       # steps (0.5 s) the right class must hold
LATENCY_MARKS_S = [0.5, 1.0, 1.5]
CV = {
    "logreg_1.0s": ["linear_logreg_glove"],
    "logreg_1.5s": ["linear_logreg_glove_w15"],
    "lda_1.5s": ["linear_lda_glove_w15"],
    "cnn_A0.5": [f"cnn_A0.5_A0.5_gesture_head_seed{s}" for s in range(3)],
    "cnn_A1": [f"cnn_A1_A1_gesture_head_seed{s}" for s in range(3)],
    "cnn_C": [f"cnn_C_C_gesture_head_seed{s}" for s in range(3)],
    "cnn_v2 (A1, rest x2, smoothing 0.1)": [f"cnn_v2_A1_gesture_head_seed{s}" for s in range(3)],
    "offline design, async (c)": ["teammate_async:Pc"],
}


def load_run(name):
    if ":" in name:
        f, key = name.split(":")
        d = np.load(OUT / "oof" / f"{f}.npz")
        return d["steps"], d[key]
    d = np.load(OUT / "oof" / f"{name}.npz")
    return d["steps"], d["P"]


def settle(pred, steps, ends, gestures):
    """Per gesture: settled right?, time to settle, first non-rest call."""
    t = ends[steps]
    out = []
    for g in gestures:
        m = (t >= g["onset"]) & (t < g["end"])
        if m.sum() < HOLD:
            continue
        p = pred[m]
        ok = np.convolve(p == g["class"], np.ones(HOLD, int), "valid") == HOLD
        first_call = next((int(c) for c in p if c != 0), None)
        out.append({"class": g["class"], "settled": bool(ok.any()),
                    "latency_s": float((t[m][int(np.argmax(ok))] - g["onset"]) / FS) if ok.any() else None,
                    "first_call_wrong": first_call is not None and first_call != g["class"],
                    "from_glove": g["source"] == "glove"})
    return out


def false_holds(pred, steps, ends, lab):
    """Non-rest classes held HOLD+ steps entirely inside rest-labelled time, per minute of rest."""
    y = lab[ends[steps]]
    contiguous = np.r_[True, np.diff(steps) != 1]
    n, run, cls = 0, 0, -1
    for i in range(len(steps)):
        if contiguous[i] or y[i] != 0:
            run, cls = 0, -1
        if y[i] != 0:
            continue
        if pred[i] != 0 and pred[i] == cls:
            run += 1
        elif pred[i] != 0:
            run, cls = 1, pred[i]
        else:
            run, cls = 0, -1
        if run == HOLD:
            n += 1
    return n, float((y == 0).sum()) * 0.1 / 60


def summary(per):
    lat = np.array([p["latency_s"] for p in per if p["settled"]])
    lat_fp = np.array([p["latency_s"] for p in per if p["settled"] and p["class"] in (1, 2)])
    n = len(per)
    s = {"n_gestures": n, "settled_rate": float(np.mean([p["settled"] for p in per])),
         "median_time_to_settle_s": float(np.median(lat)) if len(lat) else None,
         "median_time_to_settle_fist_peace_s": float(np.median(lat_fp)) if len(lat_fp) else None,
         "wrong_first_then_right": int(sum(p["first_call_wrong"] and p["settled"] for p in per)),
         "wrong_first_never_right": int(sum(p["first_call_wrong"] and not p["settled"] for p in per))}
    for x in LATENCY_MARKS_S:
        s[f"settled_within_{x}s"] = float(np.sum(lat <= x) / n)
    for c, name in ((1, "fist"), (2, "peace"), (3, "open")):
        pc = [p for p in per if p["class"] == c]
        s[f"settled_rate_{name}"] = float(np.mean([p["settled"] for p in pc])) if pc else None
    return s


def main():
    rec = load(str(OUT.parents[2] / "ECoG_Handpose.mat"))
    lab, info, _ = sample_labels(rec)
    _, ends = fold_features(rec, 0)
    out = {"config": {"hold_steps": HOLD, "hold_s": HOLD / 10, "decision": "raw per-step argmax"}}
    for split, models in (("cv", CV),):
        out[split] = {}
        for name, runs in models.items():
            try:
                loaded = [load_run(r) for r in runs]
            except FileNotFoundError as e:
                print(f"skip {name}: {e}")
                continue
            per_run = []
            for steps, P in loaded:
                sm = summary(settle(P.argmax(1), steps, ends, info))
                nf, rest_min = false_holds(P.argmax(1), steps, ends, lab)
                sm.update(false_holds=nf, false_holds_per_min=nf / rest_min)
                per_run.append(sm)
            keys = [k for k, v in per_run[0].items() if isinstance(v, (int, float)) and v is not None]
            agg = {k: [float(np.mean([r[k] for r in per_run if r[k] is not None])),
                       float(np.std([r[k] for r in per_run if r[k] is not None]))] for k in keys}
            out[split][name] = {"runs": len(per_run), "mean_sd": agg, "per_run": per_run}
            g = lambda k: agg[k][0]
            print(f"{split:8s} {name:36s} settled {g('settled_rate'):.2f} (fist {g('settled_rate_fist'):.2f} "
                  f"peace {g('settled_rate_peace'):.2f} open {g('settled_rate_open'):.2f}) | by 0.5/1.0/1.5 s "
                  f"{g('settled_within_0.5s'):.2f}/{g('settled_within_1.0s'):.2f}/{g('settled_within_1.5s'):.2f} | "
                  f"median {g('median_time_to_settle_s'):.2f} s | wrong-first-then-right "
                  f"{g('wrong_first_then_right'):.1f}, wrong-first-never {g('wrong_first_never_right'):.1f} "
                  f"of {int(g('n_gestures'))} | false holds {g('false_holds'):.1f} ({g('false_holds_per_min'):.1f}/min)", flush=True)
    (OUT / "settle.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {OUT / 'settle.json'}")


if __name__ == "__main__":
    main()
