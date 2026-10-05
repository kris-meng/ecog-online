"""Export a replay of the decoders for the 3D hand demo: results/online/demo/demo_data.js.

Only out-of-fold predictions go in: every step was predicted by a model, front end and
event stage that never trained on that stretch of the recording, and every output uses
only data up to that moment. The page replays them at recording speed, so it shows what
the decoder would have shown live.

    glove     real flexion at 30 Hz, 0-1 per finger (development-set 1st-99th percentile)
    steps     every 100 ms: decoded fingers, class probabilities (raw and smoothed),
              predicted class, true label, and the 10 x 6 high-gamma grid (z, rounded)
    gestures  true movements (onset, end, class, whether the onset came from the glove)
    events    fired detections (time, class, whether it matched the true gesture)

    python -m ecog_online.demo_export [--gestures linear_logreg_glove_w15] [--fingers linear_ridge_w15]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import OUT, detect, fold_features, runs, smooth
from .protocol import FS, fold_masks, load, sample_labels

# --- config ---------------------------------------------------------------------------
GLOVE_HZ = 30
FOLDS = [0, 1, 2, 3, 4]
EVENT_TOL_S = 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "ECoG_Handpose.mat"))
    ap.add_argument("--gestures", default="linear_logreg_glove_w15", help="saved out-of-fold classifier run")
    ap.add_argument("--gesture-desc", default="logistic regression on 1.5 s of band power")
    ap.add_argument("--finger-desc", default="Ridge, 1.5 s history")
    ap.add_argument("--fingers", default="linear_ridge_w15", help="saved out-of-fold finger run")
    ap.add_argument("--retune", default="retune_linear.json", help="event settings per fold")
    ap.add_argument("--out", default=str(OUT / "demo" / "demo_data.js"))
    args = ap.parse_args()

    rec = load(args.data)
    lab, info, _ = sample_labels(rec)
    clf = np.load(OUT / "oof" / f"{args.gestures}.npz")
    fin = np.load(OUT / "oof" / f"{args.fingers}.npz")
    tuned = json.loads((OUT / args.retune).read_text())[args.gestures]["folds"]
    _, ends = fold_features(rec, 0)

    steps = np.intersect1d(clf["steps"], fin["steps"])
    P = clf["P"][np.searchsorted(clf["steps"], steps)] if np.all(np.diff(clf["steps"]) > 0) else \
        clf["P"][[int(np.flatnonzero(clf["steps"] == s)[0]) for s in steps]]
    pos_f = {s: i for i, s in enumerate(fin["steps"])}
    Fd = fin["Yhat"][[pos_f[s] for s in steps]]

    # event stage per fold, with the settings tuned on that fold's own held-back block
    S = np.zeros_like(P)
    events = []
    grid = np.zeros((len(steps), 10, 6), np.float32)
    for f, tf in zip(FOLDS, tuned):
        feats, _ = fold_features(rec, f)
        _, te = fold_masks(ends, rec, f)
        m = te[steps]
        idx = np.flatnonzero(m)
        ev = tf["tuned"]
        for r in runs(idx):
            S[r] = smooth(P[r], ev["smoother"], ev["param"])
            for i, c in detect(S[r], ev["thr"], ev["n"]):
                events.append((int(r[i]), c))
        grid[idx] = feats[steps[idx], 1:8].mean(axis=1)        # high-gamma, this fold's front end

    t_steps = ends[steps] / FS
    tol = int(EVENT_TOL_S * FS)
    gestures = [{"class": g["class"], "onset": round(g["onset"] / FS, 3), "end": round(g["end"] / FS, 3),
                 "cue": round(g["cue"] / FS, 3), "from_glove": g["source"] == "glove"}
                for g in info if t_steps[0] <= g["onset"] / FS <= t_steps[-1]]
    ev_out = []
    for i, c in sorted(events):
        t = ends[steps[i]]
        match = next((g for g in info if g["onset"] - tol <= t <= g["end"]), None)
        ev_out.append({"t": round(t / FS, 3), "class": c, "true": None if match is None else match["class"],
                       "correct": bool(match is not None and match["class"] == c),
                       "delay": None if match is None else round((t - match["onset"]) / FS, 3)})

    cut = rec.cut
    lo = np.percentile(rec.glove[:, :cut], 1, axis=1)
    hi = np.percentile(rec.glove[:, :cut], 99, axis=1)
    t0, t1 = t_steps[0], t_steps[-1]
    tg = np.arange(t0, t1, 1 / GLOVE_HZ)
    glove = np.clip((rec.glove[:, (tg * FS).astype(int)] - lo[:, None]) / (hi - lo)[:, None], 0, 1).T

    data = {
        "meta": {"gesture_model": args.gestures, "finger_model": args.fingers, "step_s": 0.1,
                 "gesture_desc": args.gesture_desc, "finger_desc": args.finger_desc,
                 "glove_hz": GLOVE_HZ, "t0": round(float(t0), 3), "t1": round(float(t1), 3),
                 "classes": ["rest", "fist", "peace", "open"],
                 "fingers": ["thumb", "index", "middle", "ring", "little"],
                 "note": "Out-of-fold, causal predictions over the 72 development trials, replayed at recording speed."},
        "glove": np.round(glove, 3).tolist(),
        "t": np.round(t_steps, 3).tolist(),
        "fingers": np.round(np.clip(Fd, -0.2, 1.2), 3).tolist(),
        "p": np.round(P, 3).tolist(),
        "s": np.round(S, 3).tolist(),
        "true": lab[ends[steps]].tolist(),
        "grid": np.round(np.clip(grid, -3, 3), 2).reshape(len(steps), -1).tolist(),
        "gestures": gestures,
        "events": ev_out,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.DEMO = " + json.dumps(data, separators=(",", ":")) + ";\n")
    n_ok = sum(e["correct"] for e in ev_out)
    print(f"{len(steps)} steps ({t0:.1f}-{t1:.1f} s), {len(gestures)} gestures, {len(ev_out)} events "
          f"({n_ok} correct), {out.stat().st_size / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
