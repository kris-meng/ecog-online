"""Shared pieces for the online decoders: progress, cached features, windows, models, metrics."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from .features import Config, fit
from .protocol import FS, PURGE_S, final_masks, fold_masks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "online"
CACHE = OUT / "cache"
WINDOW = 10                    # steps of history per decision: 1.0 s
EVENT_TOL_S = 0.5              # an event this long before a movement onset still counts for it
REFRACTORY = 5                 # steps of predicted rest needed before the next event


# --- progress -------------------------------------------------------------------------

class Progress:
    """One line per finished unit: bar, count, elapsed, ETA, what just finished."""

    def __init__(self, total, name, width=28):
        self.total, self.name, self.width = total, name, width
        self.done, self.t0 = 0, time.time()

    def step(self, msg=""):
        self.done += 1
        el = time.time() - self.t0
        eta = el / self.done * (self.total - self.done)
        fill = int(self.width * self.done / self.total)
        bar = "#" * fill + "-" * (self.width - fill)
        print(f"{self.name} [{bar}] {self.done}/{self.total} {100 * self.done / self.total:3.0f}%  "
              f"{_fmt(el)} elapsed, ETA {_fmt(eta)}  | {msg}", flush=True)


def _fmt(s):
    return f"{int(s // 60)}m{int(s % 60):02d}s"


# --- features -------------------------------------------------------------------------

def cfg_key(cfg):
    return hashlib.sha1(repr(cfg).encode()).hexdigest()[:10]


def fold_features(rec, fold, cfg=Config()):
    """Front end fitted on the training samples of `fold` (0-4 or "final"), cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{cfg_key(cfg)}_{fold}.npz"
    if path.exists():
        d = np.load(path)
        return d["feats"], d["ends"]
    samples = np.arange(rec.ecog.shape[1])
    tr, _ = final_masks(samples, rec) if fold == "final" else fold_masks(samples, rec, fold)
    _, feats, ends = fit(rec.ecog, tr, cfg)
    feats = feats.astype(np.float32)
    np.savez(path, feats=feats, ends=ends, cfg=repr(cfg))
    return feats, ends


def windows(feats, window=WINDOW):
    """(T, ...) per-step features -> (T, window * features): the last `window` steps, oldest
    first. Rows before the first full window are zero and flagged invalid."""
    T = len(feats)
    f = feats.reshape(T, -1)
    d = f.shape[1]
    out = np.zeros((T, window * d), np.float32)
    for k in range(window):
        out[window - 1:, k * d:(k + 1) * d] = f[k: T - window + 1 + k]
    return out, np.arange(T) >= window - 1


def inner_splits(idx, ends, n=4, purge_s=PURGE_S):
    """Contiguous inner folds over training steps `idx`, purged like the outer ones."""
    idx = np.sort(idx)
    p = int(purge_s * FS)
    for c in np.array_split(idx, n):
        near = (ends[idx] >= ends[c[0]] - p) & (ends[idx] <= ends[c[-1]] + p)
        yield idx[~near], c


# --- models ---------------------------------------------------------------------------

class ShrinkLDA:
    """Shrinkage LDA, Ledoit-Wolf on standardised features, equal class priors.

    The same estimator as sklearn's `solver="lsqr", shrinkage="auto"` (which also shrinks
    the standardised covariance), solved through the n x n Gram matrix with the Woodbury
    identity. With 4800 features and ~2600 steps that is ~30x faster than forming the
    4800 x 4800 covariance. `check_lda.py`-style agreement is measured in linear.py.
    """

    def fit(self, X, y):
        X = np.asarray(X, np.float64)
        self.classes_ = np.unique(y)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12
        Z = (X - self.mu_) / self.sd_
        M = np.stack([Z[y == c].mean(0) for c in self.classes_])
        Xc = Z - M[np.searchsorted(self.classes_, y)]
        n, d = Xc.shape
        G = Xc @ Xc.T
        # Ledoit-Wolf shrinkage intensity, as sklearn.covariance.ledoit_wolf_shrinkage
        x2 = (Xc ** 2).sum(1)
        tr = x2.sum() / n
        mu = tr / d
        delta_ = (G ** 2).sum() / n ** 2
        beta_ = (x2 ** 2).sum()
        beta = (beta_ / n - delta_) / (d * n)
        delta = (delta_ - 2 * mu * tr + d * mu ** 2) / d
        s = 1.0 if delta == 0 else min(beta, delta) / delta
        self.shrinkage_ = float(s)
        # cov = (1-s) Xc'Xc/n + s mu I  ->  cov^-1 B by Woodbury
        c, a = s * mu, (1 - s) / n
        B = M.T
        inner = np.linalg.solve(c / a * np.eye(n) + G, Xc @ B) if a > 0 else 0
        W = (B - Xc.T @ inner) / c
        self.coef_ = W.T
        self.intercept_ = -0.5 * np.sum(M * self.coef_, axis=1)
        return self

    def decision_function(self, X):
        Z = (np.asarray(X, np.float64) - self.mu_) / self.sd_
        return Z @ self.coef_.T + self.intercept_

    def predict_proba(self, X):
        D = self.decision_function(X)
        D -= D.max(1, keepdims=True)
        P = np.exp(D)
        return P / P.sum(1, keepdims=True)


def make_logreg(C=0.01):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=500, class_weight="balanced"))


def ridge_path(Xtr, Ytr, Xte, alphas):
    """Ridge predictions for every alpha from one eigendecomposition of the Gram matrix.

    Returns (len(alphas), n_test, n_targets). Features are standardised and targets
    centred on the training rows.
    """
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    A = (np.asarray(Xtr, np.float64) - mu) / sd
    T = (np.asarray(Xte, np.float64) - mu) / sd
    ym = Ytr.mean(0)
    ev, Q = np.linalg.eigh(A @ A.T)
    K = T @ A.T @ Q
    QtY = Q.T @ (Ytr - ym)
    return np.stack([K @ (QtY / (ev + a)[:, None]) + ym for a in alphas])


# --- smoothing and events -------------------------------------------------------------

def runs(idx):
    """Split sorted step indices into contiguous runs."""
    cut = np.flatnonzero(np.diff(idx) != 1) + 1
    return np.split(idx, cut)


def smooth(P, kind, param):
    """Forward-only smoothing of class probabilities over one contiguous run.

    ema  s_t = param * s_{t-1} + (1 - param) * p_t
    hmm  forward filter, states rest/fist/peace/open, stay probability `param`,
         rest <-> gesture transitions only; emissions are the classifier's posteriors
         (trained with balanced classes, so its priors are flat).
    """
    if kind == "none":
        return P
    S = np.empty_like(P)
    if kind == "ema":
        s = P[0]
        for t in range(len(P)):
            s = param * s + (1 - param) * P[t] if t else P[0]
            S[t] = s
        return S
    K = P.shape[1]
    Tm = np.zeros((K, K))
    Tm[0, 0] = param
    Tm[0, 1:] = (1 - param) / (K - 1)
    for k in range(1, K):
        Tm[k, k], Tm[k, 0] = param, 1 - param
    a = np.full(K, 1.0 / K)
    for t in range(len(P)):
        a = (a @ Tm) * P[t]
        a /= a.sum()
        S[t] = a
    return S


def detect(S, thr, n):
    """Events on one run: a gesture whose smoothed probability stays >= thr for n steps.
    After an event, the next needs REFRACTORY steps where rest is the most likely class."""
    events, count, last, armed, rest_run = [], 0, -1, True, 0
    g = S[:, 1:].argmax(1) + 1
    pg = S[np.arange(len(S)), g]
    for t in range(len(S)):
        if not armed:
            rest_run = rest_run + 1 if S[t].argmax() == 0 else 0
            if rest_run >= REFRACTORY:
                armed, count = True, 0
            continue
        if pg[t] >= thr and g[t] == last:
            count += 1
        elif pg[t] >= thr:
            count, last = 1, g[t]
        else:
            count, last = 0, -1
        if count >= n:
            events.append((t, int(g[t])))
            armed, rest_run, count, last = False, 0, 0, -1
    return events


def event_metrics(events, steps, ends, truth, lab_steps):
    """Score events fired on `steps` (global step indices) against true gesture segments.

    truth: [{"class", "onset", "end"}] in samples. A true gesture is scored if its onset lies
    inside the evaluated span; it is detected by the first event in [onset - tol, end], and
    correct if that event's class matches. Events matching no gesture window are false
    detections, counted per minute of rest-labelled steps.
    """
    t_ev = [(int(ends[steps[i]]), c) for i, c in events]
    lo, hi = ends[steps[0]], ends[steps[-1]]
    tol = int(EVENT_TOL_S * FS)
    used = set()
    per = []
    for g in truth:
        if not (lo <= g["onset"] <= hi):
            continue
        hit = [(j, t, c) for j, (t, c) in enumerate(t_ev) if g["onset"] - tol <= t <= g["end"]]
        used |= {j for j, _, _ in hit}
        if hit:
            _, t, c = hit[0]
            per.append({"class": g["class"], "detected": True, "correct": c == g["class"],
                        "latency_s": (t - g["onset"]) / FS})
        else:
            per.append({"class": g["class"], "detected": False, "correct": False, "latency_s": None})
    in_window = set()
    for j, (t, c) in enumerate(t_ev):
        if any(g["onset"] - tol <= t <= g["end"] for g in truth):
            in_window.add(j)
    false = len(t_ev) - len(in_window)
    rest_min = float((lab_steps == 0).sum()) * 0.1 / 60
    return {"per_gesture": per, "n_false": false, "rest_min": rest_min, "n_events": len(t_ev)}


def summarise_events(parts):
    """Pool event_metrics over runs or folds."""
    per = [p for part in parts for p in part["per_gesture"]]
    false = sum(p["n_false"] for p in parts)
    rest = sum(p["rest_min"] for p in parts)
    out = {"n_gestures": len(per), "correct_rate": _rate(per, "correct"),
           "detected_rate": _rate(per, "detected"), "n_false": false,
           "rest_min": round(rest, 3), "false_per_min": false / rest if rest else float("nan")}
    for c, name in ((1, "fist"), (2, "peace"), (3, "open")):
        out[f"correct_rate_{name}"] = _rate([p for p in per if p["class"] == c], "correct")
    lat = [p["latency_s"] for p in per if p["correct"]]
    lat_fp = [p["latency_s"] for p in per if p["correct"] and p["class"] in (1, 2)]
    out["latency_s_median"] = float(np.median(lat)) if lat else None
    out["latency_s_median_fist_peace_glove_onset"] = float(np.median(lat_fp)) if lat_fp else None
    out["correct_rate_fist_peace"] = _rate([p for p in per if p["class"] in (1, 2)], "correct")
    return out


def _rate(per, key):
    return float(np.mean([p[key] for p in per])) if per else None


# --- sample-wise metrics --------------------------------------------------------------

def balanced_accuracy(y, p, labels):
    rec = [np.mean(p[y == c] == c) for c in labels if np.any(y == c)]
    return float(np.mean(rec))


def confusion(y, p, K=4):
    cm = np.zeros((K, K), int)
    np.add.at(cm, (y, p), 1)
    return cm


def merged(y):
    """rest + open -> 0, fist 1, peace 2: the view in which open hand is not asked of the data."""
    m = y.copy()
    m[m == 3] = 0
    return m


def regression_scores(Y, Yhat):
    r = [float(np.corrcoef(Y[:, i], Yhat[:, i])[0, 1]) if Y[:, i].std() > 1e-9 and Yhat[:, i].std() > 1e-9
         else 0.0 for i in range(Y.shape[1])]
    ss_res = ((Y - Yhat) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    return np.array(r), 1 - ss_res / ss_tot


# --- the event stage, tuned on a held-back block with a delay cap -----------------------

EVENT_SMOOTHERS = [("none", 0.0), ("ema", 0.3), ("ema", 0.5), ("ema", 0.7), ("ema", 0.85),
                   ("hmm", 0.9), ("hmm", 0.95), ("hmm", 0.98)]
EVENT_THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
EVENT_N = [1, 2, 3, 5]
FP_CAP = 2.0                   # false detections per minute of rest
DELAY_CAP = 0.8                # median delay after glove onset, fist and peace, seconds
VAL_FRAC = 0.2


def val_split(idx, ends, frac=VAL_FRAC, purge_s=PURGE_S):
    """(fit, val): the last `frac` of training steps in time is held back; fit steps
    within `purge_s` before it are dropped."""
    idx = np.sort(idx)
    k = int(round(len(idx) * (1 - frac)))
    val = idx[k:]
    fit = idx[:k]
    return fit[ends[fit] < ends[val[0]] - int(purge_s * FS)], val


def events_on(P, steps, ends, truth, lab, kind, param, thr, n):
    """Smooth and detect on each contiguous run of `steps`. Returns (smoothed, parts)."""
    S = np.empty_like(P)
    pos = {s: i for i, s in enumerate(steps)}
    parts = []
    for r in runs(steps):
        i = np.array([pos[s] for s in r])
        S[i] = smooth(P[i], kind, param)
        parts.append(event_metrics(detect(S[i], thr, n), r, ends, truth, lab[ends[r]]))
    return S, parts


def tune_capped(P, steps, ends, truth, lab, fp_cap=FP_CAP, delay_cap=DELAY_CAP):
    """Best event setting on held-back predictions: most correct detections with at most
    `fp_cap` false per minute and at most `delay_cap` s median delay. If nothing meets
    both, the setting closest to them (summed relative excess), flagged `feasible: False`."""
    import itertools
    best, best_key = None, None
    for (kind, param) in EVENT_SMOOTHERS:
        S, _ = events_on(P, steps, ends, truth, lab, kind, param, 2.0, 1)    # smoothing only
        pos = {s: i for i, s in enumerate(steps)}
        rs = [np.array([pos[s] for s in r]) for r in runs(steps)]
        for thr, n in itertools.product(EVENT_THRESHOLDS, EVENT_N):
            parts = [event_metrics(detect(S[i], thr, n), steps[i], ends, truth, lab[ends[steps[i]]]) for i in rs]
            s = summarise_events(parts)
            lat = s["latency_s_median_fist_peace_glove_onset"]
            lat = np.inf if lat is None else lat
            fp = s["false_per_min"]
            ok = fp <= fp_cap and lat <= delay_cap
            excess = max(0.0, lat - delay_cap) / delay_cap + max(0.0, fp - fp_cap) / fp_cap
            key = (1, s["correct_rate"] or 0.0, -fp) if ok else (0, -excess, s["correct_rate"] or 0.0)
            if best_key is None or key > best_key:
                best_key = key
                best = {"smoother": kind, "param": param, "thr": thr, "n": n, "feasible": ok,
                        "val_correct_rate": s["correct_rate"], "val_false_per_min": fp,
                        "val_delay_s": None if np.isinf(lat) else lat}
    return best


def near_transition(lab, ends, width_s=0.25):
    """Steps whose end lies within `width_s` of any sample-level label change."""
    change = np.flatnonzero(np.diff(lab)) + 1
    near = np.zeros(len(ends), bool)
    w = int(width_s * FS)
    idx = np.searchsorted(change, ends)
    for k in (idx - 1, idx):
        ok = (k >= 0) & (k < len(change))
        near[ok] |= np.abs(ends[ok] - change[k[ok]]) <= w
    return near
