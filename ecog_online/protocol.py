"""The evaluation protocol: contiguous folds, purging, and per-step labels.

  search   configurations were chosen by cross-validation on the first 72 trials, in 5
           contiguous blocks (15, 15, 14, 14, 14) cut in rest gaps; `rec.cut` marks the end
           of that part (338.2 s). The reported results re-run every frozen configuration
           over all 90 trials in 6 blocks (allcv.py).
  purging  training steps within PURGE_S of a test block are dropped, because a decision
           reads up to 1.5 s of history and a finger target may sit up to 0.3 s ahead.
  labels   per 100 ms step: 0 rest, or the cued gesture from glove movement onset until
           the glove has settled again after release. Open hand barely moves the glove
           (phase0.json), so its onset is the cue plus the median fist/peace delay and its
           end the cue end plus the median release-and-settle time of fist and peace.

The glove only ever becomes labels and targets here. Its normalisation and the onset
threshold use the first 72 trials; a label definition is not a fitted model input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

# --- config ---------------------------------------------------------------------------
FS = 1200
N_SEARCH_TRIALS = 72          # trials the configurations were chosen on
FOLD_SIZES = (15, 15, 14, 14, 14)
PURGE_S = 1.5
START_S = 2.0                # stream starts here: the first second holds a ~1e4 transient
GLOVE_LP_HZ = 5.0
ONSET_K = 5.0                # speed threshold = median + K robust sd of pre-cue rest
MIN_RUN_S = 0.05
SETTLE_RUN_S = 0.1           # glove counts as settled after this long below threshold
CLASSES = ["rest", "fist", "peace", "open"]


@dataclass
class Recording:
    ecog: np.ndarray         # (60, n) float64, raw
    glove: np.ndarray        # (5, n) float64, raw 0..1
    label: np.ndarray        # (n,) int, the cue channel
    cues: list               # [(class, start_sample, end_sample)] for the 90 cues
    cut: int                 # end of the configuration-search part (first 72 trials)


def load(path):
    y = loadmat(path)["y"]
    label = np.rint(y[61]).astype(int)
    b = np.r_[0, np.flatnonzero(np.diff(label)) + 1, len(label)]
    cues = [(int(label[s]), int(s), int(e)) for s, e in zip(b[:-1], b[1:]) if label[s] != 0]
    cut = (cues[N_SEARCH_TRIALS - 1][2] + cues[N_SEARCH_TRIALS][1]) // 2
    return Recording(y[1:61].astype(np.float64), y[62:67].astype(np.float64), label, cues, int(cut))


def _first_run(mask, n):
    run = np.convolve(mask, np.ones(n, int), "valid") == n
    return int(np.argmax(run)) if run.any() else None


def glove_events(rec):
    """Per cue: (movement onset, release onset, settled) in samples; None where not found."""
    dev = slice(0, rec.cut)
    lo = np.percentile(rec.glove[:, dev], 1, axis=1)
    hi = np.percentile(rec.glove[:, dev], 99, axis=1)
    g = sosfiltfilt(butter(4, GLOVE_LP_HZ, fs=FS, output="sos"),
                    (rec.glove - lo[:, None]) / (hi - lo)[:, None], axis=-1)
    speed = np.abs(np.diff(g, axis=-1, prepend=g[:, :1])).sum(0) * FS
    rest = np.concatenate([speed[s - FS // 2: s] for _, s, _ in rec.cues[:N_SEARCH_TRIALS]])
    thr = np.median(rest) + ONSET_K * 1.4826 * np.median(np.abs(rest - np.median(rest)))
    n_on, n_set = int(MIN_RUN_S * FS), int(SETTLE_RUN_S * FS)
    out = []
    for c, s, e in rec.cues:
        on = _first_run(speed[s: s + 2 * FS] > thr, n_on)
        rel = _first_run(speed[e: e + int(1.5 * FS)] > thr, n_on)
        settled = None
        if rel is not None:
            st = _first_run(speed[e + rel: e + rel + 2 * FS] <= thr, n_set)
            settled = None if st is None else e + rel + st
        out.append((None if on is None else s + on, None if rel is None else e + rel, settled))
    return out


def sample_labels(rec):
    """4-class label per sample: gesture from onset to settled, rest elsewhere."""
    ev = glove_events(rec)
    moved = [(c, s, e, o, st) for (c, s, e), (o, _, st) in zip(rec.cues, ev)
             if c in (1, 2) and o is not None and st is not None]
    on_delay = int(np.median([o - s for _, s, _, o, _ in moved]))
    off_delay = int(np.median([st - e for _, _, e, _, st in moved]))
    lab = np.zeros(rec.ecog.shape[1], int)
    info = []
    for (c, s, e), (o, _, st) in zip(rec.cues, ev):
        if c == 3 or o is None or st is None:
            o, st, src = s + on_delay, e + off_delay, "cue"
        else:
            src = "glove"
        lab[o:st] = c
        info.append({"class": c, "cue": s, "onset": o, "end": st, "source": src})
    return lab, info, {"open_onset_delay_s": on_delay / FS, "open_end_after_cue_end_s": off_delay / FS}


def fold_blocks(rec):
    """Sample ranges [a, b) of the 5 contiguous configuration-search blocks, cut mid-gap."""
    bounds = np.cumsum((0,) + FOLD_SIZES)
    edges = [int(START_S * FS)]
    for k in bounds[1:-1]:
        edges.append((rec.cues[k - 1][2] + rec.cues[k][1]) // 2)
    edges.append(rec.cut)
    return list(zip(edges[:-1], edges[1:]))


def fold_masks(step_end, rec, k, purge_s=PURGE_S):
    """Boolean (train, test) masks over steps for configuration-search fold k.

    `step_end` holds each step's last sample. Training steps within `purge_s` of either
    edge of the test block are dropped. Only the first 72 trials take part.
    """
    blocks = fold_blocks(rec)
    a, b = blocks[k]
    p = int(purge_s * FS)
    dev = (step_end >= blocks[0][0]) & (step_end < rec.cut)
    test = (step_end >= a) & (step_end < b)
    near = (step_end >= a - p) & (step_end < b + p)
    return dev & ~near, test


def search_masks(step_end, rec, purge_s=PURGE_S):
    """(train, nothing): every step of the first 72 trials, for a front end fitted on all of
    them (used by phase2b's descriptive open-vs-rest analysis)."""
    p = int(purge_s * FS)
    train = (step_end >= int(START_S * FS)) & (step_end < rec.cut - p)
    return train, np.zeros(len(step_end), bool)
