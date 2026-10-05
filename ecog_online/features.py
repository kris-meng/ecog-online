"""Causal streaming front end: raw ECoG in 50 ms chunks, a [n_bands, 10, 6] grid every 100 ms.

    raw -> high-pass 0.5 Hz -> bad channels replaced by their clean grid neighbours
        -> common average (or grid Laplacian) -> notches -> optional AR whitening
        -> band-pass per band -> square -> mean over 100 ms -> log
        -> minus trailing 25 s mean -> z-score per band (training statistics)

Every filter is forward-only `sosfilt` (or a FIR) with its state carried between chunks,
so feeding the recording in one piece or in 50 ms pieces gives the same numbers.
`python -m ecog_online.features --check` verifies that.

What is learned (bad channels, whitening coefficients, z-score statistics) comes from
`fit`, which only looks at the samples and steps a training mask allows.

    python -m ecog_online.features --check     -> results/online/features_check.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import butter, sosfilt, sosfilt_zi

from .protocol import FS, START_S, load

# --- config ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    hp_hz: float = 0.5
    reference: str = "car"                       # "car" or "laplacian"
    notches: tuple = (50, 100, 150, 200, 250)    # () for none
    notch_halfwidth: float = 2.0
    whiten: bool = False
    ar_order: int = 10
    bands: tuple = ((13, 30), (55, 85), (85, 115), (115, 145), (155, 185),
                    (185, 215), (215, 245), (255, 285))
    band_order: int = 4                          # butter order per band edge (8th-order band-pass)
    lmp: bool = False                            # add the <5 Hz local motor potential
    lmp_hz: float = 5.0
    step_s: float = 0.1
    chunk_s: float = 0.05
    drift_s: float = 25.0
    bad_mad: float = 5.0                         # std further than this many MADs from the median
    bad_bridge_r: float = 0.99                   # |r| above this between two channels = bridged
    zscore: str = "band"                         # "band" (pooled over channels) or "channel"

    @property
    def step(self):
        return int(round(self.step_s * FS))

    @property
    def n_features(self):
        return len(self.bands) + int(self.lmp)


ROWS, COLS, N_CH = 10, 6, 60
ROOT = Path(__file__).resolve().parents[1]


def to_grid(x):
    """(..., 60) -> (..., 10, 6). Channel k (1-based) sits at row (k-1)%10, column (k-1)//10."""
    return x.reshape(*x.shape[:-1], COLS, ROWS).swapaxes(-1, -2)


def grid_neighbours():
    """4-neighbours of every channel on the grid, as 0-based channel indices."""
    out = []
    for i in range(N_CH):
        r, c = i % ROWS, i // ROWS
        out.append([cc * ROWS + rr for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                    if 0 <= rr < ROWS and 0 <= cc < COLS])
    return out


# --- streaming stages -----------------------------------------------------------------

class Sos:
    """sosfilt along time with carried state; `x0` starts it at steady state on a level."""

    def __init__(self, sos, n_ch, x0=None):
        self.sos = sos
        self.zi = np.zeros((len(sos), n_ch, 2))
        if x0 is not None:
            self.zi = sosfilt_zi(sos)[:, None, :] * np.asarray(x0)[None, :, None]

    def __call__(self, x):
        y, self.zi = sosfilt(self.sos, x, axis=-1, zi=self.zi)
        return y


class Fir:
    """Per-channel FIR y[n] = sum_p a[c, p] x[n-p], history carried between chunks."""

    def __init__(self, a):
        self.a = a
        self.hist = np.zeros((a.shape[0], a.shape[1] - 1))

    def __call__(self, x):
        P = self.a.shape[1] - 1
        xh = np.concatenate([self.hist, x], axis=1)
        n = x.shape[1]
        y = sum(self.a[:, p, None] * xh[:, P - p: P - p + n] for p in range(P + 1))
        self.hist = xh[:, -P:] if P else self.hist
        return y


class Drift:
    """Subtract the trailing mean over `w` steps, the current step included; expanding at first."""

    def __init__(self, w):
        self.w, self.hist = w, None

    def __call__(self, f):
        if len(f) == 0:
            return f
        allf = f if self.hist is None else np.concatenate([self.hist, f])
        h = 0 if self.hist is None else len(self.hist)
        cs = np.concatenate([np.zeros((1,) + f.shape[1:]), np.cumsum(allf, axis=0)])
        j = np.arange(h, len(allf)) + 1
        lo = np.maximum(j - self.w, 0)
        mean = (cs[j] - cs[lo]) / (j - lo).reshape(-1, *([1] * (f.ndim - 1)))
        self.hist = allf[-(self.w - 1):] if self.w > 1 else allf[:0]
        return f - mean


# --- fitted parameters ----------------------------------------------------------------

@dataclass
class Params:
    cfg: Config
    bad: list = field(default_factory=list)       # 0-based channel indices
    spatial: np.ndarray = None                    # (60, 60): interpolation then reference
    ar: np.ndarray = None                         # (60, P+1) whitening FIR, or None
    z_mean: np.ndarray = None                     # (n_features, 1 or 60)
    z_sd: np.ndarray = None


def spatial_matrix(bad, reference):
    """Interpolate bad channels from clean neighbours, then re-reference over clean ones."""
    clean = np.setdiff1d(np.arange(N_CH), bad)
    M = np.eye(N_CH)
    for b in bad:
        nb = [n for n in grid_neighbours()[b] if n not in bad] or list(clean)
        M[b] = 0.0
        M[b, nb] = 1.0 / len(nb)
    if reference == "car":
        R = np.eye(N_CH)
        R[:, clean] -= 1.0 / len(clean)
    elif reference == "laplacian":
        R = np.eye(N_CH)
        for i, nb in enumerate(grid_neighbours()):
            R[i, nb] -= 1.0 / len(nb)
    else:
        raise ValueError(reference)
    return R @ M


def find_bad(x, cfg):
    """Bad channels from training samples of the high-passed signal: amplitude outliers,
    flat channels, and all but the first of every bridged pair."""
    sd = x.std(axis=1)
    med = np.median(sd)
    mad = np.median(np.abs(sd - med)) + 1e-12
    bad = set(np.flatnonzero(np.abs(sd - med) / mad > cfg.bad_mad).tolist())
    bad |= set(np.flatnonzero(sd < 0.1 * med).tolist())
    r = np.corrcoef(x)
    for i in range(N_CH):
        for j in range(i + 1, N_CH):
            if abs(r[i, j]) > cfg.bad_bridge_r and i not in bad:
                bad.add(j)
    return sorted(bad)


def yule_walker(x, order):
    x = x - x.mean()
    n = len(x)
    spec = np.fft.rfft(x, 2 * n)
    r = np.fft.irfft(spec * spec.conj())[: order + 1] / n
    return np.r_[1.0, -solve_toeplitz(r[:order], r[1: order + 1])]


def notch_sos(cfg):
    return np.vstack([butter(2, [f - cfg.notch_halfwidth, f + cfg.notch_halfwidth],
                             btype="bandstop", fs=FS, output="sos") for f in cfg.notches])


# --- the front end --------------------------------------------------------------------

class FrontEnd:
    """Feed raw chunks (60, n); get (steps, n_features, 10, 6) and each step's last sample."""

    def __init__(self, params, x0, start=int(START_S * FS)):
        cfg = self.cfg = params.cfg
        self.p = params
        self.hp = Sos(butter(2, cfg.hp_hz, btype="high", fs=FS, output="sos"), N_CH, x0=x0)
        self.notch = Sos(notch_sos(cfg), N_CH) if cfg.notches else None
        self.fir = Fir(params.ar) if cfg.whiten else None
        self.bands = [Sos(butter(cfg.band_order, b, btype="bandpass", fs=FS, output="sos"), N_CH)
                      for b in cfg.bands]
        self.lmp = Sos(butter(4, cfg.lmp_hz, fs=FS, output="sos"), N_CH) if cfg.lmp else None
        self.pending = np.zeros((cfg.n_features, N_CH, 0))
        self.drift = Drift(int(round(cfg.drift_s / cfg.step_s)))
        self.next_end = start + cfg.step - 1            # last sample of the next step
        self.zscore = params.z_mean is not None

    def process(self, raw):
        x = self.p.spatial @ self.hp(raw)
        parts = []
        y = self.notch(x) if self.notch else x
        y = self.fir(y) if self.fir else y
        for f in self.bands:
            parts.append(f(y) ** 2)
        if self.lmp:
            parts.append(self.lmp(x))
        buf = np.concatenate([self.pending, np.stack(parts)], axis=2)
        k = buf.shape[2] // self.cfg.step
        frames = buf[:, :, : k * self.cfg.step].reshape(*buf.shape[:2], k, self.cfg.step).mean(-1)
        self.pending = buf[:, :, k * self.cfg.step:]
        frames = frames.transpose(2, 0, 1)                # (k, n_features, 60)
        nb = len(self.cfg.bands)
        frames[:, :nb] = np.log(frames[:, :nb] + 1e-20)
        frames = self.drift(frames)
        if self.zscore:
            frames = (frames - self.p.z_mean) / self.p.z_sd
        ends = self.next_end + self.cfg.step * np.arange(k)
        self.next_end += self.cfg.step * k
        return to_grid(frames), ends


def stream(ecog, params, start=int(START_S * FS), stop=None, chunk=None):
    """Run the front end over ecog[:, start:stop]; `chunk` samples at a time (None = one go)."""
    stop = ecog.shape[1] if stop is None else stop
    fe = FrontEnd(params, ecog[:, start], start)
    chunk = chunk or (stop - start)
    out, ends = [], []
    for a in range(start, stop, chunk):
        f, e = fe.process(ecog[:, a: min(a + chunk, stop)])
        out.append(f)
        ends.append(e)
    return np.concatenate(out), np.concatenate(ends)


def fit(ecog, train_samples, cfg, start=int(START_S * FS)):
    """Learn bad channels, whitening and z-score statistics from training samples only.

    `train_samples` is a boolean mask over samples. Returns (params, features, step_ends)
    with the features of the whole stream, so a caller does not run it twice.
    """
    hp = Sos(butter(2, cfg.hp_hz, btype="high", fs=FS, output="sos"), N_CH, x0=ecog[:, start])
    x = hp(ecog[:, start:])
    m = train_samples[start:]
    bad = find_bad(x[:, m], cfg)
    p = Params(cfg, bad, spatial_matrix(bad, cfg.reference))
    if cfg.whiten:
        y = p.spatial @ x
        if cfg.notches:
            y = Sos(notch_sos(cfg), N_CH)(y)
        p.ar = np.stack([yule_walker(y[c, m], cfg.ar_order) for c in range(N_CH)])
    del x
    feats, ends = stream(ecog, p, start)
    tr = train_samples[ends]
    f = feats[tr].reshape(tr.sum(), cfg.n_features, N_CH)
    axes = (0, 2) if cfg.zscore == "band" else (0,)
    shape = (cfg.n_features, N_CH)
    p.z_mean = np.broadcast_to(f.mean(axis=axes, keepdims=True)[0], shape).copy()
    p.z_sd = np.broadcast_to(f.std(axis=axes, keepdims=True)[0] + 1e-12, shape).copy()
    feats = (feats - to_grid(p.z_mean)) / to_grid(p.z_sd)
    return p, feats, ends


# --- the check ------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT.parent / "ECoG_Handpose.mat"))
    ap.add_argument("--out", default=str(ROOT / "results" / "online" / "features_check.json"))
    ap.add_argument("--check", action="store_true", help="streaming == batch, every variant")
    ap.add_argument("--seconds", type=float, default=60.0, help="length of the check segment")
    args = ap.parse_args()

    rec = load(args.data)
    start = int(START_S * FS)
    stop = start + int(args.seconds * FS)
    train = np.zeros(rec.ecog.shape[1], bool)
    train[start: start + (stop - start) // 2] = True       # fit on the first half only
    variants = {
        "default": Config(),
        "laplacian": Config(reference="laplacian"),
        "no_notch": Config(notches=()),
        "whiten+lmp": Config(whiten=True, lmp=True),
    }
    r = {"segment_s": args.seconds, "fit_on_first_half": True, "variants": {}}
    for name, cfg in variants.items():
        e = rec.ecog[:, :stop]
        p, _, _ = fit(e, train[:stop], cfg)
        t0 = time.time()
        batch, be = stream(e, p, start, stop)
        tb = time.time() - t0
        t0 = time.time()
        live, le = stream(e, p, start, stop, chunk=int(cfg.chunk_s * FS))
        tl = time.time() - t0
        n_chunks = int(np.ceil((stop - start) / int(cfg.chunk_s * FS)))
        diff = float(np.max(np.abs(batch - live)))
        r["variants"][name] = {
            "shape": list(batch.shape), "same_step_ends": bool(np.array_equal(be, le)),
            "max_abs_diff": diff, "max_abs_value": float(np.max(np.abs(batch))),
            "bad_channels_1based": [b + 1 for b in p.bad],
            "ms_per_50ms_chunk": round(1000 * tl / n_chunks, 3), "batch_s": round(tb, 2),
            "finite": bool(np.isfinite(batch).all()),
        }
        print(f"{name:11s} {batch.shape} max|batch-stream| = {diff:.2e} "
              f"(values up to {np.max(np.abs(batch)):.1f}), {1000 * tl / n_chunks:.2f} ms per 50 ms chunk, "
              f"bad {[b + 1 for b in p.bad]}", flush=True)
    r["pass"] = all(v["max_abs_diff"] < 1e-8 and v["same_step_ends"] and v["finite"]
                    for v in r["variants"].values())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(r, indent=1) + "\n")
    print(f"pass: {r['pass']}, wrote {args.out}")


if __name__ == "__main__":
    main()
