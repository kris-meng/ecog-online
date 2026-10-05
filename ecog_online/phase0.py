"""Phase 0: verify the recording's stated facts and measure what the online pipeline needs.

Nothing here is a model. It checks the file, the cue train, the grid layout, channel
quality, line noise and the glove, and it detects movement onsets from the glove so the
cue-to-movement delay is measured rather than assumed. Offline, zero-phase filtering is
allowed here because none of it reaches a decoder's input: the glove only ever becomes
labels and targets.

    python -m ecog_online.phase0 --data ../ECoG_Handpose.mat   -> results/online/phase0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, sosfiltfilt, welch

# --- config ---------------------------------------------------------------------------
FS = 1200
N_ECOG = 60
FINGERS = ["thumb", "index", "middle", "ring", "little"]
SKIP_S = 2.0                 # startup transient excluded from channel statistics
BRIDGE_R = 0.99              # |r| above this on high-passed signal = likely bridged
AMP_FACTOR = 3.0             # std above this multiple of the median = noisy
GLOVE_LP_HZ = 5.0            # glove smoothing before differentiation
ONSET_K = 5.0                # onset threshold = median + K robust sd of pre-cue rest speed
ONSET_MIN_S = 0.05           # ...and speed stays above it this long
SEARCH_S = (0.0, 2.0)        # onset search window relative to the cue; release after cue end
N_DEV_TRIALS = 72            # chronological split: first 72 trials develop, last 18 test

ROOT = Path(__file__).resolve().parents[1]


def channel_to_grid(k):
    """ECoG channel k (1..60) -> (row, col) on the 10 x 6 grid, numbered down columns."""
    return (k - 1) % 10, (k - 1) // 10


def runs(lab):
    lab = np.rint(lab).astype(int)
    b = np.r_[0, np.flatnonzero(np.diff(lab)) + 1, len(lab)]
    return [(int(lab[s]), int(s), int(e)) for s, e in zip(b[:-1], b[1:])]


def neighbour_correlation(r, mapping):
    """Mean correlation of 4-neighbours vs all other pairs under a channel -> grid mapping."""
    pos = np.array([mapping(k) for k in range(1, N_ECOG + 1)])
    d = np.abs(pos[:, None, :] - pos[None, :, :]).sum(-1)
    iu = np.triu_indices(N_ECOG, 1)
    near = d[iu] == 1
    return float(r[iu][near].mean()), float(r[iu][~near].mean())


def line_peaks(f, psd, lo=20, hi=590, ratio=4.0):
    """Frequencies where the PSD exceeds `ratio` x the median of +/-10 Hz around it."""
    out = []
    for i in np.flatnonzero((f >= lo) & (f <= hi)):
        nb = (np.abs(f - f[i]) <= 10) & (np.abs(f - f[i]) > 2)
        if psd[i] > ratio * np.median(psd[nb]) and psd[i] == psd[max(i - 2, 0): i + 3].max():
            out.append({"hz": round(float(f[i]), 2), "x_local": round(float(psd[i] / np.median(psd[nb])), 1)})
    return out


def glove_onsets(glove, cue_starts, cue_ends, norm_lo, norm_hi, n_dev):
    """Movement onset and release per trial from summed normalised flexion speed.

    The threshold is set from the speed in the 0.5 s before each development-set cue,
    not from each trial's own peak: open hand is so close to the resting posture that a
    peak-relative rule fires on noise for most open-hand trials.
    """
    sos = butter(4, GLOVE_LP_HZ, fs=FS, output="sos")
    g = sosfiltfilt(sos, (glove - norm_lo[:, None]) / (norm_hi - norm_lo)[:, None], axis=-1)
    speed = np.abs(np.diff(g, axis=-1, prepend=g[:, :1])).sum(0) * FS     # range units / s
    rest = np.concatenate([speed[s - FS // 2: s] for s in cue_starts[:n_dev]])
    thr = float(np.median(rest) + ONSET_K * 1.4826 * np.median(np.abs(rest - np.median(rest))))
    min_len = int(ONSET_MIN_S * FS)

    def first_run(seg):
        run = np.convolve(seg > thr, np.ones(min_len, int), "valid") == min_len
        return int(np.argmax(run)) if run.any() else None

    out = []
    for s, e in zip(cue_starts, cue_ends):
        a, b = s + int(SEARCH_S[0] * FS), s + int(SEARCH_S[1] * FS)
        on, rel = first_run(speed[a:b]), first_run(speed[e: e + int(1.5 * FS)])
        out.append({"onset": None if on is None else on + a, "release": None if rel is None else rel + e,
                    "peak_speed": float(speed[a:b].max()), "pre_cue_speed": float(speed[s - FS // 2: s].mean())})
    return out, g, speed, thr


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT.parent / "ECoG_Handpose.mat"))
    ap.add_argument("--out", default=str(ROOT / "results" / "online" / "phase0.json"))
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    r = {}

    # 1. the file
    p = Path(args.data)
    r["file"] = {"bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    y = loadmat(args.data)["y"]
    t = y[0]
    r["file"].update(shape=list(y.shape), fs_from_time=float(1 / np.median(np.diff(t))),
                     duration_s=round(y.shape[1] / FS, 2), time_start=float(t[0]))
    ecog, lab, glove = y[1:61].astype(np.float64), y[61], y[62:67].astype(np.float64)
    print(f"file {r['file']}")

    # 2. the cue train
    rr = runs(lab)
    cues = [(c, s, e) for c, s, e in rr if c != 0]
    inner_gaps = [e - s for i, (c, s, e) in enumerate(rr) if c == 0 and 0 < i < len(rr) - 1]
    seq = [c for c, _, _ in cues]
    r["cues"] = {
        "values": sorted(set(np.rint(lab).astype(int).tolist())),
        "n": len(cues), "per_class": {str(k): seq.count(k) for k in (1, 2, 3)},
        "block_s": [round(min(e - s for _, s, e in cues) / FS, 3), round(max(e - s for _, s, e in cues) / FS, 3)],
        "inner_gap_s": [round(min(inner_gaps) / FS, 3), round(max(inner_gaps) / FS, 3)],
        "inner_gap_s_quantiles_5_50_95": [round(float(q) / FS, 3) for q in np.percentile(inner_gaps, [5, 50, 95])],
        "gaps_over_3s": [round(g / FS, 2) for g in inner_gaps if g > 3 * FS],
        "first_cue_s": round(cues[0][1] / FS, 3), "last_cue_end_s": round(cues[-1][2] / FS, 3),
        "sequence": seq,
    }
    dev_end, test_start = cues[N_DEV_TRIALS - 1][2], cues[N_DEV_TRIALS][1]
    r["split"] = {"dev_trials": N_DEV_TRIALS, "test_trials": len(cues) - N_DEV_TRIALS,
                  "cut_sample": int((dev_end + test_start) // 2),
                  "cut_s": round((dev_end + test_start) / 2 / FS, 3),
                  "gap_at_cut_s": round((test_start - dev_end) / FS, 3),
                  "test_per_class": {str(k): seq[N_DEV_TRIALS:].count(k) for k in (1, 2, 3)}}
    print(f"cues {r['cues']}\nsplit {r['split']}")

    # 3. channel quality on the high-passed signal, startup transient excluded
    b, a = butter(2, 0.5, btype="high", fs=FS)
    x = filtfilt(b, a, ecog, axis=-1)[:, int(SKIP_S * FS):]
    sd = x.std(axis=1)
    rc = np.corrcoef(x)
    bridged = [[i + 1, j + 1, round(float(rc[i, j]), 5)]
               for i in range(N_ECOG) for j in range(i + 1, N_ECOG) if abs(rc[i, j]) > BRIDGE_R]
    xh = filtfilt(*butter(4, [70, 170], btype="band", fs=FS), x, axis=-1)
    rh = np.corrcoef(xh)
    bridged_hg = [[i + 1, j + 1, round(float(rh[i, j]), 4)]
                  for i in range(N_ECOG) for j in range(i + 1, N_ECOG) if abs(rh[i, j]) > 0.9]
    med = np.median(sd)
    mad = np.median(np.abs(sd - med))
    r["channels"] = {
        "dc_offset_range": [float(ecog.mean(1).min()), float(ecog.mean(1).max())],
        "std_hp": [round(float(s), 2) for s in sd],
        "std_over_median": [round(float(s / med), 2) for s in sd],
        f"over_{AMP_FACTOR}x_median": [k + 1 for k in np.flatnonzero(sd > AMP_FACTOR * med)],
        "teammate_mad_rule_channels_1based": [k + 1 for k in np.flatnonzero(np.abs(sd - med) / mad > 5)],
        f"pairs_r_over_{BRIDGE_R}_broadband": bridged,
        "pairs_r_over_0.9_high_gamma_70_170": bridged_hg,
        "first_4s_ch1_4_r": np.corrcoef(filtfilt(b, a, ecog[:4, : 4 * FS], axis=-1)).round(5).tolist(),
    }
    # grid: neighbours should correlate more under the documented mapping than a row-wise one
    r["grid"] = {
        "documented_down_columns": neighbour_correlation(rh, channel_to_grid),
        "alternative_across_rows": neighbour_correlation(rh, lambda k: ((k - 1) // 6, (k - 1) % 6)),
        "note": "(mean r of 4-neighbours, mean r of other pairs), high gamma 70-170 Hz",
        "channel_numbers": [[c * 10 + rw + 1 for c in range(6)] for rw in range(10)],
    }
    print(f"channels: >3x median {r['channels'][f'over_{AMP_FACTOR}x_median']}, "
          f"MAD rule {r['channels']['teammate_mad_rule_channels_1based']}, "
          f"bridged broadband {bridged[:12]}{'...' if len(bridged) > 12 else ''}, hg {bridged_hg[:12]}")
    print(f"grid {r['grid']['documented_down_columns']} vs {r['grid']['alternative_across_rows']}")

    # 4. line noise
    f, P = welch(x, fs=FS, nperseg=4 * FS, axis=-1)
    psd = np.median(P, axis=0)
    r["spectrum"] = {"peaks": line_peaks(f, psd),
                     "median_psd_db_at": {str(h): round(float(10 * np.log10(psd[np.argmin(np.abs(f - h))])), 1)
                                          for h in (50, 60, 100, 120, 150, 180, 200, 240, 250, 300, 350, 400, 450, 500, 550)},
                     "freqs": f[f <= 600][::2].round(2).tolist(),
                     "median_psd_db": (10 * np.log10(psd[f <= 600][::2])).round(2).tolist()}
    print(f"line-noise peaks {[pk['hz'] for pk in r['spectrum']['peaks']]}")
    del x, xh

    # 5. glove
    pct = np.percentile(glove, [0, 1, 50, 99, 100], axis=1)
    steps = [int(np.count_nonzero(np.diff(g))) for g in glove]
    r["glove"] = {f: {"min": float(pct[0, i]), "p1": float(pct[1, i]), "median": float(pct[2, i]),
                      "p99": float(pct[3, i]), "max": float(pct[4, i]),
                      "value_changes_per_s": round(steps[i] / (glove.shape[1] / FS), 1),
                      "n_unique": int(len(np.unique(glove[i])))}
                  for i, f in enumerate(FINGERS)}
    print(f"glove {r['glove']}")

    # 6. movement onsets from the glove (normalised on the dev part only)
    dev = slice(0, r["split"]["cut_sample"])
    lo, hi = np.percentile(glove[:, dev], 1, axis=1), np.percentile(glove[:, dev], 99, axis=1)
    ons, g, speed, thr = glove_onsets(glove, [s for _, s, _ in cues], [e for _, _, e in cues], lo, hi, N_DEV_TRIALS)
    delay = np.array([(o["onset"] - s) / FS if o["onset"] is not None else np.nan
                      for o, (_, s, _) in zip(ons, cues)])
    rel = np.array([(o["release"] - e) / FS if o["release"] is not None else np.nan
                    for o, (_, _, e) in zip(ons, cues)])
    # per-class glove posture during the hold (1.0-2.0 s after cue), normalised
    rest_posture = np.mean([g[:, s - FS // 2: s].mean(1) for _, s, _ in cues], 0).round(3).tolist()
    posture = {str(c): np.mean([g[:, s + FS: s + 2 * FS].mean(1) for cc, s, _ in cues if cc == c], 0).round(3).tolist()
               for c in (1, 2, 3)}
    q = lambda v: [round(float(z), 3) for z in np.nanpercentile(v, [0, 5, 25, 50, 75, 95, 100])]
    r["movement"] = {
        "method": f"sum over fingers of |d/dt| of 1st-99th-percentile normalised flexion, "
                  f"{GLOVE_LP_HZ} Hz zero-phase low-pass; onset = first {ONSET_MIN_S*1000:.0f} ms "
                  f"run above median + {ONSET_K} robust sd of dev-set pre-cue speed, "
                  f"in [{SEARCH_S[0]}, {SEARCH_S[1]}] s after the cue",
        "threshold_speed": round(thr, 4),
        "cue_to_onset_s_quantiles_0_5_25_50_75_95_100": q(delay),
        "cue_to_onset_s_by_class_quantiles_5_50_95": {str(c): [round(float(z), 3) for z in np.nanpercentile(delay[np.array(seq) == c], [5, 50, 95])] for c in (1, 2, 3)},
        "no_onset_found": [i + 1 for i, d in enumerate(delay) if np.isnan(d)],
        "no_onset_found_by_class": {str(c): int(np.isnan(delay[np.array(seq) == c]).sum()) for c in (1, 2, 3)},
        "cue_end_to_release_s_quantiles_0_5_25_50_75_95_100": q(rel),
        "per_trial_delay_s": [None if np.isnan(d) else round(float(d), 3) for d in delay],
        "per_trial_release_s": [None if np.isnan(d) else round(float(d), 3) for d in rel],
        "rest_posture_norm": rest_posture,
        "hold_posture_by_class_norm": posture,
    }
    print("movement")
    for k, v in r["movement"].items():
        if not k.startswith("per_trial"):
            print(f"  {k}: {v}")

    Path(args.out).write_text(json.dumps(r, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
