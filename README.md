# ECoG hand decoding, online and asynchronous

Rock-paper-scissors gestures and finger flexion decoded from a 60-electrode ECoG grid over
sensorimotor cortex (the BR41N.IO `ECoG_Handpose.mat` recording, one patient, 90 trials).

The team's offline pipeline ([biomenace/gtech-hackathon-ecog](https://github.com/biomenace/gtech-hackathon-ecog))
classifies a gesture once per trial, at a moment chosen by the cue, with zero-phase filters
that read samples after that moment. A brain-computer interface gets none of that: no
trigger, no future samples, and it has to say "rest" most of the time. This repository
rebuilds the decoder under those conditions. It decodes **every 100 ms**, from **causal**
features only, and decides **by itself** when a gesture starts.

The recording is not included. Download it from
[g.tec](https://www.gtec.at/downloads_QyTs23/Hackathon/ecog-hand-pose.rar) and put
`ECoG_Handpose.mat` one folder above this repository, or pass `--data` to any script.

## Results

### The offline design, step by step into online use

Same features (three cue-relative windows, six high-gamma bands) and same shrinkage LDA:

| Setting | Development CV | Held-out test |
|---|---|---|
| Published: zero-phase filters, shuffled folds, trigger | 0.963 | n/a |
| (a) Trigger kept, causal filters, contiguous folds | 0.944 | 0.944 (17/18) |
| (b) Same model, sliding every 100 ms, no trigger | 0.379 (4-class balanced) | 0.379 |
| (c) Retrained with a rest class, sliding | 0.446 | 0.457 |

Causal filtering costs about 2 points. Losing the trigger is what breaks it. In (b) the
model calls 963 of 1,250 rest steps "open hand": it learned that open hand means "a cue
appeared and nothing happened", which no longer holds without a cue.

### Online decoders, development cross-validation

72 trials, 5 contiguous blocks with 1.5 s purge gaps, pooled out-of-fold, scored against
glove-based movement labels. Event metrics use the same tuning for every row
(cross-fold, at most 2 false detections per minute of rest and 0.8 s median delay).
For the CNN rows, the accuracy columns are means over 3 seeds, and the event columns use
the 3 seeds' averaged probabilities.

| Model | 4-class balanced accuracy | Excluding ±0.25 s around label changes | Rest+open, fist, peace | Gestures detected with the right label | False detections per minute | Median delay |
|---|---|---|---|---|---|---|
| Logistic regression, 1.0 s | 0.622 | 0.670 | 0.778 | 0.64 | 1.9 | 0.80 s |
| Logistic regression, 1.5 s | **0.658** | **0.713** | **0.785** | **0.65** | 2.4 | 0.84 s |
| Shrinkage LDA, 1.5 s | 0.566 | 0.612 | 0.713 | 0.53 | 3.8 | 0.80 s |
| CNN + GRU, multi-task λ=0.5 | 0.642 | 0.688 | 0.764 | 0.53 | 2.4 | 0.77 s |
| CNN + GRU, multi-task λ=1 | 0.630 | 0.673 | 0.741 | 0.58 | 2.4 | 0.65 s |
| CNN + GRU, gesture only | 0.605 | 0.649 | 0.716 | 0.47 | 3.8 | 0.78 s |

Finger flexion, continuous, 100 ms resolution:

| Model | Mean r | Mean R² |
|---|---|---|
| Ridge, 1.0 s | 0.70 | 0.49 |
| Ridge, 1.5 s | 0.70 | 0.50 |
| CNN + GRU, fingers only | **0.80** | **0.62** |

### Held-out test

Trained on the 72 development trials (0-338 s), scored once on the last 18 trials
(338-422 s: 4 fist, 6 peace, 8 open hand).

| Model | 4-class balanced accuracy | Excluding ±0.25 s | Rest+open, fist, peace | Gestures right (of 18) | False detections (33 s of rest) |
|---|---|---|---|---|---|
| Logistic regression, 1.0 s | 0.571 | 0.607 | 0.755 | 13 | 4 |
| Logistic regression, 1.5 s | 0.577 | 0.606 | 0.749 | 10 | 5 |
| CNN + GRU, λ=0.5 (5 seeds) | **0.637 ± 0.026** | **0.684** | **0.769** | 8.4 ± 2.1 | 0–3 |
| Offline design, made asynchronous (c) | 0.457 | 0.481 | 0.587 | 7 | 6 |

The CNN's finger decoding on the held-out trials: mean r = 0.76, R² = 0.39.

The test set holds 18 gestures and 33 s of rest, so one gesture is 5.6 % and one false
detection is about 1.8 per minute. Read the event columns as counts.

### What limits it

- **Open hand cannot be told from rest.** The glove barely moves (open hand is within 0.02
  of the resting posture on every finger), and no channel, band or moment separates open
  hand from the same trial's own pre-cue rest (`open_vs_rest.svg`). Fist and peace are
  separable at 0.96-1.00. Most 4-class errors are open hand called rest.
- **Onset lag.** In the first 0.5 s after movement onset only 14-23 % of fist and peace steps
  are right. Leaving out ±0.25 s around every label change adds about 5 points everywhere.
- **Speed against false alarms.** Shorter delays cost false alarms (`tradeoff.svg`).
- **The CNN classifies steps best but fires fewer clean events.** Its gesture probabilities
  stay near 0.35 during rest where logistic regression's sit near 0.2, so a threshold
  detector needs a stricter setting and misses more gestures.

## Protocol

- **Split.** The first 72 trials are for development and the last 18 are the test set. The
  cut is at 338.2 s, in a rest gap. The test set was scored once (`final.py`).
- **Folds.** 5 contiguous blocks of development trials (15, 15, 14, 14, 14), cut in rest gaps.
  Training steps within 1.5 s of a test block are dropped, because a decision reads up to
  1.5 s of history.
- **Labels.** Per 100 ms step: rest, or the cued gesture from glove movement onset until the
  glove settles after release. Open hand barely moves the glove, so its window is the cue
  plus the median fist/peace delay (0.22 s) to the cue end plus the median settle time (1.06 s).
  **The glove is never a model input.** It provides labels and regression targets only.
- **Front end** (`features.py`). Every filter is forward-only `sosfilt` with its state carried
  between chunks, so 50 ms chunks give the same output as one pass (max difference about
  5e-13, checked by `features.py --check`), at 2-3 ms per chunk. The chain is: 0.5 Hz
  high-pass; bad channels replaced by clean grid neighbours (none were flagged); common
  average; notches at 50 to 250 Hz; band power in beta 13-30 Hz and seven high-gamma bands
  from 55 to 285 Hz; 100 ms means; log; trailing 25 s mean removed; z-score. The output is a
  [8 bands, 10, 6] grid every 100 ms. Everything learned is fitted on training data only.

## Running it

From the repository root, with the recording one folder up (`../ECoG_Handpose.mat`) or
passed with `--data`. You need numpy, scipy and scikit-learn; `torch` for the CNN and
`matplotlib` for the figures. Times are for an 8-core laptop.

```bash
python3 -m ecog_online.phase0                  # verify the recording, glove onsets       1 min
python3 -m ecog_online.features --check        # streaming equals batch                   30 s
python3 -m ecog_online.phase1                  # folds, labels, feature response          30 s
python3 -m ecog_online.linear --task classify  # LDA and logistic regression              10 min
python3 -m ecog_online.linear --task classify --labels glove --window 15 --tag _w15
python3 -m ecog_online.linear --task regress   # ridge to the five fingers                 8 min
python3 -m ecog_online.retune                  # event stage tuned like the CNN's          5 min
python3 -m ecog_online.phase2b                 # transitions, phases, sweep, open vs rest  3 min
python3 -m ecog_online.cnn --variants A0.5 A1 B C --seeds 0 1 2    # CNN + GRU            40 min
python3 -m ecog_online.teammate_async          # the offline design, made asynchronous     5 min
python3 -m ecog_online.eventfix                # cross-fold event tuning, calibration      2 min
python3 -m ecog_online.figures                 # tradeoff.svg, open_vs_rest.svg
python3 -m ecog_online.demo_export             # data for the 3D demo
```

`python3 -m ecog_online.final --i-mean-it` scores the held-out test set. It refuses to run
a second time once `results/online/final.json` exists.

Every script has its configuration at the top, fixed seeds, and writes JSON to
`results/online/`. Figures are drawn only from those files. Out-of-fold predictions are
in `results/online/oof/` and held-out ones in `results/online/final/`. Feature caches
(`results/online/cache/`, about 230 MB) are rebuilt on first use and are not committed.

## The demo

A replay of the decoder at recording speed. It shows two 3D hands (data glove and decoded
fingers), the true and decoded gesture with live class probabilities, every fired
detection with its delay, the 10 × 6 high-gamma grid and a scrolling
timeline. Every value on screen was computed from the signal up to that moment, by a model
that never trained on that stretch of the recording.

**Open it:** double-click `results/online/demo/index.html`, or run

```bash
open results/online/demo/index.html        # macOS
```

It works offline. three.js and the data ship in the same folder; fonts fall back to
system fonts without a connection.

**Controls:**
- Space: play or pause
- ← and →: skip 5 s
- Speed buttons: 0.5× to 4×
- "Next fist / peace / open hand": jump to the next gesture of that type
- Bottom bar: click to seek

The page opens paused just before a correctly detected fist.

The current export replays logistic regression (1.5 s) for gestures and ridge (1.5 s) for fingers
over the development trials. To show other saved runs:

```bash
python3 -m ecog_online.demo_export --gestures linear_logreg_glove \
    --gesture-desc "logistic regression on 1.0 s of band power"
```

## Files

| Path (in `ecog_online/`) | Contents |
|---|---|
| `protocol.py` | split, contiguous folds, purging, glove-based labels |
| `features.py` | the causal streaming front end and its streaming check |
| `common.py` | progress bars, feature cache, windows, fast shrinkage LDA, ridge, smoothing, event detection and scoring |
| `phase0.py`, `phase1.py` | the recording verified; folds, labels and feature response checked |
| `linear.py`, `retune.py` | LDA, logistic regression, ridge; event stage re-tuned |
| `phase2b.py`, `figures.py` | lag and phase analysis, the delay sweep, open vs rest; figures |
| `cnn.py` | CNN + GRU (43.5k parameters), variants A (multi-task), B (fingers, then a classifier), C (gestures) |
| `teammate_async.py` | the offline design, triggered and asynchronous |
| `eventfix.py` | event stage tuned across folds, calibration, seed ensembles |
| `final.py` | the held-out test |
| `demo_export.py`, `results/online/demo/` | the 3D replay |
