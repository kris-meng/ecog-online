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

All numbers below come from contiguous-block cross-validation over all 90 trials: 6
blocks of 15 trials, cut in rest gaps, with training steps within 1.5 s of a test block
dropped. Every step is predicted once, by models that never saw its block. Every setting
(windows, bands, model sizes, regularisation, decision-rule grid) was fixed beforehand
on a 5-fold cross-validation of the first 72 trials; nothing was tuned on these results.
CNN figures are over 3 seeds.

### The offline design, step by step into online use

Same features (three cue-relative windows, six high-gamma bands) and same shrinkage LDA:

| Setting | Score |
|---|---|
| The paper's protocol: triggered, shuffled 10-fold, 10 repeats | 0.949 ± 0.012 |
| Triggered, causal filters, contiguous blocks | 0.956 |
| Same model applied every 100 ms, no trigger | 0.378 (4-class balanced) |
| Retrained with a rest class, applied every 100 ms | 0.468 |
| *Chance* | *0.333 triggered (3 classes), 0.25 every 100 ms (4 classes)* |

Causal filters and contiguous blocks cost nothing. Losing the trigger is what breaks it.
Applied every 100 ms, the triggered model calls 1,232 of 1,579 rest steps "open hand": it
learned that open hand means "a cue appeared and nothing happened", which no longer holds
without a cue.

### Decoding every 100 ms

| Model | 4-class balanced accuracy | Excluding ±0.25 s around label changes | Rest+open, fist, peace |
|---|---|---|---|
| Logistic regression, 1.0 s | 0.604 | 0.646 | 0.774 |
| Logistic regression, 1.5 s | 0.626 | 0.672 | 0.779 |
| Shrinkage LDA, 1.5 s | 0.548 | 0.584 | 0.721 |
| CNN + GRU, 1.5 s (per seed) | **0.647 ± 0.006** | **0.692 ± 0.010** | **0.770 ± 0.016** |
| CNN + GRU, 1.5 s (3 seeds averaged) | **0.669** | **0.717** | **0.792** |
| Offline design, retrained with rest | 0.468 | 0.501 | 0.601 |
| *Chance* | *0.25* | *0.25* | *0.333* |

### From the user's side

How each movement is handled, scored per true movement (90 in total, with 2.6 min of rest):

- **Strict:** the first detection fired for the movement has the right label.
- **Settled:** the decoder's top class becomes the right gesture and holds for 0.5 s; time is
  measured from movement onset. **False holds** count a gesture held for 0.5 s during rest.
- **Gated:** the decision rule system. A rest-vs-movement gate (the model's own rest
  probability, smoothed, with hysteresis) opens, the gesture label is committed from the
  evidence gathered since, and the label may change before the gate closes. Its settings
  are tuned per block on the other blocks, for at most 1 false activation per minute of rest.

| Model | Strict | Settled, median time | False holds | Gated: first call right | Final call right | Label switches per movement | False activations | Delay of a right first call |
|---|---|---|---|---|---|---|---|---|
| Logistic regression, 1.0 s | 0.51 | 0.84, 0.38 s | 16.6 /min | 0.59 | 0.63 | 0.10 | 5 (1.9 /min) | 1.14 s |
| Logistic regression, 1.5 s | 0.57 | **0.92**, 0.40 s | 15.5 /min | **0.62** | 0.68 | 0.19 | 6 (2.3 /min) | 1.09 s |
| CNN + GRU (3 seeds averaged) | **0.60** | 0.90, **0.25 s** | 19.0 /min | 0.61 | **0.72** | 0.34 | **4 (1.5 /min)** | **0.89 s** |
| Offline design, retrained with rest | 0.34 | 0.73, 0.64 s | 22.4 /min | | | | | |
| *Chance (CNN output shifted in time)* | *0.10* | *0.43, 0.39 s* | *31.8 /min* | *0.18* | *0.14* | *0.60* | *17 (6.5 /min)* | |

With 90 movements a rate is uncertain by about ±0.05, so differences between the
logistic regression and the CNN in this table are within noise. The false-activation
counts rest on 2.6 min of rest and are rough. The chance row scores the CNN's own output
shifted by 60, 120 and 180 s, which keeps its statistics but breaks its alignment with
the movements (`chance.py`); "settled" is high by chance because over a 3 s movement
some gesture often holds for 0.5 s at random.

### Finger flexion, continuous, every 100 ms

| Model | Mean r | Mean R² |
|---|---|---|
| Ridge, 1.5 s | 0.715 | 0.518 |
| CNN + GRU, 1.5 s | **0.736 ± 0.005** | **0.534 ± 0.005** |
| *Chance (outputs shifted in time)* | *−0.04* | *−0.69* |

### What limits it

- **Open hand cannot be told from rest.** The glove barely moves (open hand is within 0.02
  of the resting posture on every finger), and no channel, band or moment separates open
  hand from the same trial's own pre-cue rest. Fist and peace are separable at 0.96-1.00.
  Most 4-class errors are open hand called rest.
- **Onset lag.** In the first 0.5 s after movement onset only 14-23 % of fist and peace steps
  are right. Leaving out ±0.25 s around every label change adds about 5 points everywhere.
- **Quiet during rest costs time.** The raw output holds a wrong gesture 15-19 times a minute
  during rest. A gate and an evidence-based commit bring false activations to about 2 a
  minute, but the first right call then comes about 1 s after movement onset.
- **The CNN reads the signal best but is restless during rest.** It has the best step
  accuracy, the fastest settling and the best finger decoding. Its gesture probabilities
  stay near 0.35 during rest where logistic regression's sit near 0.2, so for one-shot
  gesture commands it is no better than logistic regression.

## Protocol

- **Choosing settings.** All configurations were chosen on the first 72 trials: 5 contiguous
  blocks (15, 15, 14, 14, 14), cut in rest gaps. The results above then re-run every frozen
  configuration on 6 blocks of 15 over all 90 trials (`allcv.py`).
- **Purge gaps.** Training steps within 1.5 s of a test block are dropped, because a
  decision reads up to 1.5 s of history.
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
# the reported results: every frozen configuration over all 90 trials           ~15 min
python3 -m ecog_online.allcv
python3 -m ecog_online.allcv --only-cnn        # saves the CNN's outputs (for chance.py and the demo)
python3 -m ecog_online.chance                  # chance baselines for the tables            3 min

# how the configurations were chosen, on the first 72 trials
python3 -m ecog_online.phase0                  # verify the recording, glove onsets       1 min
python3 -m ecog_online.features --check        # streaming equals batch                   30 s
python3 -m ecog_online.phase1                  # folds, labels, feature response          30 s
python3 -m ecog_online.linear --task classify  # LDA and logistic regression              10 min
python3 -m ecog_online.linear --task classify --labels glove --window 15 --tag _w15
python3 -m ecog_online.linear --task regress --window 15 --tag _w15   # ridge to the fingers
python3 -m ecog_online.retune                  # event stage tuned like the CNN's          5 min
python3 -m ecog_online.phase2b                 # transitions, phases, sweep, open vs rest  3 min
python3 -m ecog_online.cnn --variants A0.5 A1 B C --seeds 0 1 2    # CNN + GRU            40 min
python3 -m ecog_online.teammate_async          # the offline design, made asynchronous     5 min
python3 -m ecog_online.eventfix                # cross-fold event tuning, calibration      2 min
python3 -m ecog_online.settle                  # time to settle, false holds               1 min
python3 -m ecog_online.gated                   # gated decision rule, first and final call 5 min
python3 -m ecog_online.figures                 # tradeoff.svg, open_vs_rest.svg

# the demo data (needs allcv.py --only-cnn first)
python3 -m ecog_online.demo_export
```

Every script has its configuration at the top and fixed seeds, and writes its results
(JSON, saved predictions, figures) to `results/online/`. Figures are drawn only from those
files. Only the demo is committed; run the scripts to regenerate everything else. Feature
caches (`results/online/cache/`, about 230 MB) are rebuilt on first use.

## The demo

A replay of the decoder at recording speed over all 90 trials. It shows two 3D hands (data
glove and decoded fingers), the true and decoded gesture with live class probabilities,
every fired detection with its delay, the 10 × 6 high-gamma grid and a scrolling
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

The current export replays the CNN + GRU (3 seeds averaged) for both gestures and fingers,
from the 90-trial cross-validation. To show logistic regression and ridge from the
72-trial runs instead:

```bash
python3 -m ecog_online.demo_export --source dev --gestures linear_logreg_glove_w15 \
    --fingers linear_ridge_w15 --gesture-desc "logistic regression on 1.5 s of band power" \
    --finger-desc "Ridge, 1.5 s history"
```

## Files

| Path (in `ecog_online/`) | Contents |
|---|---|
| `protocol.py` | contiguous folds, purging, glove-based labels |
| `features.py` | the causal streaming front end and its streaming check |
| `common.py` | progress bars, feature cache, windows, fast shrinkage LDA, ridge, smoothing, event detection and scoring |
| `allcv.py`, `chance.py` | every frozen configuration over all 90 trials: the reported results; chance baselines |
| `phase0.py`, `phase1.py` | the recording verified; folds, labels and feature response checked |
| `linear.py`, `retune.py` | LDA, logistic regression, ridge; event stage re-tuned |
| `phase2b.py`, `figures.py` | lag and phase analysis, the delay sweep, open vs rest; figures |
| `cnn.py` | CNN + GRU (43.5k parameters), variants A (multi-task), B (fingers, then a classifier), C (gestures) |
| `teammate_async.py` | the offline design, triggered and asynchronous |
| `eventfix.py`, `settle.py`, `gated.py` | event tuning across folds; time to settle; the gated decision rule |
| `demo_export.py`, `results/online/demo/` | the 3D replay |
