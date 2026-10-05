"""Phase 3: a small CNN + GRU on the causal feature grid, 1.5 s of history per decision.

    input     last 15 steps x [n_bands, 10, 6] grid images (bands are the image channels)
    encoder   shared per step: Conv(n_bands->16) GN ELU Drop2d, Conv(16->32) GN ELU AvgPool(2),
              Linear(480->48) ELU Drop  ->  GRU(48->48), last hidden state
    heads     fingers Linear(48->5) (glove at lag 0, 0-1 on training data)
              gesture Linear(48->4) (rest, fist, peace, open)
    ~43k parameters at 8 bands.

  A  multi-task, loss = MSE(fingers) + lambda * weighted CE(gesture), lambda in {0.5, 1}
  B  fingers only (lambda 0), then a classifier on the decoded fingers of the last 0.5 s:
     nearest template and shrinkage LDA, trained on fingers decoded OUT OF FOLD (3 inner
     folds over the fitting part of the training fold)
  C  gesture head only

Per outer fold the last 20 % of training steps is held back (purged): it early-stops the
network and tunes the event stage under the caps of common.tune_capped. Every fit runs
in its own process with one thread and deterministic torch, so seeds reproduce.

    python -m ecog_online.cnn --variants A0.5 A1 B C --seeds 0 1 2
    progress: results/online/logs/cnn.log  (or the --tag'ed name)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .common import (OUT, Progress, ShrinkLDA, balanced_accuracy, confusion, events_on,
                     fold_features, merged, near_transition, regression_scores,
                     summarise_events, tune_capped, val_split)
from .protocol import FS, final_masks, fold_masks, load, sample_labels

# --- config ---------------------------------------------------------------------------
WINDOW = 15
FOLDS = [0, 1, 2, 3, 4]
VARIANTS = {"A0.5": 0.5, "A1": 1.0, "B": 0.0, "C": None}     # lambda; None = gesture only
HIDDEN = 48
LR, WD, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 1e-3, 64, 100, 10
AUG = {"jitter": 1, "drop_cells": 0.10, "band_scale": 0.10, "noise": 0.05}
CHUNK = 16                     # consecutive decisions per training chunk
CHUNKS_PER_BATCH = 4           # 4 x 16 = 64 windows per batch
B_INNER = 3
B_STEPS = 5                    # decoded-finger history for the variant-B classifier: 0.5 s
WORKERS = 4
FINGERS = ["thumb", "index", "middle", "ring", "little"]


# --- data -----------------------------------------------------------------------------

def fold_data(rec, fold):
    """Features, step ends, labels, normalised glove targets (lag 0) and masks for a fold."""
    feats, ends = fold_features(rec, fold)
    lab, info, _ = sample_labels(rec)
    y = lab[ends]
    g = np.stack([rec.glove[:, max(e - 119, 0): e + 1].mean(1) for e in ends]).astype(np.float32)
    tr, te = final_masks(ends, rec) if fold == "final" else fold_masks(ends, rec, fold)
    valid = np.arange(len(ends)) >= WINDOW - 1
    tr, te = np.flatnonzero(tr & valid), np.flatnonzero(te & valid)
    lo, hi = g[tr].min(0), g[tr].max(0)
    g = (g - lo) / (hi - lo)
    return {"feats": feats, "ends": ends, "y": y, "g": g, "tr": tr, "te": te, "lab": lab, "info": info}


# --- the network ----------------------------------------------------------------------

def build(n_bands, hidden=HIDDEN):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(n_bands, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.ELU(), nn.Dropout2d(0.25),
                nn.Conv2d(16, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ELU(), nn.AvgPool2d(2),
                nn.Flatten(), nn.Linear(32 * 5 * 3, hidden), nn.ELU(), nn.Dropout(0.3))
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.fingers = nn.Linear(hidden, 5)
            self.gesture = nn.Linear(hidden, 4)

        def encode(self, imgs):                    # (N, bands, 10, 6) -> (N, hidden)
            return self.enc(imgs)

        def heads(self, zw):                       # (M, WINDOW, hidden) -> fingers, gesture logits
            _, h = self.gru(zw)
            return self.fingers(h[-1]), self.gesture(h[-1])

        def forward(self, x):                      # (n, T, bands, 10, 6), for completeness
            n, T = x.shape[:2]
            return self.heads(self.encode(x.reshape(n * T, *x.shape[2:])).reshape(n, T, -1))

    return Net()


def n_params(n_bands):
    return sum(p.numel() for p in build(n_bands).parameters())


def train_one(job):
    """One network: fit on job['fit'] steps, early-stop on job['val']; predict job['pred'] steps.

    Training batches are CHUNKS_PER_BATCH chunks of CHUNK consecutive fitting steps, so each
    grid image is encoded once per chunk and the GRU reads the overlapping 15-step windows
    from those encodings. Augmentation is drawn per chunk: an electrode zeroed, a band
    scaled or the window shifted applies to every decision in the chunk. Prediction encodes
    the whole stream once. Runs in a worker process; returns probabilities and finger
    predictions for every index in `pred`, plus the validation curve.
    """
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(job["seed"])
    rng = np.random.default_rng(job["seed"])

    d = job["data"]
    feats = torch.from_numpy(d["feats"])                 # (T, bands, 10, 6)
    y, g = torch.from_numpy(d["y"]).long(), torch.from_numpy(d["g"])
    fit, val = np.sort(job["fit"]), np.sort(job["val"])
    in_fit = np.zeros(len(d["y"]), bool)
    in_fit[fit] = True
    lam = job["lambda"]
    counts = np.bincount(d["y"][fit], minlength=4).astype(np.float64)
    w = len(fit) / (4 * np.maximum(counts, 1))
    w[0] *= job.get("rest_weight", 1.0)            # extra weight on rest, against gesture-ish output at rest
    w_cls = torch.tensor(w, dtype=torch.float32)
    ls = job.get("label_smoothing", 0.0)
    net = build(feats.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    W, C, K = WINDOW, CHUNK, CHUNKS_PER_BATCH

    # chunk starts: C consecutive fitting steps, tiled with a fresh random offset each epoch
    fit_runs = [r for r in np.split(fit, np.flatnonzero(np.diff(fit) != 1) + 1) if len(r) >= C]

    def epoch_chunks():
        starts = []
        for r in fit_runs:
            off = rng.integers(0, C)
            starts += list(r[off: len(r) - C + 1: C])
        starts = np.array(starts)
        return starts[rng.permutation(len(starts))]

    def chunk_batch(starts):
        """Input images (K, C+W-1, bands, 10, 6) and target steps (K*C,) for chunk starts."""
        imgs, targets = [], []
        for s in starts:
            t = np.arange(s, s + C)
            shift = int(rng.integers(-AUG["jitter"], AUG["jitter"] + 1)) if AUG["jitter"] else 0
            if not in_fit[np.clip(t + shift, 0, len(in_fit) - 1)].all():
                shift = 0                       # never let a window end on a non-fitting step
            e0 = s + shift
            imgs.append(feats[e0 - W + 1: e0 + C])
            targets.append(t)
        x = torch.stack(imgs)
        k, L, B = x.shape[:3]
        keep = (torch.rand(k, 1, 1, 10, 6) >= AUG["drop_cells"]).float()
        scale = 1 + AUG["band_scale"] * (2 * torch.rand(k, 1, B, 1, 1) - 1)
        x = x * keep * scale + AUG["noise"] * torch.randn_like(x)
        return x, np.concatenate(targets)

    def windows_of(z):                          # (k, C+W-1, H) -> (k*C, W, H)
        return z.unfold(1, W, 1).permute(0, 1, 3, 2).reshape(-1, W, z.shape[-1])

    def loss_of(fo, go, idx):
        t = torch.from_numpy(np.asarray(idx))
        ce = F.cross_entropy(go, y[t], weight=w_cls, label_smoothing=ls)
        if lam is None:
            return ce
        return F.mse_loss(fo, g[t]) + lam * ce

    offs = torch.arange(-W + 1, 1)

    def stream_outputs(idx):
        """Encode every step once, then run the heads on the windows ending at `idx`."""
        net.eval()
        with torch.no_grad():
            z = torch.cat([net.encode(feats[a: a + 2048]) for a in range(0, len(feats), 2048)])
            fo, go = [], []
            for a in range(0, len(idx), 1024):
                f_, g_ = net.heads(z[torch.from_numpy(np.asarray(idx[a: a + 1024]))[:, None] + offs])
                fo.append(f_)
                go.append(g_)
        return torch.cat(fo), torch.cat(go)

    best, best_state, bad, curve = np.inf, None, 0, []
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        net.train()
        starts = epoch_chunks()
        for a in range(0, len(starts), K):
            x, t = chunk_batch(starts[a: a + K])
            k, L = x.shape[:2]
            z = net.encode(x.reshape(k * L, *x.shape[2:])).reshape(k, L, -1)
            fo, go = net.heads(windows_of(z))
            loss = loss_of(fo, go, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        fo, go = stream_outputs(val)
        with torch.no_grad():
            v = float(loss_of(fo, go, val))
        curve.append(round(v, 5))
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k_: t_.clone() for k_, t_ in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        if job.get("max_epochs") and epoch + 1 >= job["max_epochs"]:
            break
    net.load_state_dict(best_state)
    out = {}
    for name, idx in job["pred"].items():
        fo, go = stream_outputs(np.asarray(idx))
        out[name] = (torch.softmax(go, 1).numpy(), fo.numpy())
    return {"key": job["key"], "out": out, "epochs": len(curve), "best_epoch": int(np.argmin(curve)) + 1,
            "seconds": round(time.time() - t0, 1), "curve": curve}


# --- variant B's top classifier -------------------------------------------------------

def finger_history(Fg, steps):
    """(n, 5) decoded fingers on contiguous `steps` -> (n, B_STEPS*5) last-0.5 s history,
    repeating the first value at the start of each run."""
    out = np.zeros((len(steps), B_STEPS * 5), np.float32)
    for k in range(B_STEPS):
        src = np.arange(len(steps)) - (B_STEPS - 1 - k)
        same_run = np.ones(len(steps), bool)
        for j in range(1, B_STEPS - k):          # stay inside the run of consecutive steps
            prev = np.arange(len(steps)) - j
            same_run &= (prev >= 0) & (steps[np.maximum(prev, 0)] == steps - j)
        src = np.where(same_run & (src >= 0), src, np.arange(len(steps)))
        out[:, k * 5:(k + 1) * 5] = Fg[src]
    return out


class Template:
    """Nearest class centroid in standardised space; soft scores from squared distances."""

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.classes_ = np.unique(y)
        self.C = np.stack([Z[y == c].mean(0) for c in self.classes_])
        self.s2 = np.mean([((Z[y == c] - Z[y == c].mean(0)) ** 2).sum(1).mean() for c in self.classes_])
        return self

    def predict_proba(self, X):
        Z = (X - self.mu) / self.sd
        d2 = ((Z[:, None, :] - self.C[None]) ** 2).sum(-1)
        L = -0.5 * d2 / self.s2
        L -= L.max(1, keepdims=True)
        P = np.exp(L)
        return P / P.sum(1, keepdims=True)


# --- scoring --------------------------------------------------------------------------

def score(P, steps, d, ev):
    """Sample-level and event metrics for probabilities P on test `steps`."""
    y, ends, lab = d["y"][steps], d["ends"], d["lab"]
    S, parts = events_on(P, steps, ends, d["info"], lab, ev["smoother"], ev["param"], ev["thr"], ev["n"])
    near = near_transition(lab, ends)[steps]
    sm = S.argmax(1)
    return {"bal_acc_raw": balanced_accuracy(y, P.argmax(1), range(4)),
            "bal_acc_smoothed": balanced_accuracy(y, sm, range(4)),
            "bal_acc_no_transitions": balanced_accuracy(y[~near], sm[~near], range(4)),
            "bal_acc_merged_3class": balanced_accuracy(merged(y), merged(sm), range(3)),
            "bal_acc_merged_no_transitions": balanced_accuracy(merged(y[~near]), merged(sm[~near]), range(3)),
            "events": summarise_events(parts), "tuned": ev}, parts, S


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "ECoG_Handpose.mat"))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--folds", type=int, nargs="*", default=FOLDS)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--max-epochs", type=int, default=0, help="cap epochs (smoke tests)")
    ap.add_argument("--rest-weight", type=float, default=1.0, help="multiplies the rest class weight")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    (OUT / "oof").mkdir(parents=True, exist_ok=True)

    rec = load(args.data)
    data = {f: fold_data(rec, f) for f in args.folds}
    nb = data[args.folds[0]]["feats"].shape[1]
    print(f"bands {nb}, window {WINDOW} steps, parameters {n_params(nb)}, workers {args.workers}", flush=True)

    # jobs: one outer network per (variant, fold, seed); B adds B_INNER inner networks
    jobs = []
    for f in args.folds:
        d = data[f]
        fit, val = val_split(d["tr"], d["ends"])
        dd = {k: d[k] for k in ("feats", "y", "g")}
        for v in args.variants:
            lam = VARIANTS[v]
            for s in args.seeds:
                jobs.append({"key": (v, f, s, "outer"), "seed": s, "lambda": lam, "data": dd,
                             "fit": fit, "val": val, "pred": {"val": val, "te": d["te"]},
                             "max_epochs": args.max_epochs,
                             "rest_weight": args.rest_weight, "label_smoothing": args.label_smoothing})
                if v == "B":
                    # out-of-fold decoded fingers over the fitting steps, for the top classifier
                    for k, chunk in enumerate(np.array_split(fit, B_INNER)):
                        p = int(1.5 * FS)
                        e = d["ends"]
                        itr = fit[(e[fit] < e[chunk[0]] - p) | (e[fit] > e[chunk[-1]] + p)]
                        ifit, ival = val_split(itr, e)
                        jobs.append({"key": (v, f, s, f"inner{k}"), "seed": s, "lambda": lam, "data": dd,
                                     "fit": ifit, "val": ival, "pred": {"chunk": chunk},
                                     "max_epochs": args.max_epochs})
    name = f"cnn{args.tag}"
    prog = Progress(len(jobs), name)
    res = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(train_one, j) for j in jobs]
        for fu in as_completed(futs):
            r = fu.result()
            res[r["key"]] = r
            prog.step(f"{'/'.join(map(str, r['key']))}: {r['epochs']} epochs (best {r['best_epoch']}), {r['seconds']} s")

    # assemble
    out = {"config": {"window_steps": WINDOW, "bands": int(nb), "parameters": n_params(nb), "hidden": HIDDEN,
                      "lr": LR, "wd": WD, "batch": BATCH, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                      "aug": AUG, "variants": {v: VARIANTS[v] for v in args.variants}, "folds": args.folds,
                      "seeds": args.seeds, "b_inner": B_INNER, "b_steps": B_STEPS, "max_epochs_cap": args.max_epochs,
                      "rest_weight": args.rest_weight, "label_smoothing": args.label_smoothing},
           "results": {}}
    heads = []
    for v in args.variants:
        heads += [(v, "gesture_head")] if v != "B" else [(v, "template"), (v, "lda")]
    for v, head in heads:
        per_seed = []
        for s in args.seeds:
            folds, parts_all, oof = [], [], {"steps": [], "P": [], "S": [], "y": [], "F": [], "G": []}
            for f in args.folds:
                d = data[f]
                fit, val = val_split(d["tr"], d["ends"])
                r = res[(v, f, s, "outer")]
                Pv, Fv = r["out"]["val"]
                Pt, Ft = r["out"]["te"]
                if v == "B":
                    chunks = [res[(v, f, s, f"inner{k}")]["out"]["chunk"][1] for k in range(B_INNER)]
                    Ffit = np.concatenate(chunks)
                    clf = (Template() if head == "template" else ShrinkLDA()).fit(
                        finger_history(Ffit, fit), d["y"][fit])
                    Pv = clf.predict_proba(finger_history(Fv, val))
                    Pt = clf.predict_proba(finger_history(Ft, d["te"]))
                ev = tune_capped(Pv, val, d["ends"], d["info"], d["lab"])
                m, parts, S = score(Pt, d["te"], d, ev)
                if VARIANTS[v] is not None:
                    rr, r2 = regression_scores(d["g"][d["te"]], Ft)
                    m.update(r=rr.round(4).tolist(), r2=r2.round(4).tolist(), mean_r=float(rr.mean()), mean_r2=float(r2.mean()))
                m.update(fold=f, epochs=r["epochs"], best_epoch=r["best_epoch"])
                folds.append(m)
                parts_all += parts
                for k_, a in (("steps", d["te"]), ("P", Pt), ("S", S), ("y", d["y"][d["te"]]), ("F", Ft), ("G", d["g"][d["te"]])):
                    oof[k_].append(a)
            o = {k_: np.concatenate(a) for k_, a in oof.items()}
            d0 = data[args.folds[0]]
            near = near_transition(d0["lab"], d0["ends"])[o["steps"]]
            sm = o["S"].argmax(1)
            pooled = {"bal_acc_raw": balanced_accuracy(o["y"], o["P"].argmax(1), range(4)),
                      "bal_acc_smoothed": balanced_accuracy(o["y"], sm, range(4)),
                      "bal_acc_no_transitions": balanced_accuracy(o["y"][~near], sm[~near], range(4)),
                      "bal_acc_merged_3class": balanced_accuracy(merged(o["y"]), merged(sm), range(3)),
                      "bal_acc_merged_no_transitions": balanced_accuracy(merged(o["y"][~near]), merged(sm[~near]), range(3)),
                      "confusion_smoothed": confusion(o["y"], sm).tolist(),
                      "events": summarise_events(parts_all),
                      "n_folds_cap_met": int(sum(fm["tuned"]["feasible"] for fm in folds))}
            if VARIANTS[v] is not None:
                rr, r2 = regression_scores(o["G"], o["F"])
                pooled.update(r=dict(zip(FINGERS, rr.round(4).tolist())), r2=dict(zip(FINGERS, r2.round(4).tolist())),
                              mean_r=float(rr.mean()), mean_r2=float(r2.mean()))
            np.savez(OUT / "oof" / f"cnn{args.tag}_{v}_{head}_seed{s}.npz", **o)
            per_seed.append({"seed": s, "pooled": pooled, "folds": folds})
        keys = ["bal_acc_raw", "bal_acc_smoothed", "bal_acc_no_transitions", "bal_acc_merged_3class",
                "bal_acc_merged_no_transitions"] + (["mean_r", "mean_r2"] if VARIANTS[v] is not None else [])
        summary = {k: [float(np.mean([p["pooled"][k] for p in per_seed])), float(np.std([p["pooled"][k] for p in per_seed]))]
                   for k in keys}
        for k in ("correct_rate", "false_per_min", "latency_s_median_fist_peace_glove_onset"):
            vals = [p["pooled"]["events"][k] for p in per_seed if p["pooled"]["events"][k] is not None]
            summary[f"events_{k}"] = [float(np.mean(vals)), float(np.std(vals))] if vals else None
        summary["per_fold_bal_acc_smoothed"] = [float(np.mean([fm["bal_acc_smoothed"] for p in per_seed for fm in p["folds"]])),
                                                float(np.std([fm["bal_acc_smoothed"] for p in per_seed for fm in p["folds"]]))]
        summary["cap_met_folds"] = [p["pooled"]["n_folds_cap_met"] for p in per_seed]
        out["results"][f"{v}/{head}"] = {"mean_sd_over_seeds": summary, "seeds": per_seed}
        print(f"{v}/{head}: " + ", ".join(f"{k} {a[0]:.3f}±{a[1]:.3f}" for k, a in summary.items()
                                           if isinstance(a, list) and len(a) == 2 and isinstance(a[0], float))
              + f", cap met {summary['cap_met_folds']}", flush=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
