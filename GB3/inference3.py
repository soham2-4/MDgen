#!/usr/bin/env python3
"""
inference3.py - inference2.py + state-dependent, correlated, bounded rollout noise.

Changes vs inference2.py:
  a. Residuals are measured over the WHOLE trajectory (cached teacher forcing),
     not just the first diag_len frames after the prefix, which are all folded.
  b. Noise is state-dependent: residual quantiles are binned by predicted value,
     so the unfolded basin gets its own (wider, skewed) noise.
  c. Optional AR(1) temporal correlation (Gaussian copula) so that noise survives
     the rolling-mean smoothing used for the FES.
  d. Optional hard bounds (native contacts cannot exceed a ceiling).
  e. Positional-encoding capacity check also covers --full-rollout.


Replaces inference1.py. Nothing about training or data generation is changed,
so the known upstream problems (identical synthetic trajectories, MSE training
-> conditional-mean predictor) are NOT fixed here. What is fixed:

  1. No label leakage. inference1.py fed the shifted ground truth to the decoder
     with tgt_mask=None, so position t could read y[t] and the "prediction" was a
     lag-1 copy of the answer. Here the decoder is fed only its own samples, with
     the causal mask on, exactly as at training time.

  2. Optional stochastic rollout. An MSE-trained model outputs E[y_t | y_<t], which
     has lower variance than y_t. Injecting noise with the model's own one-step
     residual scale partially restores the width of the distribution. This is a
     patch, not a fix: it assumes the residual is homoscedastic and white, which
     it generally is not.

  3. FES from an ensemble of rollouts, with block error bars and a stationarity
     (drift) check, instead of from a single trace.

  4. Diagnostics that tell you whether the model has collapsed to the identity
     map y_hat[t] ~= y[t-1], which is the failure mode teacher forcing rewards.

Usage:
    python inference2.py --n-rollouts 32 --rollout-len 2000 --noise-mode residual
    python inference2.py --noise-mode none          # deterministic, for comparison
"""

from __future__ import annotations

import argparse
import os

# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")  # choose GPUs with the CUDA_VISIBLE_DEVICES environment variable instead

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gb3_trans1 import TrajectoryTransformer, causal_mask

KB = 0.0083144621  # kJ / (mol K)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_model(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TrajectoryTransformer(
        input_len=ckpt["input_len"],
        output_len=ckpt["output_len"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()  # matters: dropout must be off during rollout
    return model, ckpt


def load_reference(data_dir: str, index: int) -> np.ndarray:
    """Prefer the original (un-noised) series; fall back to a traj_*.npy."""
    for name in ("N_Native_Contacts.npy", "Q_Native.npy"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            print(f"Reference series: {p}")
            return np.load(p).astype(np.float64).reshape(-1)
    p = os.path.join(data_dir, f"traj_{index:03d}.npy")
    print(f"Reference series: {p}")
    return np.load(p).astype(np.float64).reshape(-1)


def check_synthetic_diversity(data_dir: str, n_check: int = 8) -> None:
    """Report whether the synthetic trajectories are actually distinct."""
    import glob

    files = sorted(glob.glob(os.path.join(data_dir, "traj_*.npy")))[:n_check]
    if len(files) < 2:
        return
    ref = np.load(files[0])
    identical = sum(np.array_equal(ref, np.load(f)) for f in files[1:])
    print(f"\n[data check] {identical}/{len(files) - 1} of the first {len(files)} "
          f"trajectories are bit-identical to traj_000.")
    if identical == len(files) - 1:
        print("[data check] The augmentation noise is being rounded away. Your "
              "'ensemble' is one trajectory repeated; train/val split is meaningless.")


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
@torch.no_grad()
def diagnostics(model, x, y_norm, input_len, device, diag_len):
    """
    Teacher-forced pass WITH the causal mask (as in training) over a prefix.
    Returns the one-step residual std in normalized units, and prints tests for
    the identity-map failure mode.
    """
    # diag_len counts frames AFTER the prefix, so the window must extend past
    # input_len. Getting this wrong silently produced empty slices and NaNs.
    T = int(min(y_norm.shape[1], input_len + diag_len))
    n_eval = T - input_len
    if n_eval < 100:
        raise SystemExit(
            f"Diagnostic window too small: input_len={input_len}, sequence "
            f"length={y_norm.shape[1]}, usable frames={n_eval}. "
            f"Lower --diag-len or use a longer reference trajectory."
        )
    y = y_norm[:, :T, :]
    dec = torch.zeros_like(y)
    dec[:, 1:, :] = y[:, :-1, :]
    pred = model(x, dec, tgt_mask=causal_mask(T, device))

    p = pred[0, input_len:, 0].cpu().numpy().astype(np.float64)
    t = y[0, input_len:, 0].cpu().numpy().astype(np.float64)
    lag1 = y[0, input_len - 1 : T - 1, 0].cpu().numpy().astype(np.float64)

    resid = t - p
    sigma = float(resid.std())

    mse_model = float(np.mean((t - p) ** 2))
    mse_lag1 = float(np.mean((t - lag1) ** 2))
    corr_lag1 = float(np.corrcoef(p, lag1)[0, 1])
    ratio = float(p.std() / t.std())

    print("\n" + "=" * 62)
    print("DIAGNOSTICS  (teacher-forced, causal mask on, normalized units)")
    print("=" * 62)
    print(f"  frames used                     : {T - input_len}")
    print(f"  one-step residual std           : {sigma:.5f}")
    print(f"  model MSE                       : {mse_model:.6f}")
    print(f"  lag-1 copy baseline MSE         : {mse_lag1:.6f}")
    print(f"  corr(prediction, y[t-1])        : {corr_lag1:.4f}")
    print(f"  std(prediction) / std(truth)    : {ratio:.4f}")
    if corr_lag1 > 0.98 and mse_model > 0.5 * mse_lag1:
        print("\n  -> The model is essentially the identity map y_hat[t] = y[t-1].")
        print("     It has learned a smoother, not dynamics. A free-running rollout")
        print("     will be a random walk and will drift. Fix training, not inference.")
    elif ratio < 0.9:
        print("\n  -> Predictions are variance-compressed (expected for MSE training).")
        print("     Basins in the FES will be too narrow and the tails too shallow.")
    print("=" * 62)
    return sigma


# ----------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------
@torch.no_grad()
def rollout(model, x, prefix_norm, n_steps, sigma, device, generator, noise_model=None):
    """
    Free-running autoregressive generation.

    Training convention: decoder_input[t] = y[t-1], decoder_input[0] = 0, and
    pred[t] approximates y[t]. So to generate y[k] we feed
    [0, y[0], ..., y[k-1]] and read the last output position.

    The decoder is never shown a ground-truth value beyond the prefix.
    """
    dec = torch.cat(
        [torch.zeros(1, 1, 1, device=device, dtype=prefix_norm.dtype), prefix_norm],
        dim=1,
    )
    if noise_model is not None:
        noise_model.reset()
    out = np.empty(n_steps, dtype=np.float64)
    for i in range(n_steps):
        mask = causal_mask(dec.size(1), device)
        nxt = model(x, dec, tgt_mask=mask)[:, -1:, :]
        if noise_model is not None:
            nxt = noise_model.apply(nxt)
        elif sigma > 0.0:
            noise = torch.randn(nxt.shape, device=device, generator=generator)
            nxt = nxt + sigma * noise
        out[i] = float(nxt.item())
        dec = torch.cat([dec, nxt], dim=1)
    return out


# ----------------------------------------------------------------------
# Incremental (cached) decoding
# ----------------------------------------------------------------------
# The naive rollout re-runs the whole decoder every step: O(L^2) per step,
# O(L^3) overall, and the dense -inf mask blocks memory-efficient attention.
# At L = 50000 the mask alone is 10 GB.
#
# With causal masking, the representation at position t never changes as later
# tokens are appended. So we cache each layer's INPUT sequence and attend the
# single new token against it. Per-step cost becomes O(L); cache memory is
# L * d_model * n_layers * 4 bytes ~ 51 MB at L = 50000.
def _embed_dec(model, tok: torch.Tensor, pos: int) -> torch.Tensor:
    """tok: [B, S, 1] raw values -> [B, S, D] with absolute positions pos..pos+S-1."""
    h = model.input_proj(tok)
    return h + model.pos_decoder.pe[:, pos : pos + h.size(1), :]


@torch.no_grad()
def prefill(model, memory, dec_tokens, device):
    """Run the true prefix once, storing each layer's input sequence."""
    h = _embed_dec(model, dec_tokens, 0)
    mask = causal_mask(h.size(1), device)
    caches = []
    for layer in model.decoder.layers:
        caches.append(h)
        h = layer(h, memory, tgt_mask=mask)
    return caches, h[:, -1:, :]


@torch.no_grad()
def cached_step(model, memory, caches, tok, pos):
    """One token through the decoder using the caches. Post-norm layout only."""
    h = _embed_dec(model, tok, pos)
    for i, layer in enumerate(model.decoder.layers):
        caches[i] = torch.cat([caches[i], h], dim=1)
        c = caches[i]
        sa, _ = layer.self_attn(h, c, c, need_weights=False)
        h = layer.norm1(h + sa)
        ca, _ = layer.multihead_attn(h, memory, memory, need_weights=False)
        h = layer.norm2(h + ca)
        ff = layer.linear2(layer.activation(layer.linear1(h)))
        h = layer.norm3(h + ff)
    return h


@torch.no_grad()
def cached_rollout(model, x, prefix_norm, n_steps, sigma, device, generator,
                   progress_every: int = 5000, noise_model=None):
    for layer in model.decoder.layers:
        if getattr(layer, "norm_first", False):
            raise SystemExit("Cached decoding assumes post-norm layers.")
    if model.decoder.norm is not None:
        raise SystemExit("Cached decoding assumes no final decoder norm.")

    memory = model.encoder(model.pos_encoder(model.input_proj(x)))
    dec = torch.cat(
        [torch.zeros(1, 1, 1, device=device, dtype=prefix_norm.dtype), prefix_norm],
        dim=1,
    )
    caches, h_last = prefill(model, memory, dec, device)
    pos = dec.size(1)
    nxt = model.output_proj(h_last)

    out = np.empty(n_steps, dtype=np.float64)
    if noise_model is not None:
        noise_model.reset()
    for i in range(n_steps):
        if noise_model is not None:
            nxt = noise_model.apply(nxt)
        elif sigma > 0.0:
            nxt = nxt + sigma * torch.randn(nxt.shape, device=device, generator=generator)
        out[i] = float(nxt.item())
        h = cached_step(model, memory, caches, nxt, pos)
        pos += 1
        nxt = model.output_proj(h)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    step {i + 1}/{n_steps}  value {out[i] :.1f}", flush=True)
    return out


@torch.no_grad()
def teacher_forced_cached(model, x, y_norm, input_len, device, max_frames=None,
                          progress_every: int = 20000):
    """
    One-step predictions over the whole trajectory, feeding the TRUE value back
    each step (teacher forcing), using the O(L) cache. Same convention as
    cached_rollout: pred[k] approximates y[k] given y[:k].

    inference2.diagnostics() only looked at frames input_len..input_len+diag_len,
    which for GB3 are all in the folded basin. The residual scale there says
    nothing about the unfolded basin, which is exactly where the FES disagrees.
    """
    memory = model.encoder(model.pos_encoder(model.input_proj(x)))
    prefix = y_norm[:, :input_len, :]
    dec = torch.cat([torch.zeros(1, 1, 1, device=device, dtype=prefix.dtype), prefix], dim=1)
    caches, h_last = prefill(model, memory, dec, device)
    pos = dec.size(1)
    nxt = model.output_proj(h_last)

    T = y_norm.shape[1]
    end = T if max_frames is None else min(T, input_len + max_frames)
    preds = np.empty(end - input_len, dtype=np.float64)
    for j, k in enumerate(range(input_len, end)):
        preds[j] = float(nxt.item())
        true_tok = y_norm[:, k:k + 1, :]
        h = cached_step(model, memory, caches, true_tok, pos)
        pos += 1
        nxt = model.output_proj(h)
        if progress_every and (j + 1) % progress_every == 0:
            print(f"    teacher-forced step {j + 1}/{end - input_len}", flush=True)
    truth = y_norm[0, input_len:end, 0].cpu().numpy().astype(np.float64)
    return preds, truth


class StateNoise:
    """
    State-dependent, optionally time-correlated, optionally bounded noise.

    Marginals: empirical residual quantiles per bin of the PREDICTED value
    (the only state known at generation time). This keeps the skew: residuals in
    the unfolded basin are wider and lean downward, which Gaussian noise with
    one global sigma cannot represent.

    Time correlation: a latent AR(1) Gaussian e_t = phi*e_{t-1} + sqrt(1-phi^2) z_t
    is mapped to a uniform and then through the bin's empirical quantile function
    (Gaussian copula). With phi = 0 this is state-dependent white noise. White
    noise is reduced by ~sqrt(w) under a w-frame rolling mean, so multi-frame
    excursions (the dips that populate the left FES tail) are lost; phi > 0
    keeps them.

    Bounds: values are reflected back into [lo, hi] (normalized units).
    """

    def __init__(self, preds, resid, n_bins, phi, scale, lo, hi, seed, n_q=201):
        self.edges = np.unique(np.quantile(preds, np.linspace(0, 1, n_bins + 1)))
        self.edges[0], self.edges[-1] = -np.inf, np.inf
        idx = np.clip(np.searchsorted(self.edges, preds, side="right") - 1,
                      0, len(self.edges) - 2)
        qs = np.linspace(0, 1, n_q)
        self.qtab = np.vstack([
            np.quantile(resid[idx == b], qs) if np.sum(idx == b) >= 50
            else np.quantile(resid, qs)
            for b in range(len(self.edges) - 1)
        ])
        self.qs, self.phi, self.scale = qs, float(phi), float(scale)
        self.lo, self.hi = lo, hi
        self.rng = np.random.default_rng(seed)
        self.e = 0.0

    def reset(self):
        self.e = self.rng.standard_normal()

    def sample(self, mu: float) -> float:
        from math import erf, sqrt
        self.e = self.phi * self.e + sqrt(1.0 - self.phi ** 2) * self.rng.standard_normal()
        u = 0.5 * (1.0 + erf(self.e / sqrt(2.0)))
        b = int(np.clip(np.searchsorted(self.edges, mu, side="right") - 1,
                        0, len(self.edges) - 2))
        r = float(np.interp(u, self.qs, self.qtab[b]))
        v = mu + self.scale * r
        if self.hi is not None and v > self.hi:
            v = 2 * self.hi - v
        if self.lo is not None and v < self.lo:
            v = 2 * self.lo - v
        return v

    def apply(self, nxt: torch.Tensor) -> torch.Tensor:
        return torch.full_like(nxt, self.sample(float(nxt.item())))

    def report(self, mean, std):
        print("\n" + "=" * 62)
        print("STATE-DEPENDENT RESIDUALS (raw contact units)")
        print("=" * 62)
        print(f"  AR(1) phi = {self.phi:.3f}   noise scale = {self.scale:.2f}")
        mid = len(self.qs) // 2
        i16, i84 = int(0.16 * (len(self.qs) - 1)), int(0.84 * (len(self.qs) - 1))
        for b in range(len(self.edges) - 1):
            a, c = self.edges[b], self.edges[b + 1]
            a_s = "-inf" if not np.isfinite(a) else f"{a * std + mean:.0f}"
            c_s = "+inf" if not np.isfinite(c) else f"{c * std + mean:.0f}"
            q = self.qtab[b] * std
            print(f"  pred in [{a_s:>5}, {c_s:>5}):  median {q[mid]:+6.2f}  "
                  f"16-84% [{q[i16]:+6.2f}, {q[i84]:+6.2f}]")
        print("  If the widths differ a lot between bins, a single global sigma")
        print("  (inference2 'residual' mode) was the cause of the tail mismatch.")
        print("=" * 62)


def build_state_noise(model, x, y_norm, y_true, input_len, device, args, mean, std):
    print(f"\nMeasuring teacher-forced residuals over "
          f"{'all' if args.resid_frames <= 0 else args.resid_frames} frames (cached)...")
    preds, truth = teacher_forced_cached(
        model, x, y_norm, input_len, device,
        max_frames=None if args.resid_frames <= 0 else args.resid_frames)
    resid = truth - preds
    if args.noise_mode == "state_ar":
        phi = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if args.ar_phi < 0 else args.ar_phi
        phi = float(np.clip(phi, 0.0, 0.99))
    else:
        phi = 0.0
    lo = hi = None
    if args.bounds:
        b_lo, b_hi = (float(v) if v.strip().lower() != "none" else None
                      for v in args.bounds.split(","))
        lo = None if b_lo is None else (b_lo - mean) / std
        hi = None if b_hi is None else (b_hi - mean) / std
    nm = StateNoise(preds, resid, args.state_bins, phi, args.noise_scale,
                    lo, hi, args.seed)
    nm.report(mean, std)
    return nm


@torch.no_grad()
def verify_cache(model, x, prefix_norm, device, n: int = 24):
    """Cached and naive decoding must agree to floating-point noise."""
    a = rollout(model, x, prefix_norm, n, 0.0, device, None)
    b = cached_rollout(model, x, prefix_norm, n, 0.0, device, None, progress_every=0)
    d = float(np.max(np.abs(a - b)))
    print(f"  cache verification over {n} steps: max |naive - cached| = {d:.3e}")
    if d > 1e-3:
        raise SystemExit("Cached decoding disagrees with the naive path. Do not trust it.")
    return d


# ----------------------------------------------------------------------
# Segmented (multi-start) rollouts
# ----------------------------------------------------------------------
def parse_starts(spec: str, input_len: int, max_start: int, rollout_len: int, T: int):
    hi = min(max_start, T - rollout_len - 1)
    if spec.startswith("auto:"):
        n = int(spec.split(":", 1)[1])
        if hi <= input_len:
            raise SystemExit("No room for segmented starts; lower --rollout-len.")
        return sorted(set(np.linspace(input_len, hi, n).astype(int).tolist()))
    starts = [int(s) for s in spec.split(",") if s.strip()]
    return [p for p in starts if input_len <= p <= hi]


def run_segmented(model, y_norm, y_true, starts, args, sigma, mean, std,
                  input_len, device, gen, noise_model=None):
    """
    Seed the decoder with the TRUE trajectory up to frame p, then generate
    freely for rollout_len steps. Repeat across many p so every basin gets
    visited.

    What this measures: short-horizon fidelity -- given a correct starting
    context, does the model stay in the right basin with the right local width?

    What it does NOT measure: whether the model can find the basins on its own.
    The basin coverage here comes from the seeds, i.e. from your data. A pooled
    FES that matches the reference is evidence of good local propagation, not
    evidence that the model reproduces the landscape unaided. Label it as such
    in anything you publish.
    """
    src = y_norm[:, :input_len, :]
    segs, truth_segs, used = [], [], []

    print(f"\nSegmented rollouts: {len(starts)} starts x {args.per_start} each "
          f"x {args.rollout_len} steps")
    for p in starts:
        prefix = y_norm[:, :p, :]
        try:
            batch = [rollout(model, src, prefix, args.rollout_len, sigma, device, gen, noise_model)
                     * std + mean for _ in range(args.per_start)]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  start {p:6d}: OOM at context length {p}; skipped. "
                  f"Lower --max-start.")
            continue
        tw = y_true[p:p + args.rollout_len]
        pm = float(np.mean([b.mean() for b in batch]))
        segs.extend(batch)
        truth_segs.append(tw)
        used.append(p)
        print(f"  start {p:6d}: truth mean {tw.mean():7.1f}  pred mean {pm:7.1f}  "
              f"offset {pm - tw.mean():+7.1f}  |  truth std {tw.std():5.1f}  "
              f"pred std {np.mean([b.std() for b in batch]):5.1f}")

    if not segs:
        raise SystemExit("All starts failed. Lower --max-start or --rollout-len.")

    offsets = [np.mean([b.mean() for b in segs[i * args.per_start:(i + 1) * args.per_start]])
               - truth_segs[i].mean() for i in range(len(used))]
    print(f"\n  basin retention: mean |offset| = {np.mean(np.abs(offsets)):.1f} contacts, "
          f"worst = {np.max(np.abs(offsets)):.1f} at start {used[int(np.argmax(np.abs(offsets)))]}")
    if np.max(np.abs(offsets)) > 50:
        print("  -> At least one segment left the basin it was seeded in. The model")
        print("     does not hold the minority state; its FES will be biased.")
    return segs, truth_segs, used


# ----------------------------------------------------------------------
# Memorization test
# ----------------------------------------------------------------------
@torch.no_grad()
def memorization_test(model, x, prefix_norm, y_norm, truth_window, input_len,
                      n_steps, mean, std, device):
    """
    Does the model condition on its context, or is it replaying the training
    trajectory indexed by absolute decoder position?

    Two ablations, both deterministic so the comparison is clean:
      A. encoder source zeroed, prefix kept
      B. prefix replaced by a distant, physically different stretch of the
         trajectory, encoder source kept

    If the output barely moves, the model is a position-indexed lookup table and
    nothing it produces is a dynamical prediction.
    """
    def corr(a, b):
        n = min(len(a), len(b))
        if n < 2 or np.std(a[:n]) == 0 or np.std(b[:n]) == 0:
            return float("nan")
        return float(np.corrcoef(a[:n], b[:n])[0, 1])

    base = rollout(model, x, prefix_norm, n_steps, 0.0, device, None) * std + mean

    src_zero = torch.zeros_like(x)
    abl_a = rollout(model, src_zero, prefix_norm, n_steps, 0.0, device, None) * std + mean

    # Most distant window of the same length as the prefix.
    T = y_norm.shape[1]
    ref_mu = float(prefix_norm.mean())
    best_off, best_d = 0, -1.0
    for off in range(0, max(1, T - input_len), max(1, (T - input_len) // 40)):
        d = abs(float(y_norm[0, off:off + input_len, 0].mean()) - ref_mu)
        if d > best_d:
            best_d, best_off = d, off
    alt_prefix = y_norm[:, best_off:best_off + input_len, :]
    alt_mean_raw = float(y_norm[0, best_off:best_off + input_len, 0].mean()) * std + mean
    abl_b = rollout(model, x, alt_prefix, n_steps, 0.0, device, None) * std + mean

    c_a, c_b = corr(abl_a, base), corr(abl_b, base)
    print("\n" + "=" * 62)
    print("MEMORIZATION TEST (deterministic, no noise)")
    print("=" * 62)
    print(f"  corr(base rollout, matched truth window) : {corr(base, truth_window):+.4f}")
    print(f"  A. encoder zeroed   -> corr with base    : {c_a:+.4f}   "
          f"mean shift {abl_a.mean() - base.mean():+.2f}")
    print(f"  B. prefix from frame {best_off} (window mean {alt_mean_raw:.0f}) "
          f"-> corr with base: {c_b:+.4f}")
    print(f"     B rollout mean {abl_b.mean():.1f} vs base {base.mean():.1f}")
    if c_a > 0.9 and c_b > 0.9:
        print("\n  -> Output is nearly independent of the context. The model is")
        print("     replaying the training trajectory by decoder position. Any")
        print("     agreement with ground truth is memorization, and the FES is")
        print("     just a histogram of the training data over these frames.")
    print("=" * 62)
    return base, abl_a, abl_b


# ----------------------------------------------------------------------
# Free energy
# ----------------------------------------------------------------------
def rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return a
    k = np.ones(w) / w
    return np.convolve(a, k, mode="valid")


def fes(values: np.ndarray, edges: np.ndarray, kT: float) -> np.ndarray:
    """-kT ln P, shifted so the minimum is zero. Empty bins -> NaN."""
    h, _ = np.histogram(values, bins=edges)
    total = h.sum()
    if total == 0:
        return np.full(len(edges) - 1, np.nan)
    p = h / total
    with np.errstate(divide="ignore"):
        f = -kT * np.log(p)
    finite = np.isfinite(f)
    if finite.any():
        f = f - f[finite].min()
    f[~finite] = np.nan
    return f


def block_fes(traces: list[np.ndarray], edges: np.ndarray, kT: float, n_blocks: int):
    """Mean and std of the FES across blocks of rollouts."""
    n_blocks = max(2, min(n_blocks, len(traces)))
    groups = np.array_split(np.arange(len(traces)), n_blocks)
    curves = []
    for g in groups:
        if len(g) == 0:
            continue
        pooled = np.concatenate([traces[i] for i in g])
        curves.append(fes(pooled, edges, kT))
    curves = np.vstack(curves)
    # Bins empty in every block are legitimately NaN; don't spam warnings.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(curves, axis=0)
            std = np.nanstd(curves, axis=0)
    return mean, std


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gb3_checkpoints/gb3_transformer_best.pt")
    ap.add_argument("--data-dir", default="synthetic_gb3_trajectories")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--outdir", default="inference_output_v3")
    ap.add_argument("--temperature", type=float, default=300.0, help="K")
    ap.add_argument("--n-rollouts", type=int, default=16)
    ap.add_argument("--rollout-len", type=int, default=2000,
                    help="Steps per rollout. Cost is quadratic in this; start small.")
    ap.add_argument("--noise-mode", choices=["none", "residual", "state", "state_ar"],
                    default="state_ar",
                    help="'none' = deterministic conditional mean (variance-collapsed). "
                         "'residual' = add noise at the model's own one-step residual scale.")
    ap.add_argument("--noise-scale", type=float, default=1.0,
                    help="Multiplier on the estimated residual sigma.")
    ap.add_argument("--diag-len", type=int, default=4000,
                    help="Frames AFTER the prefix used for the teacher-forced "
                         "diagnostic pass. Cost is quadratic in input_len+diag_len.")
    ap.add_argument("--roll-window", type=int, default=10,
                    help="Rolling-average window, to match your existing plots.")
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--skip-memtest", action="store_true",
                    help="Skip the context-ablation test for positional memorization.")
    ap.add_argument("--start-frames", default="",
                    help="Enable segmented mode. 'auto:N' spreads N starts over the "
                         "trajectory, or give an explicit comma list of frame indices. "
                         "Each start is seeded with the TRUE context up to that frame.")
    ap.add_argument("--max-start", type=int, default=12000,
                    help="Largest seed frame to attempt. Decoder cost is quadratic in "
                         "context length; raise until you hit OOM.")
    ap.add_argument("--per-start", type=int, default=4,
                    help="Rollouts per start frame.")
    ap.add_argument("--full-rollout", action="store_true",
                    help="Generate to the end of the trajectory with cached decoding, "
                         "then compare the FES against the full reference.")
    ap.add_argument("--full-reps", type=int, default=1,
                    help="Independent full-length rollouts (each takes a while).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resid-frames", type=int, default=0,
                    help="Frames after the prefix used to fit state noise (0 = all). "
                         "Must span both basins, or the unfolded tail stays wrong.")
    ap.add_argument("--state-bins", type=int, default=8,
                    help="Quantile bins of predicted value for state-dependent noise.")
    ap.add_argument("--ar-phi", type=float, default=-1.0,
                    help="AR(1) coefficient for state_ar noise; <0 = estimate from residuals.")
    ap.add_argument("--bounds", default="",
                    help="'lo,hi' in raw units, e.g. 'none,3500'. Values are reflected "
                         "back inside. Native contacts have a hard ceiling.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model, ckpt = load_model(args.model, device)
    mean, std = float(ckpt["mean"]), float(ckpt["std"])
    input_len, output_len = int(ckpt["input_len"]), int(ckpt["output_len"])
    kT = KB * args.temperature
    print(f"Checkpoint: input_len={input_len}, output_len={output_len}, "
          f"mean={mean:.4f}, std={std:.4f}, kT={kT:.4f} kJ/mol")

    check_synthetic_diversity(args.data_dir)

    y_true = load_reference(args.data_dir, args.index)
    y_norm_np = (y_true - mean) / std
    y_norm = torch.tensor(y_norm_np, dtype=torch.float32, device=device)[None, :, None]
    x = y_norm[:, :input_len, :]  # encoder input, same convention as training

    # Positional encoding capacity check.
    pe_max = model.pos_decoder.pe.size(1)
    need = (len(y_true) + 1) if args.full_rollout else (input_len + args.rollout_len + 1)
    if need > pe_max:
        raise SystemExit(f"Sequence too long: need {need} decoder positions, "
                         f"positional encoding holds {pe_max}.")

    sigma_resid = diagnostics(model, x, y_norm, input_len, device, args.diag_len)
    sigma = 0.0 if args.noise_mode == "none" else args.noise_scale * sigma_resid
    # NaN > 0 is False, so a bad sigma would otherwise disable noise in silence.
    noise_model = None
    if args.noise_mode in ("state", "state_ar"):
        noise_model = build_state_noise(model, x, y_norm, y_true, input_len, device,
                                        args, mean, std)
        sigma = 0.0
    if args.noise_mode == "residual" and not (np.isfinite(sigma) and sigma > 0.0):
        raise SystemExit(
            f"Estimated residual sigma is invalid ({sigma}). The diagnostic pass "
            f"failed, so the rollout would run deterministically without saying so."
        )
    print(f"\nRollout noise sigma (normalized units): {sigma:.5f}")
    if noise_model is None and sigma == 0.0 and args.n_rollouts > 1:
        print("NOTE: noise-mode=none is deterministic, so all rollouts will be "
              "identical and the error bars will be zero. n_rollouts>1 buys nothing here.")
    print(f"Generating {args.n_rollouts} rollouts x {args.rollout_len} steps "
          f"(free-running, causal mask on)...")

    gen = torch.Generator(device=device).manual_seed(args.seed)
    prefix = y_norm[:, :input_len, :]

    if args.full_rollout:
        n_steps = len(y_true) - input_len
        print(f"\nFull-length cached rollout: {n_steps} steps x {args.full_reps} rep(s)")
        verify_cache(model, x, prefix, device)

        reps = []
        for r in range(args.full_reps):
            print(f"  rep {r + 1}/{args.full_reps}")
            reps.append(cached_rollout(model, x, prefix, n_steps, sigma, device, gen,
                                      noise_model=noise_model)
                        * std + mean)

        w = args.roll_window
        pred_s = [rolling_mean(t, w) for t in reps]
        true_s = rolling_mean(y_true[input_len:], w)

        lo = min(true_s.min(), min(p.min() for p in pred_s))
        hi = max(true_s.max(), max(p.max() for p in pred_s))
        pad = 0.02 * (hi - lo)
        edges = np.linspace(lo - pad, hi + pad, args.bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        f_true = fes(true_s, edges, kT)
        f_pred, f_err = block_fes(pred_s, edges, kT, n_blocks=min(4, max(2, len(pred_s))))
        both = np.isfinite(f_true) & np.isfinite(f_pred)
        mae = float(np.mean(np.abs(f_true[both] - f_pred[both]))) if both.any() else float("nan")

        # Frames are now identical, so these comparisons are finally like-for-like.
        pooled = np.concatenate(pred_s)
        r_traj = float(np.corrcoef(pred_s[0][: len(true_s)], true_s[: len(pred_s[0])])[0, 1])

        print("\n" + "=" * 62)
        print("FULL-TRAJECTORY COMPARISON (identical frames)")
        print("=" * 62)
        print(f"  truth      mean={true_s.mean():8.2f}  std={true_s.std():6.2f}  "
              f"range=[{true_s.min():.0f}, {true_s.max():.0f}]")
        print(f"  generated  mean={pooled.mean():8.2f}  std={pooled.std():6.2f}  "
              f"range=[{pooled.min():.0f}, {pooled.max():.0f}]")
        print(f"  std ratio                       : {pooled.std() / true_s.std():.3f}")
        print(f"  FES mean abs deviation          : {mae:.2f} kJ/mol")
        print(f"  frames below 3200 (unfolded)    : truth {100 * (true_s < 3200).mean():.1f}%  "
              f"generated {100 * (pooled < 3200).mean():.1f}%")
        print(f"  corr(generated, truth) per-frame: {r_traj:+.4f}")
        if r_traj > 0.8:
            print("\n  -> The generated trajectory follows the reference frame by frame.")
            print("     That is replay of the training sequence, not sampling. A matching")
            print("     FES here is memorization; report it as such or retrain.")
        elif abs(pooled.std() / true_s.std() - 1.0) < 0.15 and mae < 1.5:
            print("\n  -> Distribution reproduced without frame-by-frame agreement.")
            print("     That is the outcome you actually want.")
        print("=" * 62)

        np.save(os.path.join(args.outdir, "full_rollouts.npy"), np.vstack(reps))
        np.savez(os.path.join(args.outdir, "fes_full.npz"),
                 centers=centers, f_true=f_true, f_pred=f_pred, f_err=f_err)

        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(np.arange(len(true_s)) + input_len, true_s, color="teal", lw=0.6,
                label="Ground truth")
        ax.plot(np.arange(len(pred_s[0])) + input_len, pred_s[0], color="dimgray",
                lw=0.6, alpha=0.75, label="Generated (full length)")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Value")
        ax.set_title(f"GB3: full-length generation ({w}-frame rolling mean)")
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "timeseries_full.png"), dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(centers, f_true, color="teal", lw=2.5, label="Truth, full trajectory")
        ax.plot(centers, f_pred, color="dimgray", lw=2, label="Generated, full trajectory")
        ax.fill_between(centers, f_pred - f_err, f_pred + f_err,
                        color="dimgray", alpha=0.25, lw=0)
        ax.set_xlabel(f"{w}-frame rolling-average value")
        ax.set_ylabel(r"Free energy (kJ mol$^{-1}$)")
        ax.set_title(f"GB3 free energy at {args.temperature:.0f} K (matched frames)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "fes_full.png"), dpi=200)
        plt.close(fig)

        if not args.skip_memtest:
            memorization_test(model, x, prefix, y_norm, y_true[input_len:],
                              input_len, 500, mean, std, device)
        print(f"\nWrote results to {os.path.abspath(args.outdir)}")
        return

    if args.start_frames:
        starts = parse_starts(args.start_frames, input_len, args.max_start,
                              args.rollout_len, len(y_true))
        segs, truth_segs, used = run_segmented(
            model, y_norm, y_true, starts, args, sigma, mean, std,
            input_len, device, gen, noise_model)

        w = args.roll_window
        pred_s = [rolling_mean(s, w) for s in segs]
        # Reference = union of the matched truth windows, NOT the whole
        # trajectory. Same frames, same weighting, so the comparison is fair.
        truth_s = np.concatenate([rolling_mean(t, w) for t in truth_segs])
        full_s = rolling_mean(y_true[input_len:], w)

        lo = min(full_s.min(), min(p.min() for p in pred_s))
        hi = max(full_s.max(), max(p.max() for p in pred_s))
        pad = 0.02 * (hi - lo)
        edges = np.linspace(lo - pad, hi + pad, args.bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        f_seg_true = fes(truth_s, edges, kT)
        f_seg_pred, f_seg_err = block_fes(pred_s, edges, kT,
                                          n_blocks=min(4, len(pred_s)))
        f_full = fes(full_s, edges, kT)

        both = np.isfinite(f_seg_true) & np.isfinite(f_seg_pred)
        mae = float(np.mean(np.abs(f_seg_true[both] - f_seg_pred[both]))) if both.any() else float("nan")
        print(f"\n  FES mean abs deviation over populated bins: {mae:.2f} kJ/mol")
        print(f"  seeds span truth means {min(t.mean() for t in truth_segs):.0f} "
              f"to {max(t.mean() for t in truth_segs):.0f}")

        np.savez(os.path.join(args.outdir, "fes_segmented.npz"),
                 centers=centers, f_seg_true=f_seg_true, f_seg_pred=f_seg_pred,
                 f_seg_err=f_seg_err, f_full=f_full, starts=np.array(used))

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(centers, f_full, color="teal", lw=1.2, alpha=0.4,
                label="Truth, full trajectory")
        ax.plot(centers, f_seg_true, color="teal", lw=2.5,
                label="Truth, union of seeded windows")
        ax.plot(centers, f_seg_pred, color="dimgray", lw=2,
                label="Prediction, pooled segments")
        ax.fill_between(centers, f_seg_pred - f_seg_err, f_seg_pred + f_seg_err,
                        color="dimgray", alpha=0.25, lw=0)
        ax.set_xlabel(f"{w}-frame rolling-average value")
        ax.set_ylabel(r"Free energy (kJ mol$^{-1}$)")
        ax.set_title(f"GB3 short-horizon segmented FES at {args.temperature:.0f} K")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "fes_segmented.png"), dpi=200)
        plt.close(fig)

        if not args.skip_memtest:
            memorization_test(model, x, prefix, y_norm, y_true[input_len:],
                              input_len, min(args.rollout_len, 500), mean, std, device)
        print(f"\nWrote results to {os.path.abspath(args.outdir)}")
        return

    traces = []
    for r in range(args.n_rollouts):
        raw = rollout(model, x, prefix, args.rollout_len, sigma, device, gen, noise_model)
        traces.append(raw * std + mean)
        print(f"  rollout {r + 1}/{args.n_rollouts}  "
              f"mean={traces[-1].mean():.1f}  std={traces[-1].std():.1f}")

    if len(traces) > 1 and all(np.array_equal(traces[0], t) for t in traces[1:]):
        print("\nWARNING: all rollouts are identical. No noise was actually injected, "
              "so this is a single deterministic trace and the error bars are meaningless.")

    # Rolling average, matching how the reference FES was built.
    w = args.roll_window

    # CRITICAL: the rollout covers frames [input_len, input_len+rollout_len).
    # Comparing it against the ENTIRE remainder of the trajectory is a window
    # mismatch, not a model result -- the remainder may contain basins the
    # rollout never had the chance to reach. Compare against the matched window,
    # and show the full-trajectory reference separately for context.
    end = min(len(y_true), input_len + args.rollout_len)
    y_window = y_true[input_len:end]
    pred_smooth = [rolling_mean(t, w) for t in traces]
    true_win_smooth = rolling_mean(y_window, w)
    true_full_smooth = rolling_mean(y_true[input_len:], w)

    # Shared bin edges, spanning everything so the plots are comparable.
    lo = min(true_full_smooth.min(), min(p.min() for p in pred_smooth))
    hi = max(true_full_smooth.max(), max(p.max() for p in pred_smooth))
    pad = 0.02 * (hi - lo)
    edges = np.linspace(lo - pad, hi + pad, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    f_true = fes(true_full_smooth, edges, kT)
    f_true_win = fes(true_win_smooth, edges, kT)
    f_pred, f_err = block_fes(pred_smooth, edges, kT, n_blocks=min(4, args.n_rollouts))

    covered = (end - input_len) / max(1, len(y_true) - input_len)
    print(f"\nRollout covers frames {input_len}-{end} "
          f"({100 * covered:.1f}% of the remaining trajectory).")
    if covered < 0.5:
        print("Only the matched-window comparison is meaningful. To test whether the "
              "model reproduces the full landscape you must roll out far enough to "
              "reach the other basins.")

    # Stationarity check: first vs second half of each rollout.
    half = args.rollout_len // 2
    f_h1 = fes(np.concatenate([t[:half] for t in traces]), edges, kT)
    f_h2 = fes(np.concatenate([t[half:] for t in traces]), edges, kT)
    both = np.isfinite(f_h1) & np.isfinite(f_h2)
    drift = float(np.abs(f_h1[both] - f_h2[both]).max()) if both.any() else float("nan")
    m1 = np.mean([t[:half].mean() for t in traces])
    m2 = np.mean([t[half:].mean() for t in traces])

    # The truth over the same window is not stationary either; report both so the
    # rollout's drift can be judged against the reference rather than against zero.
    hw = len(y_window) // 2
    tm1, tm2 = y_window[:hw].mean(), y_window[hw:].mean()

    print("\n" + "=" * 62)
    print("STATIONARITY  (rollout vs matched truth window)")
    print("=" * 62)
    print(f"  rollout mean, 1st / 2nd half : {m1:.2f} / {m2:.2f}  (drift {m2 - m1:+.2f})")
    print(f"  truth   mean, 1st / 2nd half : {tm1:.2f} / {tm2:.2f}  (drift {tm2 - tm1:+.2f})")
    print(f"  max |dF| between halves      : {drift:.2f} kJ/mol")
    if abs((m2 - m1) - (tm2 - tm1)) < 0.3 * abs(tm2 - tm1) + 1.0:
        print("  -> Rollout drift tracks the truth's own drift over this window, so")
        print("     this is trajectory structure, not the model wandering.")
    elif drift > 2.0:
        print("  -> Drifting independently of the reference. No detailed balance;")
        print("     this FES is not converged.")
    print("=" * 62)

    print("\nDISTRIBUTION")
    pooled = np.concatenate(pred_smooth)
    print(f"  truth, matched window mean={true_win_smooth.mean():8.2f}  "
          f"std={true_win_smooth.std():6.2f}  "
          f"range=[{true_win_smooth.min():.0f}, {true_win_smooth.max():.0f}]")
    print(f"  prediction            mean={pooled.mean():8.2f}  std={pooled.std():6.2f}  "
          f"range=[{pooled.min():.0f}, {pooled.max():.0f}]")
    print(f"  std ratio (pred / matched truth): "
          f"{pooled.std() / max(1e-12, true_win_smooth.std()):.3f}")
    print(f"\n  [context] truth, FULL remainder mean={true_full_smooth.mean():8.2f}  "
          f"std={true_full_smooth.std():6.2f}  "
          f"range=[{true_full_smooth.min():.0f}, {true_full_smooth.max():.0f}]")
    print("  Do NOT compare the prediction against this line -- different window.")

    if not args.skip_memtest:
        memorization_test(model, x, prefix, y_norm, y_window, input_len,
                          min(args.rollout_len, 500), mean, std, device)

    # ---------------- save ----------------
    np.save(os.path.join(args.outdir, "rollouts.npy"), np.vstack(traces))
    np.savez(
        os.path.join(args.outdir, "fes.npz"),
        centers=centers, f_true=f_true, f_true_window=f_true_win,
        f_pred=f_pred, f_err=f_err, f_half1=f_h1, f_half2=f_h2,
    )

    # ---------------- plots ----------------
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(np.arange(len(true_win_smooth)) + input_len, true_win_smooth,
            color="teal", lw=0.8, label="Ground truth (matched window)")
    for t in pred_smooth[:4]:
        ax.plot(np.arange(len(t)) + input_len, t, color="dimgray", lw=0.8, alpha=0.6)
    ax.plot([], [], color="dimgray", lw=0.8, label="Free-running rollouts")
    ax.axvline(input_len, color="k", ls="--", lw=1, label="End of true prefix")
    ax.set_xlim(0, input_len + args.rollout_len)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Value")
    ax.set_title(f"GB3: free-running rollout ({args.roll_window}-frame rolling mean)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "timeseries.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(centers, f_true, color="teal", lw=1.2, alpha=0.45,
            label="Truth, full trajectory (not the comparison)")
    ax.plot(centers, f_true_win, color="teal", lw=2.5,
            label="Truth, matched window")
    ax.plot(centers, f_pred, color="dimgray", lw=2, label="Prediction (ensemble)")
    ax.fill_between(centers, f_pred - f_err, f_pred + f_err,
                    color="dimgray", alpha=0.25, lw=0, label="Block std")
    ax.plot(centers, f_h1, color="darkorange", lw=1, ls=":", label="Pred, 1st half")
    ax.plot(centers, f_h2, color="crimson", lw=1, ls=":", label="Pred, 2nd half")
    ax.set_xlabel(f"{args.roll_window}-frame rolling-average value")
    ax.set_ylabel(r"Free energy (kJ mol$^{-1}$)")
    ax.set_title(f"GB3 free energy at {args.temperature:.0f} K")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fes.png"), dpi=200)
    plt.close(fig)

    print(f"\nWrote results to {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()