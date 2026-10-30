#!/usr/bin/env python3
"""
aib9_common.py

Shared pieces for the dihedral pipeline: angle encoding, binning,
the windowed dataset, and the model.

Design notes
------------

Representation
    Each dihedral is fed to the network as (sin, cos), so the +/-180
    wrap is continuous and the network never sees a discontinuity.
    D dihedrals -> 2D input features.

Prediction target
    A *distribution*, not a point. Each dihedral's next value is
    predicted as a categorical over NBINS circular bins, trained with
    cross-entropy.

    This is the central fix. The old model minimised MSE, which learns
    E[x_t+1 | context]. For a system with rare transitions, the
    conditional mean of "stay in this basin" and "flip to the other" is
    always "stay", so an MSE model can never sample a rare event -
    no amount of training or tuning changes that. A categorical head
    keeps the full multimodal distribution, so a flip is sampled
    whenever the model assigns it probability.

Positional information
    Decoder-only transformer over a FIXED-length window, with the model
    only ever asked for the prediction at the LAST position at
    generation time. The old code decoded 500-frame blocks that each
    restarted the positional encoding at index 0, which locked the
    rollout into a 500-frame limit cycle (autocorrelation showed peaks
    at exactly 500, 1000, 1500, 2000). Sliding by one frame makes every
    generated frame come from the same position index the model was
    trained on, so no periodic artifact is possible.

Mirror symmetry
    AIB9 is achiral: (phi, psi) -> (-phi, -psi) maps a structure onto
    its physically equivalent mirror image. Negating all dihedrals is
    therefore an exact data augmentation, and it enforces the left/right
    basin symmetry that the reference trajectory only satisfies to
    within its own sampling error (the 5000 ns reference gave a
    0.601/0.399 split where the true answer is 0.500/0.500).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


NBINS_DEFAULT = 36          # 10 degree resolution
KB_KCAL = 0.0019872041


# ============================================================
# Angle helpers
# ============================================================

def angles_to_features(angles_deg):
    """
    [T, D] degrees -> [T, 2D] float32 of (sin, cos) pairs.
    Column layout: [sin_0, cos_0, sin_1, cos_1, ...]
    """

    rad = np.radians(np.asarray(angles_deg, dtype=np.float32))

    out = np.empty(
        (rad.shape[0], rad.shape[1] * 2), dtype=np.float32
    )

    out[:, 0::2] = np.sin(rad)
    out[:, 1::2] = np.cos(rad)

    return out


def features_to_angles(features):
    """
    [T, 2D] (sin, cos) -> [T, D] degrees.
    """

    f = np.asarray(features)

    return np.degrees(np.arctan2(f[:, 0::2], f[:, 1::2]))


def angles_to_bins(angles_deg, nbins):
    """
    [T, D] degrees -> [T, D] int64 bin indices in [0, nbins).
    Bin b covers [-180 + b*w, -180 + (b+1)*w).
    """

    width = 360.0 / nbins

    b = np.floor((np.asarray(angles_deg) + 180.0) / width)

    return np.mod(b, nbins).astype(np.int64)


def bins_to_angles(bin_idx, nbins, rng=None, jitter=True):
    """
    Bin indices -> degrees. Jitter uniformly inside the bin so the
    output is not stuck on a lattice of bin centres.
    """

    width = 360.0 / nbins

    lo = -180.0 + np.asarray(bin_idx) * width

    if jitter:
        rng = rng or np.random.default_rng()
        return lo + rng.random(np.shape(bin_idx)) * width

    return lo + 0.5 * width


def circular_mean_deg(angles_deg, axis=-1):
    """
    Mean of angles respecting the wrap.
    """

    rad = np.radians(angles_deg)

    return np.degrees(
        np.arctan2(
            np.sin(rad).mean(axis=axis),
            np.cos(rad).mean(axis=axis),
        )
    )


def count_transitions(phi_bar, cut=25.0):
    """
    Count genuine left<->right helix transitions.

    A naive sign change of mean phi massively overcounts, because a
    trajectory sitting near phi = 0 flickers across the axis without
    committing to either basin. This uses core sets: a frame counts as
    right-handed below -cut and left-handed above +cut, the region
    between is unassigned, and a transition is recorded only when the
    trajectory leaves one core and later enters the other.

    phi_bar : [T] or [n_chains, T] mean phi in degrees
    returns : total number of committed transitions
    """

    x = np.atleast_2d(np.asarray(phi_bar))

    total = 0

    for row in x:

        state = np.zeros(row.shape[0], dtype=np.int8)
        state[row < -cut] = -1
        state[row > cut] = 1

        committed = state[state != 0]

        if committed.size < 2:
            continue

        total += int(np.sum(np.abs(np.diff(committed)) > 1))

    return total


def load_meta(data_dir):
    path = os.path.join(data_dir, "meta.json")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run aib9_dihedral_data.py first."
        )

    with open(path) as fh:
        return json.load(fh)


def phi_columns(meta):
    return [
        i for i, n in enumerate(meta["dihedral_names"])
        if n.startswith("phi")
    ]


def psi_columns(meta):
    return [
        i for i, n in enumerate(meta["dihedral_names"])
        if n.startswith("psi")
    ]


def mean_phi_psi(angles_deg, meta):
    """
    Per-frame circular mean of phi and of psi. This is the helix
    handedness coordinate for AIB9.
    """

    return (
        circular_mean_deg(angles_deg[:, phi_columns(meta)], axis=1),
        circular_mean_deg(angles_deg[:, psi_columns(meta)], axis=1),
    )


# ============================================================
# Dataset
# ============================================================

class DihedralWindowDataset(Dataset):
    """
    Sliding windows over real trajectories.

    Every window of length `window` starting at every offset is a valid
    sample (stride is 1 by default), so no absolute position within the
    trajectory is privileged. The model predicts the bin of frame t+1
    from frames <= t, for every t in the window.

    Splitting
    ---------
    `frame_range` selects a contiguous FRACTION of each trajectory, so a
    train/validation split can be made in time while still using every
    replica for training. `gap` frames are dropped from the end of the
    segment, which prevents a training window from overlapping - or
    sitting immediately adjacent to - a validation window. Set gap to at
    least `window`.

    Returns
    -------
    x : [window, 2D]   float32 (sin, cos) input frames
    y : [window, D]    int64 bin index of the NEXT frame at each position
    """

    def __init__(
        self,
        files,
        nbins=NBINS_DEFAULT,
        window=64,
        stride=1,
        mirror=True,
        frame_range=(0.0, 1.0),
        gap=0,
    ):

        self.nbins = nbins
        self.window = window
        self.mirror = mirror

        lo_frac, hi_frac = frame_range

        self.angles = []

        for path in files:
            arr = np.load(path)["angles_deg"].astype(np.float32)
            if arr.ndim != 2:
                raise ValueError(f"{path}: expected [T, D], got {arr.shape}")

            T = arr.shape[0]
            lo = int(round(lo_frac * T))
            hi = int(round(hi_frac * T)) - gap

            if hi - lo < window + 1:
                raise ValueError(
                    f"{path}: segment [{lo}, {hi}) is shorter than "
                    f"window+1 = {window + 1}"
                )

            self.angles.append(arr[lo:hi])

        if not self.angles:
            raise FileNotFoundError("No trajectory files given")

        self.n_dihedrals = self.angles[0].shape[1]

        for a in self.angles:
            if a.shape[1] != self.n_dihedrals:
                raise ValueError("Replicas disagree on the dihedral count")

        self.n_frames = sum(a.shape[0] for a in self.angles)

        # index map: (traj_id, start)
        self.index_map = []

        for tid, a in enumerate(self.angles):
            # need window+1 frames: window inputs plus one more target
            last = a.shape[0] - window - 1
            for start in range(0, max(0, last), stride):
                self.index_map.append((tid, start))

        if not self.index_map:
            raise ValueError(
                f"No windows: segments shorter than window+1 = {window + 1}"
            )

    @property
    def feature_dim(self):
        return self.n_dihedrals * 2

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):

        tid, start = self.index_map[i]

        seg = self.angles[tid][start : start + self.window + 1]

        # exact symmetry of an achiral peptide
        if self.mirror and np.random.rand() < 0.5:
            seg = -seg

        x = angles_to_features(seg[:-1])
        y = angles_to_bins(seg[1:], self.nbins)

        return torch.from_numpy(x), torch.from_numpy(y)


# ============================================================
# Model
# ============================================================

class SinusoidalPositions(nn.Module):

    def __init__(self, d_model, max_len=4096):

        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()

        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class DihedralTransformer(nn.Module):
    """
    Causal decoder-only transformer over frames.

    Input  : [B, T, 2D] (sin, cos) features
    Output : [B, T, D, nbins] logits for the NEXT frame at each position
    """

    def __init__(
        self,
        n_dihedrals,
        nbins=NBINS_DEFAULT,
        d_model=256,
        nhead=8,
        num_layers=6,
        dropout=0.1,
        max_len=4096,
    ):

        super().__init__()

        self.n_dihedrals = n_dihedrals
        self.nbins = nbins
        self.d_model = d_model

        self.input_proj = nn.Linear(n_dihedrals * 2, d_model)
        self.pos = SinusoidalPositions(d_model, max_len=max_len)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.blocks = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, n_dihedrals * nbins)

    def forward(self, x):

        B, T, _ = x.shape

        h = self.input_proj(x)
        h = self.drop(self.pos(h))

        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )

        h = self.blocks(h, mask=mask, is_causal=False)
        h = self.norm(h)

        return self.head(h).view(B, T, self.n_dihedrals, self.nbins)


# ============================================================
# Loss
# ============================================================

def circular_smooth_targets(y, nbins, eps=0.1):
    """
    One-hot targets softened onto the two neighbouring bins (wrapping at
    the ends). Reflects that adjacent bins are adjacent angles, which a
    plain one-hot does not know.
    """

    oh = F.one_hot(y, nbins).float()

    left = torch.roll(oh, shifts=-1, dims=-1)
    right = torch.roll(oh, shifts=1, dims=-1)

    return (1.0 - eps) * oh + 0.5 * eps * (left + right)


def dihedral_loss(logits, targets, nbins, smoothing=0.1):
    """
    Mean cross-entropy over all dihedrals and positions.
    """

    logp = F.log_softmax(logits, dim=-1)

    if smoothing > 0:
        soft = circular_smooth_targets(targets, nbins, eps=smoothing)
        return -(soft * logp).sum(-1).mean()

    return F.nll_loss(
        logp.reshape(-1, nbins), targets.reshape(-1)
    )


# ============================================================
# Sampling
# ============================================================

@torch.no_grad()
def sample_next(model, window_feats, temperature=1.0, generator=None):
    """
    Sample the next frame's bin indices from the last position.

    window_feats : [B, T, 2D] tensor
    returns      : [B, D] int64 bin indices
    """

    logits = model(window_feats)[:, -1]        # [B, D, nbins]

    if temperature <= 0:
        return logits.argmax(-1)

    probs = F.softmax(logits / temperature, dim=-1)

    B, D, K = probs.shape

    flat = probs.reshape(B * D, K)

    idx = torch.multinomial(flat, 1, generator=generator)

    return idx.view(B, D)


# ============================================================
# Data discovery
# ============================================================

def find_replica_files(data_dir):
    """
    All <replica>.npz files in data_dir, sorted by name.
    """

    files = sorted(
        p for p in Path(data_dir).glob("*.npz") if p.name != "meta.json"
    )

    if not files:
        raise FileNotFoundError(
            f"No .npz trajectories in {data_dir}. "
            "Run aib9_dihedral_data.py first."
        )

    return files
