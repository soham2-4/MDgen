#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASYN_PCCA_plot.py
Plots metastable state map for alpha-synuclein (asyn) MSM using pyEMMA.
"""

import os
import sys
import gc
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

try:
    import pyemma
except ImportError:
    raise ImportError("Please activate your 'msm' conda environment with pyemma installed.")

try:
    from colorama import Fore, Style
except Exception:
    class _C:
        GREEN = MAGENTA = RESET_ALL = ""
    Fore = _C()
    Style = _C()


# ========== USER SETTINGS ==========
protein = "asyn_gen"     # prefix for MSM analysis files
data_file = "asyn.npy"   # simulation data (in same dir)
n_clusters = 400
lag = 50
n_states = 5
time_step = 1000  # ps per frame (not used for plotting)
# ===================================

# Resolve paths relative to script
base_dir = os.path.dirname(os.path.abspath(__file__))
its_dir = os.path.join(base_dir, "ITS")
msm_dir = os.path.join(base_dir, "MSM_ANALYSIS", f"{protein}_nclu_{n_clusters}_lag_{lag}_state_size_{n_states}")

cluster_file = os.path.join(its_dir, f"{protein}_nclu_{n_clusters}_kmeans.pkl")
dtrajs_file = os.path.join(its_dir, f"{protein}_nclu_{n_clusters}_dtrajs.pkl")
msm_file = os.path.join(msm_dir, f"{protein}_nclu_{n_clusters}_lag_{lag}_msm.pkl")
metastable_traj_file = os.path.join(msm_dir, f"metastable_traj_{protein}_nclu_{n_clusters}_lag_{lag}_state_size_{n_states}.npy")
coarse_centers_file = os.path.join(msm_dir, f"coarse_centers_{protein}_nclu_{n_clusters}_lag_{lag}_state_size_{n_states}.npy")

# Verify files exist
for f in [data_file, cluster_file, dtrajs_file, msm_file, metastable_traj_file, coarse_centers_file]:
    if not os.path.exists(f):
        print(f"{Fore.RED}ERROR: Missing file -> {f}{Style.RESET_ALL}")
        sys.exit(1)

# Load data
data = np.load(data_file)
if data.ndim != 2 or data.shape[1] < 2:
    raise ValueError(f"Expected data with shape (N,2). Got {data.shape} from {data_file}")

with open(cluster_file, "rb") as f:
    cluster = pkl.load(f)
with open(dtrajs_file, "rb") as f:
    dtrajs = pkl.load(f)
with open(msm_file, "rb") as f:
    msm_data = pkl.load(f)

metastable_traj = np.load(metastable_traj_file)
coarse_state_centers = np.load(coarse_centers_file)

# Warn if data length doesn't match metastable_traj
if len(metastable_traj) != data.shape[0]:
    print(f"{Fore.YELLOW}Warning:{Style.RESET_ALL} data ({len(data)}) and metastable_traj ({len(metastable_traj)}) lengths differ.")
    print("→ Plot will still proceed using overlapping range.")
    min_len = min(len(data), len(metastable_traj))
    data = data[:min_len]
    metastable_traj = metastable_traj[:min_len]

# Set up colorbar ticks (1-based)
tick_labels = np.arange(1, n_states + 1)

# ---------- Plot -------------
fig, ax = plt.subplots(figsize=(10, 7))
cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.05])  # horizontal colorbar axis

# Plot metastable states
_, _, misc = pyemma.plots.plot_state_map(
    *data.T,
    metastable_traj,
    ax=ax,
    zorder=-1,
    cmap="jet",
    cbar=True,
    cbar_orientation="horizontal",
    cax=cbar_ax,
    antialiased=False
)

# Overlay coarse centers
if coarse_state_centers.ndim >= 2 and coarse_state_centers.shape[1] >= 2:
    ax.plot(
        coarse_state_centers[:, 0],
        coarse_state_centers[:, 1],
        marker="o",
        linestyle="",
        color="magenta",
        ms=8,
        mew=1,
        lw=0,
        label="Coarse centers"
    )

# Format colorbar
cbar = misc.get("cbar", None)
if cbar is not None:
    ticks = np.linspace(cbar.vmin, cbar.vmax, len(tick_labels))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Metastable States", fontsize=14, labelpad=8)

ax.set_xlabel("Dimension 1", fontsize=14)
ax.set_ylabel("Dimension 2", fontsize=14)
ax.set_axis_off()

# ---------- Save ----------
out_prefix = f"{protein}_msm_lag_{lag}_nclu_{n_clusters}_nstates_{n_states}"
for ext in ["png", "pdf", "svg"]:
    out_file = f"{out_prefix}.{ext}"
    plt.savefig(out_file, dpi=300, bbox_inches="tight", format=ext)
    print(f"{Fore.GREEN}Saved:{Style.RESET_ALL} {out_file}")

plt.show()
plt.close()
gc.collect()

