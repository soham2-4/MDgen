#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MFPT network + PCCA map overlay for alpha-synuclein (asyn)
- Robust to inactive microstates (maps original labels -> active indices)
- Plots PCCA state map
- Computes MFPT matrix (in microseconds)
- Draws transition network (full and thresholded overlays)
"""

import os
import sys
import gc
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

# deps
try:
    import pyemma
    import pyemma.msm as msm
    import pyemma.plots as mplt
except Exception as e:
    raise ImportError("pyemma is required. Activate your MSM env.") from e

# ----- USER PARAMS (edit as needed) -----
protein   = 'asyn_gen'
n_clusters = 400
lag        = 50      # MSM lag used when building the model (in frames)
n_states   = 5       # PCCA macrostates for visualization
dt_ps      = 1000    # ps per frame in the raw trajectories
# data file (2D embedding used for plots)
data_rel   = os.path.join('..', 'asyn.npy')   # change if your 2D coords live elsewhere
# ----------------------------------------

# Resolve paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file  = os.path.join(script_dir, data_rel)

msm_dir = os.path.join(
    script_dir, '..', 'MSM_ANALYSIS',
    f'{protein}_nclu_{n_clusters}_lag_{lag}_state_size_{n_states}'
)
its_dir = os.path.join(script_dir, '..', 'ITS')

msm_file = os.path.join(msm_dir, f'{protein}_nclu_{n_clusters}_lag_{lag}_msm.pkl')
kmeans_file = os.path.join(its_dir, f'{protein}_nclu_{n_clusters}_kmeans.pkl')
dtrajs_file = os.path.join(its_dir, f'{protein}_nclu_{n_clusters}_dtrajs.pkl')
coarse_centers_file = os.path.join(
    msm_dir, f'coarse_centers_{protein}_nclu_{n_clusters}_lag_{lag}_state_size_{n_states}.npy'
)

# ----- existence checks -----
for pth in [data_file, msm_file, kmeans_file, dtrajs_file, coarse_centers_file]:
    if not os.path.exists(pth):
        print(f'ERROR: missing file:\n  {pth}')
        sys.exit(1)

# ----- load inputs -----
data = np.load(data_file)
if data.ndim != 2 or data.shape[1] < 2:
    raise ValueError(f'Expected data shape (N,2+); got {data.shape} from {data_file}')
xy = data[:, :2]  # use first two columns for plotting

with open(msm_file, 'rb') as f:
    msm_data = pkl.load(f)

with open(kmeans_file, 'rb') as f:
    cluster = pkl.load(f)

with open(dtrajs_file, 'rb') as f:
    dtrajs = pkl.load(f)
# dtrajs is usually list of 1D arrays of original microstate labels

# ----- PCCA and robust metastable assignments -----
msm_data.pcca(n_states)

# Concatenate original labels
dcat = np.concatenate(dtrajs).astype(int)

# Build mapping original_label -> active_index using active_set (version-safe)
# active_set contains original labels of active microstates
active_set = np.asarray(msm_data.active_set, dtype=int)
map_size = int(max(dcat.max() + 1, n_clusters))  # cover all observed labels
active_idx_map = np.full(map_size, -1, dtype=int)
active_idx_map[active_set] = np.arange(active_set.shape[0], dtype=int)

# Per-frame active-state index or -1
ai = active_idx_map[dcat]
mask_active = ai != -1

# Metastable macro assignments; -1 for inactive frames
metastable_traj = np.full(dcat.shape, -1, dtype=np.int32)
metastable_traj[mask_active] = msm_data.metastable_assignments[ai[mask_active]]

# highest-membership microstate per macrostate; get coarse centers in original label space
highest_membership = msm_data.metastable_distributions.argmax(1)  # active-state IDs
center_orig_states = active_set[highest_membership]               # original k-means labels
coarse_state_centers = cluster.clustercenters[center_orig_states]

# ----- quick MSM diagnostics -----
try:
    print(f'Active state fraction:  {msm_data.active_state_fraction:.3f}')
    print(f'Active count fraction:  {msm_data.active_count_fraction:.3f}')
except Exception:
    pass

# ----- length alignment for plotting -----
# If your xy and metastable_traj lengths differ, trim to the min
nmin = min(len(xy), len(metastable_traj))
if nmin < len(xy) or nmin < len(metastable_traj):
    print(f'Warning: trimming plot arrays to min length = {nmin}')
xy_plot = xy[:nmin]
macro_plot = metastable_traj[:nmin]
mask_plot = mask_active[:nmin]

# ----- Plot 1: PCCA map with coarse centers -----
fig, ax = plt.subplots(figsize=(10, 7))
cbar_ax = fig.add_axes([0.15, 0.01, 0.7, 0.05])  # horizontal cbar axis

# To avoid coloring inactive frames as "-1", you can mask them out; here we keep them,
# but plot_state_map can handle the label -1 (it just assigns a color). If you prefer
# to drop them, pass xy_plot[mask_plot], macro_plot[mask_plot].
_, _, misc = pyemma.plots.plot_state_map(
    *xy_plot.T,
    macro_plot,
    ax=ax,
    zorder=-1,
    cmap='nipy_spectral',
    cbar=True,
    cbar_orientation='horizontal',
    cax=cbar_ax,
    antialiased=False,
    alpha=0.9
)

# overlay coarse centers
if coarse_state_centers.ndim >= 2 and coarse_state_centers.shape[1] >= 2:
    ax.plot(
        coarse_state_centers[:, 0],
        coarse_state_centers[:, 1],
        marker='o', linestyle='',
        color='black', ms=8, mew=1, lw=0, label='Coarse centers'
    )

# colorbar labels: 1..n_states
cbar = misc.get('cbar', None)
if cbar is not None:
    ticks = np.linspace(cbar.vmin, cbar.vmax, n_states)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(np.arange(1, n_states + 1))
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('States', size=16, labelpad=8)

ax.set_axis_off()
ax.set_xlabel('IC1')
ax.set_ylabel('IC2')
plt.tight_layout()
plt.savefig(f'pcca_map_{protein}_nclu{n_clusters}_lag{lag}_nstates{n_states}.png',
            dpi=300, bbox_inches='tight')
plt.show()

# ----- MFPT computation (μs) -----
# mfpt_ij_steps is the mean number of lag-steps from macro i to macro j.
# Convert to microseconds: steps * (lag * dt_ps) / 1e6
mfpt_us = np.zeros((n_states, n_states), dtype=float)
for i in range(n_states):
    for j in range(n_states):
        steps_ij = msm_data.mfpt(msm_data.metastable_sets[i], msm_data.metastable_sets[j])
        mfpt_us[i, j] = steps_ij * (lag * dt_ps) / 1e6  # μs

# inverse (1/μs), zero where undefined
inv_mfpt = np.zeros_like(mfpt_us)
nz = mfpt_us > 0
inv_mfpt[nz] = 1.0 / mfpt_us[nz]

# ----- Plot 2: network with full MFPT labels -----
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

pyemma.plots.plot_network(
    inv_mfpt,
    pos=coarse_state_centers,
    arrow_label_format='%.1f μs',
    state_sizes=np.array(msm_data.stationary_distribution),
    arrow_labels=mfpt_us,
    arrow_scale=2.0,
    state_labels=np.arange(1, n_states + 1),
    size=15
)
plt.title('MFPT network (all edges)', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(f'mfpt_network_all_{protein}_nclu{n_clusters}_lag{lag}_nstates{n_states}.png',
            dpi=300, bbox_inches='tight')
plt.show()

# ----- Plot 3: thresholded overlay on PCCA map -----
fig, ax = plt.subplots(figsize=(10, 8))
cbar_ax = fig.add_axes([0.22, 0.15, 0.7, 0.03])

_, _, misc = pyemma.plots.plot_state_map(
    *xy_plot.T,
    macro_plot,
    ax=ax,
    zorder=-1,
    cmap='nipy_spectral',
    cbar=True,
    cbar_orientation='horizontal',
    cax=cbar_ax,
    antialiased=False,
    alpha=0.7
)

cbar = misc.get('cbar', None)
if cbar is not None:
    ticks = np.linspace(cbar.vmin, cbar.vmax, n_states)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(np.arange(1, n_states + 1))
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('States', size=16, labelpad=8)

ax.set_axis_off()
ax.set_xlabel('IC1')
ax.set_ylabel('IC2')

# Threshold in 1/μs (keep strong edges)
cutoff_inv = 2e-1    # adjust as needed
weights_thresh = np.where(inv_mfpt > cutoff_inv, inv_mfpt, 0.0)
labels_thresh  = np.where(inv_mfpt > cutoff_inv, mfpt_us, 0.0)

pyemma.plots.plot_network(
    weights_thresh,
    pos=coarse_state_centers,
    arrow_label_format='%.1f μs',
    state_sizes=np.array(msm_data.stationary_distribution),
    arrow_labels=labels_thresh,
    arrow_scale=2.0,
    state_labels=np.arange(1, n_states + 1),
    size=15,
    ax=ax
)

plt.tight_layout()
plt.savefig(f'mfpt_overlay_thresh_{protein}_nclu{n_clusters}_lag{lag}_nstates{n_states}.png',
            dpi=300, bbox_inches='tight')
plt.savefig(f'mfpt_overlay_thresh_{protein}_nclu{n_clusters}_lag{lag}_nstates{n_states}.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig(f'mfpt_overlay_thresh_{protein}_nclu{n_clusters}_lag{lag}_nstates{n_states}.svg',
            dpi=300, bbox_inches='tight')
plt.show()

gc.collect()

