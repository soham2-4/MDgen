#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SPECTRAL ANALYSIS for ASYN (α-synuclein)
----------------------------------------
Computes implied timescales and eigenvalue ratios for MSM model.

Settings:
    n_clusters = 500
    msm_lag    = 50
    n_states   = 7
"""

import os
import gc
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pyemma

# ==========================================
# USER PARAMETERS
# ==========================================

protein = "asyn_gen"
dt_ps = 1000              # ps per frame of trajectory

n_clusters = 500         # << requested
msm_lag = 50             # << requested
n_states = 4             # << requested

script_dir = os.path.dirname(os.path.abspath(__file__))

# Location of MSM data and ITS data
msm_dir = os.path.join(script_dir, "..", "MSM_ANALYSIS",
                       f"{protein}_nclu_{n_clusters}_lag_{msm_lag}_state_size_{n_states}")

its_dir = os.path.join(script_dir, "..", "ITS")

# MSM + cluster files
msm_file = os.path.join(msm_dir, f"{protein}_nclu_{n_clusters}_lag_{msm_lag}_msm.pkl")
cluster_file = os.path.join(its_dir, f"{protein}_nclu_{n_clusters}_kmeans.pkl")
dtrajs_file = os.path.join(its_dir, f"{protein}_nclu_{n_clusters}_dtrajs.pkl")

# raw simulation coordinates (IC / PCA / 2D projection)
data_file = os.path.join(script_dir, "..", "MSM_ANALYSIS", "asyn.npy")

# ==========================================
# CHECK INPUT FILES
# ==========================================

for f in [msm_file, data_file, cluster_file, dtrajs_file]:
    if not os.path.exists(f):
        print(f"\n❌ ERROR: Missing file:\n{f}\n")
        exit(1)

print("\n✅ All files found, running analysis...\n")

# ==========================================
# LOAD DATA
# ==========================================

with open(msm_file, "rb") as f:
    M = pkl.load(f)

with open(cluster_file, "rb") as f:
    cluster = pkl.load(f)

with open(dtrajs_file, "rb") as f:
    dtrajs = pkl.load(f)

data = np.load(data_file)

if data.ndim != 2 or data.shape[1] < 2:
    raise ValueError(f"asyn.npy must be (N,2) or (N,>=2). Got: {data.shape}")

print(f"➡ Loaded asyn.npy  shape: {data.shape}")
print(f"➡ Using MSM model: n_clusters={n_clusters}, lag={msm_lag}, n_states={n_states}")

# ==========================================
# IMPLIED TIMESCALES (show only first K)
# ==========================================
K_SHOW = 35

ts = M.timescales()  # in units of lag
ts_ns = ts * (msm_lag * dt_ps) / 1000.0

kmax_ts = min(K_SHOW, len(ts_ns))
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(np.arange(1, kmax_ts + 1), ts_ns[:kmax_ts], marker='o', color='blue')

for val in ts_ns[:kmax_ts]:
    ax.axhline(val, color='red', lw=0.5, alpha=0.5)

ax.set_yscale("log")
ax.set_ylabel("Timescale $t_i$ (ns)", fontsize=18)
ax.set_xlabel("IM eigenvalue index $i$", fontsize=16)
ax.set_xlim(0.5, K_SHOW + 0.5)
ax.set_xticks(np.arange(1, kmax_ts + 1))
ax.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.savefig(f"SPECTRAL_IMPLIED_TIMESCALES_nclu{n_clusters}_lag{msm_lag}_nstates{n_states}.png",
            dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# EIGENVALUE RATIO (GAP HEURISTIC) — show only first K
# ==========================================
# ratios are t_i / t_{i+1}, so there are (len(ts) - 1) of them
kmax_ratio = min(K_SHOW, max(1, len(ts) - 1))
ratios = ts[:kmax_ratio] / ts[1:kmax_ratio + 1]
x = np.arange(2, kmax_ratio + 2)  # 2..(K_SHOW+1)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, ratios, marker="o", color='red')

ax.set_xlabel("number of states", fontsize=16)   # or "index $i+1$"
ax.set_ylabel("$t_i / t_{i+1}$", fontsize=18)
ax.set_xlim(1.5, K_SHOW + 1.5)
ax.set_xticks(x)
ax.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.savefig(f"SPECTRAL_EIGEN_RATIO_nclu{n_clusters}_lag{msm_lag}_nstates{n_states}.png",
            dpi=300, bbox_inches='tight')
plt.show()

