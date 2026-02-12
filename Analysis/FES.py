#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Free Energy Surface (FES) plot using pyEMMA for NOT-REWEIGHTED FEL.
Inputs:
    data_c     -> (N, 2) array of 2D projected coordinates (z1, z2)
    protein    -> name used for output filenames
"""

import numpy as np
import matplotlib.pyplot as plt
import pyemma


############### USER INPUT #################
# data_c must be your 2D embedding (shape: N x 2)
# Replace this with your actual loaded data
# Example:
data_c = np.load("toy.npy")[:, :2]
protein = "toy_gen"
#############################################


def plot_fes(data_c, protein):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Free energy landscape plot
    _, _, misc = pyemma.plots.plot_free_energy(
        data_c[:, 0], data_c[:, 1],
        weights=None,          # Not reweighted (simple FEL)
        ax=ax,
        legacy=False,
        cbar=False,            # Turn off auto colorbar
        vmin=0,
        vmax=5
    )

    # Pretty axis formatting
    ax.tick_params(
        axis='both', which='major',
        top=True, right=True,
        direction='in',
        length=12, width=3,
        pad=10, labelsize=20
    )
    ax.ticklabel_format(axis='both', style='sci', scilimits=[-2,2], useOffset=None)

    ax.xaxis.get_offset_text().set_fontsize(20)
    ax.yaxis.get_offset_text().set_fontsize(20)

    ax.locator_params(axis='x', nbins=6)
    ax.locator_params(axis='y', nbins=6)

    ax.set_xlabel("$z_1$", size=25)
    ax.set_ylabel("$z_2$", size=25)

    # Custom colorbar (ΔG in kcal/mol)
    cbar = fig.colorbar(misc['mappable'], ax=ax)
    cbar.set_label("ΔG [kcal/mol]", size=20)
    cbar.set_ticks(np.arange(0, 5.1, 0.5))
    cbar.ax.tick_params(labelsize=15)

    # Save output
    plt.savefig(f"NOT_REWEIGHTED_FEL_{protein}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"NOT_REWEIGHTED_FEL_{protein}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"NOT_REWEIGHTED_FEL_{protein}.svg", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Free energy surface saved as NOT_REWEIGHTED_FEL_{protein}.png / pdf / svg")


# Example usage:
data_c = np.load("toy.npy")[:, :2]
plot_fes(data_c, "toy_gen")

