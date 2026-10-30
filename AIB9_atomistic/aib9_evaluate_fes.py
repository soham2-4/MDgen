#!/usr/bin/env python3
"""
aib9_evaluate_fes.py

Step 4: compare the generated ensemble against the reference MD.

What it reports
---------------
1. Free energy surfaces in (mean phi, mean psi) and pooled Ramachandran.

2. Basin populations with an honest error bar. AIB9 is achiral, so the
   true left/right split is exactly 0.500/0.500. The deviation of the
   REFERENCE from 0.500 is a direct measure of how well converged the
   reference is - in the earlier run it was 0.601/0.399, i.e. only
   converged to about 0.25 kcal/mol. No model claim finer than that is
   measurable, so this number is printed as the resolution floor.

3. A symmetrised reference. Averaging the reference FES with its own
   point reflection (phi, psi) -> (-phi, -psi) imposes the exact symmetry
   and halves the sampling variance for free.

4. Basin crossing counts, which say whether either ensemble contains
   any helix flips at all.

5. Jensen-Shannon divergence between the distributions.

Usage
-----
    python aib9_evaluate_fes.py
    python aib9_evaluate_fes.py --reference data_dihedral/replica5.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aib9_common import (
    KB_KCAL,
    circular_mean_deg,
    count_transitions,
    find_replica_files,
    load_meta,
    phi_columns,
    psi_columns,
)


# ============================================================
# Free energy
# ============================================================

def fes_2d(x, y, bins, kt):

    counts, _, _ = np.histogram2d(
        x, y, bins=bins, range=[(-180, 180), (-180, 180)]
    )

    p = counts / max(counts.sum(), 1)

    with np.errstate(divide="ignore"):
        f = -kt * np.log(p)

    f[~np.isfinite(f)] = np.nan

    return f.T - np.nanmin(f), p


def symmetrise(p):
    """
    Impose the achiral symmetry (phi, psi) -> (-phi, -psi) on a
    histogram defined on a symmetric grid: average with its 180-degree
    rotation.
    """

    return 0.5 * (p + p[::-1, ::-1])


def fes_from_prob(p, kt):

    with np.errstate(divide="ignore"):
        f = -kt * np.log(p)

    f[~np.isfinite(f)] = np.nan

    return f.T - np.nanmin(f)


def fes_1d(x, bins, kt):

    counts, edges = np.histogram(x, bins=bins, range=(-180, 180))

    p = counts / max(counts.sum(), 1)

    with np.errstate(divide="ignore"):
        f = -kt * np.log(p)

    f[~np.isfinite(f)] = np.nan

    return 0.5 * (edges[:-1] + edges[1:]), f - np.nanmin(f)


def jensen_shannon(p, q):

    p = p.ravel() / max(p.sum(), 1e-30)
    q = q.ravel() / max(q.sum(), 1e-30)

    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Compare generated and reference free energy surfaces"
    )

    parser.add_argument("--data-dir", default="data_dihedral")
    parser.add_argument(
        "--generated", default="generated_dihedral/generated_angles.npz"
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Reference .npz (default: all replicas pooled)",
    )
    parser.add_argument("--outdir", default="generated_dihedral")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--bins", type=int, default=72)

    args = parser.parse_args()

    kt = KB_KCAL * args.temperature

    os.makedirs(args.outdir, exist_ok=True)

    meta = load_meta(args.data_dir)
    pcols, scols = phi_columns(meta), psi_columns(meta)

    # --------------------------------------------------
    # Reference
    # --------------------------------------------------

    if args.reference:
        ref_files = [args.reference]
    else:
        ref_files = [str(f) for f in find_replica_files(args.data_dir)]

    ref_chains = []

    for f in ref_files:
        a = np.load(f)["angles_deg"].astype(np.float32)
        ref_chains.append(a)

    ref_all = np.concatenate(ref_chains, axis=0)

    ref_phi = circular_mean_deg(ref_all[:, pcols], axis=1)
    ref_psi = circular_mean_deg(ref_all[:, scols], axis=1)

    ref_phi_chains = [
        circular_mean_deg(a[:, pcols], axis=1) for a in ref_chains
    ]

    # --------------------------------------------------
    # Generated
    # --------------------------------------------------

    g = np.load(args.generated)["angles_deg"].astype(np.float32)

    if g.ndim == 2:
        g = g[None]                       # single chain -> [1, T, D]

    n_chains, n_frames, _ = g.shape

    gen_flat = g.reshape(-1, g.shape[-1])

    gen_phi = circular_mean_deg(gen_flat[:, pcols], axis=1)
    gen_psi = circular_mean_deg(gen_flat[:, scols], axis=1)

    gen_phi_chains = circular_mean_deg(g[:, :, pcols], axis=-1)

    # --------------------------------------------------
    # Numbers
    # --------------------------------------------------

    print()
    print("=" * 72)
    print("EVALUATION")
    print("=" * 72)

    ref_right = float(np.mean(ref_phi < 0))
    gen_right = float(np.mean(gen_phi < 0))

    # The reference's own deviation from 0.5 is the resolution floor.
    floor = abs(-kt * np.log(max(ref_right, 1e-9) / max(1 - ref_right, 1e-9)))

    print()
    print(f"  reference: {ref_all.shape[0]:,} frames from "
          f"{len(ref_chains)} replica(s)")
    print(f"    right-handed  {ref_right:.4f}   (exact answer 0.500)")
    print(f"    crossings     {sum(count_transitions(p) for p in ref_phi_chains)}")
    print(f"    -> reference is converged to about {floor:.3f} kcal/mol;")
    print(f"       no model claim finer than that is measurable")

    print()
    print(f"  generated: {n_chains} chains x {n_frames:,} frames")
    print(f"    right-handed  {gen_right:.4f}")
    print(f"    crossings     {count_transitions(gen_phi_chains)}")

    if abs(gen_right - 0.5) < abs(ref_right - 0.5):
        print("    (closer to the exact 0.500 than the reference itself)")

    # --------------------------------------------------
    # Surfaces
    # --------------------------------------------------

    f_ref, p_ref = fes_2d(ref_phi, ref_psi, args.bins, kt)
    f_gen, p_gen = fes_2d(gen_phi, gen_psi, args.bins, kt)

    p_ref_sym = symmetrise(p_ref)
    f_ref_sym = fes_from_prob(p_ref_sym, kt)

    f_ref_pool, p_ref_pool = fes_2d(
        ref_all[:, pcols].ravel(), ref_all[:, scols].ravel(), args.bins, kt
    )
    f_gen_pool, p_gen_pool = fes_2d(
        gen_flat[:, pcols].ravel(), gen_flat[:, scols].ravel(), args.bins, kt
    )

    js_mean = jensen_shannon(p_ref_sym, p_gen)
    js_pool = jensen_shannon(p_ref_pool, p_gen_pool)

    print()
    print("  Jensen-Shannon divergence (bits; 0 identical, 1 disjoint)")
    print(f"    mean dihedral, vs symmetrised reference  {js_mean:.4f}")
    print(f"    pooled Ramachandran                      {js_pool:.4f}")

    # --------------------------------------------------
    # Plots
    # --------------------------------------------------

    def panel(ax, f, title, vmax):
        im = ax.imshow(
            f, origin="lower", extent=[-180, 180, -180, 180],
            aspect="equal", cmap="viridis", vmin=0, vmax=vmax,
        )
        ax.set_xlabel(r"$\phi$ (deg)")
        ax.set_ylabel(r"$\psi$ (deg)")
        ax.set_title(title)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-180, -90, 0, 90, 180])
        plt.colorbar(im, ax=ax, label="F (kcal/mol)")

    for tag, (a, b, c) in {
        "mean_dihedral": (f_ref, f_ref_sym, f_gen),
        "ramachandran": (f_ref_pool, None, f_gen_pool),
    }.items():

        panels = [(a, "reference")]
        if b is not None:
            panels.append((b, "reference (symmetrised)"))
        panels.append((c, "generated"))

        vmax = np.nanpercentile(
            np.concatenate([p[np.isfinite(p)] for p, _ in panels]), 98
        )

        fig, axes = plt.subplots(
            1, len(panels) + 1, figsize=(6 * (len(panels) + 1), 5.2)
        )

        for ax, (surf, title) in zip(axes, panels):
            panel(ax, surf, title, vmax)

        diff = panels[-1][0] - panels[0][0]
        lim = np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 95)

        im = axes[-1].imshow(
            diff, origin="lower", extent=[-180, 180, -180, 180],
            aspect="equal", cmap="RdBu_r", vmin=-lim, vmax=lim,
        )
        axes[-1].set_xlabel(r"$\phi$ (deg)")
        axes[-1].set_ylabel(r"$\psi$ (deg)")
        axes[-1].set_title("generated - reference")
        plt.colorbar(im, ax=axes[-1], label=r"$\Delta F$ (kcal/mol)")

        fig.suptitle(tag.replace("_", " "), fontsize=13)
        fig.tight_layout()

        out = os.path.join(args.outdir, f"fes_{tag}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"  saved {out}")

    # 1-D profile
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for label, x in [("reference", ref_phi), ("generated", gen_phi)]:
        c, f = fes_1d(x, args.bins, kt)
        ax.plot(c, f, lw=2, label=label)

    c, f = fes_1d(
        np.concatenate([ref_phi, -ref_phi]), args.bins, kt
    )
    ax.plot(c, f, lw=1.5, ls="--", color="k", label="reference (symmetrised)")

    ax.set_xlabel(r"mean $\phi$ (deg)")
    ax.set_ylabel("F (kcal/mol)")
    ax.set_title("Free energy along the helix-handedness coordinate")
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out = os.path.join(args.outdir, "fes_profile_1d.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  saved {out}")

    # --------------------------------------------------
    # Autocorrelation sanity check for the periodicity bug
    # --------------------------------------------------

    x = np.sin(np.radians(gen_phi_chains[0]))
    x = x - x.mean()
    v = max((x ** 2).mean(), 1e-12)

    print()
    print("  generated autocorrelation (mean phi, chain 0):")
    for lag in [1, 10, 50, 100, 250, 500, 1000, 2000]:
        if lag < len(x):
            c = float((x[:-lag] * x[lag:]).mean() / v)
            print(f"    lag {lag:5d}   {c:+.4f}")

    print()
    print("  A smooth decay is correct. A spike at one specific lag would")
    print("  mean the old block-periodicity artifact has returned.")
    print("=" * 72)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
