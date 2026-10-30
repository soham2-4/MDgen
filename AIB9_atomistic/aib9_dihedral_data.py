#!/usr/bin/env python3
"""
aib9_dihedral_data.py

Step 1 of the pipeline: extract backbone dihedrals from the raw MD
trajectories.

What changed relative to the Cartesian pipeline
-----------------------------------------------
1. Coordinates are replaced by backbone (phi, psi) angles.
   Dihedrals are invariant to global translation AND rotation. The old
   pipeline removed translation but not rotation, and a diagnostic showed
   ~78% of the frame-to-frame displacement was global tumbling. That
   signal is unpredictable in principle, so the model was spending its
   capacity on noise.

2. The 150 noise-augmented copies per replica are gone.
   Adding N(0, 0.02 A) noise 150 times adds no information (0.02 A is
   below bond vibration amplitude) and it leaked between the train and
   validation splits, because near-identical windows landed on both
   sides. Five real replicas are five real trajectories.

3. Dimensionality drops from 387 (129 atoms x 3) to 36
   (18 dihedrals x [sin, cos]), which is trainable from 5 trajectories.

Output
------
  <outdir>/<replica>.npz   angles_deg [T, D], float32
  <outdir>/meta.json       dihedral definitions and atom indices

Usage
-----
    python aib9_dihedral_data.py
    python aib9_dihedral_data.py --stride 1 --outdir data_dihedral
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np


# ============================================================
# Replica definitions
# ============================================================

DEFAULT_REPLICAS = [
    ("replica1", "only_prot.gro", "replica1_first_5000ns.xtc"),
    ("replica2", "only_prot.gro", "replica2_first_5000ns.xtc"),
    ("replica3", "only_prot.gro", "replica3_first_5000ns.xtc"),
    ("replica4", "only_prot.gro", "replica4_first_5000ns.xtc"),
    ("replica5", "only_prot.gro", "replica5_first_5000ns.xtc"),
]


# ============================================================
# Backbone dihedral definitions
# ============================================================

def backbone_dihedral_indices(universe):
    """
    Build phi and psi atom-index quadruples from a Universe.

    phi(i) = C(i-1) - N(i) - CA(i) - C(i)
    psi(i) = N(i)   - CA(i) - C(i)  - N(i+1)

    Terminal caps (ACE / NME) contribute their C and N atoms, so a capped
    9-residue peptide yields 9 phi and 9 psi.

    Returns
    -------
    idx    : [D, 4] int array of atom indices
    names  : list of D strings, e.g. "phi_3"
    """

    def pick(residue, name):
        sel = residue.atoms.select_atoms(f"name {name}")
        return int(sel[0].index) if len(sel) else None

    residues = list(universe.residues)

    bb = [
        (pick(r, "N"), pick(r, "CA"), pick(r, "C"))
        for r in residues
    ]

    idx = []
    names = []

    for i, res in enumerate(residues):

        n_i, ca_i, c_i = bb[i]

        if None in (n_i, ca_i, c_i):
            # cap residue: no full backbone triple of its own
            continue

        if i > 0 and bb[i - 1][2] is not None:
            idx.append([bb[i - 1][2], n_i, ca_i, c_i])
            names.append(f"phi_{res.resid}")

        if i < len(residues) - 1 and bb[i + 1][0] is not None:
            idx.append([n_i, ca_i, c_i, bb[i + 1][0]])
            names.append(f"psi_{res.resid}")

    if not idx:
        raise RuntimeError(
            "No backbone dihedrals found. Check that the topology uses "
            "standard atom names N, CA, C."
        )

    return np.array(idx, dtype=np.int64), names


# ============================================================
# Vectorised dihedral computation
# ============================================================

def dihedrals_from_frames(frames, idx):
    """
    Dihedral angles in degrees.

    frames : [T, n_atoms, 3]
    idx    : [D, 4] atom indices
    returns: [T, D] degrees in (-180, 180]
    """

    p0 = frames[:, idx[:, 0], :]
    p1 = frames[:, idx[:, 1], :]
    p2 = frames[:, idx[:, 2], :]
    p3 = frames[:, idx[:, 3], :]

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1 = b1 / np.maximum(
        np.linalg.norm(b1, axis=-1, keepdims=True), 1e-12
    )

    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1

    x = (v * w).sum(-1)
    y = (np.cross(b1, v) * w).sum(-1)

    return np.degrees(np.arctan2(y, x))


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Extract backbone dihedrals from MD trajectories"
    )

    parser.add_argument(
        "--outdir",
        default="data_dihedral",
        help="Directory for the extracted .npz files",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Read every Nth frame (1 = keep all)",
    )

    parser.add_argument(
        "--chunk",
        type=int,
        default=2000,
        help="Frames held in memory at once while reading",
    )

    args = parser.parse_args()

    import MDAnalysis as mda

    os.makedirs(args.outdir, exist_ok=True)

    print()
    print("=" * 70)
    print("DIHEDRAL EXTRACTION")
    print("=" * 70)
    print()

    meta = None
    written = []

    for name, topology, trajectory in DEFAULT_REPLICAS:

        if not (os.path.exists(topology) and os.path.exists(trajectory)):
            print(f"  SKIP {name}: missing {topology} or {trajectory}")
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = mda.Universe(topology, trajectory)

        idx, dih_names = backbone_dihedral_indices(u)

        if meta is None:
            meta = {
                "n_atoms": int(len(u.atoms)),
                "dihedral_names": dih_names,
                "atom_indices": idx.tolist(),
                "n_dihedrals": int(len(dih_names)),
            }
            print(f"  Atoms      {meta['n_atoms']}")
            print(f"  Dihedrals  {meta['n_dihedrals']}  ({', '.join(dih_names)})")
            print()

        # ---- stream the trajectory in chunks ----

        angles = []
        buffer = []

        for i, ts in enumerate(u.trajectory):

            if i % args.stride:
                continue

            buffer.append(u.atoms.positions.copy())

            if len(buffer) >= args.chunk:
                angles.append(
                    dihedrals_from_frames(
                        np.asarray(buffer, dtype=np.float64), idx
                    )
                )
                buffer = []

        if buffer:
            angles.append(
                dihedrals_from_frames(
                    np.asarray(buffer, dtype=np.float64), idx
                )
            )

        angles = np.concatenate(angles, axis=0).astype(np.float32)

        out_path = os.path.join(args.outdir, f"{name}.npz")

        np.savez_compressed(out_path, angles_deg=angles)

        # quick handedness readout
        phi_cols = [j for j, nm in enumerate(dih_names) if nm.startswith("phi")]
        rad = np.radians(angles[:, phi_cols])
        phi_bar = np.degrees(
            np.arctan2(np.sin(rad).mean(1), np.cos(rad).mean(1))
        )
        right = float(np.mean(phi_bar < 0))

        print(f"  {name}: {angles.shape[0]:,} frames x {angles.shape[1]} dihedrals")
        print(f"    right-handed fraction {right:.3f}   -> {out_path}")

        written.append(name)

    if meta is not None:
        with open(os.path.join(args.outdir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

    print()
    print("=" * 70)
    print(f"Wrote {len(written)} replicas to {args.outdir}/")
    print("Next: python aib9_train_dihedral.py")
    print("=" * 70)
    print()

    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
