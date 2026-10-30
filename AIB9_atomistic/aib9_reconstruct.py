#!/usr/bin/env python3
"""
aib9_reconstruct.py

Rebuild all-atom Cartesian coordinates from generated backbone dihedrals
and write PDB / XTC files.

How this works
--------------
The model predicts torsions, not coordinates. This script *computes*
coordinates from those torsions - there is nothing learned or stochastic
here, it is deterministic geometry. Feed it the same torsions twice and
you get identical output.

Two stages:

1. Backbone (N, CA, C) is built atom by atom with NeRF (natural
   extension reference frame). Given three already-placed atoms plus a
   bond length, bond angle and torsion, the fourth atom's position is
   fully determined by trigonometry.

2. Every remaining atom (O, CB1, CB2, amide H, methyl H, caps) is placed
   from its position in a LOCAL FRAME of three nearby already-placed
   atoms, with that local position measured once from the reference
   structure. This is exact for rigid groups, which is what all of these
   are.

Bond lengths, bond angles, omega, and the local frames are all measured
from YOUR topology file, not hardcoded, so the output matches your force
field.

What is approximated
--------------------
Bond lengths and angles are held at their reference values, so the
output has no bond vibration (real MD has about 0.03 A). Those degrees
of freedom are stiff, sub-picosecond, and carry no information about the
helix flip, so nothing relevant is lost.

Omega is taken from the reference structure rather than generated, since
the current model predicts phi and psi only. Peptide bonds stay trans.

Usage
-----
    python aib9_reconstruct.py --chain 0
    python aib9_reconstruct.py --chain 0 --stride 10 --max-frames 5000
    python aib9_reconstruct.py --angles data_dihedral/replica1.npz --chain -1
    python aib9_reconstruct.py --self-test
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np


# ============================================================
# Geometry primitives
# ============================================================

def place_nerf(a, b, c, bond, angle, torsion):
    """
    Place atom D given three placed atoms A, B, C.

    bond    : |C-D|
    angle   : angle B-C-D   (radians)
    torsion : dihedral A-B-C-D (radians)

    Vectorised over a leading batch dimension.
    """

    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    c = np.atleast_2d(c)

    bond = np.atleast_1d(bond)
    angle = np.atleast_1d(angle)
    torsion = np.atleast_1d(torsion)

    d2 = np.stack(
        [
            -bond * np.cos(angle),
            bond * np.sin(angle) * np.cos(torsion),
            bond * np.sin(angle) * np.sin(torsion),
        ],
        axis=-1,
    )

    bc = c - b
    bc = bc / np.linalg.norm(bc, axis=-1, keepdims=True)

    ab = b - a

    n = np.cross(ab, bc)
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)

    m = np.cross(n, bc)

    # columns [bc, m, n]
    basis = np.stack([bc, m, n], axis=-1)

    return c + np.einsum("bij,bj->bi", basis, d2)


def local_frame(b, a, p):
    """
    Orthonormal frame anchored at p, built from three placed atoms.
    Returns (origin, R) with R's columns the basis vectors.
    """

    e1 = p - a
    e1 = e1 / np.maximum(np.linalg.norm(e1, axis=-1, keepdims=True), 1e-12)

    v2 = b - a
    v2 = v2 - (v2 * e1).sum(-1, keepdims=True) * e1
    e2 = v2 / np.maximum(np.linalg.norm(v2, axis=-1, keepdims=True), 1e-12)

    e3 = np.cross(e1, e2)

    return p, np.stack([e1, e2, e3], axis=-1)


def to_local(x, b, a, p):
    origin, R = local_frame(b, a, p)
    return np.einsum("bji,bj->bi", R, x - origin)


def from_local(loc, b, a, p):
    origin, R = local_frame(b, a, p)
    return origin + np.einsum("bij,bj->bi", R, loc)


def dihedral(p0, p1, p2, p3):
    """Signed dihedral in radians, batched."""

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1 = b1 / np.maximum(np.linalg.norm(b1, axis=-1, keepdims=True), 1e-12)

    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1

    return np.arctan2(
        (np.cross(b1, v) * w).sum(-1), (v * w).sum(-1)
    )


def angle_between(p0, p1, p2):
    """Angle at p1, radians, batched."""

    v1 = p0 - p1
    v2 = p2 - p1

    v1 = v1 / np.maximum(np.linalg.norm(v1, axis=-1, keepdims=True), 1e-12)
    v2 = v2 / np.maximum(np.linalg.norm(v2, axis=-1, keepdims=True), 1e-12)

    return np.arccos(np.clip((v1 * v2).sum(-1), -1.0, 1.0))


# ============================================================
# Topology analysis
# ============================================================

class MoleculeTemplate:
    """
    Everything needed to rebuild the molecule: the backbone chain order,
    reference internal geometry, and local-frame positions for every
    non-backbone atom.
    """

    def __init__(self, universe):

        self.u = universe
        self.n_atoms = len(universe.atoms)

        ref = universe.atoms.positions.astype(np.float64)[None]  # [1, N, 3]

        self._build_backbone_chain()
        self._measure_backbone(ref)

    # ---------- backbone ----------

    def _build_backbone_chain(self):
        """
        Ordered list of backbone atom indices:
        [C_cap, N1, CA1, C1, N2, CA2, C2, ..., N_cap]

        and the phi/psi slot each depends on.
        """

        def pick(res, name):
            sel = res.atoms.select_atoms(f"name {name}")
            return int(sel[0].index) if len(sel) else None

        residues = list(self.u.residues)

        core = []          # residues with a full N/CA/C triple
        for i, r in enumerate(residues):
            n, ca, c = pick(r, "N"), pick(r, "CA"), pick(r, "C")
            if None not in (n, ca, c):
                core.append((i, r.resid, n, ca, c))

        if not core:
            raise RuntimeError("No residues with N, CA and C found")

        self.core = core

        first_i = core[0][0]
        last_i = core[-1][0]

        # N-terminal cap carbonyl carbon, needed for phi of residue 1
        self.cap_c = (
            pick(residues[first_i - 1], "C") if first_i > 0 else None
        )

        # C-terminal cap nitrogen, produced by psi of the last residue
        self.cap_n = (
            pick(residues[last_i + 1], "N")
            if last_i + 1 < len(residues) else None
        )

        chain = []

        if self.cap_c is not None:
            chain.append(self.cap_c)

        for _, _, n, ca, c in core:
            chain += [n, ca, c]

        if self.cap_n is not None:
            chain.append(self.cap_n)

        self.chain = chain
        self.chain_names = [self.u.atoms[i].name for i in chain]
        self.chain_resids = [int(self.u.atoms[i].resid) for i in chain]
        self.resids = [r for _, r, _, _, _ in core]

        # names of the torsions the model provides, in extraction order
        self.torsion_names = []
        for _, resid, _, _, _ in core:
            self.torsion_names.append(f"phi_{resid}")
            self.torsion_names.append(f"psi_{resid}")

    def _measure_backbone(self, ref):
        """
        Reference bond lengths and angles along the backbone chain, plus
        the omega torsions, which the model does not predict.
        """

        c = self.chain

        self.bonds = np.array([
            np.linalg.norm(ref[0, c[i + 1]] - ref[0, c[i]])
            for i in range(len(c) - 1)
        ])

        self.angles = np.array([
            angle_between(
                ref[:, c[i]], ref[:, c[i + 1]], ref[:, c[i + 2]]
            )[0]
            for i in range(len(c) - 2)
        ])

        self.ref_torsions = np.array([
            dihedral(
                ref[:, c[i]], ref[:, c[i + 1]],
                ref[:, c[i + 2]], ref[:, c[i + 3]]
            )[0]
            for i in range(len(c) - 3)
        ])

        # Classify each torsion from the names of its two central atoms.
        # This is robust to whether the peptide is capped: an uncapped
        # peptide simply has no phi for the first residue and no psi for
        # the last, and the corresponding chain positions never arise.
        #
        #   N -> CA  : phi of that residue
        #   CA -> C  : psi of that residue
        #   C  -> N  : omega, not predicted, taken from the reference

        self.torsion_kind = []

        for k in range(len(c) - 3):

            n1 = self.chain_names[k + 1]
            n2 = self.chain_names[k + 2]

            if (n1, n2) == ("N", "CA"):
                self.torsion_kind.append(("phi", self.chain_resids[k + 1]))
            elif (n1, n2) == ("CA", "C"):
                self.torsion_kind.append(("psi", self.chain_resids[k + 1]))
            else:
                self.torsion_kind.append(("fixed", k))

    # ---------- everything else ----------

    def _build_placement_plan(self, ref):
        """
        For each non-backbone atom, choose three already-placed anchors
        and record its position in their local frame.
        """



# ============================================================
# Template construction without networkx
# ============================================================

def build_template(universe):
    """
    Returns a dict describing how to rebuild every atom.
    """

    tpl = MoleculeTemplate.__new__(MoleculeTemplate)
    tpl.u = universe
    tpl.n_atoms = len(universe.atoms)

    ref = universe.atoms.positions.astype(np.float64)[None]

    MoleculeTemplate._build_backbone_chain(tpl)
    MoleculeTemplate._measure_backbone(tpl, ref)

    # ---- bond graph ----

    neighbours = {i: set() for i in range(tpl.n_atoms)}

    for b in universe.bonds:
        i, j = int(b.atoms[0].index), int(b.atoms[1].index)
        neighbours[i].add(j)
        neighbours[j].add(i)

    placed = list(tpl.chain)
    placed_set = set(placed)

    plan = []          # (atom, b, a, p, local_xyz)

    frontier = list(placed)

    while True:

        progressed = False

        for p in list(frontier):

            for x in sorted(neighbours[p]):

                if x in placed_set:
                    continue

                anchors = _choose_anchors(p, x, neighbours, placed_set, ref)

                if anchors is None:
                    continue

                b_i, a_i = anchors

                loc = to_local(
                    ref[:, x], ref[:, b_i], ref[:, a_i], ref[:, p]
                )[0]

                plan.append((x, b_i, a_i, p, loc))

                placed.append(x)
                placed_set.add(x)
                frontier.append(x)

                progressed = True

        if not progressed:
            break

    missing = set(range(tpl.n_atoms)) - placed_set

    if missing:
        names = [universe.atoms[i].name for i in sorted(missing)]
        raise RuntimeError(
            f"Could not place {len(missing)} atoms: {names[:10]}. "
            "Check that bonds were guessed correctly."
        )

    tpl.plan = plan

    return tpl


def _choose_anchors(p, x, neighbours, placed_set, ref):
    """
    Pick two placed atoms (b, a) so that (b, a, p) is a frame in which x
    is RIGID.

    Anchor choice is not arbitrary. Both anchors must be rigidly related
    to x, otherwise x is frozen against the wrong reference and moves
    incorrectly whenever a torsion changes.

    The two candidate kinds are:

      siblings          other substituents of the same parent p. These
                        are rigid with respect to x for any hybridisation
                        of p, because rotating about a bond to p moves
                        x and its siblings together.

      grandparents      neighbours of a. These are NOT generally rigid
                        with respect to x: the a-p bond is usually
                        rotatable, so the dihedral (that atom)-a-p-x is
                        a torsional degree of freedom.

    Siblings must therefore be tried FIRST. Preferring grandparents is
    what produced impossible carbonyl geometry: the amide oxygen bonded
    to C(i) was anchored to (N(i), CA(i), C(i)), which freezes the
    N-CA-C-O dihedral at its reference value. That dihedral is psi + 180,
    so whenever the generated psi differed from the reference, O rotated
    away from the peptide plane and could land on top of N(i+1),
    producing a three-membered C-O-N ring.
    """

    candidates_a = [
        n for n in sorted(neighbours[p]) if n in placed_set and n != x
    ]

    for a in candidates_a:

        siblings = [n for n in candidates_a if n != a]

        grandparents = [
            n for n in sorted(neighbours[a])
            if n in placed_set and n not in (p, x)
        ]

        for b in siblings + grandparents:

            v1 = ref[0, p] - ref[0, a]
            v2 = ref[0, b] - ref[0, a]

            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)

            if n1 < 1e-6 or n2 < 1e-6:
                continue

            cosang = abs(float(v1 @ v2) / (n1 * n2))

            if cosang < 0.98:          # not close to collinear
                return b, a

    return None


# ============================================================
# Reconstruction
# ============================================================

def reconstruct(tpl, angles_deg, name_to_col):
    """
    Build Cartesian coordinates for a batch of frames.

    angles_deg  : [B, D] degrees, columns as in meta.json
    name_to_col : {"phi_2": 0, "psi_1": 1, ...}
    returns     : [B, n_atoms, 3]
    """

    angles = np.radians(np.atleast_2d(angles_deg))

    B = angles.shape[0]

    chain = tpl.chain

    xyz = np.zeros((B, tpl.n_atoms, 3), dtype=np.float64)

    # --- seed the first three chain atoms in a canonical frame ---

    xyz[:, chain[0]] = np.array([0.0, 0.0, 0.0])
    xyz[:, chain[1]] = np.array([tpl.bonds[0], 0.0, 0.0])

    a2 = tpl.angles[0]
    xyz[:, chain[2]] = xyz[:, chain[1]] + np.array([
        -tpl.bonds[1] * np.cos(a2),
        tpl.bonds[1] * np.sin(a2),
        0.0,
    ])

    # --- NeRF along the backbone ---

    for k, (kind, ref_k) in enumerate(tpl.torsion_kind):

        target = chain[k + 3]

        if kind == "fixed":
            tau = np.full(B, tpl.ref_torsions[k])
        else:
            col = name_to_col.get(f"{kind}_{ref_k}")
            if col is None:
                # not predicted by the model: hold at the reference value
                tau = np.full(B, tpl.ref_torsions[k])
            else:
                tau = angles[:, col]

        xyz[:, target] = place_nerf(
            xyz[:, chain[k]],
            xyz[:, chain[k + 1]],
            xyz[:, chain[k + 2]],
            np.full(B, tpl.bonds[k + 2]),
            np.full(B, tpl.angles[k + 1]),
            tau,
        )

    # --- everything else, from local frames ---

    for x, b_i, a_i, p_i, loc in tpl.plan:
        xyz[:, x] = from_local(
            np.repeat(loc[None], B, axis=0),
            xyz[:, b_i], xyz[:, a_i], xyz[:, p_i],
        )

    return xyz


# ============================================================
# Self test
# ============================================================

def self_test():
    """
    Round trip on synthetic peptides, capped and uncapped: target
    torsions in, coordinates out, torsions measured back out of the
    coordinates. The uncapped case matters because an uncapped peptide
    has no phi for its first residue and no psi for its last, so the
    torsion columns do not line up one-to-one with residues.
    """

    print("\nSELF TEST: torsion round trip (capped)\n" + "-" * 46)

    rng = np.random.default_rng(0)

    # ---- build a synthetic capped 4-residue peptide ----
    # atoms: CAP_C, CAP_O, CAP_CH3, then per residue N, H, CA, CB, C, O,
    # then CAP_N, CAP_H, CAP_CH3B
    n_res = 4

    names, resids, resnames = [], [], []

    names += ["C", "O", "CH3"]
    resids += [0, 0, 0]
    resnames += ["ACE"] * 3

    for r in range(1, n_res + 1):
        names += ["N", "H", "CA", "CB", "C", "O"]
        resids += [r] * 6
        resnames += ["AIB"] * 6

    names += ["N", "H", "CH3"]
    resids += [n_res + 1] * 3
    resnames += ["NME"] * 3

    n_atoms = len(names)

    idx = {}
    for i, (nm, rid) in enumerate(zip(names, resids)):
        idx[(rid, nm)] = i

    bonds = []
    bonds += [(idx[(0, "C")], idx[(0, "O")]), (idx[(0, "C")], idx[(0, "CH3")])]

    prev_c = idx[(0, "C")]

    for r in range(1, n_res + 1):
        N, H = idx[(r, "N")], idx[(r, "H")]
        CA, CB = idx[(r, "CA")], idx[(r, "CB")]
        C, O = idx[(r, "C")], idx[(r, "O")]
        bonds += [(prev_c, N), (N, H), (N, CA), (CA, CB), (CA, C), (C, O)]
        prev_c = C

    Nc = idx[(n_res + 1, "N")]
    bonds += [(prev_c, Nc), (Nc, idx[(n_res + 1, "H")]),
              (Nc, idx[(n_res + 1, "CH3")])]

    import MDAnalysis as mda

    u = mda.Universe.empty(
        n_atoms,
        n_residues=n_res + 2,
        atom_resindex=[
            0 if r == 0 else (r if r <= n_res else n_res + 1)
            for r in resids
        ],
        trajectory=True,
    )

    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resid", list(range(0, n_res + 2)))
    u.add_TopologyAttr("resname", ["ACE"] + ["AIB"] * n_res + ["NME"])
    u.add_TopologyAttr("type", ["C"] * n_atoms)
    u.add_bonds(bonds)

    # plausible starting coordinates: a crude extended chain plus noise
    pos = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        pos[i] = [1.5 * i, 0.8 * np.sin(i), 0.8 * np.cos(i)]
    pos += rng.normal(0, 0.15, pos.shape)

    u.atoms.positions = pos.astype(np.float32)

    tpl = build_template(u)

    print(f"  atoms {n_atoms}   backbone chain {len(tpl.chain)}   "
          f"other atoms {len(tpl.plan)}")

    # ---- target torsions ----

    B = 5
    phi_t = rng.uniform(-180, 180, (B, n_res))
    psi_t = rng.uniform(-180, 180, (B, n_res))

    # interleave into a single [B, D] array with a name map, matching
    # what meta.json provides
    names, cols = [], []
    for r in range(1, n_res + 1):
        names += [f"phi_{r}", f"psi_{r}"]
        cols += [phi_t[:, r - 1], psi_t[:, r - 1]]

    angles = np.stack(cols, axis=1)
    name_to_col = {nm: i for i, nm in enumerate(names)}

    xyz = reconstruct(tpl, angles, name_to_col)

    # ---- measure them back ----

    phi_m = np.zeros_like(phi_t)
    psi_m = np.zeros_like(psi_t)

    prev_c = idx[(0, "C")]

    for r in range(1, n_res + 1):
        N, CA, C = idx[(r, "N")], idx[(r, "CA")], idx[(r, "C")]
        nxt = idx[(r + 1, "N")]
        phi_m[:, r - 1] = np.degrees(
            dihedral(xyz[:, prev_c], xyz[:, N], xyz[:, CA], xyz[:, C])
        )
        psi_m[:, r - 1] = np.degrees(
            dihedral(xyz[:, N], xyz[:, CA], xyz[:, C], xyz[:, nxt])
        )
        prev_c = C

    def wrap(d):
        return (d + 180) % 360 - 180

    e_phi = np.abs(wrap(phi_m - phi_t)).max()
    e_psi = np.abs(wrap(psi_m - psi_t)).max()

    print(f"  max |phi error| = {e_phi:.2e} deg")
    print(f"  max |psi error| = {e_psi:.2e} deg")

    # ---- bond lengths preserved ----

    d = np.array([
        np.linalg.norm(xyz[:, i] - xyz[:, j], axis=-1) for i, j in bonds
    ])
    d_ref = np.array([
        np.linalg.norm(pos[i] - pos[j]) for i, j in bonds
    ])

    e_bond = np.abs(d - d_ref[:, None]).max()

    print(f"  max bond length drift = {e_bond:.2e} A")

    # 1e-4 A is far below bond vibration (~0.03 A); this is float roundoff
    ok = e_phi < 1e-6 and e_psi < 1e-6 and e_bond < 1e-4

    print("\n  " + ("PASS" if ok else "FAIL"))

    ok_uncapped = _self_test_uncapped()

    both = ok and ok_uncapped

    print("\n  OVERALL: " + ("PASS" if both else "FAIL") + "\n")

    return 0 if both else 1


def _self_test_uncapped(n_res=9):
    """
    Uncapped peptide: 2*n_res - 2 torsions, offset from the residue index.
    """

    print("\nSELF TEST: torsion round trip (uncapped)\n" + "-" * 46)

    import MDAnalysis as mda

    rng = np.random.default_rng(1)

    names, resids = [], []
    for r in range(1, n_res + 1):
        names += ["N", "H", "CA", "CB1", "CB2", "C", "O"]
        resids += [r] * 7

    n_atoms = len(names)
    idx = {(rid, nm): i for i, (nm, rid) in enumerate(zip(names, resids))}

    bonds, prev_c = [], None
    for r in range(1, n_res + 1):
        N, CA, C = idx[(r, "N")], idx[(r, "CA")], idx[(r, "C")]
        bonds += [
            (N, idx[(r, "H")]), (N, CA),
            (CA, idx[(r, "CB1")]), (CA, idx[(r, "CB2")]),
            (CA, C), (C, idx[(r, "O")]),
        ]
        if prev_c is not None:
            bonds.append((prev_c, N))
        prev_c = C

    u = mda.Universe.empty(
        n_atoms, n_residues=n_res,
        atom_resindex=[r - 1 for r in resids], trajectory=True,
    )
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resid", list(range(1, n_res + 1)))
    u.add_TopologyAttr("resname", ["AIB"] * n_res)
    u.add_TopologyAttr("type", ["C"] * n_atoms)
    u.add_bonds(bonds)

    pos = np.array([
        [1.5 * i, 0.8 * np.sin(i), 0.8 * np.cos(i)] for i in range(n_atoms)
    ]) + rng.normal(0, 0.15, (n_atoms, 3))

    u.atoms.positions = pos.astype(np.float32)

    tpl = build_template(u)

    dnames = []
    for r in range(1, n_res + 1):
        if r > 1:
            dnames.append(f"phi_{r}")
        if r < n_res:
            dnames.append(f"psi_{r}")

    name_to_col = {nm: i for i, nm in enumerate(dnames)}

    needed = {f"{k}_{r}" for k, r in tpl.torsion_kind if k != "fixed"}

    print(f"  atoms {n_atoms}   backbone chain {len(tpl.chain)}   "
          f"other atoms {len(tpl.plan)}")
    print(f"  torsions needed {len(needed)}   model provides {len(dnames)}")

    if needed != set(dnames):
        print(f"  mismatch: {needed ^ set(dnames)}")
        print("\n  FAIL")
        return False

    B = 6
    targets = rng.uniform(-180, 180, (B, len(dnames)))

    xyz = reconstruct(tpl, targets, name_to_col)

    def wrap(d):
        return (d + 180) % 360 - 180

    err, prev_c = [], None

    for r in range(1, n_res + 1):
        N, CA, C = idx[(r, "N")], idx[(r, "CA")], idx[(r, "C")]
        if r > 1:
            m = np.degrees(
                dihedral(xyz[:, prev_c], xyz[:, N], xyz[:, CA], xyz[:, C])
            )
            err.append(
                np.abs(wrap(m - targets[:, name_to_col[f"phi_{r}"]])).max()
            )
        if r < n_res:
            m = np.degrees(
                dihedral(xyz[:, N], xyz[:, CA], xyz[:, C],
                         xyz[:, idx[(r + 1, "N")]])
            )
            err.append(
                np.abs(wrap(m - targets[:, name_to_col[f"psi_{r}"]])).max()
            )
        prev_c = C

    e = max(err)

    d = np.array([
        np.linalg.norm(xyz[:, i] - xyz[:, j], axis=-1) for i, j in bonds
    ])
    d_ref = np.array([np.linalg.norm(pos[i] - pos[j]) for i, j in bonds])
    e_bond = np.abs(d - d_ref[:, None]).max()

    # 1-3 distances: these fix the BOND ANGLES, and unlike bond lengths
    # they are what a wrong anchor choice destroys. Checking only bond
    # lengths passes happily while the carbonyl oxygen swings out of the
    # peptide plane and collapses onto the next nitrogen.
    nb = {}
    for i, j in bonds:
        nb.setdefault(i, set()).add(j)
        nb.setdefault(j, set()).add(i)

    pairs_13 = set()
    for mid, ends in nb.items():
        ends = sorted(ends)
        for u in range(len(ends)):
            for v in range(u + 1, len(ends)):
                pairs_13.add((ends[u], ends[v]))

    pairs_13 = sorted(pairs_13)

    g = np.array([
        np.linalg.norm(xyz[:, i] - xyz[:, j], axis=-1) for i, j in pairs_13
    ])
    e_13 = float(g.std(axis=1).max())

    worst = int(np.argmax(g.std(axis=1)))
    wi, wj = pairs_13[worst]

    print(f"  max |torsion error| = {e:.2e} deg")
    print(f"  max bond length drift = {e_bond:.2e} A")
    print(f"  max 1-3 distance drift = {e_13:.2e} A "
          f"(worst: atoms {wi}-{wj})")

    ok = e < 1e-6 and e_bond < 1e-4 and e_13 < 1e-4

    print("\n  " + ("PASS" if ok else "FAIL"))

    return ok


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Rebuild Cartesian coordinates from generated dihedrals"
    )

    parser.add_argument("--topology", default="only_prot.gro")
    parser.add_argument("--data-dir", default="data_dihedral")
    parser.add_argument(
        "--angles", default="generated_dihedral/generated_angles.npz"
    )
    parser.add_argument("--outdir", default="generated_dihedral")

    parser.add_argument("--chain", type=int, default=0,
                        help="Which generated chain to rebuild (-1 = all)")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--batch", type=int, default=500,
                        help="Frames rebuilt per batch")

    parser.add_argument("--self-test", action="store_true")

    args = parser.parse_args()

    if args.self_test:
        return self_test()

    import MDAnalysis as mda

    from aib9_common import load_meta

    # --------------------------------------------------
    # Template
    # --------------------------------------------------

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            u = mda.Universe(args.topology, to_guess=("bonds",))
        except TypeError:
            u = mda.Universe(args.topology, guess_bonds=True)

    tpl = build_template(u)

    print()
    print("=" * 70)
    print("RECONSTRUCTION")
    print("=" * 70)
    print(f"  topology        {args.topology}")
    print(f"  atoms           {tpl.n_atoms}")
    print(f"  backbone chain  {len(tpl.chain)}")
    print(f"  placed by frame {len(tpl.plan)}")
    print(f"  residues        {tpl.resids}")

    omega = np.array([
        tpl.ref_torsions[k]
        for kind, k in tpl.torsion_kind if kind == "fixed"
    ])
    if len(omega):
        # circular mean: a plain average of angles near +/-180 wraps to 0
        omega_bar = np.degrees(
            np.arctan2(np.sin(omega).mean(), np.cos(omega).mean())
        )
        print(f"  omega (fixed)   mean {omega_bar:.1f} deg "
              f"({len(omega)} peptide bonds)")

    # --------------------------------------------------
    # Angles
    # --------------------------------------------------

    meta = load_meta(args.data_dir)

    name_to_col = {nm: i for i, nm in enumerate(meta["dihedral_names"])}

    needed = [
        f"{kind}_{r}" for kind, r in tpl.torsion_kind if kind != "fixed"
    ]
    missing = [nm for nm in needed if nm not in name_to_col]

    print(f"  torsions needed {len(needed)}  "
          f"provided by model {len(needed) - len(missing)}")

    if missing:
        print(f"  held at reference: {', '.join(missing)}")

    data = np.load(args.angles)
    ang = data["angles_deg"].astype(np.float64)

    if ang.ndim == 2:
        ang = ang[None]

    if args.chain >= 0:
        ang = ang[args.chain : args.chain + 1]

    ang = ang.reshape(-1, ang.shape[-1])

    if args.stride > 1:
        ang = ang[:: args.stride]

    if args.max_frames:
        ang = ang[: args.max_frames]

    print(f"\n  frames to build {ang.shape[0]:,}")

    # --------------------------------------------------
    # Build and write
    # --------------------------------------------------

    os.makedirs(args.outdir, exist_ok=True)

    pdb_path = os.path.join(args.outdir, "generated.pdb")
    xtc_path = os.path.join(args.outdir, "generated.xtc")

    first = reconstruct(tpl, ang[:1], name_to_col)

    u.atoms.positions = first[0].astype(np.float32)
    u.atoms.write(pdb_path)

    print(f"  wrote {pdb_path}")

    with mda.Writer(xtc_path, n_atoms=tpl.n_atoms) as w:

        for lo in range(0, ang.shape[0], args.batch):

            chunk = ang[lo : lo + args.batch]

            xyz = reconstruct(tpl, chunk, name_to_col)

            for frame in xyz:
                u.atoms.positions = frame.astype(np.float32)
                w.write(u.atoms)

            print(f"    {min(lo + args.batch, ang.shape[0]):,} / "
                  f"{ang.shape[0]:,}")

    print(f"  wrote {xtc_path}")

    # --------------------------------------------------
    # Verify
    # --------------------------------------------------

    check = reconstruct(tpl, ang[:200], name_to_col)

    bond_idx = np.array([
        [int(b.atoms[0].index), int(b.atoms[1].index)] for b in u.bonds
    ])

    d = np.linalg.norm(
        check[:, bond_idx[:, 0]] - check[:, bond_idx[:, 1]], axis=-1
    )

    print()
    print(f"  bond lengths    {d.mean():.3f} +/- {d.std():.3f} A"
          f"   range [{d.min():.2f}, {d.max():.2f}]")
    print()
    print("  Load in VMD with:")
    print(f"    vmd {pdb_path} {xtc_path}")
    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
