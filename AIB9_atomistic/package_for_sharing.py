#!/usr/bin/env python3
"""
package_for_sharing.py

Build a self-contained folder of trajectories to send to a colleague.

Produces matched pairs so the comparison is fair: the reference is passed
through the SAME reconstruction as the generated data, so both have
identical bond lengths, no tumbling, and the same frozen terminal groups.
The raw MD trajectory is included too, but it differs from the generated
one in ways that have nothing to do with the model, so it should not be
the primary comparison.

Usage:
    python package_for_sharing.py
    python package_for_sharing.py --stride 10 --chain 0
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import date


README = """AIB9 trajectory comparison
==========================

Generated {today}

WHAT IS HERE
------------

reference_reconstructed.pdb / .xtc
    The original MD trajectory ({ref_name}), converted to backbone
    torsions and rebuilt back into Cartesian coordinates.

generated_reconstructed.pdb / .xtc
    Trajectory sampled from the trained model, rebuilt the same way.

original_raw.gro / original_raw.xtc
    The untouched MD trajectory, for reference only. See the caveat below
    before comparing this one directly against the generated data.

HOW TO OPEN
-----------

    vmd   reference_reconstructed.pdb reference_reconstructed.xtc
    pymol reference_reconstructed.pdb reference_reconstructed.xtc

Any topology + trajectory viewer works. The .pdb is the topology; the
.xtc holds the coordinates.

WHICH PAIR TO COMPARE
---------------------

Compare reference_reconstructed against generated_reconstructed.

Both went through identical geometry handling, so any difference you see
is the model's doing. Comparing against original_raw.xtc instead would
show three differences that are artifacts of the pipeline rather than
the model:

  1. Tumbling. The raw trajectory has full global rotation. Reconstructed
     frames are built in a canonical frame, so they do not tumble at all.
     Side by side, the raw one spins and the other does not.

  2. Bond vibration. Real MD has roughly 0.03 A of thermal fluctuation in
     bond lengths. Reconstructed frames hold bonds at fixed ideal values,
     because bond lengths are stiff, sub-picosecond, and carry no
     information about the conformational dynamics being modelled.

  3. Terminal groups. This peptide is uncapped, so phi of residue 1 and
     psi of residue 9 are undefined. Those two groups are held at
     reference values in BOTH reconstructed trajectories and will look
     artificially rigid.

WHAT THE MODEL DOES
-------------------

The model predicts a probability distribution over the next frame's
backbone torsions given recent history, and trajectories are produced by
sampling from it one frame at a time. Cartesian coordinates are then
computed from those torsions by deterministic geometry (NeRF plus rigid
local frames measured from the original topology). Nothing about the
coordinates is learned.

Representation: {n_torsions} backbone torsions (phi and psi).
Frame spacing: {dt} in the original MD.

Note: viewers may report a 1 ps timestep. That is a placeholder written
into the .xtc, not the real spacing.

KNOWN LIMITATION
----------------

The generated free energy surface has the correct two-basin shape and
symmetry, but its barriers are roughly 1.4 kcal/mol too low, so the
transition region is over-populated. Sampling temperature calibration is
in progress.
"""


def run(cmd):
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"Command failed: {' '.join(cmd)}")


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="share_aib9")
    ap.add_argument("--reference", default="data_dihedral/replica1.npz")
    ap.add_argument(
        "--generated", default="generated_dihedral/generated_angles.npz"
    )
    ap.add_argument("--raw-topology", default="only_prot.gro")
    ap.add_argument("--raw-trajectory", default="replica1_first_5000ns.xtc")
    ap.add_argument("--chain", type=int, default=0)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--dt", default="0.1 ns")
    ap.add_argument("--include-raw", action="store_true", default=True)
    ap.add_argument("--no-raw", dest="include_raw", action="store_false")
    ap.add_argument("--zip", action="store_true",
                    help="Also produce a .zip of the folder")

    args = ap.parse_args()

    out = args.outdir
    os.makedirs(out, exist_ok=True)

    common = ["--stride", str(args.stride)]
    if args.max_frames:
        common += ["--max-frames", str(args.max_frames)]

    print("\nRebuilding reference...")
    run([sys.executable, "aib9_reconstruct.py",
         "--angles", args.reference,
         "--chain", "0",
         "--outdir", os.path.join(out, "_ref")] + common)

    print("\nRebuilding generated...")
    run([sys.executable, "aib9_reconstruct.py",
         "--angles", args.generated,
         "--chain", str(args.chain),
         "--outdir", os.path.join(out, "_gen")] + common)

    # rename into something a colleague can read
    for src, dst in [
        ("_ref/generated.pdb", "reference_reconstructed.pdb"),
        ("_ref/generated.xtc", "reference_reconstructed.xtc"),
        ("_gen/generated.pdb", "generated_reconstructed.pdb"),
        ("_gen/generated.xtc", "generated_reconstructed.xtc"),
    ]:
        shutil.move(os.path.join(out, src), os.path.join(out, dst))

    shutil.rmtree(os.path.join(out, "_ref"), ignore_errors=True)
    shutil.rmtree(os.path.join(out, "_gen"), ignore_errors=True)

    if args.include_raw:
        for src, dst in [
            (args.raw_topology, "original_raw.gro"),
            (args.raw_trajectory, "original_raw.xtc"),
        ]:
            if os.path.exists(src):
                print(f"  copying {src}")
                shutil.copy2(src, os.path.join(out, dst))
            else:
                print(f"  SKIP {src} (not found)")

    # figure out the torsion count for the README
    n_torsions = "?"
    try:
        import json
        meta = json.load(open("data_dihedral/meta.json"))
        n_torsions = meta["n_dihedrals"]
    except Exception:
        pass

    with open(os.path.join(out, "README.txt"), "w") as fh:
        fh.write(README.format(
            today=date.today().isoformat(),
            ref_name=os.path.basename(args.reference),
            n_torsions=n_torsions,
            dt=args.dt,
        ))

    # also include the free energy figures if they exist
    for f in ["fes_mean_dihedral.png", "fes_ramachandran.png",
              "fes_profile_1d.png"]:
        src = os.path.join("generated_dihedral", f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, f))

    print("\n" + "=" * 60)
    print(f"Bundle: {out}/")
    total = 0
    for f in sorted(os.listdir(out)):
        size = os.path.getsize(os.path.join(out, f)) / 1e6
        total += size
        print(f"  {f:38s} {size:8.1f} MB")
    print(f"  {'TOTAL':38s} {total:8.1f} MB")

    if args.zip:
        shutil.make_archive(out, "zip", out)
        print(f"\nZipped: {out}.zip "
              f"({os.path.getsize(out + '.zip') / 1e6:.1f} MB)")

    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
