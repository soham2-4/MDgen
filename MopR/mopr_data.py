"""
mopr_data.py  -  Step 1 of the MopR-phenol pipeline (replaces data.py of the P450 pipeline).

Reads the 13 GROMACS `gmx rms` ligand-RMSD series (.xvg), block-averages them to a
coarser time step, writes one folder per source trajectory containing `time.npy`
and N augmented copies `traj_000.npy ... traj_{N-1}.npy`, and writes `split.json`
that assigns whole source trajectories to train / test.

Folder layout is identical to the P450 pipeline so mopr_trans.py / mopr_inference.py
are drop-in replacements for trans1.py / inference2.py:

    synthetic_trajectories/
        split.json
        r1110/      time.npy  traj_000.npy ...
        r217/       ...
        ...

Differences from data.py (all deliberate):
  * block-average (mean of `downsample` consecutive 10-ps frames) instead of striding,
    so the 1-ns value is a real average rather than one noisy frame;
  * noise amplitude is a parameter in nm and defaults to 0.02 nm (data.py used
    1e-4 .. 1e-3 nm, which is far below the thermal fluctuation of the signal and
    produces 1000 copies that are numerically the same curve);
  * optional time-shift augmentation: drop a random slice of the unbound prefix so the
    binding event moves earlier (the unbound phase is stationary, so this is a
    physically meaningful way to create new-looking trajectories);
  * a train/test split BY SOURCE TRAJECTORY, so the model is evaluated on binding
    events it has never seen (in the P450 pipeline, inference2.py reads traj_003 of a
    training folder, i.e. a near-identical copy of a training curve).

Usage:
    python mopr_data.py --xvg_dir sahil_MopR_phenol_raw_data --out synthetic_trajectories \
        --downsample 100 --n_copies 50 --noise 0.02 --max_shift_ns 150 \
        --test r1110 rmsd_1016 rmsd_1125 --exclude r223
"""
import argparse
import json
import os
import glob
import numpy as np


def load_xvg(filename):
    time, value = [], []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith(("#", "@")):
                continue
            parts = line.split()
            if len(parts) >= 2:
                time.append(float(parts[0]))
                value.append(float(parts[1]))
    return np.asarray(time), np.asarray(value)


def block_average(t, x, factor):
    """Mean over consecutive blocks of `factor` frames; drops the incomplete tail."""
    n = (len(x) // factor) * factor
    xb = x[:n].reshape(-1, factor).mean(axis=1)
    tb = t[:n].reshape(-1, factor)[:, 0]          # time stamp = start of block
    return tb, xb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xvg_dir", required=True, help="folder with the 13 .xvg files")
    ap.add_argument("--out", default="synthetic_trajectories")
    ap.add_argument("--downsample", type=int, default=100,
                    help="frames per block; 100 x 10 ps = 1 ns per step (default)")
    ap.add_argument("--n_copies", type=int, default=50, help="augmented copies per source trajectory")
    ap.add_argument("--noise", type=float, default=0.02, help="Gaussian noise SD added to each copy (nm)")
    ap.add_argument("--max_shift_ns", type=float, default=0.0,
                    help="if >0, each copy drops a random 0..max_shift prefix (ns) of the unbound phase")
    ap.add_argument("--clip", type=float, default=None,
                    help="optional upper clip for RMSD (nm); values above are 'far away' anyway")
    ap.add_argument("--test", nargs="*", default=["r428", "rmsd_716"],
                    help="source names held out for testing (never copied into training)")
    ap.add_argument("--exclude", nargs="*", default=[], help="source names to leave out entirely (e.g. outliers)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    files = sorted(glob.glob(os.path.join(args.xvg_dir, "*.xvg")))
    if not files:
        raise SystemExit(f"no .xvg files in {args.xvg_dir}")
    os.makedirs(args.out, exist_ok=True)

    names = []
    for fn in files:
        name = os.path.splitext(os.path.basename(fn))[0]
        if name in args.exclude:
            print(f"{name:12s} excluded"); continue
        t, x = load_xvg(fn)
        dt_ps = np.median(np.diff(t)) * 1000.0
        tb, xb = block_average(t, x, args.downsample)
        if args.clip is not None:
            xb = np.minimum(xb, args.clip)
        step_ns = tb[1] - tb[0]

        folder = os.path.join(args.out, name)
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "time.npy"), tb)
        np.save(os.path.join(folder, "original.npy"), xb)       # clean, un-augmented series

        is_test = name in args.test
        n_copies = 1 if is_test else args.n_copies             # test folders: one clean copy only
        max_shift = 0 if is_test else int(round(args.max_shift_ns / step_ns))
        for i in range(n_copies):
            shift = rng.integers(0, max_shift + 1) if max_shift > 0 else 0
            y = xb[shift:].copy()
            if not is_test and args.noise > 0:
                y = y + rng.normal(0.0, args.noise, size=y.shape)
                y = np.maximum(y, 0.0)                           # RMSD cannot be negative
            np.save(os.path.join(folder, f"traj_{i:03d}.npy"), y.astype(np.float32))

        names.append(name)
        print(f"{name:12s} {len(x):7d} frames @ {dt_ps:.0f} ps -> {len(xb):5d} steps @ {step_ns:.2f} ns "
              f"| {'TEST ' if is_test else 'train'} | {n_copies} copies saved")

    split = {"train": [n for n in names if n not in args.test],
             "test": [n for n in names if n in args.test],
             "step_ns": float(step_ns), "downsample": args.downsample,
             "noise_nm": args.noise, "max_shift_ns": args.max_shift_ns, "n_copies": args.n_copies}
    with open(os.path.join(args.out, "split.json"), "w") as f:
        json.dump(split, f, indent=2)
    missing = [n for n in args.test if n not in names]
    if missing:
        print(f"WARNING: requested test trajectories not found: {missing}")
    print(f"split.json written: {len(split['train'])} train / {len(split['test'])} test source trajectories")


if __name__ == "__main__":
    main()
