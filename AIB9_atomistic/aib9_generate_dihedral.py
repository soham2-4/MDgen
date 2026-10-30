#!/usr/bin/env python3
"""
aib9_generate_dihedral.py

Step 3: generate trajectories by sampling one frame at a time.

What changed relative to the old inference script
-------------------------------------------------
1. Frames are SAMPLED from the predicted categorical distribution, not
   taken as a point estimate. This is what lets a rare transition happen.

2. Generation slides the context window by one frame. The old code
   produced 500-frame blocks, each restarting the decoder's positional
   encoding at index 0, which locked the output into a 500-frame limit
   cycle (autocorrelation peaked at exactly 500, 1000, 1500, 2000 with
   no decay across four periods).

3. Many independent chains run in parallel as a batch. Free energy
   estimates need independent samples; one long correlated chain gives
   a far smaller effective sample size than its frame count suggests.
   Each chain is seeded from a different point in the reference.

4. Temperature is exposed. T=1 samples the learned distribution, which
   is what you want for a free energy surface. Lower T sharpens toward
   the mode and will under-sample the minor basin.

Usage
-----
    python aib9_generate_dihedral.py --n-chains 16 --n-frames 20000
    python aib9_generate_dihedral.py --temperature 1.0 --seed-from replica5
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from aib9_common import (
    DihedralTransformer,
    angles_to_features,
    bins_to_angles,
    circular_mean_deg,
    count_transitions,
    find_replica_files,
    load_meta,
    phi_columns,
    sample_next,
)


def main():

    parser = argparse.ArgumentParser(
        description="Sample dihedral trajectories from a trained model"
    )

    parser.add_argument(
        "--model", default="checkpoints_dihedral/aib9_dihedral_best.pt"
    )
    parser.add_argument("--data-dir", default="data_dihedral")
    parser.add_argument("--outdir", default="generated_dihedral")

    parser.add_argument(
        "--seed-from",
        default=None,
        help="Replica name to draw seed windows from "
             "(default: the held-out validation replica)",
    )

    parser.add_argument(
        "--seed-mode",
        default="random",
        choices=["random", "first"],
        help=(
            "random: each chain starts from a different random offset in the "
            "reference (best for free energy estimates, gives independent "
            "samples). first: every chain continues the SAME opening segment "
            "of the reference, reproducing the 'generate the rest of the "
            "trajectory from its first part' workflow."
        ),
    )

    parser.add_argument(
        "--seed-frames",
        type=int,
        default=None,
        help=(
            "How many opening frames to treat as the given segment "
            "(seed-mode=first). Note that only the last `window` frames "
            "actually condition the model."
        ),
    )

    parser.add_argument("--n-chains", type=int, default=16,
                        help="Independent chains sampled in parallel")
    parser.add_argument("--n-frames", type=int, default=20000,
                        help="Frames generated per chain")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    ckpt = torch.load(args.model, map_location=device, weights_only=False)

    n_dihedrals = ckpt["n_dihedrals"]
    nbins = ckpt["nbins"]
    window = ckpt["window"]
    cfg = ckpt["config"]

    model = DihedralTransformer(
        n_dihedrals=n_dihedrals,
        nbins=nbins,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        max_len=max(4096, window + 8),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = load_meta(args.data_dir)

    print()
    print("=" * 70)
    print("GENERATION")
    print("=" * 70)
    print(f"  device       {device}")
    print(f"  dihedrals    {n_dihedrals}   bins {nbins}   window {window}")
    print(f"  chains       {args.n_chains}")
    print(f"  frames/chain {args.n_frames:,}")
    print(f"  temperature  {args.temperature}")
    print()

    # --------------------------------------------------
    # Seeds: distinct windows from a reference replica
    # --------------------------------------------------

    seed_name = args.seed_from or cfg.get("val_replica", "replica5")

    files = find_replica_files(args.data_dir)
    match = [f for f in files if f.stem == seed_name]

    if not match:
        match = [files[-1]]
        print(f"  (replica '{seed_name}' not found; seeding from {match[0].stem})")

    ref = np.load(match[0])["angles_deg"].astype(np.float32)

    if ref.shape[0] < window + 1:
        raise SystemExit("Reference trajectory shorter than the context window")

    if args.seed_mode == "first":

        given = args.seed_frames or window

        if given < window:
            raise SystemExit(
                f"--seed-frames {given} is shorter than the model's context "
                f"window {window}"
            )

        given = min(given, ref.shape[0])

        # Only the last `window` frames enter the context, but the whole
        # opening segment is kept so the saved output can be stitched
        # onto it.
        prefix = ref[:given]

        seed_angles = np.repeat(
            prefix[-window:][None], args.n_chains, axis=0
        )

        print(f"  seeded from  {match[0].stem}, first {given:,} frames "
              f"({given / ref.shape[0]:.1%} of the trajectory)")
        print(f"               all {args.n_chains} chains continue the same "
              f"segment, sampling different futures")
        print(f"               model conditions on the last {window} frames")

    else:

        prefix = None

        starts = rng.choice(
            ref.shape[0] - window, size=args.n_chains, replace=False
        )

        seed_angles = np.stack(
            [ref[s : s + window] for s in starts], axis=0
        )                                              # [B, W, D]

        # Half the chains start mirrored, so the initial handedness is
        # balanced and any basin asymmetry in the output is the model's,
        # not the seed's.
        flip = rng.random(args.n_chains) < 0.5
        seed_angles[flip] = -seed_angles[flip]

        print(f"  seeded from  {match[0].stem} at {args.n_chains} "
              f"random offsets")

    seed_phi = circular_mean_deg(
        seed_angles[:, -1, phi_columns(meta)], axis=-1
    )
    print(f"  seed handedness: right {np.mean(seed_phi < 0):.2f} "
          f"/ left {np.mean(seed_phi > 0):.2f}")
    print()

    # --------------------------------------------------
    # Roll forward, one frame at a time
    # --------------------------------------------------

    ctx = torch.from_numpy(
        np.stack([angles_to_features(s) for s in seed_angles], axis=0)
    ).to(device)                                       # [B, W, 2D]

    out = np.empty(
        (args.n_chains, args.n_frames, n_dihedrals), dtype=np.float32
    )

    t0 = time.time()

    with torch.no_grad():

        for step in range(args.n_frames):

            bins = sample_next(model, ctx, temperature=args.temperature)

            angles = bins_to_angles(
                bins.cpu().numpy(), nbins, rng=rng, jitter=True
            ).astype(np.float32)                        # [B, D]

            out[:, step] = angles

            feats = torch.from_numpy(
                np.stack([angles_to_features(a[None]) for a in angles])
            ).to(device)                                # [B, 1, 2D]

            ctx = torch.cat([ctx[:, 1:], feats], dim=1)

            if (step + 1) % 1000 == 0:
                rate = (step + 1) / (time.time() - t0)
                eta = (args.n_frames - step - 1) / max(rate, 1e-9)
                print(f"    {step + 1:7,d} / {args.n_frames:,}"
                      f"   {rate:.0f} frames/s   eta {eta / 60:.1f} min")

    # --------------------------------------------------
    # Save and report
    # --------------------------------------------------

    path = os.path.join(args.outdir, "generated_angles.npz")

    save_kwargs = dict(
        angles_deg=out,
        seed_angles=seed_angles,
        temperature=args.temperature,
        seed_mode=args.seed_mode,
    )

    if prefix is not None:
        # the given opening segment, so the full trajectory can be
        # reassembled as prefix + generated
        save_kwargs["prefix_angles"] = prefix

    np.savez_compressed(path, **save_kwargs)

    pcols = phi_columns(meta)

    phi_bar = circular_mean_deg(out[:, :, pcols], axis=-1)   # [B, T]

    right = float(np.mean(phi_bar < 0))

    crossings = count_transitions(phi_bar)

    print()
    print("=" * 70)
    print(f"  saved {path}   shape {out.shape}")
    print(f"  right-handed fraction {right:.4f}   (true value 0.500)")
    print(f"  basin crossings across all chains: {crossings}")

    if crossings == 0:
        print("    WARNING: no helix flips. The model is not sampling the "
              "rare event; try temperature 1.0, a longer run, or more "
              "training.")

    print("Next: python aib9_evaluate_fes.py")
    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
