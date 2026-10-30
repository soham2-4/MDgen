#!/usr/bin/env python3
"""
export_trajectories.py - package the generated and reference trajectories for sharing.

Handles the frame alignment (the generated series starts at frame `input_len`,
not at frame 0) and records the diagnostics alongside the data so whoever opens
the file can tell how it was produced.

    python export_trajectories.py
    python export_trajectories.py --outdir inference_output_v2 --rep 0
"""

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="inference_output_v2",
                    help="Directory holding full_rollouts.npy / fes_full.npz")
    ap.add_argument("--data-dir", default="synthetic_gb3_trajectories")
    ap.add_argument("--checkpoint", default="gb3_checkpoints/gb3_transformer_best.pt")
    ap.add_argument("--rep", type=int, default=0, help="Which full rollout to export")
    ap.add_argument("--prefix", default="gb3_generated_vs_truth",
                    help="Base name for the optional combined file.")
    ap.add_argument("--truth-prefix", default="gb3_ground_truth",
                    help="Base name for the reference trajectory files.")
    ap.add_argument("--gen-prefix", default="gb3_generated",
                    help="Base name for the model trajectory files.")
    ap.add_argument("--combined", action="store_true",
                    help="Also write the paired two-column file.")
    ap.add_argument("--roll-window", type=int, default=10)
    args = ap.parse_args()

    gen_all = np.load(os.path.join(args.outdir, "full_rollouts.npy"))
    if gen_all.ndim == 1:
        gen_all = gen_all[None, :]
    generated = gen_all[args.rep].astype(np.float64)

    truth_full = None
    for name in ("N_Native_Contacts.npy", "Q_Native.npy"):
        p = os.path.join(args.data_dir, name)
        if os.path.exists(p):
            truth_full = np.load(p).astype(np.float64).reshape(-1)
            observable = name.replace(".npy", "")
            break
    if truth_full is None:
        raise SystemExit(f"No reference series found in {args.data_dir}")

    # The generated series covers the frames AFTER the seeded prefix.
    input_len = len(truth_full) - len(generated)
    if input_len < 0:
        raise SystemExit("Generated series is longer than the reference.")
    truth = truth_full[input_len:]
    frames = np.arange(input_len, len(truth_full))

    time_path = os.path.join(args.data_dir, "time_ns.npy")
    time_ns = (np.load(time_path).astype(np.float64)[input_len:]
               if os.path.exists(time_path) else np.full(len(truth), np.nan))

    n = min(len(truth), len(generated), len(time_ns), len(frames))
    truth, generated = truth[:n], generated[:n]
    time_ns, frames = time_ns[:n], frames[:n]

    r = float(np.corrcoef(generated, truth)[0, 1])
    meta = {
        "observable": observable,
        "n_frames": int(n),
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),
        "seeded_prefix_frames": int(input_len),
        "prefix_note": ("Frames 0..input_len-1 are the TRUE trajectory, supplied to the "
                        "model as context. They are not included in these arrays."),
        "truth_mean": float(truth.mean()), "truth_std": float(truth.std()),
        "generated_mean": float(generated.mean()), "generated_std": float(generated.std()),
        "per_frame_correlation": r,
        "provenance": ("Autoregressive rollout with cached decoding. The model was "
                       "trained on a single MD trajectory duplicated 1000x with "
                       "rounding-suppressed noise, so the training set contained one "
                       "distinct sequence."),
        "interpretation": ("Per-frame correlation of %.4f indicates the model reproduces "
                           "the training sequence rather than sampling independently from "
                           "the equilibrium distribution. Agreement of the free-energy "
                           "surface follows from this reconstruction and should not be "
                           "read as validation of learned dynamics." % r),
    }

    npz_path = f"{args.prefix}.npz"
    dat_path = f"{args.prefix}.dat"

    def write_one(tag, values, role):
        m = dict(meta)
        m["series"] = role
        np.savez_compressed(f"{tag}.npz", frame=frames, time_ns=time_ns,
                            **{observable: values}, metadata=json.dumps(m))
        header = (f"# GB3 {observable} -- {role}\n"
                  f"# frames {frames[0]}-{frames[-1]}\n"
                  f"# the first {input_len} frames of the run were supplied to the model "
                  f"as context and are excluded here, so both files cover identical frames\n"
                  f"# mean {values.mean():.4f}  std {values.std():.4f}\n"
                  f"# Frame  Time_ns  {observable}")
        np.savetxt(f"{tag}.dat", np.column_stack([frames, time_ns, values]),
                   fmt=["%d", "%.6f", "%.6f"], header=header, comments="")
        with open(f"{tag}_metadata.json", "w") as f:
            json.dump(m, f, indent=2)
        print(f"  {tag}.npz / {tag}.dat / {tag}_metadata.json")

    print("Wrote:")
    write_one(args.truth_prefix, truth, "reference MD trajectory")
    write_one(args.gen_prefix, generated, "model reconstruction")

    if args.combined:
        np.savez_compressed(npz_path, frame=frames, time_ns=time_ns,
                            ground_truth=truth, generated=generated,
                            metadata=json.dumps(meta))
        header = (f"# GB3 {observable}: reference vs model reconstruction\n"
                  f"# frames {frames[0]}-{frames[-1]}; first {input_len} frames were "
                  f"supplied as context and are excluded\n"
                  f"# per-frame correlation = {r:.4f} (reconstruction, not independent sampling)\n"
                  f"# Frame  Time_ns  Ground_Truth  Generated")
        np.savetxt(dat_path, np.column_stack([frames, time_ns, truth, generated]),
                   fmt=["%d", "%.6f", "%.6f", "%.6f"], header=header, comments="")
        with open(json_path := f"{args.prefix}_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  {npz_path} / {dat_path} / {json_path}")

    print(f"\n  {n} frames, {frames[0]}-{frames[-1]}, per-frame r = {r:.4f} (raw, unsmoothed)")
    print(f"  truth  mean {truth.mean():.2f}  std {truth.std():.2f}")
    print(f"  gen    mean {generated.mean():.2f}  std {generated.std():.2f}")


if __name__ == "__main__":
    main()