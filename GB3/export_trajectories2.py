#!/usr/bin/env python3
"""
export_trajectories2.py - collect ALL generated rollouts and the matching ground
truth into one folder.

Layout written to --export-dir:
    ground_truth/gb3_ground_truth.{npz,dat}  + _metadata.json   (full reference)
    generated/<run>_rep000.{npz,dat}         + _metadata.json   (one per rollout)
    paired/<run>_rep000_paired.dat           (frame, time, truth, generated)  [--paired]
    summary.csv                              (one row per rollout)
    summary.json

Usage:
    python export_trajectories2.py --rollout-dirs inference_output_v2 inference_output_v3
    python export_trajectories2.py --rollout-dirs inference_output_v3 --export-dir gb3_export --paired
"""

import argparse
import csv
import glob
import json
import os

import numpy as np


def load_reference(data_dir):
    for name in ("N_Native_Contacts.npy", "Q_Native.npy"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return np.load(p).astype(np.float64).reshape(-1), name.replace(".npy", "")
    raise SystemExit(f"No reference series found in {data_dir}")


def rolling_mean(a, w):
    return a if w <= 1 else np.convolve(a, np.ones(w) / w, mode="valid")


def write_series(base, frames, time_ns, values, observable, meta):
    np.savez_compressed(f"{base}.npz", frame=frames, time_ns=time_ns,
                        **{observable: values}, metadata=json.dumps(meta))
    header = (f"# GB3 {observable} -- {meta['series']}\n"
              f"# frames {frames[0]}-{frames[-1]}\n"
              f"# mean {values.mean():.4f}  std {values.std():.4f}\n"
              f"# Frame  Time_ns  {observable}")
    np.savetxt(f"{base}.dat", np.column_stack([frames, time_ns, values]),
               fmt=["%d", "%.6f", "%.6f"], header=header, comments="")
    with open(f"{base}_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dirs", nargs="+", default=["inference_output_v2"],
                    help="One or more inference output directories to collect from.")
    ap.add_argument("--data-dir", default="synthetic_gb3_trajectories")
    ap.add_argument("--export-dir", default="gb3_trajectory_export",
                    help="Folder that will receive everything.")
    ap.add_argument("--files", nargs="+", default=["full_rollouts.npy", "rollouts.npy"],
                    help="Rollout file names to look for in each directory.")
    ap.add_argument("--input-len", type=int, default=None,
                    help="Frame where generation starts. Required for short "
                         "rollouts.npy; inferred from length for full rollouts.")
    ap.add_argument("--paired", action="store_true",
                    help="Also write a 4-column truth-vs-generated file per rollout.")
    ap.add_argument("--roll-window", type=int, default=10,
                    help="Window for the smoothed correlation in the summary.")
    ap.add_argument("--replay-threshold", type=float, default=0.8)
    ap.add_argument("--provenance", default="",
                    help="Free-text note on how the model was trained, stored in metadata.")
    args = ap.parse_args()

    truth_full, observable = load_reference(args.data_dir)
    T = len(truth_full)
    time_path = os.path.join(args.data_dir, "time_ns.npy")
    time_full = (np.load(time_path).astype(np.float64).reshape(-1)
                 if os.path.exists(time_path) else np.full(T, np.nan))
    if len(time_full) < T:
        time_full = np.r_[time_full, np.full(T - len(time_full), np.nan)]
    frames_full = np.arange(T)

    gt_dir = os.path.join(args.export_dir, "ground_truth")
    gen_dir = os.path.join(args.export_dir, "generated")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    if args.paired:
        os.makedirs(os.path.join(args.export_dir, "paired"), exist_ok=True)

    # Ground truth written once, full length; each generated file records its start frame.
    write_series(os.path.join(gt_dir, "gb3_ground_truth"), frames_full, time_full[:T],
                 truth_full, observable,
                 {"series": "reference MD trajectory", "observable": observable,
                  "n_frames": int(T), "source": os.path.abspath(args.data_dir)})

    rows = []
    for rdir in args.rollout_dirs:
        run = os.path.basename(os.path.normpath(rdir))
        found = [os.path.join(rdir, f) for f in args.files if os.path.exists(os.path.join(rdir, f))]
        if not found:
            print(f"  [skip] {rdir}: none of {args.files} found")
            continue
        for path in found:
            kind = os.path.basename(path).replace(".npy", "")
            arr = np.load(path)
            arr = arr[None, :] if arr.ndim == 1 else arr
            L = arr.shape[1]

            if args.input_len is not None:
                start = args.input_len
            elif kind == "full_rollouts":
                start = T - L
            else:
                print(f"  [skip] {path}: pass --input-len to align short rollouts")
                continue
            if start < 0 or start + L > T:
                print(f"  [skip] {path}: frames {start}-{start + L - 1} don't fit in "
                      f"reference of length {T}")
                continue

            sl = slice(start, start + L)
            frames, time_ns, truth = frames_full[sl], time_full[sl], truth_full[sl]

            for rep in range(arr.shape[0]):
                gen = arr[rep].astype(np.float64)
                r = float(np.corrcoef(gen, truth)[0, 1])
                w = args.roll_window
                r_s = float(np.corrcoef(rolling_mean(gen, w), rolling_mean(truth, w))[0, 1])
                replay = r > args.replay_threshold
                tag = f"{run}_{kind}_rep{rep:03d}"
                meta = {
                    "series": "model generated trajectory",
                    "observable": observable,
                    "source_file": os.path.abspath(path),
                    "rep": rep,
                    "first_frame": int(frames[0]), "last_frame": int(frames[-1]),
                    "n_frames": int(L),
                    "seeded_prefix_frames": int(start),
                    "prefix_note": (f"Frames 0..{start - 1} of the TRUE trajectory were "
                                    "given to the model as context and are not included."),
                    "generated_mean": float(gen.mean()), "generated_std": float(gen.std()),
                    "truth_mean_same_frames": float(truth.mean()),
                    "truth_std_same_frames": float(truth.std()),
                    "per_frame_correlation": r,
                    f"per_frame_correlation_rolling{w}": r_s,
                    "interpretation": (
                        f"Per-frame correlation {r:.3f} > {args.replay_threshold}: the rollout "
                        "tracks the reference frame by frame, i.e. it reproduces the training "
                        "sequence. FES agreement is not evidence of learned dynamics."
                        if replay else
                        f"Per-frame correlation {r:.3f}: not a frame-by-frame copy. Compare "
                        "distributions (FES), not individual frames."),
                    "provenance": args.provenance,
                }
                write_series(os.path.join(gen_dir, tag), frames, time_ns, gen, observable, meta)

                if args.paired:
                    header = (f"# GB3 {observable}: reference vs {tag}\n"
                              f"# frames {frames[0]}-{frames[-1]}; per-frame r = {r:.4f}\n"
                              f"# Frame  Time_ns  Ground_Truth  Generated")
                    np.savetxt(os.path.join(args.export_dir, "paired", f"{tag}_paired.dat"),
                               np.column_stack([frames, time_ns, truth, gen]),
                               fmt=["%d", "%.6f", "%.6f", "%.6f"], header=header, comments="")

                rows.append({"run": run, "file": kind, "rep": rep, "tag": tag,
                             "first_frame": int(frames[0]), "last_frame": int(frames[-1]),
                             "n_frames": int(L), "gen_mean": round(float(gen.mean()), 3),
                             "gen_std": round(float(gen.std()), 3),
                             "truth_mean": round(float(truth.mean()), 3),
                             "truth_std": round(float(truth.std()), 3),
                             "r_per_frame": round(r, 4), f"r_rolling{w}": round(r_s, 4),
                             "replay_flag": replay})
                print(f"  {tag:40s} frames {frames[0]}-{frames[-1]}  r={r:+.3f}"
                      f"{'  [replay]' if replay else ''}")

    if not rows:
        raise SystemExit("Nothing exported.")
    with open(os.path.join(args.export_dir, "summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(args.export_dir, "summary.json"), "w") as f:
        json.dump({"observable": observable, "reference_frames": int(T),
                   "rollouts": rows}, f, indent=2)
    print(f"\nExported {len(rows)} generated trajectories + ground truth to "
          f"{os.path.abspath(args.export_dir)}")


if __name__ == "__main__":
    main()
