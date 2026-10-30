#!/usr/bin/env python3
"""
plot_training.py

Plot training and validation loss.

Reads either:
  - checkpoints_dihedral/history.json  (written by the patched trainer), or
  - a captured stdout log, parsed with a regex

The log parser exists so a run that was already underway when history
saving was added is not lost: paste the terminal scrollback into a file
and point this script at it.

Usage:
    python plot_training.py
    python plot_training.py --log train_log.txt
    python plot_training.py --history checkpoints_dihedral/history.json
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCH_RE = re.compile(
    r"epoch\s+(\d+)\s*/\s*\d+\s+"
    r"train\s+([0-9.]+)\s+"
    r"val\s+([0-9.]+)\s+"
    r"val-acc\s+([0-9.]+|nan)"
)


def from_log(path):

    rows = []

    with open(path, errors="ignore") as fh:
        for line in fh:
            m = EPOCH_RE.search(line)
            if m:
                rows.append({
                    "epoch": int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "val_loss": float(m.group(3)),
                    "val_acc": float(m.group(4)) if m.group(4) != "nan" else np.nan,
                })

    return rows


def main():

    parser = argparse.ArgumentParser(description="Plot training curves")

    parser.add_argument(
        "--history", default="checkpoints_dihedral/history.json"
    )
    parser.add_argument(
        "--log", default=None,
        help="Captured stdout log, used if history.json is absent",
    )
    parser.add_argument("--nbins", type=int, default=36,
                        help="Bins used in training, for the uniform baseline")
    parser.add_argument("--out", default="training_curves.png")

    args = parser.parse_args()

    if args.log:
        rows = from_log(args.log)
        source = args.log
    elif os.path.exists(args.history):
        rows = json.load(open(args.history))
        source = args.history
    else:
        raise SystemExit(
            f"Neither {args.history} nor a --log file was found.\n"
            "If the run predates history saving, paste the terminal output "
            "into train_log.txt and rerun with --log train_log.txt"
        )

    if not rows:
        raise SystemExit(f"No epoch lines parsed from {source}")

    ep = np.array([r["epoch"] for r in rows])
    tr = np.array([r["train_loss"] for r in rows])
    va = np.array([r["val_loss"] for r in rows])
    ac = np.array([r.get("val_acc", np.nan) for r in rows], dtype=float)

    uniform = float(np.log(args.nbins))
    best = int(np.nanargmin(va))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.plot(ep, tr, lw=2, label="train")
    ax.plot(ep, va, lw=2, label="validation (held-out replica)")
    ax.axhline(uniform, color="k", ls=":", lw=1.5,
               label=f"uniform baseline = {uniform:.2f}")
    ax.axvline(ep[best], color="r", ls="--", lw=1.2,
               label=f"best val @ epoch {ep[best]}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy (nats)")
    ax.set_title("Loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    if np.isfinite(ac).any():
        ax.plot(ep, ac, lw=2, color="tab:green")
        ax.axhline(1.0 / args.nbins, color="k", ls=":", lw=1.5,
                   label=f"chance = {1 / args.nbins:.3f}")
        ax.set_ylabel("per-dihedral top-1 bin accuracy")
        ax.legend(fontsize=9)
    ax.set_xlabel("epoch")
    ax.set_title("Validation accuracy")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)

    gap = tr[best] - va[best]

    print()
    print(f"  source            {source}")
    print(f"  epochs parsed     {len(ep)}")
    print(f"  uniform baseline  {uniform:.4f}")
    print(f"  best val loss     {va[best]:.4f} at epoch {ep[best]}")
    print(f"  train at that pt  {tr[best]:.4f}   (train - val = {gap:+.4f})")
    print(f"  final val loss    {va[-1]:.4f}")
    print()

    if ep[best] == ep[-1]:
        print("  Validation was still improving at the last epoch.")
        print("  Training longer would likely help.")
    elif ep[best] < 0.5 * ep[-1]:
        print("  Validation bottomed out in the first half of the run.")
        print("  Fewer epochs would have reached the same checkpoint.")
    else:
        print("  Validation flattened near the end. The epoch budget was")
        print("  about right.")

    if va[-1] > va[best] * 1.05:
        print("  Val loss rose after the minimum: overfitting. The saved")
        print("  best checkpoint is the one to use, not the final one.")

    print(f"\n  saved {args.out}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
