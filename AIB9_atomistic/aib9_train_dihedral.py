#!/usr/bin/env python3
"""
aib9_train_dihedral.py

Step 2: train a causal transformer to predict the next frame's dihedral
distribution.

What changed relative to the old training script
------------------------------------------------
1. Loss is cross-entropy over binned angles, not MSE on coordinates.
   MSE learns the conditional mean, which for a two-basin system always
   says "stay put" and structurally cannot sample a helix flip.

2. Validation holds out an ENTIRE replica. The old split was a random
   split over windows drawn from 150 noise-copies of the same five
   trajectories, so near-identical windows sat on both sides and the
   validation loss was measuring memorisation.

3. Mirror augmentation exploits the exact achiral symmetry of AIB9.

4. CUDA_VISIBLE_DEVICES is no longer hard-coded at import; use the
   --device flag or set the environment variable yourself.

Usage
-----
    python aib9_train_dihedral.py --epochs 60
    python aib9_train_dihedral.py --val-replica replica5 --window 64
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aib9_common import (
    NBINS_DEFAULT,
    DihedralWindowDataset,
    DihedralTransformer,
    dihedral_loss,
    find_replica_files,
    load_meta,
)


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, nbins, device, smoothing):

    model.eval()

    total = 0.0
    correct = 0
    count = 0

    for x, y in loader:

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        total += dihedral_loss(logits, y, nbins, smoothing).item()

        correct += (logits.argmax(-1) == y).sum().item()
        count += y.numel()

    n = max(1, len(loader))

    return total / n, correct / max(1, count)


def main():

    parser = argparse.ArgumentParser(
        description="Train a dihedral next-frame transformer"
    )

    parser.add_argument("--data-dir", default="data_dihedral")
    parser.add_argument("--output-dir", default="checkpoints_dihedral")

    parser.add_argument(
        "--split",
        default="temporal",
        choices=["temporal", "replica"],
        help=(
            "temporal: train on the first part of EVERY replica, validate "
            "on the tail of every replica (all replicas contribute to "
            "training). replica: hold out one whole replica."
        ),
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Tail fraction of each replica used for validation "
             "(temporal split only)",
    )

    parser.add_argument(
        "--val-replica",
        default="replica5",
        help="Replica held out (replica split only)",
    )

    parser.add_argument("--window", type=int, default=64,
                        help="Context length in frames")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride between training windows")

    parser.add_argument("--nbins", type=int, default=NBINS_DEFAULT)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=500,
                        help="Linear warmup steps")

    parser.add_argument("--no-mirror", action="store_true",
                        help="Disable the achiral mirror augmentation")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(args.data_dir)
    files = find_replica_files(args.data_dir)

    if args.split == "replica":
        val_files = [f for f in files if f.stem == args.val_replica]
        train_files = [f for f in files if f.stem != args.val_replica]
        train_range, val_range = (0.0, 1.0), (0.0, 1.0)
        if not train_files:
            raise SystemExit("No training replicas left after holding one out")
    else:
        train_files = val_files = files
        cut = 1.0 - args.val_fraction
        train_range, val_range = (0.0, cut), (cut, 1.0)

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print(f"  device        {device}")
    print(f"  split         {args.split}")
    print(f"  train         {[f.stem for f in train_files]}"
          f"  frames {train_range[0]:.0%}-{train_range[1]:.0%}")
    print(f"  validation    {[f.stem for f in val_files] or 'NONE'}"
          f"  frames {val_range[0]:.0%}-{val_range[1]:.0%}")
    print(f"  window        {args.window} frames")
    print(f"  bins          {args.nbins} ({360 / args.nbins:.1f} deg)")
    print(f"  mirror aug    {not args.no_mirror}")
    print()

    train_set = DihedralWindowDataset(
        train_files,
        nbins=args.nbins,
        window=args.window,
        stride=args.stride,
        mirror=not args.no_mirror,
        frame_range=train_range,
        # keep a window-sized buffer so no training window touches the
        # validation segment
        gap=(args.window if args.split == "temporal" else 0),
    )

    print(f"  dihedrals     {train_set.n_dihedrals}")
    print(f"  train windows {len(train_set):,}"
          f"  ({train_set.n_frames:,} frames)")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = None

    if val_files:
        val_set = DihedralWindowDataset(
            val_files,
            nbins=args.nbins,
            window=args.window,
            stride=max(1, args.window // 2),   # validation needs no overlap
            mirror=False,
            frame_range=val_range,
        )
        print(f"  val windows   {len(val_set):,}")

        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    model = DihedralTransformer(
        n_dihedrals=train_set.n_dihedrals,
        nbins=args.nbins,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_len=max(4096, args.window + 8),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters    {n_params:,}")

    # A uniform predictor scores ln(nbins); anything above that is broken.
    print(f"  uniform-baseline loss = {np.log(args.nbins):.4f}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    total_steps = max(1, args.epochs * len(train_loader))

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best = float("inf")
    best_path = out_dir / "aib9_dihedral_best.pt"
    history = []
    history_path = out_dir / "history.json"

    ckpt_meta = {
        "n_dihedrals": train_set.n_dihedrals,
        "nbins": args.nbins,
        "window": args.window,
        "config": vars(args),
        "split": args.split,
        "dihedral_names": meta["dihedral_names"],
    }

    for epoch in range(1, args.epochs + 1):

        model.train()

        running = 0.0
        t0 = time.time()

        for i, (x, y) in enumerate(train_loader):

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)

            loss = dihedral_loss(logits, y, args.nbins, args.smoothing)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            running += loss.item()

            if i % 200 == 0:
                print(
                    f"    epoch {epoch:3d}  batch {i:5d}/{len(train_loader)}"
                    f"  loss {loss.item():.4f}"
                )

        train_loss = running / max(1, len(train_loader))

        if val_loader is not None:
            val_loss, val_acc = evaluate(
                model, val_loader, args.nbins, device, args.smoothing
            )
        else:
            val_loss, val_acc = train_loss, float("nan")

        print(
            f"  epoch {epoch:3d}/{args.epochs}"
            f"  train {train_loss:.4f}"
            f"  val {val_loss:.4f}"
            f"  val-acc {val_acc:.3f}"
            f"  lr {optimizer.param_groups[0]['lr']:.2e}"
            f"  {time.time() - t0:.0f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        # written every epoch so a killed run still leaves a usable curve
        with open(history_path, "w") as fh:
            json.dump(history, fh, indent=2)

        if val_loss < best:
            best = val_loss
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "epoch": epoch, "val_loss": val_loss, **ckpt_meta},
                best_path,
            )
            print(f"    new best -> {best_path}")

    torch.save(
        {"model_state_dict": model.state_dict(),
         "epoch": args.epochs, "val_loss": best, **ckpt_meta},
        out_dir / "aib9_dihedral_final.pt",
    )

    with open(out_dir / "train_config.json", "w") as fh:
        json.dump(vars(args), fh, indent=2)

    print()
    print("=" * 70)
    print(f"Best validation loss {best:.4f}   (uniform = {np.log(args.nbins):.4f})")
    print(f"Best checkpoint: {best_path}")
    print(f"Loss history:    {history_path}")
    print("Next: python aib9_generate_dihedral.py")
    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
