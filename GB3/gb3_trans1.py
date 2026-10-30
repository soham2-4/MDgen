#!/usr/bin/env python3
"""
Train a Transformer on synthetic GB3 native-contact trajectories.

This version matches the old synthetic workflow:
1) gb3_synthetic_data.py creates many traj_*.npy files from the GB3
   native-contact time series.
2) gb3_trans1.py loads those synthetic trajectories and trains a
   seq2seq Transformer to reconstruct/forecast the full trajectory.

Default expectation:
    synthetic_gb3_trajectories/
        time_ns.npy
        N_Native_Contacts.npy   or   Q_Native.npy
        traj_000.npy
        traj_001.npy
        ...

Each traj_*.npy is one synthetic trajectory with shape [T].
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # choose GPUs with the CUDA_VISIBLE_DEVICES environment variable instead
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


# ------------------------------
# Utilities
# ------------------------------
def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """Subsequent mask for causal decoding."""
    return torch.triu(
        torch.full((size, size), float("-inf"), device=device),
        diagonal=1,
    )


def fit_standardizer(data: np.ndarray) -> Tuple[float, float]:
    mean = float(data.mean())
    std = float(data.std())
    if std < 1e-8:
        std = 1.0
    return mean, std


def apply_standardizer(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (data - mean) / std


# ------------------------------
# Dataset
# ------------------------------
class SyntheticGB3TrajectoryDataset(Dataset):
    """
    Loads many synthetic trajectories from traj_*.npy.

    Each sample:
      x = first input_len points
      y = full trajectory
    """

    def __init__(self, data_dir: str = "synthetic_gb3_trajectories", input_pct: float = 0.1):
        data_path = Path(data_dir)
        files = sorted(data_path.glob("traj_*.npy"))
        if len(files) == 0:
            raise FileNotFoundError(
                f"No traj_*.npy files found in {data_path.resolve()}. "
                "Run gb3_synthetic_data.py first."
            )

        trajectories = []
        for f in files:
            arr = np.load(f)
            if arr.ndim != 1:
                arr = np.asarray(arr).reshape(-1)
            trajectories.append(arr.astype(np.float32))

        lengths = {len(t) for t in trajectories}
        if len(lengths) != 1:
            raise ValueError(f"All trajectories must have the same length. Found lengths: {sorted(lengths)}")

        self.data = np.stack(trajectories)  # [N, T]
        self.input_len = max(1, int(round(self.data.shape[1] * input_pct)))

        print(f"Loaded {len(trajectories)} synthetic trajectories with shape: {self.data.shape}")

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        full_seq = self.data[idx]  # [T]
        input_seq = full_seq[: self.input_len]
        return (
            torch.tensor(input_seq[:, None], dtype=torch.float32),   # [S, 1]
            torch.tensor(full_seq[:, None], dtype=torch.float32),     # [T, 1]
        )


# ------------------------------
# Positional Encoding
# ------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ------------------------------
# Transformer Model
# ------------------------------
class TrajectoryTransformer(nn.Module):
    def __init__(
        self,
        input_len: int,
        output_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.output_proj = nn.Linear(d_model, 1)

        max_len = max(input_len, output_len) + 1024
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor | None = None) -> torch.Tensor:
        src = self.input_proj(src)
        tgt = self.input_proj(tgt)

        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)

        memory = self.encoder(src)
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.output_proj(out)


# ------------------------------
# Train / Eval
# ------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    input_len: int,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        decoder_input = torch.zeros_like(y)
        decoder_input[:, 1:, :] = y[:, :-1, :]

        tgt_mask = causal_mask(y.size(1), device)
        pred = model(x, decoder_input, tgt_mask=tgt_mask)

        loss = criterion(pred[:, input_len:, :], y[:, input_len:, :])
        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


def train() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer on synthetic GB3 trajectories.")
    parser.add_argument("--data-dir", type=str, default="synthetic_gb3_trajectories")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--input-pct", type=float, default=0.1)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="gb3_checkpoints")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gb3-timeseries")
    parser.add_argument("--wandb-entity", type=str, default="")
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SyntheticGB3TrajectoryDataset(
        data_dir=args.data_dir,
        input_pct=args.input_pct,
    )

    # Standardize using the training split only.
    n_total = len(dataset)
    n_train = max(1, int(round(n_total * args.train_frac)))
    n_val = max(1, n_total - n_train)
    if n_train + n_val != n_total:
        n_train = n_total - n_val

    train_subset, val_subset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_data = np.stack([dataset.data[i] for i in train_subset.indices])
    mean, std = fit_standardizer(train_data)

    dataset.data = apply_standardizer(dataset.data, mean, std).astype(np.float32)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    input_len = dataset.input_len
    output_len = dataset.data.shape[1]

    model = TrajectoryTransformer(
        input_len=input_len,
        output_len=output_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, config=vars(args))
    elif args.use_wandb:
        print("wandb requested but not available; continuing without logging.")

    print(f"Train trajectories: {len(train_subset)} | Val trajectories: {len(val_subset)}")
    print(f"Trajectory shape: {dataset.data.shape[1]}")
    print(f"Input length: {input_len} | Forecast length: {output_len - input_len}")
    print(f"Normalization: mean={mean:.6f}, std={std:.6f}")

    best_val = float("inf")
    best_path = output_dir / "gb3_transformer_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            decoder_input = torch.zeros_like(y)
            decoder_input[:, 1:, :] = y[:, :-1, :]

            tgt_mask = causal_mask(y.size(1), device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x, decoder_input, tgt_mask=tgt_mask)
            loss = criterion(pred[:, input_len:, :], y[:, input_len:, :])
            loss.backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            running_loss += float(loss.item())

        train_loss = running_loss / max(1, len(train_loader))
        val_loss = evaluate(model, val_loader, device, criterion, input_len=input_len)

        print(f"Epoch [{epoch:03d}/{args.epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "input_len": input_len,
                    "output_len": output_len,
                    "config": vars(args),
                },
                best_path,
            )

    final_path = output_dir / "gb3_transformer_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "input_len": input_len,
            "output_len": output_len,
            "config": vars(args),
        },
        final_path,
    )

    print(f"Best checkpoint:  {best_path.resolve()}")
    print(f"Final checkpoint: {final_path.resolve()}")
    print(f"Best val loss: {best_val:.6f}")

    # Optional qualitative plot.
    try:
        import matplotlib.pyplot as plt
        from PIL import Image

        model.eval()
        x, y = val_subset[0]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        decoder_input = torch.zeros_like(y)
        decoder_input[:, 1:, :] = y[:, :-1, :]
        tgt_mask = causal_mask(y.size(1), device)

        pred = model(x, decoder_input, tgt_mask=tgt_mask).detach().cpu().numpy()[0, :, 0]
        y_true = y.detach().cpu().numpy()[0, :, 0]

        pred = pred * std + mean
        y_true = y_true * std + mean

        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="true")
        plt.plot(pred, label="pred")
        plt.title("GB3 synthetic trajectory reconstruction/forecast")
        plt.xlabel("Time index")
        plt.ylabel("Native-contact value")
        plt.legend()
        plt.tight_layout()

        plot_path = output_dir / "gb3_validation_forecast.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        img = Image.open(plot_path)
        plt.figure(figsize=(10, 4))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        print(f"Saved validation plot: {plot_path.resolve()}")
    except Exception as e:
        print(f"Plotting skipped: {e}")


if __name__ == "__main__":
    train()