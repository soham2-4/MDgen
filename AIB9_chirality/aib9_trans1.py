#!/usr/bin/env python3

from __future__ import annotations

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # choose GPUs with the CUDA_VISIBLE_DEVICES environment variable instead

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
)

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


# ============================================================
# Utility
# ============================================================

def seed_everything(seed=42):

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_standardizer(data):

    mean = float(data.mean())
    std = float(data.std())

    if std < 1e-8:
        std = 1.0

    return mean, std


def apply_standardizer(data, mean, std):

    return (data - mean) / std


# ============================================================
# Dataset
# ============================================================

class SyntheticAIB9Dataset(Dataset):

    def __init__(
        self,
        data_dir="synthetic_aib9_trajectories",
        input_pct=0.10
    ):

        data_path = Path(data_dir)

        files = sorted(
            data_path.glob("traj_*.npy")
        )

        if len(files) == 0:
            raise FileNotFoundError(
                f"No traj_*.npy found in {data_dir}"
            )

        trajectories = []

        for f in files:

            arr = np.load(f)

            if arr.ndim != 1:
                arr = arr.reshape(-1)

            trajectories.append(
                arr.astype(np.float32)
            )

        lengths = {len(x) for x in trajectories}

        if len(lengths) != 1:

            raise ValueError(
                f"Different lengths detected: {lengths}"
            )

        self.data = np.stack(trajectories)

        self.input_len = max(
            1,
            int(
                round(
                    self.data.shape[1] * input_pct
                )
            )
        )

        print(
            f"Loaded {len(trajectories)} trajectories"
        )

        print(
            f"Shape = {self.data.shape}"
        )

    def __len__(self):

        return self.data.shape[0]

    def __getitem__(self, idx):

        full_seq = self.data[idx]

        input_seq = full_seq[:self.input_len]

        return (
            torch.tensor(
                input_seq[:, None],
                dtype=torch.float32
            ),
            torch.tensor(
                full_seq[:, None],
                dtype=torch.float32
            ),
        )


# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_len=100000
    ):

        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            )
            *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(
            position.float() * div_term
        )

        pe[:, 1::2] = torch.cos(
            position.float() * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):

        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)


# ============================================================
# Transformer
# ============================================================

class TrajectoryTransformer(nn.Module):

    def __init__(
        self,
        input_len,
        output_len,
        d_model=64,
        nhead=4,
        num_layers=4,
        dropout=0.1
    ):

        super().__init__()

        self.input_proj = nn.Linear(
            1,
            d_model
        )

        self.output_proj = nn.Linear(
            d_model,
            1
        )

        max_len = max(
            input_len,
            output_len
        ) + 1024

        self.pos_encoder = PositionalEncoding(
            d_model,
            dropout,
            max_len
        )

        self.pos_decoder = PositionalEncoding(
            d_model,
            dropout,
            max_len
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

    def forward(
        self,
        src,
        tgt,
        tgt_mask=None
    ):

        src = self.input_proj(src)
        tgt = self.input_proj(tgt)

        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)

        memory = self.encoder(src)

        out = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask
        )

        return self.output_proj(out)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    criterion,
    input_len
):

    model.eval()

    total_loss = 0

    count = 0

    for x, y in dataloader:

        x = x.to(device)
        y = y.to(device)

        decoder_input = torch.zeros_like(y)

        decoder_input[:,1:,:] = y[:,:-1,:]

        pred = model(
            x,
            decoder_input,
            tgt_mask=None
        )

        loss = criterion(
            pred[:,input_len:,:],
            y[:,input_len:,:]
        )

        total_loss += loss.item()

        count += 1

    return total_loss / max(1,count)


# ============================================================
# Training
# ============================================================

def train():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="synthetic_aib9_trajectories"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=500
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=64
    )

    parser.add_argument(
        "--nhead",
        type=int,
        default=4
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--input-pct",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--output-dir",
        default="aib9_checkpoints"
    )

    args = parser.parse_args()

    seed_everything()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset = SyntheticAIB9Dataset(
        args.data_dir,
        args.input_pct
    )

    train_size = int(
        0.8 * len(dataset)
    )

    val_size = len(dataset) - train_size

    train_set, val_set = random_split(
        dataset,
        [train_size,val_size]
    )

    train_data = np.stack(
        [
            dataset.data[i]
            for i in train_set.indices
        ]
    )

    mean, std = fit_standardizer(
        train_data
    )

    dataset.data = apply_standardizer(
        dataset.data,
        mean,
        std
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size
    )

    input_len = dataset.input_len
    output_len = dataset.data.shape[1]

    model = TrajectoryTransformer(
        input_len=input_len,
        output_len=output_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr
    )

    best_val = float("inf")

    best_path = (
        output_dir /
        "aib9_transformer_best.pt"
    )

    for epoch in range(
        1,
        args.epochs+1
    ):

        model.train()

        running_loss = 0

        for x,y in train_loader:

            x = x.to(device)
            y = y.to(device)

            decoder_input = torch.zeros_like(y)

            decoder_input[:,1:,:] = y[:,:-1,:]

            optimizer.zero_grad()

            pred = model(
                x,
                decoder_input,
                tgt_mask=None
            )

            loss = criterion(
                pred[:,input_len:,:],
                y[:,input_len:,:]
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = (
            running_loss /
            len(train_loader)
        )

        val_loss = evaluate(
            model,
            val_loader,
            device,
            criterion,
            input_len
        )

        print(
            f"Epoch {epoch} "
            f"train={train_loss:.6f} "
            f"val={val_loss:.6f}"
        )

        if epoch % 50 == 0:

            torch.save(
                {
                    "epoch":epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "mean":mean,
                    "std":std,
                    "input_len":input_len,
                    "output_len":output_len,
                    "config":vars(args),
                },
                output_dir /
                f"aib9_epoch_{epoch}.pt"
            )

        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                {
                    "epoch":epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "mean":mean,
                    "std":std,
                    "input_len":input_len,
                    "output_len":output_len,
                    "config":vars(args),
                },
                best_path
            )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "mean":mean,
            "std":std,
            "input_len":input_len,
            "output_len":output_len,
            "config":vars(args),
        },
        output_dir /
        "aib9_transformer_final.pt"
    )


if __name__ == "__main__":
    train()