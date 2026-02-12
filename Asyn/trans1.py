# trans1.py  — two-channel (z1, z2) version, minimal change from your old code
import os
import glob
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# Setup
# ------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"  # keep your original behavior
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(
    project="timeseries",
    entity="choudhurisoham",
    config={
        "epochs": 700,
        "batch_size": 4,
        "lr": 1e-4,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 4,
        "dropout": 0.1,
        "downsample_factor": 2,   # if you didn't downsample when creating TRP-cage synthetics, set this to 1
        "input_pct": 0.1,
        "data_dir": "synthetic_trpcage"  # <— point to your TRP-cage synthetic set (traj_XXX.npy with shape [T,2])
    }
)
config = wandb.config

# ------------------------------
# Dataset  (expects each traj_*.npy to be [T, 2] = [z1, z2])
# ------------------------------
class TrajectoryDataset(Dataset):
    def __init__(self, data_dir=None, input_pct=0.1, downsample_factor=1):
        data_dir = data_dir or "synthetic_trpcage"
        files = sorted(glob.glob(os.path.join(data_dir, "traj_*.npy")))
        if not files:
            raise FileNotFoundError(f"No traj_*.npy found in {data_dir}")

        mats = []
        for f in files:
            arr = np.load(f)  # expected shape [T, 2]
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"{f} has shape {arr.shape}, expected [T, 2] with columns [z1, z2]")
            if downsample_factor > 1:
                arr = arr[::downsample_factor]
            mats.append(arr.astype(np.float32))

        # enforce equal lengths
        lengths = {m.shape[0] for m in mats}
        if len(lengths) != 1:
            raise ValueError(f"All trajectories must have same T, got lengths: {sorted(lengths)}")

        data = np.stack(mats, axis=0)  # [N, T, 2]
        print(f"Loaded {len(mats)} trajectories with shape: {data.shape}")

        self.data = torch.tensor(data, dtype=torch.float32)
        self.input_len = int(self.data.shape[1] * input_pct)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        full_seq = self.data[idx]                     # [T, 2]
        input_seq = full_seq[:self.input_len]         # [S, 2]
        return input_seq, full_seq

# ------------------------------
# Positional Encoding
# ------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def generate_subsequent_mask(sz: int) -> torch.Tensor:
    # Bool mask where True marks masked (future) positions
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)

# ------------------------------
# Transformer Model  (now 2→D→2)
# ------------------------------
class TrajectoryTransformer(nn.Module):
    def __init__(self, input_len, output_len, d_model=64, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)   # was (1 → d_model)
        self.output_proj = nn.Linear(d_model, 2)  # was (d_model → 1)

        max_len = max(input_len, output_len) + 1000
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128, dropout=dropout, batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=128, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, src, tgt):
        # src: [B, S, 2], tgt: [B, T, 2]
        src = self.input_proj(src)
        tgt = self.input_proj(tgt)

        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)

        memory = self.encoder(src)
        T = tgt.size(1)
        tgt_mask = generate_subsequent_mask(T).to(tgt.device)  # prevent peeking into the future
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.output_proj(out)  # [B, T, 2]

# ------------------------------
# Training Loop
# ------------------------------
def train():
    dataset = TrajectoryDataset(
        data_dir=config.get("data_dir", "synthetic_trpcage"),
        input_pct=config.input_pct,
        downsample_factor=config.downsample_factor
    )

    input_len = dataset.input_len
    output_len = dataset.data.shape[1]

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = TrajectoryTransformer(
        input_len=input_len,
        output_len=output_len,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout
    )

    model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0

        for x, y in dataloader:
            # x: [B, S, 2], y: [B, T, 2]
            x, y = x.to(device), y.to(device)

            # Prepare decoder input: shift ground truth by 1 with zero at start
            decoder_input = torch.zeros_like(y)
            decoder_input[:, 1:, :] = y[:, :-1, :]

            optimizer.zero_grad()
            pred = model(x, decoder_input)  # [B, T, 2]
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config.epochs}] Loss: {avg_loss:.6f}")
        wandb.log({"loss": avg_loss, "epoch": epoch+1})

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"transformer_epoch_{epoch+1}.pt")

# ------------------------------
# Entry Point
# ------------------------------
if __name__ == "__main__":
    train()

