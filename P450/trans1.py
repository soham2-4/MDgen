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
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(
    project="timeseries",
    entity="choudhurisoham",
    config={
        "epochs": 500,
        "batch_size": 4,
        "lr": 1e-4,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 4,
        "dropout": 0.1,
        "downsample_factor": 52,
        "input_pct": 0.1,
    }
)
config = wandb.config

# ------------------------------
# Dataset
# ------------------------------
class TrajectoryDataset(Dataset):
    def __init__(self, data_dir="synthetic_trajectories", input_pct=0.1, downsample_factor=52):
        # Recursively load all .npy files from all subfolders
        files = sorted(glob.glob(os.path.join(data_dir, "**", "traj_*.npy"), recursive=True))
        if len(files) == 0:
            raise ValueError(f"No trajectory files found in {data_dir}")

        # Load and downsample
        trajectories = [np.load(f)[::downsample_factor] for f in files]
        max_len = max(len(traj) for traj in trajectories)

        # Pad to max length
        padded = [
            np.pad(traj, (0, max_len - len(traj)), mode='constant') for traj in trajectories
        ]

        data = np.stack(padded)  # shape: [N, max_len]
        print(f"Loaded {len(padded)} padded trajectories with shape: {data.shape}")

        self.data = torch.tensor(data, dtype=torch.float32)
        self.input_len = int(self.data.shape[1] * input_pct)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        full_seq = self.data[idx]
        input_seq = full_seq[:self.input_len]
        return input_seq.unsqueeze(-1), full_seq.unsqueeze(-1)

# ------------------------------
# Positional Encoding
# ------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ------------------------------
# Transformer Model
# ------------------------------
class TrajectoryTransformer(nn.Module):
    def __init__(self, input_len, output_len, d_model=64, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.output_proj = nn.Linear(d_model, 1)

        max_len = max(input_len, output_len) + 1000
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout, batch_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, src, tgt):
        src = self.input_proj(src)
        tgt = self.input_proj(tgt)

        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)

        memory = self.encoder(src)
        out = self.decoder(tgt, memory)
        return self.output_proj(out)

# ------------------------------
# Training Loop
# ------------------------------
def train():
    dataset = TrajectoryDataset(
        data_dir="synthetic_trajectories",
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
            x, y = x.to(device), y.to(device)
            decoder_input = torch.zeros_like(y)
            decoder_input[:, 1:, :] = y[:, :-1, :]

            optimizer.zero_grad()
            pred = model(x, decoder_input)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config.epochs}] Loss: {avg_loss:.6f}")
        wandb.log({"loss": avg_loss, "epoch": epoch+1})

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f"transformer_epoch_{epoch+1}.pt")

# ------------------------------
# Entry Point
# ------------------------------
if __name__ == "__main__":
    train()

