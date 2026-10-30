"""
mopr_trans.py  -  Step 2 of the MopR-phenol pipeline (replaces trans1.py of the P450 pipeline).

Same architecture as trans1.py: a seq2seq Transformer (encoder over the first
`input_len` steps of the ligand-RMSD series, decoder generating the full series with
teacher forcing, MSE loss, Adam).  Corrections relative to trans1.py are marked
"FIX" in the code:

  FIX 1  causal mask in the decoder.  trans1.py called self.decoder(tgt, memory)
         with no tgt_mask, so position t could attend to position t+1, whose input
         is exactly the value position t has to predict.  Training loss falls for
         the wrong reason and autoregressive inference does not match training.
  FIX 2  a dedicated start token (START_VALUE = -1, impossible for an RMSD) instead
         of 0, which for MopR is a legitimate RMSD value (bound state).
  FIX 3  padding with the last real value instead of zeros, and a loss mask so padded
         steps do not contribute.  With zero padding every MopR series would appear to
         end with a drop to RMSD 0, which the model would learn as a real event.
  FIX 4  train / test split by source trajectory (split.json from mopr_data.py), with
         periodic evaluation on the held-out trajectories using the same autoregressive
         rollout that inference uses.
  FIX 5  the checkpoint stores the model configuration (input_len, output_len, step
         size ...) so mopr_inference.py rebuilds the identical model instead of guessing.
  FIX 6  wandb and DataParallel are optional flags; nothing is hard-coded.

Usage (defaults reproduce the P450 hyper-parameters: d_model 64, 4 heads, 4 layers,
dropout 0.1, batch 4, lr 1e-4, 500 epochs):

    python mopr_trans.py --data_dir synthetic_trajectories --input_ns 200 --epochs 500 \
        --eval_every 50 --out checkpoints [--wandb --wandb_project timeseries --wandb_entity ME]
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

START_VALUE = -1.0   # FIX 2: decoder start token; RMSD is never negative


# ------------------------------
# Dataset
# ------------------------------
class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, folders, input_len, downsample_factor=1, max_len=None):
        files = []
        for folder in folders:
            files += sorted(glob.glob(os.path.join(data_dir, folder, "traj_*.npy")))
        if not files:
            raise ValueError(f"No traj_*.npy files found under {data_dir} for folders {folders}")
        trajs = [np.load(f).astype(np.float32)[::downsample_factor] for f in files]
        if max_len is not None:
            trajs = [tr[:max_len] for tr in trajs]
        self.lengths = np.array([len(tr) for tr in trajs])
        if input_len >= self.lengths.min():
            raise ValueError(f"input_len={input_len} is not shorter than the shortest trajectory "
                             f"({self.lengths.min()} steps)")
        T = int(self.lengths.max())
        data = np.stack([np.pad(tr, (0, T - len(tr)), mode="edge") for tr in trajs])     # FIX 3
        mask = np.stack([np.r_[np.ones(len(tr)), np.zeros(T - len(tr))] for tr in trajs])
        self.data = torch.tensor(data, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.files = files
        self.input_len = int(input_len)
        self.output_len = T
        print(f"Loaded {len(files)} trajectories from {len(folders)} source folders; "
              f"padded shape {tuple(self.data.shape)}; input_len {self.input_len} steps")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        full = self.data[idx]
        return full[: self.input_len].unsqueeze(-1), full.unsqueeze(-1), self.mask[idx]


# ------------------------------
# Positional Encoding (unchanged)
# ------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


# ------------------------------
# Transformer Model (same layers as trans1.py)
# ------------------------------
class TrajectoryTransformer(nn.Module):
    def __init__(self, input_len, output_len, d_model=64, nhead=4, num_layers=4, dropout=0.1,
                 dim_feedforward=128):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.output_proj = nn.Linear(d_model, 1)
        max_len = max(input_len, output_len) + 1000
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
                                         dropout=dropout, batch_first=True)
        dec = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
                                         dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(dec, num_layers=num_layers)

    def encode(self, src):
        return self.encoder(self.pos_encoder(self.input_proj(src)))

    def decode(self, tgt, memory):
        T = tgt.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=tgt.device)   # FIX 1
        out = self.decoder(self.pos_decoder(self.input_proj(tgt)), memory, tgt_mask=causal)
        return self.output_proj(out)

    def forward(self, src, tgt):
        return self.decode(tgt, self.encode(src))


def shift_right(y, start_value=START_VALUE):
    """Teacher-forcing decoder input: [START, y_0, ..., y_{T-2}]."""
    d = torch.full_like(y, start_value)
    d[:, 1:, :] = y[:, :-1, :]
    return d


@torch.no_grad()
def rollout(model, prefix, total_len, start_value=START_VALUE):
    """
    Autoregressive generation consistent with training.
    prefix: [B, L, 1] known first L steps.  Returns [B, total_len, 1] where the first L
    steps are the prefix and the rest is generated.
    """
    memory = model.encode(prefix)
    # decoder input [START, y_0, ..., y_{L-1}] -> output at position L is the prediction of y_L
    tgt = torch.cat([torch.full_like(prefix[:, :1, :], start_value), prefix], dim=1)
    gen = [prefix]
    for _ in range(total_len - prefix.size(1)):
        out = model.decode(tgt, memory)
        nxt = out[:, -1:, :]
        gen.append(nxt)
        tgt = torch.cat([tgt, nxt], dim=1)
    return torch.cat(gen, dim=1)


def masked_mse(pred, y, mask):
    se = ((pred - y) ** 2).squeeze(-1) * mask
    return se.sum() / mask.sum().clamp_min(1.0)


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


# ------------------------------
# Evaluation on held-out source trajectories
# ------------------------------
@torch.no_grad()
def evaluate(model, dataset, device, batch_size=4):
    core = unwrap(model)
    core.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tf_loss, n = 0.0, 0
    abs_err, n_pred = 0.0, 0.0
    L = dataset.input_len
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        pred = core(x, shift_right(y))
        tf_loss += masked_mse(pred, y, m).item() * x.size(0); n += x.size(0)
        gen = rollout(core, x, y.size(1))
        err = (gen - y).abs().squeeze(-1)[:, L:] * m[:, L:]
        abs_err += err.sum().item(); n_pred += m[:, L:].sum().item()
    return tf_loss / max(n, 1), abs_err / max(n_pred, 1.0)


# ------------------------------
# Training
# ------------------------------
def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    with open(os.path.join(args.data_dir, "split.json")) as f:
        split = json.load(f)
    step_ns = split["step_ns"] * args.downsample_factor
    if args.input_steps is not None:
        input_len = args.input_steps
    else:
        input_len = int(round(args.input_ns / step_ns))
    max_len = None if args.max_len_ns is None else int(round(args.max_len_ns / step_ns))

    train_ds = TrajectoryDataset(args.data_dir, split["train"], input_len, args.downsample_factor, max_len)
    test_ds = None
    if split["test"]:
        test_ds = TrajectoryDataset(args.data_dir, split["test"], input_len, args.downsample_factor,
                                    max_len=train_ds.output_len)
    output_len = max(train_ds.output_len, test_ds.output_len if test_ds else 0)
    print(f"step {step_ns:.2f} ns | input {input_len} steps = {input_len*step_ns:.0f} ns | "
          f"output {output_len} steps = {output_len*step_ns:.0f} ns")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    cfg = dict(input_len=input_len, output_len=output_len, d_model=args.d_model, nhead=args.nhead,
               num_layers=args.num_layers, dropout=args.dropout, dim_feedforward=args.dim_feedforward)
    model = TrajectoryTransformer(**cfg)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    os.makedirs(args.out, exist_ok=True)
    meta = dict(config=cfg, step_ns=step_ns, downsample_factor=args.downsample_factor,
                start_value=START_VALUE, train=split["train"], test=split["test"], args=vars(args))

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            optimizer.zero_grad()
            pred = model(x, shift_right(y))
            loss = masked_mse(pred, y, m)
            loss.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            running += loss.item()
        avg = running / len(loader)
        log = {"epoch": epoch, "train_loss": avg}
        msg = f"Epoch [{epoch}/{args.epochs}] train MSE {avg:.5f}"

        if test_ds is not None and (epoch % args.eval_every == 0 or epoch == args.epochs):
            tf, mae = evaluate(model, test_ds, device, args.batch_size)
            log.update({"test_teacher_forced_mse": tf, "test_rollout_mae_nm": mae})
            msg += f" | held-out: teacher-forced MSE {tf:.5f}, rollout MAE {mae:.3f} nm"
        msg += f" | {time.time()-t0:.0f}s"
        print(msg, flush=True)
        if run is not None:
            run.log(log)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            path = os.path.join(args.out, f"transformer_epoch_{epoch}.pt")
            torch.save({"model_state": unwrap(model).state_dict(), "epoch": epoch, **meta}, path)   # FIX 5
            print(f"  saved {path}")
    if run is not None:
        run.finish()


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="synthetic_trajectories")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--dim_feedforward", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--downsample_factor", type=int, default=1,
                    help="extra striding on top of mopr_data.py's block average (P450 used 52 on raw data)")
    ap.add_argument("--input_ns", type=float, default=200.0,
                    help="length of the known prefix in ns (must end before the earliest binding, ~270 ns)")
    ap.add_argument("--input_steps", type=int, default=None, help="overrides --input_ns")
    ap.add_argument("--max_len_ns", type=float, default=None,
                    help="optionally truncate every series to this length (trj1 is 3942 ns, most are ~1000)")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--multi_gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="timeseries")
    ap.add_argument("--wandb_entity", default=None)
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
