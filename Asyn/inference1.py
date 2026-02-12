#!/usr/bin/env python3
import os
os.environ.setdefault("WANDB_DISABLED", "true")

import argparse
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import your model class (keep the same import path you used)
from trans1 import TrajectoryTransformer  # ensure this path is correct in your project


# ---- helpers ----

def load_time_if_available(data_dir):
    p = os.path.join(data_dir, "time.npy")
    return np.load(p).astype(float) if os.path.exists(p) else None

def load_state_dict_flex(path, device):
    # robust loader; strips DataParallel; drops PE buffers sized on seq length
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)  # torch>=2.4
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt
    if isinstance(sd, dict):
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
        for k in [k for k in sd.keys() if k.endswith("pos_encoder.pe") or k.endswith("pos_decoder.pe")]:
            sd.pop(k, None)
            print(f"[load] dropped '{k}' from checkpoint (positional encoding buffer)")
    return sd

def extend_pe_sinusoidal(pe_buf, new_len, device=None):
    """
    Extend a positional encoding buffer pe_buf [1, old_len, D] to [1, new_len, D]
    using standard sinusoidal positional encodings. Returns tensor on same device/dtype.
    """
    if not isinstance(pe_buf, torch.Tensor):
        pe_buf = torch.as_tensor(pe_buf)
    old_len = pe_buf.size(1)
    if new_len <= old_len:
        return pe_buf.to(device) if device is not None else pe_buf
    D = pe_buf.size(2)
    dtype = pe_buf.dtype
    device = device if device is not None else pe_buf.device

    positions = torch.arange(old_len, new_len, dtype=torch.float32, device=device).unsqueeze(1)  # [L,1]
    div_term = torch.exp(torch.arange(0, D, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / D))
    pe_extra = torch.zeros((new_len - old_len, D), dtype=dtype, device=device)
    pe_extra[:, 0::2] = torch.sin(positions * div_term)
    # handle odd D
    if D % 2 == 0:
        pe_extra[:, 1::2] = torch.cos(positions * div_term)
    else:
        pe_extra[:, 1::2] = torch.cos(positions * div_term[: (D // 2)])
    pe_extra = pe_extra.unsqueeze(0)  # [1, Lnew, D]
    return torch.cat([pe_buf.to(device), pe_extra], dim=1)  # [1, new_len, D]

def add_pos_with_offset(x_proj, pe_buf, start_pos):
    """
    x_proj: [B, L, D] already projected to model dim
    pe_buf: [1, max_len, D] positional encodings buffer (from model.pos_* .pe)
    start_pos: integer global offset of the first token in x_proj
    """
    B, L, D = x_proj.shape
    # slice pe with absolute positions
    pe_slice = pe_buf[:, start_pos:start_pos+L, :]
    return x_proj + pe_slice

@torch.no_grad()
def generate_chunked_reencode_with_offsets(
    model, src_init, total_len,
    chunk_size=10000, context_len=2048,
    reencode_every=1024, src_window=8192,
    log_every=200, outdir=None, base="000"
):
    """
    Autoregressive generation:
      - encoder re-encodes periodically from the last `src_window` steps **with absolute PE offsets**
      - decoder uses sliding context `context_len` **with absolute PE offsets**
      - saves chunks incrementally
    Returns full prediction [T,2] and list of saved part paths.
    """
    device = src_init.device
    model.eval()

    # convenience references
    in_proj  = model.input_proj
    out_proj = model.output_proj
    enc      = model.encoder
    dec      = model.decoder
    pe_enc   = model.pos_encoder.pe   # [1, max_len, D]
    pe_dec   = model.pos_decoder.pe   # [1, max_len, D]

    # make sure pe_* are tensors on device
    if not isinstance(pe_enc, torch.Tensor):
        pe_enc = torch.as_tensor(pe_enc).to(device)
    if not isinstance(pe_dec, torch.Tensor):
        pe_dec = torch.as_tensor(pe_dec).to(device)

    S = src_init.size(1)
    T = total_len

    pred = np.zeros((T, 2), dtype=np.float32)
    pred[:S] = src_init.squeeze(0).detach().cpu().numpy()

    # initial encoder memory from the last src_window of known/predicted, using absolute offsets
    def build_memory_from_global_slice(start_idx, end_idx):
        # slice [start_idx, end_idx) from the global pred buffer, convert to tensor, add PE with offset=start_idx
        sl_np = pred[start_idx:end_idx]                    # [L, 2]
        sl_t  = torch.from_numpy(sl_np).unsqueeze(0).to(device).float()  # [1,L,2]
        x     = in_proj(sl_t)                              # [1,L,D]
        x     = add_pos_with_offset(x, pe_enc, start_idx)  # absolute positions
        return enc(x)                                      # [1,L,D] -> memory

    # encoder memory seeded on the last part of the known prefix
    enc_start = max(0, S - src_window)
    memory = build_memory_from_global_slice(enc_start, S)

    # decoder buffer starts as tail of prefix; its first token is at global index (S - B)
    buffer = src_init[:, -min(context_len, S):, :].clone()  # [1,B,2]
    buffer_global_start = S - buffer.size(1)                # absolute pos of buffer[0]

    steps_done = S
    part_idx = 0
    part_paths = []

    while steps_done < T:
        part_idx += 1
        remaining = T - steps_done
        this_chunk = min(chunk_size, remaining)
        block = np.zeros((this_chunk, 2), dtype=np.float32)

        for i in range(this_chunk):

            # periodic re-encode from last src_window global steps (absolute PE)
            if reencode_every > 0 and ((steps_done - S) % reencode_every == 0 or steps_done == S):
                enc_end   = steps_done
                enc_start = max(0, enc_end - src_window)
                memory    = build_memory_from_global_slice(enc_start, enc_end)

            # one-step decode with absolute PE for decoder buffer
            tgt_proj = in_proj(buffer)                                            # [1,B,D]
            tgt_pos  = add_pos_with_offset(tgt_proj, pe_dec, buffer_global_start) # abs positions
            L = tgt_pos.size(1)
            # build causal mask for length L (True=mask)
            tgt_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
            out = dec(tgt_pos, memory, tgt_mask=tgt_mask)                         # [1,B,D]
            nxt = out_proj(out)[:, -1:, :]                                        # [1,1,2]

            # commit
            val_np = nxt.squeeze(0).squeeze(0).detach().cpu().numpy()
            pred[steps_done] = val_np
            block[i] = val_np
            steps_done += 1

            # slide decoder buffer and its global start
            buffer = torch.cat([buffer, nxt], dim=1)
            if buffer.size(1) > context_len:
                # dropping leftmost token increases global start by dropped count
                drop = buffer.size(1) - context_len
                buffer = buffer[:, -context_len:, :]
                buffer_global_start += drop
            else:
                # we appended 1 token, length grew by 1; global start unchanged
                pass

            if (steps_done % log_every) == 0 or steps_done == T:
                print(f"[decode] generated {steps_done}/{T}", flush=True)

        # save chunk
        if outdir is not None:
            part_path = os.path.join(outdir, f"predicted_traj_{base}_part{part_idx:03d}.npy")
            np.save(part_path, block)
            part_paths.append(part_path)
            print(f"[save] chunk {part_idx}: steps {steps_done - this_chunk}-{steps_done - 1} -> {part_path}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pred, part_paths


# ---- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="transformer_epoch_190.pt")
    ap.add_argument("--data_dir", type=str, default="synthetic_trpcage")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="inference_output_asyn")
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--input_pct", type=float, default=0.1)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--chunk_size", type=int, default=10000)
    ap.add_argument("--context_len", type=int, default=2048)
    ap.add_argument("--reencode_every", type=int, default=1024)
    ap.add_argument("--src_window", type=int, default=8192)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_csv", action="store_true")
    # NEW: absolute total length to generate (overrides using original T)
    ap.add_argument("--total_len", type=int, default=None,
                    help="Total number of points to generate. If omitted, uses original dataset length.")
    # NEW: alternatively, a multiplier of original length (e.g. 2 => double length)
    ap.add_argument("--multiplier", type=float, default=None,
                    help="Multiply original trajectory length by this factor to get total_len.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)

    # load trajectory [T_raw,2]
    traj_path = os.path.join(args.data_dir, f"traj_{args.index:03d}.npy")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(traj_path)
    full = np.load(traj_path).astype(np.float32)
    if full.ndim != 2 or full.shape[1] != 2:
        raise ValueError(f"Expected [T,2], got {full.shape}")

    if args.downsample > 1:
        full = full[::args.downsample]

    orig_T = full.shape[0]

    # Decide total generation length
    if args.total_len is not None:
        total_len = int(args.total_len)
    elif args.multiplier is not None:
        total_len = int(max(1, round(orig_T * float(args.multiplier))))
    else:
        total_len = orig_T

    if total_len <= 0:
        raise ValueError("total_len must be positive")

    # S = initial prefix length: follow user's original logic: 10% of original trajectory
    S = max(1, int(orig_T * args.input_pct))
    print(f"Original T = {orig_T}. Using input_pct={args.input_pct:.3f} -> S={S}. Generating total_len={total_len}")

    src_np = full[:S, :]

    # tensors
    src = torch.from_numpy(src_np).unsqueeze(0).to(device)

    # build model + load weights (ignore PE buffers)
    model = TrajectoryTransformer(
        input_len=S, output_len=total_len,
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dropout=args.dropout
    ).to(device)
    sd = load_state_dict_flex(args.model, device)
    model.load_state_dict(sd, strict=False)
    print(f"Loaded weights from {args.model} (PE buffers reinitialized).")

    # Ensure positional encoders are long enough. If not, extend them using sinusoidal PE.
    pe_enc = model.pos_encoder.pe   # [1, max_len, D]
    pe_dec = model.pos_decoder.pe   # [1, max_len, D]

    # convert to tensors if not
    if not isinstance(pe_enc, torch.Tensor):
        pe_enc = torch.as_tensor(pe_enc)
    if not isinstance(pe_dec, torch.Tensor):
        pe_dec = torch.as_tensor(pe_dec)

    # extend if needed
    if pe_enc.size(1) < total_len or pe_dec.size(1) < total_len:
        print(f"[pe] PE buffers shorter than requested total_len={total_len}. Extending PE buffers.")
        if pe_enc.size(1) < total_len:
            pe_enc = extend_pe_sinusoidal(pe_enc.to(device), total_len, device=device)
            # set back (works if attribute writable)
            model.pos_encoder.pe = pe_enc
        else:
            model.pos_encoder.pe = pe_enc.to(device)
        if pe_dec.size(1) < total_len:
            pe_dec = extend_pe_sinusoidal(pe_dec.to(device), total_len, device=device)
            model.pos_decoder.pe = pe_dec
        else:
            model.pos_decoder.pe = pe_dec.to(device)
    else:
        model.pos_encoder.pe = pe_enc.to(device)
        model.pos_decoder.pe = pe_dec.to(device)

    base = f"{args.index:03d}"
    pred, part_paths = generate_chunked_reencode_with_offsets(
        model, src, total_len=total_len,
        chunk_size=args.chunk_size,
        context_len=args.context_len,
        reencode_every=args.reencode_every,
        src_window=args.src_window,
        log_every=args.log_every,
        outdir=args.outdir,
        base=base
    )

    # save final
    pred_path = os.path.join(args.outdir, f"predicted_traj_{base}.npy")
    in_path   = os.path.join(args.outdir, f"input_traj_{base}.npy")
    gt_path   = os.path.join(args.outdir, f"groundtruth_traj_{base}.npy")
    np.save(pred_path, pred); np.save(in_path, src_np); np.save(gt_path, full)
    print(f"Saved final arrays: {pred_path}, {in_path}, {gt_path}")

    if args.save_csv:
        np.savetxt(os.path.join(args.outdir, f"predicted_traj_{base}.csv"), pred, delimiter=",", header="z1,z2", comments="")
        np.savetxt(os.path.join(args.outdir, f"input_traj_{base}.csv"), src_np, delimiter=",", header="z1,z2", comments="")
        np.savetxt(os.path.join(args.outdir, f"groundtruth_traj_{base}.csv"), full, delimiter=",", header="z1,z2", comments="")

    # plot: overlay predicted and true (true may be shorter)
    x_pred = np.arange(len(pred))
    x_true = np.arange(orig_T)
    plt.figure(figsize=(10,5))
    plt.subplot(2,1,1)
    plt.plot(x_pred, pred[:,0], "--", label="z1 pred")
    plt.plot(x_true, full[:,0], label="z1 true")
    plt.axvline(x[S-1], linestyle=":", label="prefix end")
    plt.legend(); plt.ylabel("z1")
    plt.subplot(2,1,2)
    plt.plot(x_pred, pred[:,1], "--", label="z2 pred")
    plt.plot(x_true, full[:,1], label="z2 true")
    plt.axvline(x[S-1], linestyle=":")
    plt.legend(); plt.ylabel("z2"); plt.xlabel("time")
    plt.tight_layout()
    fig_path = os.path.join(args.outdir, f"prediction_{base}.png")
    plt.savefig(fig_path, dpi=200); plt.close()
    print("Saved plot to", fig_path)


if __name__ == "__main__":
    main()

