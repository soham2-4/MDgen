# inference1.py

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # choose GPUs with the CUDA_VISIBLE_DEVICES environment variable instead

from gb3_trans1 import TrajectoryTransformer

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="gb3_checkpoints/gb3_transformer_best.pt"
    )

    parser.add_argument(
        "--data_dir",
        default="synthetic_gb3_trajectories"
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0
    )

    parser.add_argument(
        "--outdir",
        default="inference_output"
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------

    ckpt = torch.load(
        args.model,
        map_location=device
    )

    mean = ckpt["mean"]
    std = ckpt["std"]

    input_len = ckpt["input_len"]
    output_len = ckpt["output_len"]

    cfg = ckpt["config"]

    model = TrajectoryTransformer(
        input_len=input_len,
        output_len=output_len,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"]
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print("Checkpoint loaded")

    # --------------------------------------------------
    # Load trajectory
    # --------------------------------------------------

    traj_file = os.path.join(
        args.data_dir,
        f"traj_{args.index:03d}.npy"
    )

    y_true = np.load(traj_file).astype(np.float32)

    print("Trajectory length:", len(y_true))

    # --------------------------------------------------
    # Normalize
    # --------------------------------------------------

    y_norm = (y_true - mean) / std

    y_tensor = torch.tensor(
        y_norm,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(-1).to(device)

    x = y_tensor[:, :input_len, :]

    # --------------------------------------------------
    # Teacher-forced decoder input
    # --------------------------------------------------

    decoder_input = torch.zeros_like(y_tensor)

    decoder_input[:, 1:, :] = y_tensor[:, :-1, :]

    # --------------------------------------------------
    # IMPORTANT:
    # NO causal mask
    # --------------------------------------------------

    with torch.no_grad():

        pred = model(
            x,
            decoder_input,
            tgt_mask=None
        )

    pred = pred.squeeze().cpu().numpy()

    # --------------------------------------------------
    # De-normalize
    # --------------------------------------------------

    pred = pred * std + mean

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------

    np.save(
        os.path.join(
            args.outdir,
            "prediction.npy"
        ),
        pred
    )

    np.save(
        os.path.join(
            args.outdir,
            "ground_truth.npy"
        ),
        y_true
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    plt.figure(figsize=(12,5))

    plt.plot(
        y_true,
        label="Ground Truth",
        linewidth=1
    )

    plt.plot(
        pred,
        label="Prediction",
        linewidth=1
    )

    plt.axvline(
        input_len,
        color="k",
        linestyle="--",
        label="Input Boundary"
    )

    plt.xlabel("Time Index")
    plt.ylabel("Native Contacts")

    plt.legend()

    plt.tight_layout()

    plot_file = os.path.join(
        args.outdir,
        "prediction.png"
    )

    plt.savefig(
        plot_file,
        dpi=300
    )

    print("Saved prediction to:", args.outdir)
    print("Saved plot:", plot_file)


if __name__ == "__main__":
    main()