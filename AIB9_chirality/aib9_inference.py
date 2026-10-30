import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # choose GPUs with the CUDA_VISIBLE_DEVICES environment variable instead

from aib9_trans1 import TrajectoryTransformer


def load_replica(csv_file):

    df = pd.read_csv(csv_file)

    return df["Zeta_rad"].values.astype(np.float32)


def generate_trajectory(
    model,
    observed,
    total_len,
    device
):

    src = torch.tensor(
        observed,
        dtype=torch.float32
    ).reshape(1, -1, 1).to(device)

    tgt = src.clone()

    generated = observed.tolist()

    model.eval()

    with torch.no_grad():

        for step in range(
            total_len - len(observed)
        ):

            out = model(
                src,
                tgt,
                tgt_mask=None
            )

            next_val = out[0, -1, 0].item()

            generated.append(next_val)

            next_tensor = torch.tensor(
                [[[next_val]]],
                dtype=torch.float32
            ).to(device)

            tgt = torch.cat(
                [tgt, next_tensor],
                dim=1
            )

            if step % 1000 == 0:
                print(
                    f"Generated {step} / "
                    f"{total_len-len(observed)}"
                )

    return np.array(generated)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="aib9_checkpoints/aib9_transformer_best.pt"
    )

    parser.add_argument(
        "--input-pct",
        type=float,
        default=0.10
    )

    parser.add_argument(
        "--outdir",
        default="aib9_inference_output"
    )

    args = parser.parse_args()

    os.makedirs(
        args.outdir,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Using device:",
        device
    )

    # -----------------------------------
    # Load checkpoint
    # -----------------------------------

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

    print(
        "Checkpoint loaded"
    )

    replicas = [
        "aib9_chirality_analysis_data_replica_1.csv",
        "aib9_chirality_analysis_data_replica_2.csv",
        "aib9_chirality_analysis_data_replica_3.csv",
        "aib9_chirality_analysis_data_replica_4.csv",
        "aib9_chirality_analysis_data_replica_5.csv",
    ]

    originals = []
    generateds = []

    for i, replica in enumerate(replicas):

        print(
            f"\nReplica {i+1}"
        )

        traj = load_replica(replica)

        total_len = len(traj)

        observed_len = int(
            total_len *
            args.input_pct
        )

        observed = traj[
            :observed_len
        ]

        # normalize

        observed_norm = (
            observed - mean
        ) / std

        pred_norm = generate_trajectory(
            model=model,
            observed=observed_norm,
            total_len=total_len,
            device=device
        )

        pred = (
            pred_norm * std
            + mean
        )

        originals.append(
            traj
        )

        generateds.append(
            pred
        )

        np.save(
            os.path.join(
                args.outdir,
                f"generated_replica_{i+1}.npy"
            ),
            pred
        )

    # -----------------------------------
    # Plot
    # -----------------------------------

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(16,20),
        sharex=True
    )

    for i in range(5):

        ax = axes[i]

        input_boundary = int(
            len(originals[i]) *
            args.input_pct
        )

        ax.plot(
            originals[i],
            label=f"Replica {i+1} Original",
            linewidth=1
        )

        ax.plot(
            generateds[i],
            label=f"Replica {i+1} Generated",
            linewidth=1
        )

        ax.axvline(
            input_boundary,
            color="k",
            linestyle="--",
            label="Input Boundary"
        )

        ax.set_ylabel(
            "Zeta_rad"
        )

        ax.legend()

    axes[-1].set_xlabel(
        "Time Index"
    )

    plt.tight_layout()

    outfile = os.path.join(
        args.outdir,
        "all_replicas_autoregressive.png"
    )

    plt.savefig(
        outfile,
        dpi=300
    )

    print(
        "\nSaved figure:",
        outfile
    )


if __name__ == "__main__":
    main()