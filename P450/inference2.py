import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from trans1 import TrajectoryTransformer  # Adjust this if your model file has a different name


def generate_and_plot(model_path="transformer_epoch_500.pt",
                      data_dir="synthetic_trajectories",
                      folder_name="distance2",
                      filename_index=3,
                      downsample_factor=52,
                      input_pct=0.05,
                      save_plot=True,
                      show_plot=False):
    """
    Generate and plot trajectory prediction using a trained transformer model.

    Args:
        model_path: Path to the trained model (.pt file).
        data_dir: Base directory containing trajectory folders.
        folder_name: Subfolder (e.g., 'distance0', 'distance1') to load trajectory from.
        filename_index: Index for zero-padded trajectory file (e.g., 3 → traj_003.npy).
        downsample_factor: Temporal downsampling factor.
        input_pct: Fraction of the trajectory used as input.
        save_plot: Whether to save the prediction plot.
        show_plot: Whether to display the plot.
    """

    folder_path = os.path.join(data_dir, folder_name)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} does not exist.")

    # --------- Load Specific Trajectory File ---------
    file_name = f"traj_{filename_index:03d}.npy"  # zero-padded filename
    target_file = os.path.join(folder_path, file_name)
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Trajectory file '{target_file}' not found.")

    traj = np.load(target_file)[::downsample_factor]  # Downsample
    print(f"Loaded trajectory '{file_name}' with length {len(traj)} after downsampling.")

    full_traj = traj.reshape(-1, 1)  # [T, 1]
    total_len = full_traj.shape[0]
    input_len = int(total_len * input_pct)
    input_seq = full_traj[:input_len]  # [input_len, 1]

    # --------- Initialize Model ---------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TrajectoryTransformer(
        input_len=input_len,
        output_len=total_len,
        d_model=64,
        nhead=4,
        num_layers=4,
        dropout=0.1
    ).to(device)

    # --------- Load Model Weights (Fix Positional Encoding if Needed) ---------
    loaded_dict = torch.load(model_path, map_location=device)

    # Remove DataParallel prefix if needed
    if list(loaded_dict.keys())[0].startswith("module."):
        loaded_dict = {k.replace("module.", ""): v for k, v in loaded_dict.items()}

    model_state = model.state_dict()

    # Truncate positional encodings to match current model
    for key in ['pos_encoder.pe', 'pos_decoder.pe']:
        if key in loaded_dict and key in model_state:
            ckpt_tensor = loaded_dict[key]
            expected_shape = model_state[key].shape
            if ckpt_tensor.shape != expected_shape:
                print(f"Fixing shape mismatch for {key}: {ckpt_tensor.shape} → {expected_shape}")
                if ckpt_tensor.shape[1] > expected_shape[1]:
                    loaded_dict[key] = ckpt_tensor[:, :expected_shape[1], :]
                else:
                    raise ValueError(f"{key} is too short ({ckpt_tensor.shape[1]} < {expected_shape[1]})")

    model.load_state_dict(loaded_dict)
    model.eval()

    # --------- Inference ---------
    with torch.no_grad():
        src = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)  # [1, input_len, 1]
        tgt = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)

        pred_seq = input_seq.copy().tolist()

        for _ in range(total_len - input_len):
            out = model(src, tgt)
            next_val = out[0, -1, 0].item()
            pred_seq.append([next_val])
            next_tensor = torch.tensor([[[next_val]]], dtype=torch.float32).to(device)
            tgt = torch.cat([tgt, next_tensor], dim=1)

        pred_traj = np.array(pred_seq).reshape(-1)
        full_traj = full_traj.reshape(-1)

    print(f"Prediction complete. Predicted shape: {pred_traj.shape}, Original shape: {full_traj.shape}")

    # --------- Plotting ---------
    plt.figure(figsize=(12, 5))
    plt.plot(full_traj, label="Original Trajectory", linewidth=2)
    plt.plot(pred_traj, label="Predicted Trajectory", linestyle='--')
    plt.axvline(x=input_len, color='red', linestyle=':', label="Prediction Start")
    plt.xlabel("Time step")
    plt.ylabel("Position")
    plt.legend()
    plt.title(f"Trajectory Prediction ({folder_name}/traj_{filename_index:03d}.npy from {input_pct*100:.0f}% input)")
    plt.tight_layout()

    if save_plot:
        plot_name = f"predicted_traj_{folder_name}_idx{filename_index:03d}_.05_2.png"
        plt.savefig(plot_name)
        print(f"Saved plot to {plot_name}")
    if show_plot:
        plt.show()
    plt.close()


if __name__ == "__main__":
    generate_and_plot(
        model_path="transformer_epoch_500.pt",
        data_dir="synthetic_trajectories",
        folder_name="distance2",
        filename_index=3,        # traj_003.npy
        downsample_factor=52,
        input_pct=0.1,
        save_plot=True,
        show_plot=True
    )

