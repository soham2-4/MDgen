
import argparse
import os
import numpy as np
import pandas as pd


def load_gb3_timeseries(filename: str, column: str):
    """
    Load GB3 time series file and return time (ns) and selected data column.
    Expected columns:
      Global_Frame Chunk_ID Local_Frame Absolute_Time_ps Absolute_Time_ns
      Absolute_Time_us N_Native_Contacts Q_Native
    """
    df = pd.read_csv(
        filename,
        sep=r"\s+",
        comment="#",
        header=None,
        names=[
            "Global_Frame",
            "Chunk_ID",
            "Local_Frame",
            "Absolute_Time_ps",
            "Absolute_Time_ns",
            "Absolute_Time_us",
            "N_Native_Contacts",
            "Q_Native",
        ],
    )

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")

    time_ns = df["Absolute_Time_ns"].to_numpy(dtype=float)
    values = df[column].to_numpy(dtype=float)
    return time_ns, values, df


def main():
    parser = argparse.ArgumentParser(description="Create synthetic GB3 trajectories from a native-contact time series.")
    parser.add_argument(
        "--input",
        type=str,
        default="gb3_native_contacts_vs_time.dat",
        help="Input GB3 time-series file",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="synthetic_gb3_trajectories",
        help="Folder to save synthetic trajectories",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="N_Native_Contacts",
        choices=["N_Native_Contacts", "Q_Native"],
        help="Which column to use for synthetic data generation",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=1000,
        help="Number of synthetic trajectories to generate",
    )
    parser.add_argument(
        "--noise-min",
        type=float,
        default=0.0001,
        help="Minimum uniform noise amplitude",
    )
    parser.add_argument(
        "--noise-max",
        type=float,
        default=0.001,
        help="Maximum uniform noise amplitude",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    os.makedirs(args.output_folder, exist_ok=True)

    time_ns, original_series, df = load_gb3_timeseries(args.input, args.column)

    # Save time once
    np.save(os.path.join(args.output_folder, "time_ns.npy"), time_ns)

    # Save the original series too
    np.save(os.path.join(args.output_folder, f"{args.column}.npy"), original_series)

    # Generate synthetic trajectories by adding small random noise
    for i in range(args.num_trajectories):
        noise = np.random.uniform(
            args.noise_min,
            args.noise_max,
            size=original_series.shape,
        ) * np.random.choice([-1, 1], size=original_series.shape)

        synthetic = original_series + noise

        # If using native contact counts, keep values physically meaningful
        if args.column == "N_Native_Contacts":
            synthetic = np.clip(np.rint(synthetic), 0, None).astype(np.float32)
        else:
            # Q values should stay in [0, 1]
            synthetic = np.clip(synthetic, 0.0, 1.0).astype(np.float32)

        np.save(os.path.join(args.output_folder, f"traj_{i:03d}.npy"), synthetic)

    # Save a small metadata file
    meta_path = os.path.join(args.output_folder, "metadata.txt")
    with open(meta_path, "w") as f:
        f.write(f"input_file={args.input}\n")
        f.write(f"column={args.column}\n")
        f.write(f"num_frames={len(original_series)}\n")
        f.write(f"time_start_ns={time_ns[0]}\n")
        f.write(f"time_end_ns={time_ns[-1]}\n")
        f.write(f"series_min={original_series.min()}\n")
        f.write(f"series_max={original_series.max()}\n")
        f.write(f"series_mean={original_series.mean()}\n")
        f.write(f"noise_min={args.noise_min}\n")
        f.write(f"noise_max={args.noise_max}\n")
        f.write(f"num_trajectories={args.num_trajectories}\n")

    print(f"Saved {args.num_trajectories} synthetic trajectories in '{args.output_folder}'")
    print(f"Used column: {args.column}")
    print(f"Original series saved to: {os.path.join(args.output_folder, f'{args.column}.npy')}")
    print(f"Time saved to: {os.path.join(args.output_folder, 'time_ns.npy')}")


if __name__ == "__main__":
    main()
