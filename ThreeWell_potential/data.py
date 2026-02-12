# data.py
import os
import argparse
import numpy as np

def load_trpcage(path):
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected 2D array with ≥2 columns, got shape {arr.shape}")
    if arr.shape[1] >= 3:
        # assume [time, z1, z2, ...] and take the first three
        time = arr[:, 0].astype(float)
        z = arr[:, 1:3].astype(float)
    else:
        # only [z1, z2]; create a synthetic time index
        z = arr[:, :2].astype(float)
        time = np.arange(z.shape[0], dtype=float)
    return time, z  # z shape [T,2]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--infile", type=str, default="threewell.npy", help="Path to trpcage.npy")
    p.add_argument("--outdir", type=str, default="synthetic_trpcage", help="Output folder")
    p.add_argument("--num", type=int, default=250, help="Number of synthetic trajectories")
    p.add_argument("--downsample", type=int, default=1, help="Downsample factor (e.g., 5 keeps every 5th sample)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise_min", type=float, default=0.001, help="Min noise scale (× per-channel std)")
    p.add_argument("--noise_max", type=float, default=0.01, help="Max noise scale (× per-channel std)")
    args = p.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    time, z = load_trpcage(args.infile)
    if args.downsample > 1:
        time = time[::args.downsample]
        z = z[::args.downsample]

    # Save shared timebase
    np.save(os.path.join(args.outdir, "time.npy"), time)

    # Per-channel std for noise scaling
    ch_std = np.std(z, axis=0) + 1e-8

    for i in range(args.num):
        scale = np.random.uniform(args.noise_min, args.noise_max, size=(1, 2))
        noise = np.random.randn(*z.shape) * (scale * ch_std)
        syn = z + noise
        np.save(os.path.join(args.outdir, f"traj_{i:03d}.npy"), syn.astype(np.float32))

    print(f"Saved {args.num} synthetic trajectories to '{args.outdir}' and time.npy")

if __name__ == "__main__":
    main()

