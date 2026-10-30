"""
export_generated.py - Step 4 of the MopR pipeline: generate samples from a trained checkpoint
for EVERY source trajectory (training and held-out) and write them one-file-per-trajectory,
together with the clean source series, in the layout memorization_check.py expects:

    <out>/train/<name>.npy            clean 1-ns source series, training trajectories
    <out>/test/<name>.npy             clean 1-ns source series, held-out trajectories
    <out>/gen_from_train/<name>_sNN.npy   samples conditioned on a TRAINING prefix
    <out>/gen_from_test/<name>_sNN.npy    samples conditioned on a HELD-OUT prefix

All series (clean and generated) are capped at the length the model was trained on
(--max_len_ns in mopr_trans.py), so every comparison covers the same time range.
Each generated file contains the copied prefix (first L steps) followed by the generated
continuation, so run memorization_check.py with --skip L (printed at the end).

n_samples > 1 uses MC dropout (stochastic forward passes); n_samples = 1 is deterministic.

Usage:
    python export_generated.py --ckpt checkpoints/transformer_epoch_500.pt \
        --data_dir synthetic_trajectories --n_samples 10 --out memcheck_data
"""
import argparse
import os

import numpy as np
import torch

from mopr_trans import TrajectoryTransformer, rollout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="synthetic_trajectories")
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--out", default="memcheck_data")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]; L = cfg["input_len"]; ds = ckpt["downsample_factor"]
    model = TrajectoryTransformer(**cfg).to(device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt["model_state"].items()})

    for role in ["train", "test"]:
        os.makedirs(os.path.join(a.out, role), exist_ok=True)
        os.makedirs(os.path.join(a.out, f"gen_from_{role}"), exist_ok=True)
        for name in ckpt[role]:
            x = np.load(os.path.join(a.data_dir, name, "original.npy")).astype(np.float32)[::ds]
            x = x[: cfg["output_len"]]            # cap at the trained length so all series share one time range
            np.save(os.path.join(a.out, role, f"{name}.npy"), x)
            T = len(x)
            prefix = torch.tensor(x[:L], device=device).view(1, L, 1)
            for s in range(a.n_samples):
                if a.n_samples > 1:
                    model.train(); torch.manual_seed(1000 * s + 7)
                else:
                    model.eval()
                gen = rollout(model, prefix, T, ckpt.get("start_value", -1.0)).view(-1).cpu().numpy()
                np.save(os.path.join(a.out, f"gen_from_{role}", f"{name}_s{s:02d}.npy"), gen.astype(np.float32))
            print(f"{role:5s} {name:12s} {T} steps -> {a.n_samples} samples")
    model.eval()
    print(f"\ndone. Prefix length L = {L} steps ({L*ckpt['step_ns']:.0f} ns): run memorization_check.py with --skip {L}")


if __name__ == "__main__":
    main()
