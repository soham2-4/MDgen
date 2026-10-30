"""
mopr_inference.py  -  Step 3 of the MopR-phenol pipeline (replaces inference2.py).

Loads a checkpoint written by mopr_trans.py, takes the first `input_len` steps of a
ligand-RMSD series, generates the rest autoregressively (same rollout as the training
script's evaluation), reports errors and the predicted vs. true binding time, and plots
the result in physical units (ns / nm).

Corrections relative to inference2.py:
  * the model is rebuilt from the configuration stored in the checkpoint;
  * the decoder is fed [START, y_0 ... y_{L-1}] exactly as in training (inference2.py fed
    [y_0 ... y_{L-1}] with no start token, i.e. shifted by one position relative to training);
  * the default target is a HELD-OUT source trajectory (split.json "test"), not a copy of a
    training curve;
  * optional MC-dropout ensemble (--n_samples > 1) to show the spread of predicted
    binding times - a single deterministic curve cannot represent a stochastic event.

Usage:
    python mopr_inference.py --ckpt checkpoints/transformer_epoch_500.pt --folder r428
    python mopr_inference.py --ckpt ... --xvg sahil_MopR_phenol_raw_data/r428.xvg
    python mopr_inference.py --ckpt ... --folder rmsd_716 --n_samples 20
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mopr_trans import TrajectoryTransformer, rollout
from mopr_data import load_xvg, block_average

BOUND_NM = 0.5


def first_crossing(t, x, thr=BOUND_NM):
    idx = np.where(x < thr)[0]
    return float(t[idx[0]]) if idx.size else np.nan


def load_series(args, ckpt):
    if args.xvg:
        t, x = load_xvg(args.xvg)
        tb, xb = block_average(t, x, args.downsample)
        name = os.path.splitext(os.path.basename(args.xvg))[0]
        return tb, xb.astype(np.float32), name
    folder = os.path.join(args.data_dir, args.folder)
    t = np.load(os.path.join(folder, "time.npy"))
    fn = "original.npy" if args.index is None else f"traj_{args.index:03d}.npy"
    x = np.load(os.path.join(folder, fn)).astype(np.float32)
    return t[: len(x)], x, f"{args.folder}/{fn}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="synthetic_trajectories")
    ap.add_argument("--folder", default=None, help="source folder; default = first held-out trajectory")
    ap.add_argument("--index", type=int, default=None, help="traj_XXX.npy index; default = original.npy")
    ap.add_argument("--xvg", default=None, help="instead of --folder: a raw .xvg file")
    ap.add_argument("--downsample", type=int, default=None, help="block size for --xvg (default from split.json)")
    ap.add_argument("--n_samples", type=int, default=1, help=">1 = MC-dropout ensemble")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    step_ns = ckpt["step_ns"]; L = cfg["input_len"]; start_value = ckpt.get("start_value", -1.0)
    model = TrajectoryTransformer(**cfg).to(device)
    state = {k.replace("module.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state)
    model.eval()

    if args.folder is None and args.xvg is None:
        if not ckpt["test"]:
            raise SystemExit("no held-out trajectory in the checkpoint; pass --folder or --xvg")
        args.folder = ckpt["test"][0]
    if args.downsample is None:
        with open(os.path.join(args.data_dir, "split.json")) as f:
            args.downsample = json.load(f)["downsample"]
    t, x, name = load_series(args, ckpt)
    x = x[:: ckpt["downsample_factor"]]; t = t[:: ckpt["downsample_factor"]][: len(x)]
    T_max = cfg["output_len"]                      # length the model was trained on (--max_len_ns)
    if len(x) > T_max:
        print(f"note: series has {len(x)} steps; the model was trained on <= {T_max} steps, "
              f"generating the first {T_max*step_ns:.0f} ns only")
        x, t = x[:T_max], t[:T_max]
    T = len(x)
    if L >= T:
        raise SystemExit(f"series has {T} steps but the model's input prefix is {L} steps")
    held_out = args.xvg is None and args.folder in ckpt["test"]
    print(f"series {name}: {T} steps x {step_ns:.2f} ns; prefix {L} steps = {L*step_ns:.0f} ns; "
          f"{'HELD-OUT' if held_out else 'TRAINING source (optimistic!)'}")

    prefix = torch.tensor(x[:L], dtype=torch.float32, device=device).view(1, L, 1)
    preds = []
    for s in range(args.n_samples):
        if args.n_samples > 1:
            model.train()                      # MC dropout: stochastic forward passes
            torch.manual_seed(s)
        gen = rollout(model, prefix, T, start_value)
        preds.append(gen.view(-1).cpu().numpy())
    model.eval()
    preds = np.stack(preds)                     # [S, T]
    mean = preds.mean(0)

    true_tb = first_crossing(t, x)
    pred_tbs = np.array([first_crossing(t, p) for p in preds])
    mae = np.abs(mean[L:] - x[L:]).mean(); rmse = np.sqrt(((mean[L:] - x[L:]) ** 2).mean())
    print(f"predicted region: MAE {mae:.3f} nm, RMSE {rmse:.3f} nm")
    print(f"binding time (first RMSD < {BOUND_NM} nm): true {true_tb:.0f} ns | predicted "
          + (f"{pred_tbs[0]:.0f} ns" if args.n_samples == 1 else
             f"median {np.nanmedian(pred_tbs):.0f} ns (range {np.nanmin(pred_tbs):.0f}-{np.nanmax(pred_tbs):.0f})"))

    os.makedirs(args.out, exist_ok=True)
    tag = name.replace("/", "_").replace(".npy", "")
    np.save(os.path.join(args.out, f"pred_{tag}.npy"), preds)

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)
    ax.plot(t, x, color="#2a78d6", lw=1.2, label="MD (original)")
    if args.n_samples > 1:
        ax.fill_between(t, preds.min(0), preds.max(0), color="#eb6834", alpha=0.18, lw=0,
                        label=f"predicted, range of {args.n_samples} samples")
    ax.plot(t, mean, color="#eb6834", lw=1.4, ls="--", label="predicted (mean)" if args.n_samples > 1 else "predicted")
    ax.axvline(t[L], color="#52514e", lw=0.9, ls=":", label=f"prediction start ({t[L]:.0f} ns)")
    ax.axhline(BOUND_NM, color="#c3c2b7", lw=0.8)
    ax.set_xlabel("Time (ns)"); ax.set_ylabel("Ligand RMSD to bound pose (nm)")
    ax.set_title(f"MopR-phenol | {name} | {'held-out' if held_out else 'training source'} | "
                 f"MAE {mae:.2f} nm | binding true {true_tb:.0f} ns")
    ax.legend(frameon=False, fontsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    png = os.path.join(args.out, f"pred_{tag}.png")
    fig.savefig(png)
    print(f"saved {png}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
