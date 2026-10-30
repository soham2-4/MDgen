"""
design_split.py - approach 2: build a train/test split of the MopR binding trajectories
whose two halves have matched diversity, after removing outliers.

Features per trajectory (1-ns block-averaged ligand RMSD):
  site            binding site = reference used (ref1 / ref2) read from the .xvg header
  length_ns       trajectory length
  t_bind_ns       final binding time (bound < 0.5 nm, released only if > 1.0 nm; start of the last bound run)
  near_frac       fraction of the pre-binding time spent near the site (1-ns RMSD < 3 nm)
  bound_mean      mean RMSD in the bound state
  unbound_mean    mean RMSD before binding

Outliers: robust z-score (median / MAD) on t_bind (linear) and log length; |z| > 3.5 is flagged;
  --exclude removes trajectories by hand.

Split: every C(n, k) choice of k test trajectories is scored by the multivariate energy
distance between standardized train and test feature vectors (smaller = more similar
distributions); splits that do not contain both sites are rejected.  The best split is
written to split.json (compatible with mopr_data.py --test ...).

Usage:
  python design_split.py --xvg_dir sahil_MopR_phenol_raw_data --k_test 3 --out split_design
"""
import argparse
import glob
import itertools
import json
import os
import re

import numpy as np


def load_xvg(fn):
    t, v, ref = [], [], "?"
    with open(fn) as f:
        for line in f:
            if line.startswith("#") and " -s " in line:
                ref = re.search(r"-s\s+(\S+)", line).group(1).split("/")[-1].replace(".pdb", "")
            if line[0] in "#@":
                continue
            p = line.split()
            if len(p) >= 2:
                t.append(float(p[0])); v.append(float(p[1]))
    return np.asarray(t), np.asarray(v), ref


def block(x, k):
    n = (len(x) // k) * k
    return x[:n].reshape(-1, k).mean(1)


def hyst(r, lo=0.5, hi=1.0):
    s = np.zeros(r.size, bool); b = False
    for i, x in enumerate(r):
        if not b and x < lo: b = True
        elif b and x > hi: b = False
        s[i] = b
    return s


def feats(x_raw, step_raw_ns, block_k):
    x = block(x_raw, block_k); step = step_raw_ns * block_k
    s = hyst(x_raw)                                   # on raw 10-ps data, as in the trajectory check
    d = np.diff(s.astype(int)); starts = np.where(d == 1)[0] + 1
    if s[0]: starts = np.r_[0, starts]
    i0 = int(starts[-1]); t_bind = i0 * step_raw_ns
    pre = x[: max(1, int(t_bind / step))]
    return dict(length_ns=len(x_raw) * step_raw_ns, t_bind_ns=t_bind,
                near_frac=float((pre < 3.0).mean()), bound_mean=float(x_raw[i0:].mean()),
                unbound_mean=float(pre.mean()))


def energy_distance(X, Y):
    def md(A, B): return np.mean(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1))
    return 2 * md(X, Y) - md(X, X) - md(Y, Y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xvg_dir", required=True)
    ap.add_argument("--k_test", type=int, default=3)
    ap.add_argument("--block", type=int, default=100, help="frames per 1-ns block")
    ap.add_argument("--z_outlier", type=float, default=3.5, help="robust z (median/MAD) threshold; binding time linear, length log")
    ap.add_argument("--exclude", nargs="*", default=[], help="names to remove manually")
    ap.add_argument("--out", default="split_design")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.xvg_dir, "*.xvg")))
    if not files:
        raise SystemExit(f"ERROR: no .xvg files found in '{a.xvg_dir}' (looked for {os.path.abspath(a.xvg_dir)}/*.xvg). "
                         f"Copy the folder with the 13 .xvg files here, or pass its path with --xvg_dir.")
    print(f"found {len(files)} .xvg files in {a.xvg_dir}")
    rows = []
    for fn in files:
        t, v, ref = load_xvg(fn)
        step_raw = float(np.median(np.diff(t)))
        f = feats(v, step_raw, a.block); f["site"] = 1 if ref.endswith("1") else 2
        f["name"] = os.path.basename(fn)[:-4]; rows.append(f)
    names = [r["name"] for r in rows]

    # outliers
    def rz(v, log=False):
        v = np.asarray(v, float); v = np.log(v) if log else v
        m = np.median(v); mad = np.median(np.abs(v - m)) * 1.4826
        return (v - m) / mad if mad > 0 else np.zeros_like(v)
    z_tb = rz([r["t_bind_ns"] for r in rows]); z_len = rz([r["length_ns"] for r in rows], log=True)
    out_mask = (np.abs(z_tb) > a.z_outlier) | (np.abs(z_len) > a.z_outlier)
    for i, r in enumerate(rows):
        if r["name"] in a.exclude: out_mask[i] = True
    print(f"{'name':12s} {'site':>4s} {'len ns':>7s} {'t_bind':>7s} {'near%':>6s} {'bound':>6s} {'unbnd':>6s} {'z(tb)':>6s} {'z(len)':>6s}")
    for r, zt, zl, o in zip(rows, z_tb, z_len, out_mask):
        print(f"{r['name']:12s} {r['site']:4d} {r['length_ns']:7.0f} {r['t_bind_ns']:7.0f} {100*r['near_frac']:6.0f} "
              f"{r['bound_mean']:6.2f} {r['unbound_mean']:6.2f} {zt:6.1f} {zl:6.1f} {'  <-- OUTLIER' if o else ''}")
    keep = [r for r, o in zip(rows, out_mask) if not o]
    outliers = [r["name"] for r, o in zip(rows, out_mask) if o]

    # feature matrix (standardized); site enters as a balance constraint, not as a distance
    keys = ["t_bind_ns", "length_ns", "near_frac", "bound_mean", "unbound_mean"]
    X = np.array([[np.log(r["t_bind_ns"]), np.log(r["length_ns"]), r["near_frac"], r["bound_mean"], r["unbound_mean"]] for r in keep])
    X = (X - X.mean(0)) / X.std(0)
    sites = np.array([r["site"] for r in keep]); n = len(keep)
    scored = []
    for test_idx in itertools.combinations(range(n), a.k_test):
        ti = np.array(test_idx); tr = np.setdiff1d(np.arange(n), ti)
        if len(set(sites[ti])) < 2:
            continue
        scored.append((energy_distance(X[tr], X[ti]), ti))
    scored.sort(key=lambda z: z[0])
    print(f"\n{len(scored)} admissible splits (both sites in test). Best {min(5, len(scored))}:")
    for sc, ti in scored[:5]:
        print(f"  energy distance {sc:.3f}: test = {[keep[i]['name'] for i in ti]}")
    best_sc, best = scored[0]
    test_names = [keep[i]["name"] for i in best]
    train_names = [r["name"] for i, r in enumerate(keep) if i not in best]

    # per-feature comparison of the chosen split
    print("\nChosen split - feature medians (train | test):")
    for k in keys + ["site"]:
        tv = [r[k] for r in keep if r["name"] in train_names]; sv = [r[k] for r in keep if r["name"] in test_names]
        print(f"  {k:13s} {np.median(tv):8.3g} | {np.median(sv):8.3g}   (train range {min(tv):.3g}-{max(tv):.3g}, test {min(sv):.3g}-{max(sv):.3g})")

    split = dict(train=train_names, test=test_names, outliers_removed=outliers,
                 energy_distance=float(best_sc), features=keys)
    with open(os.path.join(a.out, "split.json"), "w") as f:
        json.dump(split, f, indent=2)
    with open(os.path.join(a.out, "features.csv"), "w") as f:
        f.write("name,site,length_ns,t_bind_ns,near_frac,bound_mean,unbound_mean,outlier,role\n")
        for r, o in zip(rows, out_mask):
            role = "outlier" if o else ("test" if r["name"] in test_names else "train")
            f.write(f"{r['name']},{r['site']},{r['length_ns']:.0f},{r['t_bind_ns']:.0f},{r['near_frac']:.3f},"
                    f"{r['bound_mean']:.3f},{r['unbound_mean']:.3f},{int(o)},{role}\n")
    print(f"\nwritten {a.out}/split.json and features.csv")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#9a9992", "#0b0b0b", "#52514e"
    labels = {"t_bind_ns": "binding time (ns)", "length_ns": "length (ns)", "near_frac": "near-site fraction before binding",
              "bound_mean": "bound-state RMSD (nm)", "unbound_mean": "unbound RMSD (nm)"}
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4), dpi=150, facecolor="#fcfcfb")
    rng = np.random.default_rng(1)
    for ax, k in zip(axes, keys):
        for r in rows:
            role = "outlier" if r["name"] in outliers else ("test" if r["name"] in test_names else "train")
            c = {"train": BLUE, "test": ORANGE, "outlier": GRAY}[role]; xpos = {"train": 0, "test": 1, "outlier": 2}[role]
            mk = "o" if r["site"] == 1 else "s"
            ax.scatter(xpos + (rng.random() - 0.5) * 0.3, r[k], color=c, marker=mk, s=28, alpha=0.85, lw=0)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["train", "test", "outlier"], fontsize=8)
        ax.set_title(labels[k], fontsize=9, loc="left", color=INK)
        if k in ("t_bind_ns", "length_ns"): ax.set_yscale("log")
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    axes[0].text(0.02, 0.02, "circle = site 1, square = site 2", transform=axes[0].transAxes, fontsize=7.5, color=INK2)
    fig.suptitle(f"MopR split design: test = {', '.join(test_names)}; removed outlier(s): {', '.join(outliers) or 'none'}",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(a.out, "split_design.png"), facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
