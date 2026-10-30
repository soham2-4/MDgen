"""
memorization_check.py - data-copying / memorization test for generated MD time series.

Idea (Meehan, Chaudhuri & Dasgupta, AISTATS 2020; Alaa et al., ICML 2022): a generator
memorizes when its samples sit closer to the training set than genuinely new data would.
So the distance "generated -> nearest training trajectory" is only meaningful relative
to the distance "held-out MD trajectory -> nearest training trajectory".  Both are computed
here with two metrics:

  * pointwise   : mean |y_gen(t) - y_train(t)| at the same time t  (the PI's approach 1)
  * DTW         : dynamic time warping with a Sakoe-Chiba window - the minimum cumulative
                  distance after allowing local stretching/compression of the time axis, so a
                  time-shifted or slightly re-timed copy is still recognised as a copy.

Three groups of nearest-neighbour (NN) distances are reported:
  gen  -> train     (what we are testing)
  test -> train     (reference: how far a NEW trajectory is from the training set)
  train -> train    (leave-one-out: spread inside the training set)

Verdict logic: if median(gen->train) is far below median(test->train) and many generated
samples fall below the smallest test->train distance, the generator is copying.  If the
gen->train distribution looks like test->train, it is not.  A one-sided Mann-Whitney U test
(gen < test) gives a p-value; the NN-identity histogram shows whether all generated samples
collapse onto one or two training trajectories.

Inputs: one trajectory per file (.npy 1-D array or GROMACS .xvg).  Use the UNIQUE source
trajectories as --train, not noise-augmented copies.

Usage:
  python memorization_check.py --train "memcheck_data/train/*.npy" --test "memcheck_data/test/*.npy" \
         --gen "memcheck_data/gen_from_test/*.npy" --skip 200 --step_ns 1 --window 0.1 --thr 0.5 --out memcheck
  (--skip = prefix length in steps, so the copied 200-ns prefix is not compared)
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats
from dtaidistance import dtw


# ------------------------------------------------------------------ io
def load_xvg(fn):
    v = []
    with open(fn) as f:
        for line in f:
            if line[0] in "#@":
                continue
            p = line.split()
            if len(p) >= 2:
                v.append(float(p[1]))
    return np.asarray(v)


def load_set(pattern, downsample=1):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files match {pattern}")
    out = []
    for fn in files:
        x = load_xvg(fn) if fn.endswith(".xvg") else np.load(fn)
        x = np.asarray(x, dtype=float).ravel()
        if downsample > 1:
            n = (len(x) // downsample) * downsample
            x = x[:n].reshape(-1, downsample).mean(1)
        out.append(x)
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return names, out


# ------------------------------------------------------------------ distances
def pointwise(a, b):
    n = min(len(a), len(b))
    return float(np.mean(np.abs(a[:n] - b[:n])))


def dtw_dist(a, b, window_frac):
    n = max(len(a), len(b))
    w = max(1, int(round(window_frac * n)))
    d = dtw.distance_fast(np.ascontiguousarray(a, dtype=np.double),
                          np.ascontiguousarray(b, dtype=np.double), window=w)
    return float(d / np.sqrt(n))          # ~ RMS difference along the warping path (nm)


def nn_table(A, B, metric, exclude_self=False):
    """for each a in A: (distance to nearest b in B, index of that b)"""
    d_nn, i_nn = [], []
    for i, a in enumerate(A):
        ds = np.array([np.inf if (exclude_self and i == j) else metric(a, b) for j, b in enumerate(B)])
        j = int(np.argmin(ds)); d_nn.append(ds[j]); i_nn.append(j)
    return np.array(d_nn), np.array(i_nn)


# ------------------------------------------------------------------ features
def features(x, t_step, thr):
    idx = np.where(x < thr)[0]
    t_bind = float(idx[0] * t_step) if idx.size else np.nan
    bound = x[idx[0]:] if idx.size else np.array([])
    return dict(t_bind_ns=t_bind,
                bound_mean=float(bound.mean()) if bound.size else np.nan,
                frac_below_thr=float((x < thr).mean()),
                unbound_mean=float(x[: idx[0]].mean()) if idx.size and idx[0] > 0 else float(x.mean()))


# ------------------------------------------------------------------ main
def run(train, test, gen, names, step_ns, window, thr, out, make_plot=True):
    os.makedirs(out, exist_ok=True)
    res = {}
    for mname, metric in [("pointwise", pointwise), ("dtw", lambda a, b: dtw_dist(a, b, window))]:
        g2t, g2t_i = nn_table(gen, train, metric)
        s2t, s2t_i = nn_table(test, train, metric)
        t2t, _ = nn_table(train, train, metric, exclude_self=True)
        u = stats.mannwhitneyu(g2t, s2t, alternative="less")
        copy_frac = float(np.mean(g2t < s2t.min()))
        counts = np.bincount(g2t_i, minlength=len(train))
        res[mname] = dict(
            gen_to_train_median=float(np.median(g2t)), test_to_train_median=float(np.median(s2t)),
            train_to_train_median=float(np.median(t2t)),
            ratio_gen_over_test=float(np.median(g2t) / np.median(s2t)),
            frac_gen_below_min_test=copy_frac, mannwhitney_p_gen_less_than_test=float(u.pvalue),
            gen_nn_identity_counts={names["train"][j]: int(c) for j, c in enumerate(counts) if c > 0},
            gen_to_train=g2t.tolist(), test_to_train=s2t.tolist(), train_to_train=t2t.tolist())
        print(f"[{mname:9s}] NN distance median (nm): gen->train {np.median(g2t):.3f} | "
              f"test->train {np.median(s2t):.3f} | train->train {np.median(t2t):.3f} | "
              f"ratio gen/test {np.median(g2t)/np.median(s2t):.2f} | "
              f"gen below min test {100*copy_frac:.0f}% | MWU p(gen<test) {u.pvalue:.2g}")
        print(f"            generated samples' nearest training trajectory: "
              + ", ".join(f"{names['train'][j]}:{c}" for j, c in enumerate(counts) if c > 0))

    # feature-level comparison (binding time etc.)
    F = {k: [features(x, step_ns, thr) for x in v] for k, v in [("train", train), ("test", test), ("gen", gen)]}
    feat_summary = {}
    for key in ["t_bind_ns", "bound_mean", "frac_below_thr", "unbound_mean"]:
        vals = {k: np.array([f[key] for f in F[k]], dtype=float) for k in F}
        vals = {k: v[~np.isnan(v)] for k, v in vals.items()}
        ks = stats.ks_2samp(vals["gen"], vals["test"]).pvalue if len(vals["gen"]) > 1 and len(vals["test"]) > 1 else np.nan
        feat_summary[key] = {k: (float(np.median(v)) if v.size else np.nan) for k, v in vals.items()}
        feat_summary[key]["ks_p_gen_vs_test"] = float(ks)
        print(f"[feature {key:14s}] median train {feat_summary[key]['train']:.3g} | test {feat_summary[key]['test']:.3g} "
              f"| gen {feat_summary[key]['gen']:.3g} | KS p(gen vs test) {ks:.2g}")
    res["features"] = feat_summary
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(res, f, indent=2)

    if make_plot:
        plot(res, names, out)
    return res


def plot(res, names, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#9a9992", "#0b0b0b", "#52514e"
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=150, facecolor="#fcfcfb")
    for ax, m, title in zip(axes[:2], ["pointwise", "dtw"],
                            ["Pointwise |Δy| at same t (nm)", "DTW distance / √n (nm)"]):
        r = res[m]
        groups = [("train→train (LOO)", r["train_to_train"], GRAY), ("test→train", r["test_to_train"], BLUE),
                  ("generated→train", r["gen_to_train"], ORANGE)]
        for k, (lab, v, c) in enumerate(groups):
            v = np.asarray(v); jit = (np.random.default_rng(0).random(v.size) - 0.5) * 0.35
            ax.scatter(np.full(v.size, k) + jit, v, s=14, color=c, alpha=0.75, lw=0)
            ax.hlines(np.median(v), k - 0.3, k + 0.3, color=INK, lw=1.5)
        ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups], fontsize=8)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.set_ylabel("nearest-neighbour distance", fontsize=8.5, color=INK2); ax.set_ylim(bottom=0)
        ax.text(0.98, 0.95, f"median ratio gen/test = {r['ratio_gen_over_test']:.2f}\n"
                f"gen below min test: {100*r['frac_gen_below_min_test']:.0f}%\nMWU p = {r['mannwhitney_p_gen_less_than_test']:.1g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax = axes[2]
    counts = res["dtw"]["gen_nn_identity_counts"]
    labs = names["train"]; vals = [counts.get(l, 0) for l in labs]
    ax.barh(range(len(labs)), vals, color=ORANGE, height=0.6)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8); ax.invert_yaxis()
    ax.set_title("Which training trajectory is nearest (DTW)", fontsize=10, loc="left", color=INK)
    ax.set_xlabel("# generated samples", fontsize=8.5, color=INK2)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "memorization_check.png"), facecolor=fig.get_facecolor())
    print(f"saved {os.path.join(out, 'memorization_check.png')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help='glob, e.g. "data/train/*.npy" (unique source trajectories)')
    ap.add_argument("--test", required=True, help="glob of held-out MD trajectories")
    ap.add_argument("--gen", required=True, help="glob of generated trajectories")
    ap.add_argument("--step_ns", type=float, default=1.0, help="time per point after --downsample")
    ap.add_argument("--downsample", type=int, default=1, help="block-average factor applied to all inputs")
    ap.add_argument("--window", type=float, default=0.1, help="Sakoe-Chiba window as a fraction of length")
    ap.add_argument("--thr", type=float, default=0.5, help="threshold for the 'bound' feature (nm)")
    ap.add_argument("--skip", type=int, default=0,
                    help="drop this many leading points from EVERY series (use the prefix length L when the "
                         "generated series start with a copied prefix, so only generated parts are compared)")
    ap.add_argument("--out", default="memcheck")
    a = ap.parse_args()
    names, sets = {}, {}
    for k in ["train", "test", "gen"]:
        names[k], sets[k] = load_set(getattr(a, k), a.downsample)
        if a.skip > 0:
            sets[k] = [x[a.skip:] for x in sets[k]]
        print(f"{k}: {len(sets[k])} trajectories, length {min(map(len, sets[k]))}-{max(map(len, sets[k]))} points")
    run(sets["train"], sets["test"], sets["gen"], names, a.step_ns, a.window, a.thr, a.out)
