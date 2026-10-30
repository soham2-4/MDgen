"""
demo_controls.py - shows what memorization_check.py reports for
  (A) a COPYING generator: training trajectories, time-shifted and noised   -> must be flagged
  (B) a NON-COPYING surrogate generator: a stochastic model fitted to the training set
      (log-space AR(1) unbound phase, log-normal binding time, AR(1) bound phase)      -> must pass
on the MopR split from design_split.py.  Replace (A)/(B) by your transformer's samples.

Usage: python demo_controls.py --xvg_dir ... --split split_design/split.json --out demo
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memorization_check import load_xvg, run

ap = argparse.ArgumentParser()
ap.add_argument("--xvg_dir", required=True); ap.add_argument("--split", required=True)
ap.add_argument("--block", type=int, default=100); ap.add_argument("--n_gen", type=int, default=30)
ap.add_argument("--out", default="demo"); a = ap.parse_args()
rng = np.random.default_rng(0)
split = json.load(open(a.split))

def series(name):
    x = load_xvg(os.path.join(a.xvg_dir, name + ".xvg")); n = (len(x) // a.block) * a.block
    return x[:n].reshape(-1, a.block).mean(1)

train = [series(n) for n in split["train"]]; test = [series(n) for n in split["test"]]
names = {"train": split["train"], "test": split["test"]}

# ---- (A) copying generator: pick a training trajectory, drop a random 0-50 ns prefix, add 0.05 nm noise
gen_copy = []
for i in range(a.n_gen):
    src = train[i % len(train)]; shift = rng.integers(0, 51)
    gen_copy.append(np.maximum(src[shift:] + rng.normal(0, 0.05, len(src) - shift), 0))

# ---- (B) surrogate generator fitted to the training set (no copying of any curve)
def ar1_fit(seg):
    x = seg - seg.mean(); phi = np.corrcoef(x[:-1], x[1:])[0, 1]; return seg.mean(), seg.std(), np.clip(phi, 0, 0.99)
def ar1_sample(n, mu, sd, phi):
    e = rng.normal(0, sd * np.sqrt(1 - phi**2), n); x = np.empty(n); x[0] = rng.normal(0, sd)
    for i in range(1, n): x[i] = phi * x[i-1] + e[i]
    return mu + x
tb = []; unb, bnd, lens = [], [], []
for x in train:
    i0 = int(np.argmax(x < 0.5)); tb.append(i0); unb.append(x[:i0]); bnd.append(x[i0:]); lens.append(len(x))
mu_u, sd_u, phi_u = ar1_fit(np.log(np.concatenate(unb)))   # unbound phase modelled in log space (stays > ~1 nm)
mu_b, sd_b, phi_b = ar1_fit(np.concatenate(bnd))
log_tb = np.log(np.array(tb)); mu_t, sd_t = log_tb.mean(), log_tb.std()
gen_surr = []
for i in range(a.n_gen):
    L = int(rng.choice(lens)); t_b = int(np.clip(np.exp(rng.normal(mu_t, sd_t)), 50, L - 100))
    x = np.concatenate([np.exp(ar1_sample(t_b, mu_u, sd_u, phi_u)), ar1_sample(L - t_b, mu_b, sd_b, phi_b)])
    gen_surr.append(np.maximum(x, 0.02))
print(f"surrogate fit: log-unbound {mu_u:.2f}±{sd_u:.2f} (phi {phi_u:.2f}); bound {mu_b:.2f}±{sd_b:.2f} nm (phi {phi_b:.2f}); "
      f"binding time log-normal median {np.exp(mu_t):.0f} ns")

for tag, gen in [("A_copying_generator", gen_copy), ("B_surrogate_generator", gen_surr)]:
    print(f"\n===== {tag} =====")
    run(train, test, gen, names, step_ns=1.0, window=0.1, thr=0.5, out=os.path.join(a.out, tag))
