#!/usr/bin/env python3
"""
cktest_clean.py
Safe, robust script to load an MSM pickle and produce CK-test plots (svg/png/pdf).

Usage:
    python cktest_clean.py
    python cktest_clean.py --msm_dir /path/to/MSM_ANALYSIS --filename_init asyn_gen --n_clusters 100 --lag 10 --n_states 9
"""

import os
import sys
import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import gc

# Optional imports (fail loudly if missing)
try:
    import pyemma
except Exception as e:
    raise ImportError("pyemma is required to run this script. Install pyemma in your environment.") from e

# colorama is optional; fall back to empty strings if missing
try:
    from colorama import Fore, Style
except Exception:
    class _C:
        GREEN = MAGENTA = RESET_ALL = ""
    Fore = _C()
    Style = _C()

def build_cktest(msm_data, out_prefix):
    """
    Create CK-test plots from an MSM object and save svg/png/pdf with given prefix.
    out_prefix: string without extension, e.g. "asyn_gen_nclu_100_lag_10_state_size_9_cktest"
    """
    # create a fresh figure to avoid reusing previous figures
    plt.figure(figsize=(6.5, 4.5))

    # pyemma plotting returns matplotlib objects; units='ps' is acceptable if your lag/time is in ps
    try:
        pyemma.plots.plot_cktest(msm_data.cktest(msm_data.transition_matrix.shape[0]) if False else msm_data.cktest,
                                 units='ps')
        # NOTE: some versions of pyemma accept plot_cktest(msm_data.cktest(nstates)) or plot_cktest(ck_result)
        # using the convenience call below (which worked with many pyemma versions): pyemma.plots.plot_cktest(msm_data.cktest(n_states))
    except TypeError:
        # Fallback call (older/newer pyemma API differences)
        try:
            pyemma.plots.plot_cktest(msm_data.cktest)
        except Exception as e:
            # Last resort: raise with helpful message
            raise RuntimeError("pyemma.plot_cktest call failed. Check your pyemma version and API. Error: " + str(e))

    plt.tight_layout()

    out_svg = f"{out_prefix}.svg"
    out_png = f"{out_prefix}.png"
    out_pdf = f"{out_prefix}.pdf"

    plt.savefig(out_svg, dpi=300, format='svg', bbox_inches='tight')
    plt.savefig(out_png, dpi=300, format='png', bbox_inches='tight')
    plt.savefig(out_pdf, dpi=300, format='pdf', bbox_inches='tight')

    plt.close()  # release memory / figure handles

    print(f"\n{Fore.GREEN}CREATED CKTEST FILES:{Fore.MAGENTA} {out_svg}, {out_png}, {out_pdf} {Style.RESET_ALL}\n")


def main(args):
    # Resolve script directory and default MSM_ANALYSIS path relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_msm_dir = os.path.join(script_dir, "..", "MSM_ANALYSIS")
    msm_dir = args.msm_dir if args.msm_dir else default_msm_dir

    filename_init = args.filename_init
    n_clusters = args.n_clusters
    lag = args.lag
    n_states = args.n_states

    msm_subdir = os.path.join(msm_dir, f"{filename_init}_nclu_{n_clusters}_lag_{lag}_state_size_{n_states}")
    msm_file = os.path.join(msm_subdir, f"{filename_init}_nclu_{n_clusters}_lag_{lag}_msm.pkl")

    if not os.path.exists(msm_file):
        # Helpful error message with tried path
        raise FileNotFoundError(f"MSM file not found:\n  Tried: {msm_file}\n"
                                f"Please check --msm_dir or run script from correct working directory.")

    # Load MSM pickle
    with open(msm_file, "rb") as fh:
        msm_data = pkl.load(fh)

    # Build safe output prefix
    out_prefix = f"{filename_init}_nclu_{n_clusters}_lag_{lag}_state_size_{n_states}_cktest"

    # The pyemma API for cktest sometimes expects a cktest result object from msm_data.cktest(n_states)
    # We'll try to call msm_data.cktest(n_states) if msm_data has the method; otherwise attempt to pass msm_data directly.
    # For maximum compatibility, we'll prepare a `ck_result` variable and call pyemma.plots.plot_cktest on it.
    try:
        # prefer explicit call if available
        if hasattr(msm_data, "cktest"):
            try:
                ck_result = msm_data.cktest(n_states)
            except TypeError:
                # some pyemma versions may require no args
                ck_result = msm_data.cktest()
        else:
            raise AttributeError("MSM object missing 'cktest' method")
    except Exception as e:
        # fallback: try to let plot_cktest consume msm_data (some older API)
        ck_result = None
        print("Warning: could not compute ck_result via msm_data.cktest(...). Attempting direct plotting; if this fails, check pyemma version. Error:", e)

    # Plotting: if we have ck_result, pass it; otherwise plot with msm_data (some APIs accept this)
    if ck_result is not None:
        pyemma.plots.plot_cktest(ck_result, units='ps')
        plt.tight_layout()
        plt.savefig(out_prefix + ".svg", dpi=300, bbox_inches='tight')
        plt.savefig(out_prefix + ".png", dpi=300, bbox_inches='tight')
        plt.savefig(out_prefix + ".pdf", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n{Fore.GREEN}CREATED CKTEST FILES:{Fore.MAGENTA} {out_prefix}.svg, {out_prefix}.png, {out_prefix}.pdf {Style.RESET_ALL}\n")
    else:
        # Use helper to handle API differences and safe plotting
        build_cktest(msm_data=msm_data, out_prefix=out_prefix)

    gc.collect()
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CK-test plotting for an MSM pickle (robust path handling).")
    parser.add_argument("--msm_dir", default=None, help="Path to MSM_ANALYSIS directory (default: ../MSM_ANALYSIS relative to this script).")
    parser.add_argument("--filename_init", default="asyn_gen", help="base filename (default: asyn_gen)")
    parser.add_argument("--n_clusters", type=int, default=500, help="microstate count (default: 100)")
    parser.add_argument("--lag", type=int, default=50, help="lag used to build the MSM (default: 10)")
    parser.add_argument("--n_states", type=int, default=3, help="number of macrostates for CK-test (default: 9)")
    args = parser.parse_args()

    sys.exit(main(args) or 0)

