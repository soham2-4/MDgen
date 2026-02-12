#!/usr/bin/env python3
"""
build_msm_clean.py

Cleaned and robust version of build_msm.py
- Uses safe pickle I/O
- Resolves ITS path relative to script location
- Passes save_dir explicitly to helper functions
- Optional CLI args to run subsets/resume
- Memory-aware concatenation (uses int32 dtype)
- Try/except around PCCA
"""

import os
import sys
import gc
import argparse
import shutil
import pickle as pkl
import numpy as np
import logging
from typing import List

# Optional: import these only if available (original script used pyemma)
try:
    import pyemma
    import pyemma.msm as msm
except Exception as e:
    msm = None
    pyemma = None

# colorama optional
try:
    from colorama import Fore, Style
except Exception:
    class _C:
        GREEN = ""
        MAGENTA = ""
        RESET_ALL = ""
    Fore = _C()
    Style = _C()

# Basic logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("build_msm")

# -------------------------
# Helper functions
# -------------------------
def safe_pkl_load(path):
    with open(path, "rb") as fh:
        return pkl.load(fh)

def safe_pkl_dump(obj, path):
    with open(path, "wb") as fh:
        pkl.dump(obj, fh)

def build_markov_model(dtrajs, lag, n_clusters, filename_init, save_dir):
    """Estimate MSM and save to save_dir."""
    if msm is None:
        raise RuntimeError("pyemma (msm) is not available in the current environment.")
    outpath = os.path.join(save_dir, f"{filename_init}_gen_nclu_{n_clusters}_lag_{lag}_msm.pkl")
    if os.path.exists(outpath):
        log.info("MSM file already exists, skipping: %s", outpath)
        return outpath
    log.info("Estimating MSM: lag=%s", lag)
    mm = pyemma.msm.estimate_markov_model(dtrajs=dtrajs, lag=lag)
    safe_pkl_dump(mm, outpath)
    log.info(f"{Fore.GREEN}CREATED MSM: {Fore.MAGENTA}{outpath}{Style.RESET_ALL}")
    return outpath

def build_state_size(msm_data, n_clusters, lag, n_states, filename_init, save_dir):
    """Compute and save state sizes (pi sums for metastable sets)."""
    state_size = []
    for i, s in enumerate(msm_data.metastable_sets):
        size = float(msm_data.pi[s].sum())
        state_size.append(size)
        log.info("π_%d = %f", i + 1, size)
    out = os.path.join(save_dir, f"{filename_init}_gen_nclu_{n_clusters}_lag_{lag}_state_size_{n_states}.npy")
    np.save(out, np.array(state_size))
    log.info(f"{Fore.GREEN}CREATED FILE FOR STATE SIZE : {Fore.MAGENTA}{out}{Style.RESET_ALL}")

def build_index_file_each_macrostate(metastable_traj, dtrajs_concatenated, n_states, n_clusters, lag, filename_init, save_dir):
    """Write index files (ndx) for each macrostate."""
    for i in range(n_states):
        frame_number = []
        # Note: metastable_traj is an array of macrostate assignment per frame
        for j in range(len(dtrajs_concatenated)):
            if metastable_traj[j] == i:
                frame_number.append(j + 1)
        frame = ["`[frames]"] + [str(x) for x in frame_number] + ["`"]
        out = os.path.join(save_dir, f"{filename_init}_gen_nclu_{n_clusters}_lag_{lag}_cluster_{i+1}.ndx")
        np.savetxt(out, frame, fmt="%s")
        log.info(f"{Fore.GREEN}CREATED INDEX FILE FOR STATE %d : {Fore.MAGENTA}%s {Style.RESET_ALL}", i+1, out)

# -------------------------
# Main flow
# -------------------------
def main(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    its_dir = args.its_dir if args.its_dir else os.path.join(script_dir, "ITS")

    filename_init = args.filename_init
    n_clusters = args.n_clusters

    # Build file paths for cluster + dtrajs (match your actual file name pattern)
    cluster_file = os.path.join(its_dir, f"{filename_init}_gen_nclu_{n_clusters}_kmeans.pkl")
    dtrajs_file = os.path.join(its_dir, f"{filename_init}_gen_nclu_{n_clusters}_dtrajs.pkl")

    # Validate existence
    if not os.path.exists(cluster_file):
        log.error("Cluster file not found: %s", cluster_file)
        sys.exit(2)
    if not os.path.exists(dtrajs_file):
        log.error("Dtrajs file not found: %s", dtrajs_file)
        sys.exit(2)

    # Load pickles safely
    log.info("Loading cluster file: %s", cluster_file)
    cluster = safe_pkl_load(cluster_file)

    log.info("Loading dtrajs file: %s", dtrajs_file)
    dtrajs = safe_pkl_load(dtrajs_file)
    if not isinstance(dtrajs, (list, tuple, np.ndarray)):
        log.error("dtrajs loaded is not a list/array. Found: %s", type(dtrajs))
        sys.exit(3)

    # Estimate total length (simple check)
    total_frames = sum(len(t) for t in dtrajs)
    log.info("Estimated total frames (sum of dtrajs lengths): %d", total_frames)

    # Convert to numpy concatenation with int32 dtype to save memory (if appropriate)
    try:
        dtrajs_concatenated = np.concatenate(dtrajs).astype(np.int32)
    except Exception as e:
        log.warning("Concatenation failed or too large: %s. Will not concatenate now.", str(e))
        dtrajs_concatenated = None

    # Iterate requested lags and state counts
    for lag in args.lags:
        for n_states in args.states:
            save_dir = os.path.join(script_dir, f"{filename_init}_gen_nclu_{n_clusters}_lag_{lag}_state_size_{n_states}")
            os.makedirs(save_dir, exist_ok=True)

            # Build & save MSM
            try:
                msm_path = build_markov_model(dtrajs=dtrajs, lag=lag, n_clusters=n_clusters, filename_init=filename_init, save_dir=save_dir)
            except Exception as e:
                log.exception("Failed to build MSM for lag=%s: %s", lag, e)
                continue

            # Load the MSM object
            try:
                log.info("Loading MSM from: %s", msm_path)
                with open(msm_path, "rb") as fh:
                    msm_data = pkl.load(fh)
            except Exception as e:
                log.exception("Failed to load MSM file %s: %s", msm_path, e)
                continue

            # Basic checks / prints
            try:
                log.info("Fraction of states used (should be close to 1): %f", msm_data.active_state_fraction)
                log.info("Fraction of counts used (should be close to 1): %f", msm_data.active_count_fraction)
            except Exception:
                log.warning("MSM object missing expected attributes 'active_state_fraction' / 'active_count_fraction'")

            log.info("Shape of dtrajs (python-level): %s", np.shape(dtrajs))

            # Compute PCCA / metastable sets
            try:
                msm_data.pcca(n_states)
            except Exception as e:
                log.exception("PCCA failed for n_states=%s at lag=%s: %s", n_states, lag, e)
                # continue to next configuration
                continue

            # If we didn't manage to concatenate earlier, do it now (attempt once more)
            if dtrajs_concatenated is None:
                try:
                    dtrajs_concatenated = np.concatenate(dtrajs).astype(np.int32)
                except Exception as e:
                    log.exception("Concatenation failed again: %s", e)
                    continue

            # Metastable assignments (map active-set indices onto concatenated frames)
            try:
                metastable_traj = msm_data.metastable_assignments[dtrajs_concatenated]
            except Exception as e:
                log.exception("Failed to build metastable_traj mapping: %s", e)
                continue

            # Save metastable trajectory
            mt_out = os.path.join(save_dir, f"metastable_traj_{filename_init}_gen_nclu_{n_clusters}_lag_{lag}_state_size_{n_states}.npy")
            np.save(mt_out, metastable_traj)
            log.info("%sCREATED FILE FOR METASTABLE TRAJECTORY : %s%s", Fore.GREEN, mt_out, Style.RESET_ALL)

            # Highest membership and coarse centers
            try:
                highest_membership = msm_data.metastable_distributions.argmax(1)
                coarse_state_centers = cluster.clustercenters[msm_data.active_set[highest_membership]]
                coarse_out = os.path.join(save_dir, f"coarse_centers_{filename_init}_gen_nclu_{n_clusters}_lag_{lag}_state_size_{n_states}.npy")
                np.save(coarse_out, coarse_state_centers)
                log.info("%sCREATED FILE FOR COARSE GRAIN CENTERS : %s%s", Fore.GREEN, coarse_out, Style.RESET_ALL)
            except Exception as e:
                log.warning("Failed to compute / save coarse centers: %s", e)

            # State sizes
            try:
                build_state_size(msm_data=msm_data, n_clusters=n_clusters, lag=lag, n_states=n_states, filename_init=filename_init, save_dir=save_dir)
            except Exception as e:
                log.warning("Failed to compute state sizes: %s", e)

            # Make index files per macrostate
            try:
                build_index_file_each_macrostate(metastable_traj=metastable_traj, dtrajs_concatenated=dtrajs_concatenated,
                                                 n_states=n_states, n_clusters=n_clusters, lag=lag, filename_init=filename_init, save_dir=save_dir)
            except Exception as e:
                log.warning("Failed to create index files: %s", e)

            # Microstates per macrostate
            try:
                micro_in_macro = np.array(msm_data.metastable_sets, dtype=object)
                micro_out = os.path.join(save_dir, f"microstate_in_macrostate_{filename_init}_gen_nclu_{n_clusters}_lag_{lag}_state_size_{n_states}.npy")
                np.save(micro_out, micro_in_macro)
                log.info("%sCREATED FILE FOR MICROSTATES IN EACH MACROSTATE : %s%s", Fore.GREEN, micro_out, Style.RESET_ALL)
            except Exception as e:
                log.warning("Failed to save microstate-in-macro info: %s", e)

            # Frame-in-each-micro dictionary
            try:
                dict_frame_in_micro = {key: [] for key in range(n_clusters)}
                for idx, key in enumerate(dtrajs_concatenated):
                    dict_frame_in_micro[int(key)].append(int(idx))
                dict_out = os.path.join(save_dir, f"frame_in_each_micro_{filename_init}_gen_nclu_{n_clusters}_lag_{lag}_state_size_{n_states}.pkl")
                safe_pkl_dump(dict_frame_in_micro, dict_out)
                log.info("%sCREATED FILE FOR INDEX OF FRAMES IN EACH MICROSTATE : %s%s", Fore.GREEN, dict_out, Style.RESET_ALL)
            except Exception as e:
                log.warning("Failed to create frame_in_each_micro dict: %s", e)

            # Count frames in each macrostate (simple diagnostic)
            try:
                for macrostate_idx in range(n_states):
                    clusters_in_macrostate = [cluster_idx for cluster_idx, macrostate in enumerate(metastable_traj) if macrostate == macrostate_idx]
                    log.info("Macrostate %d : %d frames", macrostate_idx + 1, len(clusters_in_macrostate))
            except Exception as e:
                log.warning("Failed while counting frames per macrostate: %s", e)

            # Save frame cluster center files (frames that belong to cluster center indices)
            try:
                for i in range(n_states):
                    frame_number = []
                    # highest_membership[i] is the active-set microstate index for macrostate i
                    for j in range(len(dtrajs_concatenated)):
                        if dtrajs_concatenated[j] == highest_membership[i]:
                            frame_number.append(j + 1)
                    frame = ["`[frames]"] + [str(x) for x in frame_number] + ["`"]
                    out = os.path.join(save_dir, f"frame_cluster_center_state_{i+1}_{filename_init}_gen_nclu_{n_clusters}_lag_{lag}_state_size_{n_states}.ndx")
                    np.savetxt(out, frame, fmt="%s")
            except Exception as e:
                log.warning("Failed to save frame_cluster_center_state files: %s", e)

            # Print sum of pi
            try:
                log.info("Sum of weights = %f", msm_data.pi.sum())
            except Exception:
                pass

    # After loops: collect some metastable_traj files into output_dir
    output_dir = args.output_dir or os.path.join(script_dir, "metastable_traj_all")
    os.makedirs(output_dir, exist_ok=True)
    for lag in args.lags:
        for n_state in args.states:
            subdir_name = os.path.join(script_dir, f"{filename_init}_gen_nclu_{n_clusters}_lag_{lag}_state_size_{n_state}")
            if os.path.exists(subdir_name):
                for file in os.listdir(subdir_name):
                    expected_prefix = f"metastable_traj_{filename_init}_gen_nclu_{n_clusters}_lag_{lag}_state_size_{n_state}.npy"
                    if file.startswith(expected_prefix):
                        src = os.path.join(subdir_name, file)
                        dst = os.path.join(output_dir, file)
                        shutil.copy(src, dst)
                        log.info("Copied %s to %s", src, dst)
            else:
                log.debug("Directory not found (skipping): %s", subdir_name)

    # Clean exit (do not attempt to delete names)
    gc.collect()
    log.info("Done.")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build MSM from cluster/dtrajs pickles (cleaned)")
    p.add_argument("--filename_init", default="asyn", help="base filename (default: trpcage)")
    p.add_argument("--n_clusters", type=int, default=400, help="number of microstates")
    p.add_argument("--lags", nargs="+", type=int, default=[50], help="list of lags")
    p.add_argument("--states", nargs="+", type=int, default=[3, 4, 5, 6], help="list of macrostate counts")
    p.add_argument("--its_dir", default=None, help="path to ITS directory (defaults to ./ITS next to script)")
    p.add_argument("--output_dir", default=None, help="where to copy metastable_traj files (default: metastable_traj_all in script dir)")
    args = p.parse_args()

    main(args)

