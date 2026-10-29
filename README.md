# MDgen

Code and data for the paper

> **Learning Molecular Histories: Autoregressive Trajectory Generation from Limited Simulations**
> Soham Choudhuri, Subinoy Adhikari, Irawati Roy, and Jagannath Mondal

<p align="center">
  <img src="transformer_architecture.png" width="55%">
</p>

An autoregressive Transformer receives a short prefix of a molecular dynamics (MD) trajectory and generates the rest of it frame by frame. Each generated frame becomes context for the next. The repository contains the training, generation and analysis code for every system in the paper.

## Systems and where they appear in the paper

| Folder | System | Coordinate | Model | Paper |
|---|---|---|---|---|
| [`ThreeWell_potential/`](ThreeWell_potential) | 2D three-well Brownian dynamics | $(x, y)$ | encoder–decoder | Fig. 2a–c; SI Fig. S1 |
| [`AIB9_chirality/`](AIB9_chirality) | AIB9 helix handedness | chirality coordinate ζ | encoder–decoder | Fig. 2d,e; SI Fig. S2 |
| [`GB3/`](GB3) | GB3 native state | number of native contacts | encoder–decoder | Fig. 3a; SI Fig. S7 |
| [`Asyn/`](Asyn) | α-synuclein | 2D β-VAE latent space | encoder–decoder | Fig. 3b,c; SI Fig. S12 |
| [`P450/`](P450) | cytochrome P450 substrate binding | protein–ligand distance | encoder–decoder | Fig. 3d; SI Figs. S8, S13 |
| [`AIB9_atomistic/`](AIB9_atomistic) | AIB9 all-atom trajectories | 18 backbone φ, ψ dihedrals | categorical causal (decoder-only) | Fig. 4; SI Figs. S3–S6 |
| [`MopR/`](MopR) | MopR–phenol binding, held-out test of memorization | ligand RMSD | encoder–decoder | Fig. 5; SI Figs. S9–S11 |
| [`Analysis/`](Analysis) | free-energy surfaces, MSMs, implied timescales, CK test, MFPT | — | — | Figs. 2, 3; SI Fig. S12 |

Each system folder has its own README with the exact commands.

## Repository layout

```
MDgen/
├── Data/                  reference data for three-well, α-synuclein and P450
├── ThreeWell_potential/   data.py → trans1.py → inference1.py   (+ trained checkpoint)
├── Asyn/                  data.py → trans1.py → inference1.py   (+ trained checkpoint)
├── P450/                  data.py → trans1.py → inference2.py   (+ trained checkpoint)
├── AIB9_chirality/        aib9_data.py → aib9_trans1.py → aib9_inference.py (+ ζ data)
├── AIB9_atomistic/        dihedral extraction → training → sampling → reconstruction
├── GB3/                   native-contact data, training, rollouts, exported results
├── MopR/                  held-out split, training, generation, data-copying test
└── Analysis/              FES, MSM, ITS, CK test, PCCA and MFPT scripts (PyEMMA)
```

## Installation

```bash
conda create -n mdgen python=3.10
conda activate mdgen
pip install -r requirements.txt
```

The MSM analysis uses PyEMMA. PyEMMA is easiest to install from conda-forge and needs an older Python (≤ 3.9). A separate environment is the simplest option:

```bash
conda create -n msm -c conda-forge python=3.9 pyemma matplotlib colorama scikit-learn
```

GPU selection is left to you; set `CUDA_VISIBLE_DEVICES` before running a script. All scripts fall back to the CPU when no GPU is available.

The `trans1.py` scripts for three-well, α-synuclein and P450 log to [Weights & Biases](https://wandb.ai). To run them without an account:

```bash
export WANDB_MODE=disabled
```

## Data availability

| System | Included here | Source |
|---|---|---|
| Three-well | `Data/threewell.npy` (50,000 × 2) | generated for this work |
| α-synuclein | `Data/asyn.npy` (73,124 × 2 latent coordinates) | refs. in the paper |
| Cytochrome P450 | `Data/distance{0,1,2}.xvg` | Ahalawat & Mondal (2018) |
| AIB9 ζ | `AIB9_chirality/*.csv`, five 5-µs replicas | generated for this work |
| GB3 | `GB3/gb3_native_contacts_vs_time.dat` | from the a99SB-disp trajectory of Robustelli, Piana & Shaw (2018) |
| AIB9 all-atom | not included (`.xtc` + `only_prot.gro` needed) | generated for this work |
| MopR–phenol | not included (13 `gmx rms` `.xvg` files needed) | Sahil *et al.* (2023) |

## Citation

The citation will be added on publication.

## License

MIT; see [`LICENSE`](LICENSE).

## Contact

Soham Choudhuri — choudhurisoham@gmail.com
Jagannath Mondal — jmondal@tifrh.res.in
