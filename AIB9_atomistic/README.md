# AIB9 all-atom trajectories from generated backbone dihedrals

The peptide is represented by its 18 backbone φ, ψ dihedrals. A decoder-only causal Transformer predicts a categorical distribution over angle bins for every dihedral at the next frame; sampled dihedrals are converted back to all-atom Cartesian coordinates by deterministic internal-coordinate reconstruction. Paper: Fig. 4 and SI Figs. S3–S6.

## Input data (not included)

Five AIB9 replicas, `replica{1..5}_first_5000ns.xtc`, and the topology `only_prot.gro`, in this folder.

## Run

```bash
cd AIB9_atomistic
python aib9_dihedral_data.py                           # step 1: dihedrals -> data_dihedral/replica{1..5}.npz
python aib9_train_dihedral.py --split replica --val-replica replica5 --epochs 40 --window 64
python plot_training.py                                # training_curves.png
python aib9_generate_dihedral.py --n-chains 32 --n-frames 50000 --temperature 1.0
python aib9_evaluate_fes.py                            # compare free-energy surfaces with the reference
python aib9_reconstruct.py --chain 0                   # dihedrals -> PDB / XTC
```

`training_curves.png` is the training run used for the paper (36 bins, replica 5 held out, best validation loss at epoch 35 of 40).

`aib9_generate_dihedral.py --seed-mode first --seed-frames N` continues the first `N` frames of a replica instead of starting the chains from random windows.

## Settings (script defaults)

| | |
|---|---|
| representation | 18 dihedrals as (sin, cos) → 36 input features |
| bins | 36 bins of 10°; a sampled bin is mapped to an angle drawn uniformly within the bin |
| model | pre-norm decoder-only Transformer, d_model 256, 8 heads, 6 layers, feed-forward 1024, GELU, dropout 0.1 |
| context | sliding window of 64 frames (6.4 ns at 0.1 ns per frame) |
| loss | cross-entropy with targets smoothed onto the two neighboring bins (0.1) |
| augmentation | mirror image (φ, ψ) → (−φ, −ψ) with probability 0.5 |
| training | AdamW, lr 3e-4, weight decay 0.01, 500 warm-up steps then cosine decay, batch 128, gradient clipping 1.0 |
| sampling | temperature 1.0, window advanced by one frame per step |
| reconstruction | NeRF backbone build with bond lengths, angles and ω from the topology; other atoms placed in local frames |

## `legacy_cartesian/`

The earlier Cartesian-coordinate pipeline (encoder–decoder on 129 × 3 coordinates), kept for reference. It is not used for the results in the paper.
