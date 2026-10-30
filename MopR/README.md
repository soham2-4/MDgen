# MopR–phenol: held-out binding events and the test of memorization

Thirteen phenol-binding events from the unbiased simulations of Sahil *et al.* (2023), each described by the ligand RMSD to the crystallographic pose (`gmx rms`, 10 ps). Nine events are used for training and three are held out; the model receives only the first 200 ns of each held-out trajectory. Paper: Fig. 5 and SI Figs. S9–S11.

## Input data (not included)

Put the 13 `.xvg` files in `sahil_MopR_phenol_raw_data/` inside this folder.

## Run

```bash
cd MopR
pip install dtaidistance scipy

# 1  outlier removal and matched held-out split  -> split_design/
python design_split.py --xvg_dir sahil_MopR_phenol_raw_data --k_test 3 --out split_design

# 2  1-ns block averages; 50 augmented copies per training trajectory
python mopr_data.py --xvg_dir sahil_MopR_phenol_raw_data --out synthetic_trajectories \
    --downsample 100 --n_copies 50 --noise 0.02 --max_shift_ns 150 \
    --test r1110 rmsd_1016 rmsd_1125 --exclude r223

# 3  train (GPU, ~1 h for 500 epochs); see train.log for the paper run
python mopr_trans.py --data_dir synthetic_trajectories --input_ns 200 --max_len_ns 1500 \
    --epochs 500 --eval_every 50 --save_every 100 --out checkpoints

# 4  one held-out prediction with an MC-dropout ensemble
python mopr_inference.py --ckpt checkpoints/transformer_epoch_500.pt --folder r1110 --n_samples 20

# 5  10 samples per source prefix (training and held-out)
python export_generated.py --ckpt checkpoints/transformer_epoch_500.pt \
    --data_dir synthetic_trajectories --n_samples 10 --out memcheck_data

# 6  data-copying test: held-out prefixes (main text) and training prefixes (SI)
python memorization_check.py --train "memcheck_data/train/*.npy" --test "memcheck_data/test/*.npy" \
    --gen "memcheck_data/gen_from_test/*.npy" --skip 200 --step_ns 1 --out memcheck_from_test
python memorization_check.py --train "memcheck_data/train/*.npy" --test "memcheck_data/test/*.npy" \
    --gen "memcheck_data/gen_from_train/*.npy" --skip 200 --step_ns 1 --out memcheck_from_train

# 7  positive (copying) and negative (non-copying) controls
python demo_controls.py --xvg_dir sahil_MopR_phenol_raw_data --split split_design/split.json --out controls
```

## Settings

| | |
|---|---|
| held-out split | r223 excluded (robust z-score 6.1 on binding time); of the 175 three-trajectory subsets with both binding sites, the one with the smallest energy distance between training and held-out feature vectors: r1110, rmsd_1016, rmsd_1125 |
| augmentation | 50 copies per training trajectory; Gaussian noise 0.02 nm; a random 0–150 ns of the unbound phase removed |
| model | encoder–decoder Transformer, d_model 64, 4 heads, 4 + 4 layers, feed-forward 128, dropout 0.1, causal decoder mask, start token −1 |
| training | Adam, lr 1e-4, batch 4, 500 epochs, gradient clipping 1.0; padded steps masked in the loss; series truncated at 1.5 µs |
| generation | first 200 ns as prefix; 10 rollouts per prefix with dropout active (MC dropout) |
| binding time | first time the RMSD drops below 0.5 nm |
| data-copying test | nearest-training-trajectory distance (pointwise and DTW with a 10 % Sakoe–Chiba window, divided by √n), prefix excluded; one-sided Mann–Whitney U test against the held-out MD |

The same pipeline can be repeated with the other well-matched splits printed by step 1 (e.g. `--test r317 r428 r519`) to hold out every event once.
