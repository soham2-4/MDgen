# AIB9 helix handedness (chirality coordinate ζ)

Five 5-µs replicas of AIB9 at 400 K, reduced to ζ = Σ φ₃…φ₇ (radians), saved every 100 ps (`aib9_chirality_analysis_data_replica_{1..5}.csv`). Paper: Fig. 2d,e and SI Fig. S2.

## Run

```bash
cd AIB9_chirality
python aib9_data.py        # 200 noise-augmented copies per replica -> synthetic_aib9_trajectories/
python aib9_trans1.py      # -> aib9_checkpoints/aib9_transformer_best.pt (and every 50 epochs)
python aib9_inference.py --model aib9_checkpoints/aib9_transformer_best.pt --input-pct 0.10
```

`aib9_inference.py` continues each of the five replicas from its first 10 % and writes `aib9_inference_output/generated_replica_{1..5}.npy` and an overview figure.

## Settings (from the scripts)

| | |
|---|---|
| augmentation | 200 copies per replica; uniform noise of magnitude 1e-4–1e-3 rad with random sign |
| normalization | single mean/SD computed on the training split |
| model | encoder–decoder Transformer, d_model 64, 4 heads, 4 + 4 layers, feed-forward 256, dropout 0.1 |
| training | AdamW, lr 1e-4, batch 4, 500 epochs, MSE on the frames after the prefix; 80/20 random split of the copies; best validation checkpoint kept |
| prefix | 10 % of each replica (5,000 frames = 500 ns) |
| inference | deterministic rollout to the full replica length |
