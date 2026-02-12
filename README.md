# MDgen

This repository is the official implementation of the paper **"Enhanced Transformer-Based Sampling for
Long-Range Molecular Dynamics Trajectory
Generation"**.

<p align="center">
  <img src="architecture (1).png">
</p>

A Transformer-based autoregressive framework for learning and generating Molecular Dynamics (MD) trajectories. This project models long-range temporal dependencies in MD simulations and generates future trajectory segments conditioned on an initial fraction of the sequence.

The framework currently supports three systems:

1. Cytochrome P450 (P450)

2. Toy 3-Well Potential System

3. α-Synuclein (Asyn)

## Requirements

This project was developed using:

Python ≥ 3.8

PyTorch ≥ 1.13

NumPy

Matplotlib

It is recommended to use a virtual environment (e.g., Conda).. A virtual environment manager such as [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) is recommended.

## Creat environment:
conda create -n mdtraj python=3.8
conda activate mdtraj
pip install torch numpy matplotlib

## Project overview
Each system directory contains:

Raw MD trajectory data

data.py → synthetic data generation (noise augmentation)

trans1.py → Transformer model training

inference1.py or inference2.py → trajectory generation

The workflow is similar across systems, with minor differences in file names and data formats.

## Pre-trained Model Checkpoint

You can download the model weights used in the paper [here](https://drive.google.com/drive/folders/1paFyU48Rn5DHEkmc9xUrPAvtHVfcXik5) for inference.

## Configuration

Configuration management for MDgen is done through multiple .yml files located in the `configs/` directory.  
- `train.yml` specifies the parameters for training the diffusion model along with path to the processed dataset.
- `sample.yml` specifies parameters for sampling, including the number of peptides to sample for each target receptor and model weights to load.
- `reconstruct.yml` specifies the reconstruction algorithm as well as its hyperparameters.

These configurations might have to be modified to point to the dataset and checkpoint path on your system before proceeding with training and evaluation.

## Training

To train MDgen, run:

```train
python3 scripts/train.py configs/train.yml
```

Optionally, you may specify the parameter `--num_gpus N` in order to perform multi-GPU training using the Distributed Data Parallel strategy.  
The parameter `--ckpt` can be used to resume training from an existing checkpoint.

## Evaluation

Evaluating MDgen must be done in two stages: 
1. Sampling residues based on the target receptor.
2. Reconstructing generated residues into peptides.

### 1.1 Sampling for all receptors in the testset

```eval
python3 scripts/sample-testset.py configs/train.yml configs/sample.yml --out_dir ./outputs
```

### 1.2 Sampling for a receptor from PDB

```eval
python3 scripts/sample-pdb.py data/pfemp1/PF3D71150400_MEDIUM.pdb configs/train.yml configs/sample.yml --out_dir ./outputs
```

### 2. Reconstructing sampled residues into peptides
```eval
python3 scripts/reconstruct.py ./outputs configs/reconstruct.yml
```

## Contributing
All source code within this repository is licensed under the MIT License, please see the [LICENSE](LICENSE) file for more details.
