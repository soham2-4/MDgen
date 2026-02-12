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

## 1. P450 System
Raw Data (distance0.xvg, distance1.xvg, distance2.xvg) contain MD trajectory distance measurements. These training data can be found in ''

## Workflow (P450)
Step 1: data.py loads MD trajectories, adds small Gaussian noise and produces augmented synthetic trajectories data for training. 

Step 2: trans1.py trains our autoregressive Transformer model, which uses sliding-window sequence learning to learn long-range trajectory dependencies.

Step 3: inference2.py generates future trajectory segments autoregressively, allowing modification of input conditioning percentage (e.g., first 10%)

## 2 and 3. Toy 3-Well System and α-Synuclein (Asyn) System

## Data augmentation and Training
Raw Data () training data can be found in ''.

To generate synthetic data, run:

```
python data.py
```
To train, run:

```
python trans1.py
```
To do inference, run:

```
python inference1.py
```

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
