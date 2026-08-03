# cbm-reproduction-benchmark
Official implementation of the M1 Internship report: "Reproduction and Benchmarking of Concept Bottleneck Models" at **Jean Monnet University** and **Hubert Curien Laboratory**.

## Overview

This repository provides a unified benchmark framework for Concept Bottleneck Models (CBMs), including:

- Joint CBM
- Sequential CBM
- Concept Embedding Model (CEM)
- Probabilistic CBM (ProbCBM)
- Stochastic CBM (SCBM)

## Features

- Unified implementation of six Concept Bottleneck Model architectures
- Common ResNet-18 backbone for fair and reproducible comparisons
- Support for five benchmark datasets, including the proposed GEOM-3-28
- Automatic five-fold cross-validation
- Multi-seed evaluation for robust performance estimation
- Configurable benchmark pipeline through command-line arguments
- Resume interrupted cross-validation from any fold
- Automatic checkpoint selection and independent test evaluation
- Complete GEOM-3-28 synthetic dataset generator

The benchmark supports five datasets. Due to licensing restrictions, the external datasets must be downloaded from their official sources before running the experiments.

| Dataset | Download |
|----------|----------|
| CUB-200-2011 | [Official Website](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1) |
| AwA2 | [Official Website](https://cvml.ista.ac.at/AwA2/) |
| Derm7pt | [Official Website](https://derm.cs.sfu.ca/) |
| CelebA | [Official Website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| GEOM-3-28 | Generated locally |

## Repository structure
```text
datasets/                        Dataset loaders
datasets/synthetic_generator/    GEOM-3-28 dataset generator
models/                          Model implementations
training/                        Training procedures
testing/                         Evaluation scripts
transforms/                      Image preprocessing
scripts/                         Specific adjustments for CUB, AwA2 and CelebA
```

## Installation
Clone the repository

```bash
git clone https://github.com/paulovitor93/cbm-reproducibility-benchmark.git

cd cbm-reproducibility-benchmark
```

Create the environment

```bash
pip install -r requirements.txt
```

Tested with
- Python 3.11
- PyTorch 2.5.1
- CUDA 12.1

## Dataset Preparation
> **Note:** The benchmark expects all datasets to be located under the `data/` directory following the structures shown below. Dataset preparation scripts only need to be executed once after downloading the original datasets.

```text
data/
├── cub/
├── awa2/
├── derm7pt/
├── celeba/
└── synthetic_dataset/
```

After downloading the dataset, organize the files as follows:
### CUB-200-2011
The benchmark uses the train/validation/test split introduced by Koh et al. for Concept Bottleneck Models. The corresponding `class_attr_data_10` files should be downloaded separately.
[Official Split CBM CUB dataset](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1)

```text
data/
└── cub/
    ├── CUB_200_2011/
    │   ├── images/
    │   ├── attributes/
    │   ├── classes.txt
    │   ├── image_class_labels.txt
    │   ├── images.txt
    │   ├── train_test_split.txt
    │   └── ...
    │
    ├── class_attr_data_10/
    │   ├── train.pkl
    │   ├── val.pkl
    │   └── test.pkl
    │
    ├── attributes.txt
    └── attributes112.txt
```

Generate the reduced list of 112 concepts used in the benchmark:

```bash
python scripts/create_cub_112_attributes.py
```

This script extracts the subset of 112 attributes used by the original Concept Bottleneck Model benchmark from the original list of 312 CUB attributes. :contentReference[oaicite:0]{index=0}

The benchmark includes utility scripts to prepare some datasets before training. These scripts reproduce the preprocessing protocol adopted in the experiments.

| Dataset | Script | Purpose |
|---------|--------|---------|
| CUB | `scripts/create_cub_112_attributes.py` | Selects the 112 concepts used by the original CBM benchmark from the 312 available CUB attributes. |
| AwA2 | `scripts/create_awa2_split.py` | Creates the stratified train/validation/test split used throughout the experiments. |
| CelebA | `scripts/celeba_attributes_male_female_task.py` | Selects the facial attributes used as concepts and creates the binary Male/Female prediction task. |
| CelebA | `scripts/create_celeba_20_subset.py` | Generates the reproducible 20% subset used in the experiments while preserving the class distribution of each split. |
| GEOM-3-28 | `datasets/synthetic_generator/generate_dataset.py` | Generates the complete synthetic dataset, including images, concept annotations, metadata and dataset splits. |
> **Note:** These scripts only need to be executed once after downloading the original datasets. The generated files (concept lists, subsets and dataset splits) are then reused by all training and evaluation scripts.

## Running Experiments
The framework allows configuring:

- Model architecture
- Dataset
- Number of epochs
- Batch size
- Cross-validation folds
- Random seed
- Resume training from any fold

### Available models
| Mode | Description |
|------|-------------|
| `joint_cv` | Joint Concept Bottleneck Model |
| `concepts_cv` | Concept predictor only |
| `sequential_cv` | Sequential CBM |
| `independent_cv` | Independent CBM |
| `cem_cv` | Concept Embedding Model |
| `prob_concepts_cv` | ProbCBM concept encoder |
| `prob_classifier_cv` | ProbCBM classifier |
| `scbm_cv` | Stochastic CBM |

## Supported Datasets

| Dataset | Images | Classes | Concepts | Domain |
|:--------|-------:|--------:|---------:|--------|
| CUB | 11,788 | 200 | 112 | Bird Species |
| AwA2 | 37,322 | 50 | 85 | Animal Species |
| Derm7pt | 1,011 | 2 | 19 | Skin Lesions |
| CelebA | 32,544 | 2 | 18 | Facial Attributes |
| **GEOM-3-28** | **28,000** | **28** | **12** | **Synthetic** |
> **Note:** The repository provides loaders for all supported datasets. Due to licensing restrictions, CUB, AwA2, Derm7pt, and CelebA must be downloaded separately. The complete generator for the proposed GEOM-3-28 dataset is included and allows users to reproduce the synthetic dataset from scratch.

## Training and Evaluation Arguments
| Argument | Description | Default |
|-----------|-------------|---------|
| `--mode` | Model to train or evaluate | required |
| `--dataset` | Dataset | `cub` |
| `--epochs` | Training epochs | `100` |
| `--batch_size` | Batch size | `32` |
| `--folds` | Cross-validation folds | `5` |
| `--seed` | Random seed | `42` |
| `--start_fold` | Resume CV from a given fold | `1` |

### Train a model
```bash
python train.py --mode joint_cv --dataset cub --epochs 100 --batch_size 32 --folds 5 --seed 203040
```

### Evaluate a model
```bash
python test.py --mode joint_cv --dataset cub --batch_size 32 --folds 5 --seed 203040
```

### Resume cross-validation
If training is interrupted, cross-validation can be resumed from any fold.

```bash
python train.py --mode joint_cv --dataset cub --epochs 100 --folds 5 --seed 203040 --start_fold 3
```
The previous folds are loaded automatically and only the remaining folds are executed.

## Full Benchmark Pipeline
The repository provides a configurable benchmarking pipeline through `full_pipeline_cv.py`.

The pipeline automatically:

- Trains every selected model.
- Evaluates every trained model.
- Performs *k*-fold cross-validation.
- Repeats the experiments for multiple random seeds.
- Stores all checkpoints and evaluation results.
- Measures the execution time of every experiment.

The benchmark configuration is fully controlled through command-line arguments.

### Example

```bash
python full_pipeline_cv.py --datasets cub awa2 celeba --models joint_cv cem_cv scbm_cv --epochs 100 --batch_size 32 --folds 5 --seeds 203040 532164
```
### Resume an interrupted benchmark
```bash
python full_pipeline_cv.py --datasets cub --models scbm_cv --epochs 100 --folds 5 --start_fold 4 --seeds 203040
```
The pipeline automatically resumes the cross-validation from the specified fold.

## Reproducibility
All experiments are deterministic given the selected random seed. The benchmark stores the random seeds and benchmark configuration for every experiment, allowing the reported results to be reproduced.

## Synthetic dataset generation (GEOM-3-28)
This repository also contains the complete generator for the proposed **GEOM-3-28** synthetic dataset.
