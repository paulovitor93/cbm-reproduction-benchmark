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
| AwA2 | [Official Website](https://cvml.ista.ac.at/AwA2/AwA2-data.zip) |
| Derm7pt | [Official Website](https://derm.cs.sfu.ca/Download.html) |
| CelebA | [Official Website](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=drive_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ) |
| GEOM-3-28 | [Generated locally](https://doi.org/10.5281/zenodo.21777426)|

## Repository structure
```text
datasets/                        Dataset loaders
datasets/synthetic_generator/    GEOM-3-28 dataset generator
models/                          Model implementations
training/                        Training procedures
testing/                         Evaluation scripts
transforms/                      Image preprocessing
scripts/                         Dataset preparation utilities
```

## Installation
Clone the repository
```bash
git clone https://github.com/paulovitor93/cbm-reproduction-benchmark.git
cd cbm-reproduction-benchmark
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
The benchmark uses the train/validation/test split introduced by Koh et al. for Concept Bottleneck Models. The corresponding `class_attr_data_10` files are already included in this repository.
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

This script extracts the subset of 112 attributes used by the original Concept Bottleneck Model benchmark from the original list of 312 CUB attributes. 

### Animals with Attributes 2 (AwA2)
```text
data/
└── awa2/
    ├── Animals_with_Attributes2/
    │   ├── JPEGImages/
    │   ├── classes.txt
    │   ├── predicates.txt
    │   └── ...
    │
    ├── train_val_idx.npy
    └── test_idx.npy
```

Generate the benchmark train/test split:
```bash
python scripts/create_awa2_split.py
```

### Derm7pt
```text
data/
└── derm7pt/
    └── release_v0/
        ├── images/
        ├── meta/
             ├── train_indexes.csv
             ├── valid_indexes.csv
             └── test_indexes.csv
```
No additional preprocessing is required.

### CelebA
The corresponding `list_attr_celeba.txt` and `list_eval_partition.txt`, files are already included in this repository.
The file `OFFICIAL_WORK_celeba_20_percent_metadata.json` has the current configurations used in this work.
```text
data/
└── celeba/
    ├── img_align_celeba/
    ├── list_attr_celeba.txt
    ├── list_eval_partition.txt
    ├── celeba_gender_cbm_dataset.csv
    ├── selected_concepts.csv
    ├── removed_concepts.csv
    ├── celeba_20_percent_train_idx.npy
    ├── celeba_20_percent_val_idx.npy
    ├── celeba_20_percent_test_idx.npy
    └── ...
```

Prepare the dataset by selecting the concepts and creating the subset used in the experiments:
```bash
python scripts/celeba_attributes_male_female_task.py
```
```bash
python scripts/create_celeba_20_subset.py
```
The first script creates the binary Male/Female prediction task and selects the concept annotations used by the benchmark. The second script generates the reproducible 20% subset while preserving the class distribution of the official train, validation and test partitions. 

### GEOM-3-28
Generate the synthetic dataset using
```bash
python datasets/synthetic_generator/generate_dataset.py
```

The script automatically

- generates the scenes,
- extracts the concepts,
- creates the metadata,
- creates the official train/validation/test split.
  
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
| Argument | Description |
|-----------|-------------|
| `--mode` | Model to train or evaluate |
| `--dataset` | Dataset | 
| `--epochs` | Training epochs | 
| `--batch_size` | Batch size | 
| `--folds` | Cross-validation folds |
| `--seed` | Random seed |
| `--start_fold` | Resume CV from a given fold |

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

The pipeline supports running either the complete benchmark or any user-defined subset of datasets, models and random seeds.
### Run the complete benchmark

```bash
python full_pipeline_cv.py --run_all
```

This command executes all implemented models on all supported datasets using the default benchmark configuration.

### Run all models on one dataset

```bash
python full_pipeline_cv.py --datasets cub --models all
```

### Run one model on all datasets
```bash
python full_pipeline_cv.py --datasets all  --models cem_cv
```

### Run a custom benchmark
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
