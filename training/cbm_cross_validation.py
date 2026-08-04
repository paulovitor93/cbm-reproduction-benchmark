import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
import time
import os

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from datasets.cub_cv import CUBDataset
from datasets.awa2 import AwA2Dataset
from datasets.derm7pt import Derm7ptDataset
from datasets.celeba import CelebADataset
from datasets.synthetic import SyntheticDataset
from transforms.transforms import get_transforms

class TransformSubset(Dataset):
    """
    Dataset wrapper that applies transforms without duplicating samples.
    """

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.dataset.samples[real_idx]

        if "image_path" in sample:
            image_path = sample["image_path"]

        else:
            original_path = Path(sample["img_path"])
            image_path = (self.dataset.image_root / original_path.parent.name / original_path.name)

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")

        except Exception as e:
            print("=" * 80)
            print("ERROR LOADING IMAGE")
            print(image_path)
            print(e)
            print("=" * 80)
            raise

        if self.transform:
            image = self.transform(image)

        label = sample.get("label", sample.get("class_label"))
        concepts = sample.get("concepts", sample.get("attribute_label"))

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "concepts": torch.tensor(concepts, dtype=torch.float32),
            "idx": real_idx,
        }

    @property
    def num_classes(self):
        return self.dataset.num_classes

    @property
    def num_concepts(self):
        return self.dataset.num_concepts

def run_cv(trainer, experiment_name, dataset_name="cub", epochs=100, 
           batch_size=32, SEED=42, n_splits=5, start_fold=1,):
    """
    Run stratified cross-validation for a given trainer.
    """

    if dataset_name.lower() == "cub":
        SPLIT_ROOT = "data/cub/class_attr_data_10"
        IMAGE_ROOT = "data/cub/CUB_200_2011/images"

        train_ds = CUBDataset(split_root=SPLIT_ROOT, image_root=IMAGE_ROOT, split="train", transform=None)
        val_ds = CUBDataset(split_root=SPLIT_ROOT, image_root=IMAGE_ROOT, split="val", transform=None)

        # Combine train + val
        train_ds.samples.extend(val_ds.samples)
        labels = np.array(train_ds.labels)

    elif dataset_name.lower() == "awa2":
        ROOT = "data/awa2/Animals_with_Attributes2"
        train_ds = AwA2Dataset(root=ROOT, split="all", transform=None,)

        trainval_idx = np.load("data/awa2/train_val_idx.npy")
        labels = np.array(train_ds.labels)[trainval_idx]

    elif dataset_name.lower() == "derm7pt":
        ROOT = "data/derm7pt/release_v0"

        train_ds = Derm7ptDataset(root=ROOT, split="train", transform=None,)
        val_ds = Derm7ptDataset(root=ROOT, split="val", transform=None,)

        # Merge train + validation
        train_ds.samples.extend(val_ds.samples)
        labels = np.array(train_ds.labels)

    elif dataset_name.lower() == "celeba":
        ROOT = "data/celeba"

        train_ds = CelebADataset(root=ROOT, split="train", transform=None,)
        val_ds = CelebADataset(root=ROOT, split="val", transform=None,)

        train_idx = np.load("data/celeba/celeba_20_percent_train_idx.npy")
        val_idx = np.load("data/celeba/celeba_20_percent_val_idx.npy")

        train_ds.samples = [train_ds.samples[i] for i in train_idx]
        val_ds.samples = [val_ds.samples[i] for i in val_idx]

        train_ds.samples.extend(val_ds.samples)
        labels = np.array(train_ds.labels)

    elif dataset_name.lower() == "synthetic":
        ROOT = "data/synthetic_dataset"

        train_ds = SyntheticDataset(root=ROOT, split="train", transform=None,)
        val_ds = SyntheticDataset(root=ROOT, split="val", transform=None,)

        train_ds.samples.extend(val_ds.samples)
        labels = np.array(train_ds.labels)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    # Preserves the class distribution of the original dataset in each fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    os.makedirs(f"results/{experiment_name}", exist_ok=True)
    results_file = f"results/{experiment_name}/cv_results_seed{SEED}.csv"

    # If resuming, load previous results
    if start_fold > 1 and os.path.exists(results_file):
        print(f"Loading previous CV results from: {results_file}")

        previous_results = pd.read_csv(results_file)

        # Remove the summary row if it exists
        previous_results = previous_results[previous_results["fold"] != "mean±std"]
        previous_results["fold"] = previous_results["fold"].astype(int)

        # Keep only completed folds before the one we're restarting from
        previous_results = previous_results[previous_results["fold"] < start_fold]
        previous_results = previous_results.reset_index(drop=True)
        
        results = previous_results.to_dict("records")

    else:
        results = []

    cv_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(labels)), labels)):

        current_fold = fold + 1
        if current_fold < start_fold:
            print("=" * 80)
            print(f"Skipping Fold {current_fold}")
            print("=" * 80)
            continue

        print("\n" + "=" * 80)
        print(f"FOLD {current_fold}/{n_splits}")
        print("=" * 80)

        if dataset_name.lower() == "awa2":
            train_idx = trainval_idx[train_idx]
            val_idx = trainval_idx[val_idx]

        train_subset = TransformSubset(train_ds, train_idx, get_transforms("train"),)
        val_subset = TransformSubset(train_ds, val_idx, get_transforms("val"),)

        print(
            f"Train size={len(train_subset)} "
            f"Val size={len(val_subset)}"
        )

        metrics = trainer(epochs=epochs, batch_size=batch_size, SEED=SEED, train_dataset=train_subset,
                          val_dataset=val_subset, fold_id=current_fold, dataset_name=dataset_name,)
        
        results.append(metrics)
        # Save after every completed fold
        pd.DataFrame(results).to_csv(results_file, index=False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_cv_time = time.time() - cv_start
    results_df = pd.DataFrame(results)

    numeric_cols = results_df.select_dtypes(include=np.number).columns

    summary_row = {"fold": "mean±std", "best_epoch": "-"}
    summary_lines = []

    print(f"\nTotal CV Time: {total_cv_time/60:.2f} minutes")
    print("\n")
    print("=" * 80)
    print("CROSS VALIDATION RESULTS")
    print("=" * 80)

    for col in numeric_cols:
        mean = results_df[col].mean()
        std = results_df[col].std()

        summary_row[col] = f"{mean:.4f} ± {std:.4f}"
        print(f"{col}: {mean:.4f} ± {std:.4f}")

        summary_lines.append(f"{col}: {mean:.4f} ± {std:.4f}")

    mean_time = results_df["fold_time_minutes"].mean()

    print(
        f"Average Fold Time : "
        f"{format_time(mean_time * 60)}"
    )

    print(
        f"Total CV Time : "
        f"{format_time(total_cv_time)}"
    )

    summary_file = (f"results/{experiment_name}/cv_summary_seed{SEED}.txt")

    with open(summary_file, "w") as f:
        f.write(
            f"{experiment_name.upper()} "
            f"CROSS VALIDATION\n\n"
        )

        for line in summary_lines:
            f.write(line + "\n")

        f.write(
            f"\nAverage Fold Time: "
            f"{format_time(mean_time * 60)}\n"
        )

        f.write(
            f"Total CV Time: "
            f"{format_time(total_cv_time)}\n"
        )

    results_df = pd.concat([results_df, pd.DataFrame([summary_row])], ignore_index=True)
    results_df.to_csv(results_file, index=False)
    
    del train_ds
    if dataset_name.lower() != "awa2":
        del val_ds

    if torch.cuda.is_available():
        print(
            f"After Fold {current_fold}: "
            f"allocated={torch.cuda.memory_allocated()/1024**3:.2f} GB | "
            f"reserved={torch.cuda.memory_reserved()/1024**3:.2f} GB"
        )

    return results_df

def format_time(seconds):
    """
    Format seconds as minutes and seconds.
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes} min {seconds} s"

if __name__ == "__main__":
    print("Use train.py to launch cross validation.")