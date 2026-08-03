from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split

ROOT = Path("data/celeba")
SUBSET_PERCENTAGE = 0.20

SEED = np.random.randint(0, 1_000_000)
print(f"Seed: {SEED}")

# Load dataset
df = pd.read_csv(ROOT / "celeba_gender_cbm_dataset.csv")

split_df = pd.read_csv(ROOT / "list_eval_partition.txt", sep=r"\s+", names=["image_id", "partition"],)
df = df.merge(split_df, on="image_id")

partition_names = {0: "train", 1: "val", 2: "test",}
metadata = {"seed": int(SEED), "subset_percentage": SUBSET_PERCENTAGE,}

# Sample each split independently
for partition, name in partition_names.items():
    split = df[df["partition"] == partition].reset_index(drop=True)
    idx, _ = train_test_split(np.arange(len(split)), train_size=SUBSET_PERCENTAGE, stratify=split["Male"], random_state=SEED,)
    idx = np.sort(idx)
    
    np.save(ROOT / f"celeba_20_percent_{name}_idx.npy", idx)
    metadata[f"{name}_size"] = int(len(idx))

    subset = split.iloc[idx]
    print(f"{name.capitalize():10s}: {len(idx)} images")
    print(f"   Male      : {(subset['Male']==1).sum()}")
    print(f"   Not Male  : {(subset['Male']==0).sum()}")
    print(f"   Male ratio: {subset['Male'].mean():.4f}\n")

metadata["total_size"] = (metadata["train_size"] + metadata["val_size"] + metadata["test_size"])

with open(ROOT / "celeba_20_percent_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\nDone!")