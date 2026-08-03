import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import numpy as np
from sklearn.model_selection import train_test_split
from datasets.awa2 import AwA2Dataset

ROOT = "data/awa2/Animals_with_Attributes2"
SEED = 42

# Load ALL images
full_ds = AwA2Dataset(root=ROOT, split="all", transform=None,)

indices = np.arange(len(full_ds))
labels = full_ds.labels

# 80% train+val, 20% test
trainval_idx, test_idx = train_test_split(indices, test_size=0.20, stratify=labels, random_state=SEED,)

print(len(trainval_idx))
print(len(test_idx))

# Save indices
np.save("data/awa2/train_val_idx.npy", trainval_idx)
np.save("data/awa2/test_idx.npy", test_idx)

print("Done!")