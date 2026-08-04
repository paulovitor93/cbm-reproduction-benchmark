from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

NUM_CLASSES = 2

class CelebADataset(Dataset):
    def __init__(self, root, split="train", target_attribute="Male", transform=None):

        assert split in ["train", "val", "test"]
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_attribute = target_attribute

        self.image_root = self.root / "img_align_celeba"

        # Load attributes
        attr_path = self.root / "celeba_gender_cbm_dataset.csv"

        df = pd.read_csv(attr_path)

        # Load official split
        split_df = pd.read_csv(self.root / "list_eval_partition.txt", sep=r"\s+", names=["image_id", "partition"],)
        split_map = {"train": 0, "val": 1, "test": 2,}
        split_df = split_df[split_df.partition == split_map[split]]
        df = df.merge(split_df, on="image_id")

        # Class names
        self.class_names = [f"Not_{target_attribute}", target_attribute,]

        # Build samples
        self.samples = []
 
        excluded_columns = {"image_id", "partition", target_attribute,}
        self.concept_columns = [c for c in df.columns if c not in excluded_columns]

        for _, row in df.iterrows():
            image_path = self.image_root / row["image_id"]

            label = int(row[target_attribute])
            concepts = row[self.concept_columns].to_numpy(dtype=np.float32)

            self.samples.append({
                "image_path": image_path,
                "label": label,
                "concepts": concepts,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(sample["label"],dtype=torch.long,),
            "concepts": torch.tensor(sample["concepts"],dtype=torch.float32,),
            "idx": idx,
        }

    @property
    def num_classes(self):
        return NUM_CLASSES

    @property
    def num_concepts(self):
        return len(self.concept_columns)

    @property
    def labels(self):
        return [sample["label"] for sample in self.samples]

    def get_class_name(self, class_id):
        return self.class_names[class_id]

    def __repr__(self):
        return (
            f"CelebADataset("
            f"split={self.split}, "
            f"n={len(self)}, "
            f"classes={self.num_classes}, "
            f"concepts={self.num_concepts})"
        )