from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

NUM_CLASSES = 28
NUM_CONCEPTS = 12

class SyntheticDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        assert split in ["train", "val", "test"]

        self.root = Path(root)
        self.images_root = self.root / "images"

        self.transform = transform
        self.split = split

        # Load split
        split_file = self.root / f"{split}.csv"
        df = pd.read_csv(split_file)

        # Load concept names
        self.concept_names = [
            c
            for c in df.columns
            if c not in ["image_id", "image_name", "class"]
        ]

        assert len(self.concept_names) == NUM_CONCEPTS

        # Class names
        self.class_names = [f"Class {i}" for i in range(NUM_CLASSES)]

        # Build samples
        self.samples = []

        for _, row in df.iterrows():
            sample = {
                "image_path": str(self.images_root / row["image_name"]),
                "label": int(row["class"]),
                "concepts": row[self.concept_names].astype(float).tolist(),
            }

            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with Image.open(sample["image_path"]) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "concepts": torch.tensor(sample["concepts"], dtype=torch.float32),
            "idx": idx,
        }

    @property
    def num_classes(self):
        return NUM_CLASSES

    @property
    def num_concepts(self):
        return NUM_CONCEPTS

    @property
    def labels(self):
        return [sample["label"] for sample in self.samples]

    def get_class_name(self, class_id):
        return self.class_names[class_id]

    def get_concept_name(self, concept_id):
        return self.concept_names[concept_id]

    def __repr__(self):
        return (
            f"SyntheticDataset("
            f"split={self.split}, "
            f"n={len(self)}, "
            f"classes={self.num_classes}, "
            f"concepts={self.num_concepts})"
        )