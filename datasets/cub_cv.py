from pathlib import Path
import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image

NUM_CLASSES = 200
NUM_CONCEPTS = 112

class CUBDataset(Dataset):
    def __init__(self, split_root, image_root, split="train", transform=None):
        assert split in ["train", "val", "test"]

        self.split_root = Path(split_root)
        self.image_root = Path(image_root)

        self.transform = transform
        self.split = split

        # Load classes name
        classes_file = self.image_root.parent / "classes.txt"
        self.class_names = []
        with open(classes_file, "r") as f:
            for line in f:
                _, name = line.strip().split(" ", 1)
                self.class_names.append(name)
        
        assert len(self.class_names) == NUM_CLASSES
        
        # Load concept names
        attributes_file = self.image_root.parent / "attributes112.txt"
        self.concept_names = []

        with open(attributes_file, "r") as f:
            self.concept_names = [line.strip() for line in f]

        assert len(self.concept_names) == NUM_CONCEPTS

        # Load split
        split_file = self.split_root / f"{split}.pkl"

        with open(split_file, "rb") as f:
            self.samples = pickle.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        original_path = Path(sample["img_path"])
        image_path = (self.image_root / original_path.parent.name / original_path.name)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(sample["class_label"], dtype=torch.long),
            "concepts": torch.tensor(sample["attribute_label"], dtype=torch.float32),
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
        return [sample["class_label"] for sample in self.samples]

    def get_class_name(self, class_id):
        return self.class_names[class_id]
    
    def get_concept_name(self, concept_id):
        return self.concept_names[concept_id]
    
    def __repr__(self):

        return (
            f"CUBDataset("
            f"split={self.split}, "
            f"n={len(self)}, "
            f"classes={self.num_classes}, "
            f"concepts={self.num_concepts})"
        )