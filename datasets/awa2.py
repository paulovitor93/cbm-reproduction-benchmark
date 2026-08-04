from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

NUM_CLASSES = 50
NUM_CONCEPTS = 85

class AwA2Dataset(Dataset):
    def __init__(self, root, split="train", transform=None):

        assert split in ["train", "test", "all"]

        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.image_root = self.root / "JPEGImages"

        # Class names
        self.class_names = _read_namelist(self.root / "classes.txt")
        assert len(self.class_names) == NUM_CLASSES

        # Concept names
        self.concept_names = _read_namelist(self.root / "predicates.txt")
        assert len(self.concept_names) == NUM_CONCEPTS

        selected_classes = self.class_names
        train_classes = sorted(_read_namelist(self.root / "trainclasses.txt"))
        test_classes = sorted(_read_namelist(self.root / "testclasses.txt"))
        
        if split == "train":
            selected_classes = train_classes
        elif split == "test":
            selected_classes = test_classes
        else:
            selected_classes = self.class_names

        # Concept matrix
        attr_matrix = np.loadtxt(self.root / "predicate-matrix-binary.txt", dtype=np.float32,)
        
        class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        class_to_label = {c: i for i, c in enumerate(self.class_names)}

        # Build samples
        self.samples = []
        
        for cls_name in selected_classes:
            folder = self.image_root / cls_name

            if not folder.is_dir():
                continue

            concepts = attr_matrix[class_to_idx[cls_name]]
            label = class_to_label[cls_name]

            for img_path in sorted(folder.glob("*.jpg")):
                self.samples.append(
                    {
                        "image_path": img_path,
                        "label": label,
                        "concepts": concepts.copy(),
                    }
                )

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found for split '{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]
        try:
            with Image.open(sample["image_path"]) as img:
                image = img.convert("RGB")

        except Exception as e:
            raise RuntimeError(f"Failed to load image '{sample['image_path']}'") from e

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(sample["label"], dtype=torch.long,),
            "concepts": torch.tensor(sample["concepts"], dtype=torch.float32,),
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
            f"AwA2Dataset("
            f"split={self.split}, "
            f"n={len(self)}, "
            f"classes={self.num_classes}, "
            f"concepts={self.num_concepts})"
        )

def _read_namelist(path):
    names = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)

            if len(parts) == 2:
                names.append(parts[1])
            else:
                names.append(parts[0])
    return names