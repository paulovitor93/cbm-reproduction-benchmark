from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

NUM_CLASSES = 2
NUM_CONCEPTS = 19

BENIGN = {"nevus", "blue nevus", "clark nevus", "combined nevus", "congenital nevus", "dermal nevus", "recurrent nevus",
          "reed or spitz nevus", "dermatofibroma", "lentigo", "melanosis", "miscellaneous", "vascular lesion",
          "seborrheic keratosis", "basal cell carcinoma",}

class Derm7ptDataset(Dataset):
    def __init__(self, root, split="train", transform=None):

        assert split in ["train", "val", "test"]
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.class_names = ["Benign", "Malignant",]

        meta_path = self.root / "meta" / "meta.csv"
        df = pd.read_csv(meta_path)
        df.columns = (df.columns.str.lower().str.replace(" ", "_"))

        split_files = {"train": "train_indexes.csv", "val": "valid_indexes.csv", "test": "test_indexes.csv",}

        split_df = pd.read_csv(self.root / "meta" / split_files[split])
        df = df.iloc[split_df["indexes"].to_numpy()].reset_index(drop=True)

        img_col = "derm"
        diag_col = "diagnosis" if "diagnosis" in df.columns else "dx"
     
        self.samples = []

        for _, row in df.iterrows():
            concepts = []

            # Pigment Network -> 3 concepts
            pn = str(row["pigment_network"]).strip().lower()
            concepts.append(float(pn == "absent"))
            concepts.append(float(pn == "typical"))
            concepts.append(float(pn == "atypical"))

            # Blue Whitish Veil -> 2 concepts
            bwv = str(row["blue_whitish_veil"]).strip().lower()
            concepts.append(float(bwv == "absent"))
            concepts.append(float(bwv == "present"))

            # Vascular Structures -> 3 concepts
            vs = str(row["vascular_structures"]).strip().lower()
            concepts.append(float(vs == "absent"))
            concepts.append(float(vs in ["arborizing", "comma", "hairpin", "within regression", "wreath"]))
            concepts.append(float(vs in ["dotted", "linear irregular"]))

            # Pigmentation -> 3 concepts
            pig = str(row["pigmentation"]).strip().lower()
            concepts.append(float(pig == "absent"))
            concepts.append(float(pig in ["diffuse regular", "localized regular"]))
            concepts.append(float(pig in ["diffuse irregular", "localized irregular"]))

            # Streaks -> 3 concepts
            st = str(row["streaks"]).strip().lower()
            concepts.append(float(st == "absent"))
            concepts.append(float(st == "regular"))
            concepts.append(float(st == "irregular"))

            # Dots and Globules -> 3 concepts
            dag = str(row["dots_and_globules"]).strip().lower()
            concepts.append(float(dag == "absent"))
            concepts.append(float(dag == "regular"))
            concepts.append(float(dag == "irregular"))

            # Regression Structures -> 2 concepts
            rs = str(row["regression_structures"]).strip().lower()
            concepts.append(float(rs == "absent"))
            concepts.append(float(rs in ["blue areas", "white areas", "combinations"]))

            # If diagnosis BENING --> 0 | Othersise --> 1
            assert len(concepts) == NUM_CONCEPTS
            assert sum(concepts) == 7
            label = int(str(row[diag_col]).strip().lower() not in BENIGN)

            self.samples.append({
                "image_path": self.root / "images" / row[img_col],
                "label": label,
                "concepts": np.array(concepts, dtype=np.float32,),
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
    
    @property
    def concept_names(self):
        return ["PN_Absent", "PN_Typical", "PN_Atypical",
                "BWV_Absent","BWV_Present",
                "VS_Absent", "VS_Regular", "VS_Irregular",
                "PIG_Absent", "PIG_Regular", "PIG_Irregular",
                "STR_Absent", "STR_Regular", "STR_Irregular",
                "DAG_Absent", "DAG_Regular", "DAG_Irregular",
                "RS_Absent", "RS_Present",]

    def get_class_name(self, class_id):
        return self.class_names[class_id]

    def __repr__(self):
        return (
            f"Derm7ptDataset("
            f"split={self.split}, "
            f"n={len(self)}, "
            f"classes={self.num_classes}, "
            f"concepts={self.num_concepts})"
        )