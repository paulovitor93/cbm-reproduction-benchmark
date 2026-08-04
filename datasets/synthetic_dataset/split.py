from pathlib import Path
import pandas as pd
import numpy as np
import random

class DatasetSplitter:
    """
    Creates reproducible train/validation/test splits.

    The split is stratified by class so that every class has the same
    proportion of samples in each subset.
    """

    def __init__(self):
        self.seed = random.randint(0, 2**32 - 1)

    def split(self, metadata_csv, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,):
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
            raise ValueError("Split ratios must sum to 1.")

        metadata_csv = Path(metadata_csv)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(metadata_csv)
        rng = np.random.default_rng(self.seed)

        train_rows = []
        val_rows = []
        test_rows = []

        for class_id in sorted(df["class"].unique()):
            class_df = df[df["class"] == class_id].copy()

            indices = class_df.index.to_numpy(copy=True)

            rng.shuffle(indices)
            n = len(indices)

            n_train = int(train_ratio * n)
            n_val = int(val_ratio * n)

            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]

            train_rows.append(df.loc[train_idx])
            val_rows.append(df.loc[val_idx])
            test_rows.append(df.loc[test_idx])

        train_df = (pd.concat(train_rows).sort_values("image_id").reset_index(drop=True))
        val_df = (pd.concat(val_rows).sort_values("image_id").reset_index(drop=True))
        test_df = (pd.concat(test_rows).sort_values("image_id").reset_index(drop=True))

        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "val.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

        split_info = output_dir / "split_info.txt"

        with open(split_info, "w") as f:
            f.write("Dataset Split Information\n")
            f.write("=========================\n\n")

            f.write(f"Split seed: {self.seed}\n\n")

            f.write(f"Train ratio: {train_ratio}\n")
            f.write(f"Validation ratio: {val_ratio}\n")
            f.write(f"Test ratio: {test_ratio}\n\n")

            f.write(f"Train samples: {len(train_df)}\n")
            f.write(f"Validation samples: {len(val_df)}\n")
            f.write(f"Test samples: {len(test_df)}\n\n")

            f.write("Per-class distribution\n")
            f.write("----------------------\n")

            for class_id in sorted(df["class"].unique()):

                train_count = (train_df["class"] == class_id).sum()
                val_count = (val_df["class"] == class_id).sum()
                test_count = (test_df["class"] == class_id).sum()

                f.write(
                    f"Class {class_id:02d}: "
                    f"Train={train_count}, "
                    f"Validation={val_count}, "
                    f"Test={test_count}\n"
                )

        print(f"Split seed: {self.seed}")
        print(f"Train: {len(train_df)}")
        print(f"Validation: {len(val_df)}")
        print(f"Test: {len(test_df)}")