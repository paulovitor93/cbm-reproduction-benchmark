import pandas as pd
import numpy as np

TASK_LABEL = "Male"
# percentage
THRESHOLD = 15
ROOT = "data/celeba/"
subjective_concepts = ["Attractive", "Young", "Chubby", "Blurry", "Smiling",]
semantic_whitelist = ["Mustache", "Sideburns"]

# Load CelebA
df = pd.read_csv("data/celeba/list_attr_celeba.txt", sep=r"\s+", skiprows=1, index_col=0,)

# Convert {-1,1} -> {0,1}
df = df.replace(-1, 0)

# Percentage of each concept by gender
proportion = df.groupby(TASK_LABEL).mean() * 100
proportion.index = ["Female (%)", "Male (%)"]
proportion = proportion.T

# Difference between Male and Female
proportion["Absolute Difference"] = (proportion["Male (%)"] - proportion["Female (%)"]).abs()

# Sort by importance
proportion = proportion.sort_values("Absolute Difference", ascending=False)

# Removing  subjective concepts
objective = proportion[~proportion.index.isin(subjective_concepts)].copy()

# Applying treshold to select only the values with absolute difference >= 15
selected = objective[(objective["Absolute Difference"] >= THRESHOLD) | (objective.index.isin(semantic_whitelist))].copy()
removed = proportion.loc[~proportion.index.isin(selected.index)].copy()

# Associating Male and Female Concepts
selected["Association"] = np.where(selected["Male (%)"] > selected["Female (%)"], "Male", "Female",)
male_count = (selected["Association"] == "Male").sum()
female_count = (selected["Association"] == "Female").sum()

# generating final attribute list
final_concepts = selected.index.tolist()
cbm_df = df[final_concepts + [TASK_LABEL]].copy()

output_file = ROOT + "celeba_gender_cbm_dataset.csv"

print("=" * 70)
print("FINAL CBM DATASET")
print("=" * 70)
print(f"Images            : {len(cbm_df)}")
print(f"Selected concepts : {len(final_concepts)}")
print(f"Task label        : {TASK_LABEL}")
print()

for i, concept in enumerate(final_concepts, 1):
    print(f"{i:2d}. {concept}")

cbm_df.index.name = "image_id"
selected.to_csv(ROOT + "selected_concepts.csv")
removed.to_csv(ROOT + "removed_concepts.csv")
cbm_df.to_csv(output_file, index=True)

print(f"File saved at: {output_file}")
