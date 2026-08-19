from pathlib import Path
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import json
import resvg_py

class DatasetBuilder:
    def __init__(self, generator, renderer, extractor):
        self.generator = generator
        self.renderer = renderer
        self.extractor = extractor

    def build(self, output_dir, samples_per_class, generation_seed,):
        output_dir = Path(output_dir)

        random.seed(generation_seed)
        np.random.seed(generation_seed)

        images_dir = output_dir / "images"
        metadata_dir = output_dir / "metadata"
        scenes_dir = output_dir / "scenes"
        svg_dir = output_dir / "svg"

        images_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        scenes_dir.mkdir(parents=True, exist_ok=True)
        svg_dir.mkdir(parents=True, exist_ok=True)

        rows = []

        image_id = 0

        num_classes = len(self.generator.specification.CLASS_SPECS)

        for class_id in range(num_classes):
            print(f"Generating class {class_id}")

            for _ in tqdm(range(samples_per_class)):
                scene = self.generator.generate(class_id)

                image_name = f"{image_id:06d}.png"
                svg_name = f"{image_id:06d}.svg"
                scene_name = f"{image_id:06d}.json"

                self.save_scene(scene, class_id, image_name, scenes_dir / scene_name,)

                concepts = self.extractor.extract(scene)

                svg_path = svg_dir / svg_name
                png_path = images_dir / image_name

                self.renderer.render(scene, svg_path,)

                png_bytes = resvg_py.svg_to_bytes(
                    svg_path=str(svg_path),
                    width=self.renderer.image_size,
                    height=self.renderer.image_size,
                )

                with open(png_path, "wb") as f:
                    f.write(bytes(png_bytes))

                row = {
                    "image_id": image_id,
                    "image_name": image_name,
                    "class": class_id,
                }

                row.update(concepts)
                rows.append(row)
                image_id += 1

        df = pd.DataFrame(rows)
        metadata_file = metadata_dir  / "metadata.csv"
        df.to_csv(metadata_file, index=False)

        info_file = output_dir / "dataset_info.txt"
        with open(info_file, "w") as f:
            f.write("Synthetic Dataset\n")
            f.write("=================\n\n")
            f.write(f"Generation seed: {generation_seed}\n")
            f.write(f"Classes: {num_classes}\n")
            f.write(f"Samples per class: {samples_per_class}\n")
            f.write(f"Total images: {len(df)}\n")
            f.write(f"Image size: {self.renderer.image_size} x {self.renderer.image_size}\n")
            f.write("Image format: PNG\n")
            f.write("Master format: SVG\n")

        print()
        print(f"Generation seed: {generation_seed}")
        print(f"Dataset saved to: {output_dir}")
        print(f"Images : {len(df)}")
        print(f"Classes: {num_classes}")
        print(f"Metadata: {metadata_file}")

    def save_scene(self, scene, class_id, image_name, filepath):
        data = {
            "class": class_id,
            "image_name": image_name,
            "objects": []
        }

        for obj in scene:
            data["objects"].append({
                "shape": obj.shape,
                "x": obj.x,
                "y": obj.y,
                "size": obj.size,
                "angle": obj.angle,
            })

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)