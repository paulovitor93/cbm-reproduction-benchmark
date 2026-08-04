import random
import numpy as np
import matplotlib.pyplot as plt

from generator import RuleBasedSceneGenerator
from extractor import ConceptExtractor
from classifier import RuleBasedClassifier
from validator import SceneValidator
from renderer_png import SceneRenderer
from constants import CONCEPTS

# Configuration
SEED = 3938019914      
SAMPLES_PER_CLASS = 1000

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Components
generator = RuleBasedSceneGenerator()
extractor = ConceptExtractor()
classifier = RuleBasedClassifier(generator.specification)
validator = SceneValidator(extractor=extractor, classifier=classifier,)
renderer = SceneRenderer()

# Validation
NUM_CLASSES = len(generator.class_specs)

print(f"Validation seed: {SEED}")
print(f"Classes: {NUM_CLASSES}")
print(f"Samples/class: {SAMPLES_PER_CLASS}\n")

for class_id in range(NUM_CLASSES):
    print(f"Testing class {class_id}")

    for sample_id in range(SAMPLES_PER_CLASS):
        scene = generator.generate(class_id)

        valid, predicted = validator.validate(scene, expected_class=class_id,)

        if valid:
            continue

        print("\n==========================================")
        print("VALIDATION FAILED")
        print("==========================================")
        print(f"Seed      : {SEED}")
        print(f"Class     : {class_id}")
        print(f"Sample    : {sample_id}")
        print(f"Expected  : {class_id}")
        print(f"Predicted : {predicted}")

        print("\nObjects")

        for obj in scene:
            print(
                f"{obj.shape:8s} "
                f"x={obj.x:3d} "
                f"y={obj.y:3d} "
                f"radius={obj.radius:2d}"
            )

        expected = generator.expected_concepts(class_id)
        observed = extractor.extract(scene)

        print("\nExpected concepts")

        for concept in CONCEPTS:
            print(f"{concept:30} {expected[concept]}")

        print("\nObserved concepts")

        for concept in CONCEPTS:
            print(f"{concept:30} {observed[concept]}")

        image = renderer.render(scene)

        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Seed {SEED}\nExpected {class_id} | Predicted {predicted}")
        plt.show()

        raise RuntimeError(f"Validation failed for class {class_id}, sample {sample_id}.")

    print(f"✓ {SAMPLES_PER_CLASS}/{SAMPLES_PER_CLASS}")

print("\n==========================================")
print("ALL CLASSES PASSED VALIDATION!")
print("==========================================")
print(f"Seed: {SEED}")