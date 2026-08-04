from constants import SHAPES, CONCEPTS

class DatasetSpecification:
    def __init__(self):
        self.CLASS_SPECS = {
            # Single-shape classes
            0: {
                "required_shapes": ["triangle"],
                "largest_shape": "triangle",
            },

            1: {
                "required_shapes": ["circle"],
                "largest_shape": "circle",
            },

            2: {
                "required_shapes": ["square"],
                "largest_shape": "square",
            },

            # Presence only
            3: {
                "required_shapes": ["triangle", "circle", "square"],
                "largest_shape": None,
            },
        }

        # Relation classes
        self.CLASS_SPECS.update(self.relation_classes(4, "circle", "square"))
        self.CLASS_SPECS.update(self.relation_classes(12, "triangle", "circle"))
        self.CLASS_SPECS.update(self.relation_classes(20, "triangle", "square"))

    def relation_classes(self, start_id, reference_shape, target_shape):
        """
        Generate the 8 classes corresponding to one pair of shapes.
        """
        specs = {}
        class_id = start_id

        for largest_shape in [reference_shape, target_shape]:
            for above in [True, False]:
                for left in [True, False]:
                    specs[class_id] = {
                        "required_shapes": [reference_shape, target_shape,],
                        "largest_shape": largest_shape,
                        "reference_shape": reference_shape,
                        "target_shape": target_shape,
                        "layout": {
                            "above": above,
                            "left": left,
                        },
                    }

                    class_id += 1

        return specs

    def class_spec(self, class_id):
        if class_id not in self.CLASS_SPECS:
            raise ValueError(f"Unknown class {class_id}")

        return self.CLASS_SPECS[class_id]

    def expected_concepts(self, class_id):
        spec = self.class_spec(class_id)
        concepts = {concept: None for concept in CONCEPTS}

        # Presence
        for shape in SHAPES:
            concepts[f"{shape}_present"] = False

        for shape in spec["required_shapes"]:
            concepts[f"{shape}_present"] = True

        # Largest
        largest = spec["largest_shape"]

        if largest is not None:
            for shape in SHAPES:
                concepts[f"largest_{shape}"] = False

            concepts[f"largest_{largest}"] = True

        # Spatial relations
        if "reference_shape" in spec:

            reference = spec["reference_shape"]
            target = spec["target_shape"]

            concepts[f"{reference}_above_{target}"] = spec["layout"]["above"]
            concepts[f"{reference}_left_{target}"] = spec["layout"]["left"]

        return concepts