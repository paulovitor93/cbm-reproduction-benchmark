class RuleBasedClassifier:
    """
    Assigns a class by comparing the observed concepts against
    the expected concept specification for each class.
    """
    def __init__(self, specification):
        self.specification = specification

    def matches(self, observed, expected):
        for concept, expected_value in expected.items():

            # Ignore unconstrained concepts
            if expected_value is None:
                continue

            if observed[concept] != expected_value:
                return False

        return True

    def predict(self, observed, verbose=False):
        for class_id in self.specification.CLASS_SPECS:
            expected = self.specification.expected_concepts(class_id)

            if verbose:
                print(f"\nTesting class {class_id}")

            if self.matches(observed, expected):
                if verbose:
                    print(f"Matched class {class_id}")

                return class_id
            
            elif verbose:
                print("No match")

        raise ValueError("No class matches the observed concepts.")