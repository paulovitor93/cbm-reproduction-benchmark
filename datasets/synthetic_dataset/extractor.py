import numpy as np
from itertools import combinations
from constants import CONCEPTS, SHAPES

class ConceptExtractor:
    def extract(self, scene):
        concepts = {}

        concepts.update(self.presence(scene))
        concepts.update(self.size(scene))
        concepts.update(self.relations(scene))

        return concepts
    
    def extract_vector(self, scene):
        concepts = self.extract(scene)

        return np.asarray([int(concepts[name]) for name in CONCEPTS], dtype=np.uint8,)
    
    def concept_names(self):
        return CONCEPTS.copy()

    def presence(self, scene):
        concepts = {}

        for shape in SHAPES:
            concepts[f"{shape}_present"] = scene.contains(shape)

        return concepts
    
    def size(self, scene):
        concepts = {}

        largest = scene.largest()

        for shape in SHAPES:
            concepts[f"largest_{shape}"] = (largest.shape == shape)

        return concepts
    
    def relations(self, scene):
        concepts = {}

        for shape1, shape2 in combinations(SHAPES, 2):
            objects1 = scene.get(shape1)
            objects2 = scene.get(shape2)
            
            if not objects1 or not objects2:
                above = False
                left = False
            else:
                # The relation is true only if every object of shape1
                # satisfies it with every object of shape2.
                above = all(scene.above(obj1, obj2) 
                            for obj1 in objects1 
                            for obj2 in objects2)
                
                left = all(scene.left_of(obj1, obj2)
                        for obj1 in objects1
                        for obj2 in objects2)
                
            concepts[f"{shape1}_above_{shape2}"] = above
            concepts[f"{shape1}_left_{shape2}"] = left

        return concepts