from geometry import Geometry

class Scene:
    """
    Stores the objects composing the scene
    """
    # Object management
    def __init__(self):
        self.objects = []

    def add(self, obj):
        self.objects.append(obj)

    # Returns all objects with a specific shape
    def get(self, shape):
        return [obj for obj in self.objects if obj.shape == shape]

    # True if at least one object of that shape exists
    def contains(self, shape):
        return len(self.get(shape)) > 0
       
    # Geometry wrappers
    def left_of(self, obj1, obj2):
        return Geometry.left_of(obj1, obj2)

    def above(self, obj1, obj2):
        return Geometry.above(obj1, obj2)

    def largest(self):
        return Geometry.largest(self.objects)
        
    # Utilities
    def __len__(self):
        return len(self.objects)

    def __iter__(self):
        return iter(self.objects)
    
    def __getitem__(self, index):
        return self.objects[index]

    def __repr__(self):
        return f"Scene({len(self.objects)} objects)"