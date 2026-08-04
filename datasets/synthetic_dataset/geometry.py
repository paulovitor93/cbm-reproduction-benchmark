import math

class Geometry:
    """
    Makes the computations for the geometric relations
    """
    @staticmethod
    # Euclidean distance from the center of the object
    def distance(obj1, obj2):
        return math.sqrt((obj1.x - obj2.x) ** 2 + (obj1.y - obj2.y) ** 2)

    @staticmethod
    def left_of(obj1, obj2):
        return obj1.x < obj2.x

    @staticmethod
    def above(obj1, obj2):
        return obj1.y < obj2.y

    @staticmethod
    def largest(objects):
        return max(objects, key=lambda obj: obj.size,)