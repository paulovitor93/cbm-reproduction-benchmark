import random
import math
import numpy as np
from scene import Scene
from specification import DatasetSpecification
from constants import SHAPE_CLASSES, SHAPES
from geometry import Geometry
from objects import SceneObject

class RuleBasedSceneGenerator:
    # Image
    IMAGE_SIZE = 224

    # Number of objects
    MIN_OBJECTS = 3
    MAX_OBJECTS = 5

    # Logical object sizes
    MIN_LOGICAL_SIZE = 2.0
    MAX_LOGICAL_SIZE = 7.0
    MIN_LARGEST_SIZE = 5.0
    MIN_SIZE_DIFFERENCE = 2.2
    
    # Rendering
    MIN_PIXEL_RADIUS = IMAGE_SIZE // 15   
    MAX_PIXEL_RADIUS = IMAGE_SIZE // 6

    LAMBDA_MIN = 0.2

    def __init__(self, specification=None):
        if specification is None:
            specification = DatasetSpecification()

        self.specification = specification

    def generate(self, class_id):
        scene = Scene()

        # Number of objects
        n_objects = self.number_of_objects()

        # Shapes required by the class
        allowed_shapes = self.allowed_shapes(class_id)

        # Create objects
        objects = self.generate_objects(n_objects, allowed_shapes,)

        # Assign sizes
        largest = self.largest_shape(class_id)
        self.assign_sizes(objects, largest)

        # Assign positions
        self.assign_positions(objects, class_id,)

        # Build scene
        for obj in objects:
            scene.add(obj)

        return scene

    def number_of_objects(self):
        return random.randint(self.MIN_OBJECTS, self.MAX_OBJECTS,)
    
    def allowed_shapes(self, class_id):
        return self.specification.class_spec(class_id)["required_shapes"]
    
    def create_object(self, shape):
        Shape = SHAPE_CLASSES[shape]
        obj = Shape(x=0, y=0, size=0,)

        if shape == "triangle":
            obj.angle = random.uniform(0, 360)

        elif shape == "square":
            obj.angle = random.uniform(0, 90)

        return obj
    
    def generate_objects(self, n_objects, allowed_shapes):
        objects = []
        # One mandatory object of each allowed shape
        for shape in allowed_shapes:
            objects.append(self.create_object(shape))

        # Remaining objects
        remaining = n_objects - len(objects)

        for _ in range(remaining):
            shape = random.choice(allowed_shapes)
            objects.append(self.create_object(shape))

        random.shuffle(objects)

        return objects
    
    def largest_shape(self, class_id):
        return self.specification.class_spec(class_id)["largest_shape"]
    
    def logical_to_pixels(self, size):
        return int(np.interp(size, [self.MIN_LOGICAL_SIZE, self.MAX_LOGICAL_SIZE], [self.MIN_PIXEL_RADIUS, self.MAX_PIXEL_RADIUS],))
    
    def assign_sizes(self, objects, largest_shape):
        if largest_shape is None:
        # No constraint: every object receives a random size
            for obj in objects:
                obj.size = random.uniform(self.MIN_LOGICAL_SIZE, self.MAX_LOGICAL_SIZE,)
                obj.radius = self.logical_to_pixels(obj.size)

            return

        # Choose the largest size
        largest_size = random.uniform(self.MIN_LARGEST_SIZE, self.MAX_LOGICAL_SIZE,)
    
        # All remaining objects must be smaller
        max_other = largest_size - self.MIN_SIZE_DIFFERENCE
        
        # Choose which object will be the reference object
        candidates = [obj for obj in objects if obj.shape == largest_shape]
        reference_object = random.choice(candidates)

        # Assign sizes
        reference_object.size = largest_size
        reference_object.radius = self.logical_to_pixels(largest_size)

        for obj in objects:
            if obj is reference_object:
                continue

            obj.size = random.uniform(self.MIN_LOGICAL_SIZE, max_other,)
            obj.radius = self.logical_to_pixels(obj.size)

    def class_relations(self, class_id):
        spec = self.specification.class_spec(class_id)

        if "reference_shape" not in spec:
            return None

        return {"reference_shape": spec["reference_shape"], 
                "target_shape": spec["target_shape"], 
                **spec["layout"],}
        
    def sample_position(self, region, radius):
        """
        Uniformly samples one position inside a feasible region.
        """

        xmin = int(region["xmin"] + radius)
        xmax = int(region["xmax"] - radius)

        ymin = int(region["ymin"] + radius)
        ymax = int(region["ymax"] - radius)

        if xmin > xmax or ymin > ymax:
            return None

        x = random.randint(xmin, xmax)
        y = random.randint(ymin, ymax)

        return x, y

    def valid_position(self, obj, x, y, occupied_objects,):
        """
        Checks whether an object can be placed at (x,y).
        Only overlap constraints are verified here.
        """
        candidate = SceneObject(obj.shape, x, y, obj.size, radius=obj.radius,)
        r1 = self.effective_radius(obj)

        for other in occupied_objects:
            r2 = self.effective_radius(other)
            distance = Geometry.distance(candidate, other)

            # λ = d / (r1+r2)
            lam = distance / (r1 + r2)

            if lam < self.LAMBDA_MIN:
                return False

        return True

    def place_objects(self, objects, region, occupied_objects=None,):
        """
        Places a list of objects uniformly inside a feasible region.
        """

        if occupied_objects is None:
            occupied_objects = []

        # Objects already placed in this call
        placed = []

        for obj in objects:
            success = False
            # Try random locations
            for _ in range(200):
                position = self.sample_position(region, self.effective_radius(obj),)

                if position is None:
                    break

                x, y = position

                # Check against previously placed objects
                if not self.valid_position(obj, x, y, occupied_objects + placed,):
                    continue

                obj.x = x
                obj.y = y

                placed.append(obj)
                success = True
                break

            if not success:
                raise RuntimeError(f"Unable to place {obj.shape} inside feasible region.")

        return placed

    def compute_feasible_region(self, reference_objects, relations):
        """
        Computes the feasible region using only the centers of the
        reference objects.
        """

        xs = [obj.x for obj in reference_objects]
        ys = [obj.y for obj in reference_objects]

        region = {
            "xmin": 0,
            "xmax": self.IMAGE_SIZE,
            "ymin": 0,
            "ymax": self.IMAGE_SIZE,
        }

        # Horizontal relation
        if relations["left"]:
            # target must be to the right
            region["xmin"] = max(xs)

        else:
            # target must be to the left
            region["xmax"] = min(xs)

        # Vertical relation
        if relations["above"]:
            # target must be below
            region["ymin"] = max(ys)

        else:
            # target must be above
            region["ymax"] = min(ys)

        return region

    def assign_positions(self, objects, class_id):
        """
        Assign positions according to the spatial constraints of the class.
        """
        MAX_ATTEMPTS = 1000

        whole_region = {
            "xmin": 0,
            "xmax": self.IMAGE_SIZE,
            "ymin": 0,
            "ymax": self.IMAGE_SIZE,
        }

        for _ in range(MAX_ATTEMPTS):
            # Reset positions
            for obj in objects:
                obj.x = 0
                obj.y = 0

            try:
                relations = self.class_relations(class_id)

                # No positional constraints
                if relations is None:
                    self.place_objects(objects, whole_region)
                    return

                reference_shape = relations["reference_shape"]
                target_shape = relations["target_shape"]

                reference_objects = [
                    obj for obj in objects
                    if obj.shape == reference_shape
                ]

                target_objects = [
                    obj for obj in objects
                    if obj.shape == target_shape
                ]

                other_objects = [
                    obj for obj in objects
                    if obj.shape not in (reference_shape, target_shape)
                ]

                # Place reference objects
                placed_reference = self.place_objects(reference_objects, whole_region,)

                # Compute feasible region
                target_region = self.compute_feasible_region(placed_reference, relations,)

                # Place target objects
                placed_all = self.place_objects(target_objects, target_region, occupied_objects=placed_reference,)

                # Place remaining objects
                if other_objects:
                    self.place_objects(other_objects, whole_region, occupied_objects=placed_all,)

                return

            except RuntimeError:
                continue

        raise RuntimeError(f"Could not generate a valid layout after {MAX_ATTEMPTS} attempts.")
       
    def effective_radius(self, obj):
        if obj.shape == "circle":
            return obj.radius

        elif obj.shape == "square":
            return int(math.ceil(obj.radius * math.sqrt(2)))

        elif obj.shape == "triangle":
            return int(math.ceil(obj.radius * math.sqrt(2)))
        
        return obj.radius

    def expected_concepts(self, class_id):
        return self.specification.expected_concepts(class_id)

    @property
    def class_specs(self):
        return self.specification.CLASS_SPECS