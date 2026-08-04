from objects import Triangle, Circle, Square

SHAPES = ["triangle", "circle", "square",]

CONCEPTS = [

    # Presence
    "triangle_present", 
    "circle_present", 
    "square_present",

    # Largest
    "largest_triangle", 
    "largest_circle", 
    "largest_square",

    # Relations
    # Triangle + Circle
    "triangle_above_circle", 
    "triangle_left_circle", 
    
    # Triangle + Square
    "triangle_above_square", 
    "triangle_left_square",
    
    # Circle + Square
    "circle_above_square", 
    "circle_left_square",]

SHAPE_CLASSES = {
    "triangle": Triangle,
    "circle": Circle,
    "square": Square,
}