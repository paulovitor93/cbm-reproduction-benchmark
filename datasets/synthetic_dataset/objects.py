from dataclasses import dataclass

@dataclass
class SceneObject:
    shape: str
    x: int
    y: int

    # Logical size
    size: float

    # Pixel radius used only for rendering
    radius: int = 0

    # Rotation
    angle: float = 0.0

# Shape classes
class Triangle(SceneObject):
    def __init__(self, x, y, size):
        super().__init__(shape="triangle", x=x, y=y, size=size,)

class Circle(SceneObject):
    def __init__(self, x, y, size):
        super().__init__(shape="circle", x=x, y=y, size=size,)       

class Square(SceneObject):
    def __init__(self, x, y, size):
        super().__init__(shape="square", x=x, y=y, size=size,)         