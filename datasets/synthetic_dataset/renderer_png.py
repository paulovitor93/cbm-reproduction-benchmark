import random
import math
from PIL import Image, ImageDraw

class SceneRenderer:

    BACKGROUND = (255, 255, 255)

    def __init__(self, image_size=224):
        self.image_size = image_size

    def random_color(self):
        while True:

            # RGB
            color = (random.randint(30, 220), random.randint(30, 220), random.randint(30, 220),)

            # Reject colors too close to white
            if sum(color) < 550:
                return color
            
    def render(self, scene):
        image = Image.new("RGB", (self.image_size, self.image_size), self.BACKGROUND,)
        draw = ImageDraw.Draw(image)

        for obj in scene:
            color = self.random_color()

            if obj.shape == "circle":
                self.draw_circle(draw, obj, color)
            elif obj.shape == "square":
                self.draw_square(draw, obj, color)
            elif obj.shape == "triangle":
                self.draw_triangle(draw, obj, color)

        return image

    def rotate_points(self, points, center, angle):
        angle = math.radians(angle)
        cx, cy = center

        rotated = []

        for x, y in points:
            x -= cx
            y -= cy

            xr = x * math.cos(angle) - y * math.sin(angle)
            yr = x * math.sin(angle) + y * math.cos(angle)

            rotated.append((xr + cx, yr + cy))
        return rotated
    
    def draw_circle(self, draw, obj, color):
        r = obj.radius
        draw.ellipse((obj.x - r, obj.y - r, obj.x + r, obj.y + r,), fill=color,)

    def draw_square(self, draw, obj, color):
        r = obj.radius
        points = [(obj.x-r, obj.y-r), (obj.x+r, obj.y-r), (obj.x+r, obj.y+r), (obj.x-r, obj.y+r),]
        points = self.rotate_points(points, (obj.x, obj.y), obj.angle,)

        draw.polygon(points, fill=color,)
    
    def draw_triangle(self, draw, obj, color):
        r = obj.radius
        points = [(obj.x, obj.y-r), (obj.x-r, obj.y+r), (obj.x+r, obj.y+r),]
        points = self.rotate_points(points, (obj.x, obj.y), obj.angle,)

        draw.polygon(points, fill=color,)