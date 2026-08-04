import random
import math
import svgwrite

class SceneRendererSVG:
    BACKGROUND = "white"

    def __init__(self, image_size=224):
        self.image_size = image_size

    def random_color(self):
        while True:

            color = (random.randint(30,220), random.randint(30,220), random.randint(30,220),)

            if sum(color) < 550:
                return color

    def svg_color(self, color):
        return svgwrite.rgb(*color)

    def render(self, scene, filename):
        dwg = svgwrite.Drawing(str(filename), size=(self.image_size, self.image_size))
        dwg.add(dwg.rect(insert=(0,0), size=(self.image_size,self.image_size), fill=self.BACKGROUND))

        for obj in scene:
            color = self.svg_color(self.random_color())

            if obj.shape == "circle":
                self.draw_circle(dwg,obj,color)

            elif obj.shape == "square":
                self.draw_square(dwg,obj,color)

            elif obj.shape == "triangle":
                self.draw_triangle(dwg,obj,color)

        dwg.save()

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

    def draw_circle(self, dwg, obj, color):
        dwg.add(dwg.circle(center=(obj.x, obj.y),r=obj.radius,fill=color,))
    
    def draw_square(self, dwg, obj, color):
        r = obj.radius

        points = [(obj.x-r, obj.y-r), (obj.x+r, obj.y-r), (obj.x+r, obj.y+r), (obj.x-r, obj.y+r),]
        points = self.rotate_points(points, (obj.x, obj.y), obj.angle,)

        dwg.add(dwg.polygon(points=points, fill=color,))

    def draw_triangle(self, dwg, obj, color):
        r = obj.radius

        points = [(obj.x, obj.y-r), (obj.x-r, obj.y+r), (obj.x+r, obj.y+r),]
        points = self.rotate_points(points, (obj.x, obj.y), obj.angle,)

        dwg.add(dwg.polygon(points=points, fill=color,))