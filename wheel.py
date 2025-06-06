<<<<<<< HEAD
from drawscene import Drawing

class Wheel:
    def __init__(self, width=60, height=100):
        self.width = width
        self.height = height

    def draw_wheel(self, center_x, bottom_y):
        Drawing.draw_rect(center_x - self.width // 2, bottom_y - self.height, self.width, self.height, (40, 40, 40))
        # Bieżnik opony - poziome linie
        for y in range(bottom_y - self.height + 5, bottom_y, 10):
            Drawing.draw_line(center_x - self.width // 2, y, center_x + self.width // 2, y, (100, 100, 100), 2)

    def draw_platform(self, center_x, platform_y, ground_y):
        Drawing.draw_line(center_x, platform_y + 20, center_x, ground_y, (180, 180, 180), 12)
=======
from drawscene import Drawing

class Wheel:
    def __init__(self, width=60, height=100):
        self.width = width
        self.height = height

    def draw_wheel(self, center_x, bottom_y):
        Drawing.draw_rect(center_x - self.width // 2, bottom_y - self.height, self.width, self.height, (40, 40, 40))
        # Bieżnik opony - poziome linie
        for y in range(bottom_y - self.height + 5, bottom_y, 10):
            Drawing.draw_line(center_x - self.width // 2, y, center_x + self.width // 2, y, (100, 100, 100), 2)

    def draw_platform(self, center_x, platform_y, ground_y):
        Drawing.draw_line(center_x, platform_y + 20, center_x, ground_y, (180, 180, 180), 12)
>>>>>>> 92664a21a74fbb884fe937d3ed18307e76553ee4
        Drawing.draw_rect(center_x - 150, platform_y, 300, 20, (100, 100, 255))