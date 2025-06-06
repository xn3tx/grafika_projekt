<<<<<<< HEAD
from drawscene import Drawing

class Attenuator:
    def __init__(self):
        self.b = 20.0

    def draw_attenuator(self, attach_x, wheel_y, desk_y):
        spring_top = desk_y + 20
        attenuator_y = (spring_top + wheel_y) // 2
=======
from drawscene import Drawing

class Attenuator:
    def __init__(self):
        self.b = 20.0

    def draw_attenuator(self, attach_x, wheel_y, desk_y):
        spring_top = desk_y + 20
        attenuator_y = (spring_top + wheel_y) // 2
>>>>>>> 92664a21a74fbb884fe937d3ed18307e76553ee4
        Drawing.draw_rect(attach_x - 10, attenuator_y - 20, 20, 40, (255, 165, 0))