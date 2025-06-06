from drawscene import Drawing

class Attenuator:
    def __init__(self):
        self.b = 20.0

    def draw_attenuator(self, attach_x, wheel_y, desk_y):
        spring_top = desk_y + 20
        attenuator_y = (spring_top + wheel_y) // 2
        Drawing.draw_rect(attach_x - 10, attenuator_y - 20, 20, 40, (255, 165, 0))