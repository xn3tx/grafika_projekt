from drawscene import Drawing

class Spring:
    def __init__(self):
        self.k = 200.0

    def draw_spring(self, attach_x, wheel_y, desk_y):
        spring_top = desk_y + 20
        Drawing.draw_line(attach_x, spring_top, attach_x, wheel_y, (0, 255, 0), 4)