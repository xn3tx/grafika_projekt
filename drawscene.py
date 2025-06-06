from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

class Drawing:
    def __init__(self, spring, attenuator, wheel):
        self.width = 800
        self.height = 600
        self.fps = 60
        self.spring = spring
        self.attenuator = attenuator
        self.wheel = wheel

    @staticmethod   #umożliwia wywołanie bez tworzenia obiektu
    def set_2d_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, self.width, self.height, 0)  # układ współrzędnych zgodny z Pygame
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    @staticmethod
    def draw_rect(x, y, w, h, color):
        glColor3f(*[c / 255.0 for c in color])
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

    @staticmethod
    def draw_line(x1, y1, x2, y2, color, width=1):
        glLineWidth(width)
        glColor3f(*[c / 255.0 for c in color])
        glBegin(GL_LINES)
        glVertex2f(x1, y1)
        glVertex2f(x2, y2)
        glEnd()

    def draw_scene(self, y_wheel, y_desk):
        glClear(GL_COLOR_BUFFER_BIT)
        center_x = self.width // 2
        platform_y = self.height // 2 + int(y_wheel)
        desk_y = int(y_desk)
        ground_y = self.height - 40

        # Podłoże
        Drawing.draw_line(0, ground_y, self.width, ground_y, (80, 80, 80), 6)

        # Platforma
        self.wheel.draw_platform(center_x, platform_y, ground_y)

        # Koło
        wheel_bottom_y = platform_y
        self.wheel.draw_wheel(center_x, wheel_bottom_y)
        wheel_axis_y = wheel_bottom_y - self.wheel.height // 2

        # Podwozie
        beam_start_x = center_x + self.wheel.width // 2
        beam_end_x = beam_start_x + 100
        Drawing.draw_line(beam_start_x, wheel_axis_y, beam_end_x, wheel_axis_y, (150, 150, 150), 8)
        Drawing.draw_rect(center_x + 50, desk_y, 200, 20, (180, 180, 180))

        # Sprężyna i tłumik
        spring_x = beam_end_x - 60
        attenuator_x = beam_end_x + 10
        self.spring.draw_spring(spring_x, wheel_axis_y, desk_y)
        self.attenuator.draw_attenuator(attenuator_x, wheel_axis_y, desk_y)