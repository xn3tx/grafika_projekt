import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math

def draw_3d_spring(start_y=0, end_y=10, wire_radius=0.05, spring_radius=1.0, coils=10, segments_per_coil=20, circle_resolution=8):
    height = end_y - start_y
    total_segments = coils * segments_per_coil

    for i in range(total_segments):
        t0 = i / total_segments
        t1 = (i + 1) / total_segments

        angle0 = 2 * math.pi * coils * t0
        angle1 = 2 * math.pi * coils * t1

        center0 = [ #The first and sec point of the little line we're drawing
            spring_radius * math.cos(angle0),
            start_y + t0 * height,
            spring_radius * math.sin(angle0)
        ]
        center1 = [
            spring_radius * math.cos(angle1),
            start_y + t1 * height,
            spring_radius * math.sin(angle1)
        ]

        # tangent = styczna althought technically its wersor(unit vector) my bad
        tangent = [center1[j] - center0[j] for j in range(3)]
        length = math.sqrt(sum(t ** 2 for t in tangent))
        tangent = [t / length for t in tangent]

        # wektor normalny = normal; binormal = tangent x normal from lecture https://drive.google.com/drive/folders/1RqL5Nifwung5YjaiZQz-Y2T6GUSKCUzP
        normal = [-tangent[2], 0, tangent[0]]
        binormal = [
            tangent[1] * normal[2] - tangent[2] * normal[1],
            tangent[2] * normal[0] - tangent[0] * normal[2],
            tangent[0] * normal[1] - tangent[1] * normal[0]
        ]

        def point_on_circle(center, normal, binormal, dx, dy):
            return [
                center[k] + dx * normal[k] + dy * binormal[k]
                for k in range(3)
            ]

        glBegin(GL_TRIANGLE_STRIP)
        glColor3f(0, 191, 255)  # COLOUR https://www.w3schools.com/colors/colors_converter.asp
        for j in range(circle_resolution + 1):
            theta = 2 * math.pi * j / circle_resolution
            dx = wire_radius * math.cos(theta)
            dy = wire_radius * math.sin(theta)
            p1 = point_on_circle(center0, normal, binormal, dx, dy)
            p2 = point_on_circle(center1, normal, binormal, dx, dy)
            glVertex3f(*p1)
            glVertex3f(*p2)
        glEnd()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Window TEST2")

    glEnable(GL_DEPTH_TEST) # objects closer concleal the further ones
    gluPerspective(45, display[0] / display[1], 0.1, 100.0)
    glTranslatef(0.0, -5.0, -20.0) # X, Y, Z coords for the "camera" view

    clock = pygame.time.Clock()
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glPushMatrix()
        glRotatef(pygame.time.get_ticks() * 0.05, 0, 1, 0)  # Spins = to check whether 3D is correct
        draw_3d_spring()
        glPopMatrix()

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
