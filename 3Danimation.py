import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import time


# External dynamic model
class Spring:
    def __init__(self, k):
        self.k = k

class Attenuator:
    def __init__(self, b):
        self.b = b

class InputOutputFunction:
    def __init__(self, input_type, wheel, spring, attenuator):
        self.wheel = wheel
        self.spring = spring
        self.attenuator = attenuator
        self.input_type = input_type
        self.dt = 0.01
        self.frequency = 1.0
        self.amplitude = 1.0
        self.phase = 0.0
        self.mass = 1.0

    def input_function(self, t):
        if self.input_type == "sine":
            return self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)

    def euler_output(self, t):
        dt = self.dt
        k = self.spring.k
        b = self.attenuator.b
        m = self.mass
        u = self.input_function(t)
        N = len(t)
        y = np.zeros(N)
        dy1 = np.zeros(N)
        dy2 = np.zeros(N)
        for i in range(2, N):
            dy2[i] = (-k*y[i-1] - b*dy1[i-1] + u[i]) / m
            dy1[i] = dy1[i-1] + dt * dy2[i]
            y[i] = y[i-1] + dt * dy1[i]
        return y

    def simulate_live(self, total_time=10.0):
        dt = self.dt
        t = np.arange(0, total_time, dt)
        return self.euler_output(t)

# OpenGL rendering
def draw_3d_spring(start_y, end_y, wire_radius=0.05, spring_radius=0.5,
                   coils=10, segments_per_coil=20, circle_resolution=8):
    height = end_y - start_y
    total_segments = coils * segments_per_coil

    for i in range(total_segments):
        t0 = i / total_segments
        t1 = (i + 1) / total_segments

        angle0 = 2 * math.pi * coils * t0
        angle1 = 2 * math.pi * coils * t1

        center0 = [
            spring_radius * math.cos(angle0),
            start_y + t0 * height,
            spring_radius * math.sin(angle0)
        ]
        center1 = [
            spring_radius * math.cos(angle1),
            start_y + t1 * height,
            spring_radius * math.sin(angle1)
        ]

        tangent = [center1[j] - center0[j] for j in range(3)]
        length = math.sqrt(sum(t ** 2 for t in tangent))
        tangent = [t / length for t in tangent]

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
        glColor3f(140, 140, 140)
        for j in range(circle_resolution + 1):
            theta = 2 * math.pi * j / circle_resolution
            dx = wire_radius * math.cos(theta)
            dy = wire_radius * math.sin(theta)
            p1 = point_on_circle(center0, normal, binormal, dx, dy)
            p2 = point_on_circle(center1, normal, binormal, dx, dy)
            glVertex3f(*p1)
            glVertex3f(*p2)
        glEnd()
        
def draw_cylinder(y1, y2, radius):
    segments = 20
    glBegin(GL_QUAD_STRIP)
    for i in range(segments + 1):
        theta = 2.0 * math.pi * i / segments
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
        glVertex3f(x, y1, z)
        glVertex3f(x, y2, z)
    glEnd()
  
def damper(start_y, end_y, damper_radius=0.5, rod_radius=0.1, body_height=2.0):
    # Draw the rod (full height)
    glColor3f(0.7, 0.7, 0.7)  # light gray
    draw_cylinder(start_y, end_y, rod_radius)

    # Draw damper body (centered between start_y and end_y)
    center_y = (start_y + end_y) / 2
    body_start = center_y - body_height / 2
    body_end = center_y + body_height / 2

    glColor3f(1.0, 0.0, 0.0)  # red
    draw_cylinder(body_start, body_end, damper_radius)

def draw_rod(start_y, rod_length=4.0, rod_radius=0.25, segments=32):

    half_length = rod_length / 2.0
    glColor3f(0.4, 0.4, 0.4)  # reddish brown

    # Side surface
    glBegin(GL_QUAD_STRIP)
    for i in range(segments + 1):
        theta = 2 * np.pi * i / segments
        x = np.cos(theta)
        y = np.sin(theta)
        glVertex3f(-half_length, start_y + rod_radius * y, rod_radius * x)
        glVertex3f(half_length, start_y + rod_radius * y, rod_radius * x)
    glEnd()

    # Left cap
    glBegin(GL_POLYGON)
    for i in range(segments):
        theta = 2 * np.pi * i / segments
        x = np.cos(theta)
        y = np.sin(theta)
        glVertex3f(-half_length, start_y + rod_radius * y, rod_radius * x)
    glEnd()

    # Right cap
    glBegin(GL_POLYGON)
    for i in range(segments):
        theta = 2 * np.pi * i / segments
        x = np.cos(theta)
        y = np.sin(theta)
        glVertex3f(half_length, start_y + rod_radius * y, rod_radius * x)
    glEnd()


def draw_platform(y, width=3.0, depth=3.0, thickness=0.5):
    hw = width / 2
    hd = depth / 2
    h = thickness

    glColor3f(0.0, 0.0, 1.0)  # green
    glBegin(GL_QUADS)

    # Top face
    glVertex3f(-hw, y + h, -hd)
    glVertex3f(hw, y + h, -hd)
    glVertex3f(hw, y + h, hd)
    glVertex3f(-hw, y + h, hd)

    # Bottom face
    glVertex3f(-hw, y, -hd)
    glVertex3f(hw, y, -hd)
    glVertex3f(hw, y, hd)
    glVertex3f(-hw, y, hd)

    # Front face
    glVertex3f(-hw, y, hd)
    glVertex3f(hw, y, hd)
    glVertex3f(hw, y + h, hd)
    glVertex3f(-hw, y + h, hd)

    # Back face
    glVertex3f(-hw, y, -hd)
    glVertex3f(hw, y, -hd)
    glVertex3f(hw, y + h, -hd)
    glVertex3f(-hw, y + h, -hd)

    # Left face
    glVertex3f(-hw, y, -hd)
    glVertex3f(-hw, y, hd)
    glVertex3f(-hw, y + h, hd)
    glVertex3f(-hw, y + h, -hd)

    # Right face
    glVertex3f(hw, y, -hd)
    glVertex3f(hw, y, hd)
    glVertex3f(hw, y + h, hd)
    glVertex3f(hw, y + h, -hd)

    glEnd()


def draw_wheel(start_y, wheel_radius, wheel_thickness, colorR,colorG,colorB,rod_length=4.0, segments=32):

    center_x = -rod_length / 2.0  # Left end of the rod

    # Side surface (wheel body)
    glColor3f(colorR,colorG,colorB)  # dark gray
    glBegin(GL_QUAD_STRIP)
    for i in range(segments + 1):
        theta = 2 * np.pi * i / segments
        x = wheel_radius * np.cos(theta)
        y = wheel_radius * np.sin(theta)
        glVertex3f(center_x - wheel_thickness/2, start_y + y, x)
        glVertex3f(center_x + wheel_thickness/2, start_y + y, x)
    glEnd()

    # Left cap
    glBegin(GL_POLYGON)
    for i in range(segments):
        theta = 2 * np.pi * i / segments
        x = wheel_radius * np.cos(theta)
        y = wheel_radius * np.sin(theta)
        glVertex3f(center_x - wheel_thickness/2, start_y + y, x)
    glEnd()

    # Right cap
    glBegin(GL_POLYGON)
    for i in range(segments):
        theta = 2 * np.pi * i / segments
        x = wheel_radius * np.cos(theta)
        y = wheel_radius * np.sin(theta)
        glVertex3f(center_x + wheel_thickness/2, start_y + y, x)
    glEnd()


# Main app
def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D Spring Dynamic Simulation")

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, display[0] / display[1], 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

    # Sim model
    spring = Spring(k=100) # higer stiffer
    attenuator = Attenuator(b=1.0)
    sim = InputOutputFunction("sine", None, spring, attenuator)
    output = sim.simulate_live(20.0)
    current_index = 0

    # Camera rotation around Y axis
    camera_angle = 0.0
    rotate_left, rotate_right = False, False
    rotation_speed = 60  # degrees per second

    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_a:
                    rotate_left = True
                elif event.key == K_d:
                    rotate_right = True
            elif event.type == KEYUP:
                if event.key == K_a:
                    rotate_left = False
                elif event.key == K_d:
                    rotate_right = False

        
        if rotate_left:
            camera_angle += rotation_speed * dt
        if rotate_right:
            camera_angle -= rotation_speed * dt

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Convert polar to Cartesian for camera
        radius = 20
        cam_x = radius * math.sin(math.radians(camera_angle))
        cam_z = radius * math.cos(math.radians(camera_angle))
        gluLookAt(cam_x, 5, cam_z, 0, 5, 0, 0, 1, 0)

        # Animate spring base (start_y) and top (end_y)
        current_index = (current_index + 1) % len(output)
        input_value = sim.input_function(np.array([current_index * sim.dt]))[0]
        start_y = 5 + input_value
        end_y = (start_y + output[current_index]) *2 #*2 bc i could not see the spring coils

        draw_3d_spring(start_y=start_y, end_y=end_y)
        damper(start_y, end_y, damper_radius=0.3, rod_radius=0.05)
        draw_platform(end_y)
        draw_rod(start_y)
        draw_wheel(start_y,2.0,1.0,0.7,0.7,0.7)
        draw_wheel(start_y,3.0,0.9,0.3,0.3,0.3)


        pygame.display.flip()


    pygame.quit()

if __name__ == "__main__":
    main()
