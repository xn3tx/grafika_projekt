import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import tkinter as tk
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from spring import Spring
from attenuator import Attenuator
from wheel import Wheel
from drawscene import Drawing
from input_output import InputOutputFunction
import ctypes

class ParameterControl:
    def __init__(self):
        self.width = 800
        self.height = 1050
        self.fps = 60
        self.spring = Spring()
        self.attenuator = Attenuator()
        self.wheel = Wheel()
        self.drawing = Drawing(self.spring, self.attenuator, self.wheel)
        self.input_type = "sine"
        self.in_out = InputOutputFunction(self.input_type, self.wheel, self.spring, self.attenuator)
        self.error_label = None

        self.simulation_data = None  # dodaj to!
        self.frame = 0    

    def update_parameters(self):
        try:
            new_k = float(self.k_entry.get())
            new_b = float(self.b_entry.get())
            new_a = float(self.amp_entry.get())
            new_f = float(self.freq_entry.get())
            new_m = float(self.m_entry.get())
            new_phase = float(self.phase_entry.get())
            new_pulse_width = float(self.pulse_width_entry.get())
            new_type = self.input_type.get().lower()

            if not (10.0 <= new_k <= 1000.0):
                raise ValueError("Error - k must be in range [10, 1000]")
            if not (0.0 <= new_b <= 100.0):
                raise ValueError("Error - b must be in range [0, 100]")
            if not (0.0 <= new_a <= 200.0):
                raise ValueError("Error - A must be in range [0, 200]")
            if not (0.1 <= new_f <= 10.0):
                raise ValueError("Error - f must be in range [0.1, 10]")
            if not (-3.14 <= new_phase <= 3.14):
                raise ValueError("Error - phase must be in range [-π, π]")
            if not (0.0 <= new_pulse_width <= 10.0):
                raise ValueError("Error - pulse width must be in range [0,10]")
            if not (1.0 <= new_m <= 1000.0):
                raise ValueError("Error - mass must be in range [1,1000]")

            #Save correct values
            self.spring.k = new_k
            self.attenuator.b = new_b
            self.in_out.frequency = new_f
            self.in_out.amplitude = new_a
            self.in_out.phase = new_phase
            self.in_out.pulse_width = new_pulse_width
            self.in_out.mass = new_m
            self.in_out.input_type = new_type

            if self.error_label is not None:
                self.error_label.config(text="Correct parameters", foreground="green")
            return True

        except ValueError as e:
            if self.error_label is not None:
                self.error_label.config(text=str(e), foreground="red")
            return False

    def update_simulation_data(self):
        t = np.arange(0.0, 10.0, self.in_out.dt)
        u = self.in_out.input_function(t)
        y = self.in_out.euler_output(t)
        self.simulation_data = (t, u, y)
        self.frame = 0

    def simulate_and_plot(self):
        if self.update_parameters():  
            self.in_out.input_output_plot()

    def simulate(self):
        if self.update_parameters():  
            self.update_simulation_data()

    def update_visibility(self, *args):
        type = self.input_type.get().lower()
        if type in ("sine", "square"):
            self.freq_label.grid()
            self.freq_entry.grid()
            self.amp_label.grid()
            self.amp_entry.grid()
            self.phase_label.grid()
            self.phase_entry.grid()
            self.pulse_width_label.grid_remove()
            self.pulse_width_entry.grid_remove()
        elif type in ("triangle", "sawtooth"):
            self.freq_label.grid()
            self.freq_entry.grid()
            self.amp_label.grid()
            self.amp_entry.grid()
            self.phase_label.grid_remove()
            self.phase_entry.grid_remove()
            self.pulse_width_label.grid_remove()
            self.pulse_width_entry.grid_remove()
        elif type in ("step", "impulse"):
            self.freq_label.grid_remove()
            self.freq_entry.grid_remove()
            self.amp_label.grid()
            self.amp_entry.grid()
            self.phase_label.grid_remove()
            self.phase_entry.grid_remove()
            self.pulse_width_label.grid_remove()
            self.pulse_width_entry.grid_remove()
        elif type == "rectangle impulse":
            self.freq_label.grid_remove()
            self.freq_entry.grid_remove()
            self.amp_label.grid()
            self.amp_entry.grid()
            self.phase_label.grid_remove()
            self.phase_entry.grid_remove()
            self.pulse_width_label.grid()
            self.pulse_width_entry.grid()

        if self.error_label and self.error_label["text"]:
            msg = self.error_label["text"].lower()
            if "phase" in msg and type not in ("sine", "square"):
                self.error_label.config(text="", foreground="red")
                self.phase_entry.delete(0, tk.END)
                self.phase_entry.insert(0, "0.0")
            elif "frequency" in msg and type not in ("sine", "square", "triangle", "sawtooth"):
                self.error_label.config(text="", foreground="red")
                self.freq_entry.delete(0, tk.END)
                self.freq_entry.insert(0, "1.0")
            elif "pulse width" in msg and type != "rectangle impulse":
                self.error_label.config(text="", foreground="red")
                self.pulse_width_entry.delete(0, tk.END)
                self.pulse_width_entry.insert(0, "1.0")

    def on_input_type_change(self, *args):
        new_type = self.input_type.get().lower()
        self.in_out.input_type = new_type
        self.update_visibility()

    def open_control_window(self):
        window = tk.Tk()
        window.title("Suspension parameters")
        window.geometry(f"{self.width}x{self.height}")
        window.configure(bg="white")
        self.input_type = tk.StringVar(value="sine")
        self.input_type.trace_add("write", self.on_input_type_change)

        #Better quality of text in window
        font_settings = ("Helvetica", 11)
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass
        window.tk.call('tk', 'scaling', 2.5)

        tk.Label(window, text="Spring constant (k):", bg="white", font=font_settings).pack(pady=(30,10))
        self.k_entry = tk.Entry(window)
        self.k_entry.insert(0, str(self.spring.k))
        self.k_entry.pack()

        tk.Label(window, text="Damping ratio (b):", bg="white", font=font_settings).pack(pady=(0,10))
        self.b_entry = tk.Entry(window)
        self.b_entry.insert(0, str(self.attenuator.b))
        self.b_entry.pack()

        tk.Label(window, text="Mass (m):", bg="white", font=font_settings).pack(pady=(0,10))
        self.m_entry = tk.Entry(window)
        self.m_entry.insert(0, str(self.in_out.mass))
        self.m_entry.pack()

        self.input_type.trace_add("write", self.update_visibility)

        signal_types = ["sine", "square", "sawtooth", "triangle", "rectangle impulse", "impulse", "step"]
        radio_frame = tk.LabelFrame(window, text="Input signal type", bg="white", font=font_settings, fg="#3e8ef7", bd=2, relief="groove", padx=10, pady=10)
        radio_frame.pack(pady=(10, 10), padx=20, fill="x")
        for signal in signal_types:
            tk.Radiobutton(radio_frame, text=signal, variable=self.input_type, value=signal, bg="white", font=font_settings).pack()

        self.param_frame = tk.Frame(window, bg="white")
        self.param_frame.pack(pady=10)

        self.amp_label = tk.Label(self.param_frame, text="Amplitude:", bg="white", font=font_settings)
        self.amp_label.grid(row=0, column=0, padx=10)
        self.amp_entry = tk.Entry(self.param_frame, width=7)
        self.amp_entry.insert(0, str(self.in_out.amplitude))
        self.amp_entry.grid(row=0, column=1, padx=10)

        self.freq_label = tk.Label(self.param_frame, text="Frequency [Hz]:", bg="white", font=font_settings)
        self.freq_label.grid(row=1, column=0, padx=10)
        self.freq_entry = tk.Entry(self.param_frame, width=7)
        self.freq_entry.insert(0, str(self.in_out.frequency))
        self.freq_entry.grid(row=1, column=1, padx=10)

        self.phase_label = tk.Label(self.param_frame, text="Phase [rad]:", bg="white", font=font_settings)
        self.phase_label.grid(row=2, column=0, padx=10)
        self.phase_entry = tk.Entry(self.param_frame, width=7)
        self.phase_entry.insert(0, str(self.in_out.phase))
        self.phase_entry.grid(row=2, column=1, padx=10)

        self.pulse_width_label = tk.Label(self.param_frame, text="Pulse width:", bg="white", font=font_settings)
        self.pulse_width_label.grid(row=3, column=0, padx=10)
        self.pulse_width_entry = tk.Entry(self.param_frame, width=7)
        self.pulse_width_entry.insert(0, str(self.in_out.pulse_width))
        self.pulse_width_entry.grid(row=3, column=1, padx=10)

        self.update_visibility()

        tk.Button(window, text="Simulate", command=self.simulate, bg="#3e8ef7", fg="white", padx=15, pady=8).pack(pady=(30,10))
        tk.Button(window, text="Input and output", command=self.simulate_and_plot, bg="#3e8ef7", fg="white", padx=15, pady=8).pack(pady=(0,20))

        self.error_label = tk.Label(window, text="", bg="white", foreground="red")
        self.error_label.pack(pady=5)
        window.mainloop()


def main():
    control = ParameterControl()
    os.environ['SDL_VIDEO_WINDOW_POS'] = "1200,200"
    pygame.init()
    width = 1500
    height = 1000
    screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Car suspension simulator")
    control.drawing.set_2d_projection(control.drawing)

    clock = pygame.time.Clock()
    running = True
    scale = 50
    threading.Thread(target=control.open_control_window, daemon=True).start()

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        if control.simulation_data is not None:
            t, u, y = control.simulation_data
            frame = control.frame

            if frame < len(t):
                y_wheel = u[frame] * scale
                y_desk = (y[frame] + u[frame]) * scale
                control.frame += 1
            else:
                y_wheel = 0
                y_desk = 0
        else:
            y_wheel = 0
            y_desk = 0

        control.drawing.draw_scene(y_wheel, y_desk)
        pygame.display.flip()
        clock.tick(control.fps)
    pygame.quit()

if __name__ == "__main__":
    main()

