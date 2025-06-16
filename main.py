import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import freqs
from scipy.signal import TransferFunction
import tkinter as tk
import numpy as np
import threading
import ctypes


class Attenuator:
    def __init__(self):
        self.b = 20.0
    
    def draw_cylinder(self,y1, y2, radius):
        segments = 20
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            theta = 2.0 * math.pi * i / segments
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            glVertex3f(x, y1, z)
            glVertex3f(x, y2, z)
        glEnd()
    
    def damper(self,start_y, end_y, damper_radius=0.5, rod_radius=0.1, body_height=1.2):
        # Draw the rod (full height)
        glColor3f(0.7, 0.7, 0.7)  # light gray
        self.draw_cylinder(start_y, end_y, rod_radius)

        # Draw damper body (centered between start_y and end_y)
        center_y = (start_y + end_y) / 2
        body_start = center_y - body_height / 2
        body_end = center_y + body_height / 2

        glColor3f(1.0, 0.0, 0.0)  # red
        self.draw_cylinder(body_start, body_end, damper_radius)


class Spring:
    def __init__(self):
        self.k = 200.0

    def draw_3d_spring(self,start_y, end_y, wire_radius=0.05, spring_radius=0.5,
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


class Wheel:
    def __init__(self):
        pass

    def draw_wheel(self,start_y, wheel_radius, wheel_thickness, colorR,colorG,colorB,rod_length=4.0, segments=32):

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


class Other_elements:
    def __init__(self):
        pass
    
    def draw_rod(self,start_y, rod_length=4.0, rod_radius=0.25, segments=32):

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


    def draw_platform(self,y, width, depth, thickness,x,c_r,c_g,c_b):
        hw = width / 2 #half-width
        hd = depth / 2 #half-depth
        h = thickness

        glColor3f(c_r,c_g,c_b) 
        glBegin(GL_QUADS)

        # Top face
        glVertex3f(-hw -x, y + h, -hd)
        glVertex3f(hw-x, y + h, -hd)
        glVertex3f(hw-x, y + h, hd)
        glVertex3f(-hw-x, y + h, hd)

        # Bottom face
        glVertex3f(-hw-x, y, -hd)
        glVertex3f(hw-x, y, -hd)
        glVertex3f(hw-x, y, hd)
        glVertex3f(-hw-x, y, hd)

        # Front face
        glVertex3f(-hw-x, y, hd)
        glVertex3f(hw-x, y, hd)
        glVertex3f(hw-x, y + h, hd)
        glVertex3f(-hw-x, y + h, hd)

        # Back face
        glVertex3f(-hw-x, y, -hd)
        glVertex3f(hw-x, y, -hd)
        glVertex3f(hw-x, y + h, -hd)
        glVertex3f(-hw-x, y + h, -hd)

        # Left face
        glVertex3f(-hw-x, y, -hd)
        glVertex3f(-hw-x, y, hd)
        glVertex3f(-hw-x, y + h, hd)
        glVertex3f(-hw-x, y + h, -hd)

        # Right face
        glVertex3f(hw-x, y, -hd)
        glVertex3f(hw-x, y, hd)
        glVertex3f(hw-x, y + h, hd)
        glVertex3f(hw-x, y + h, -hd)

        glEnd()


class BodePlot:
    def __init__(self, attenuator, spring, inout):
      self.attenuator = attenuator
      self.spring = spring
      self.inout = inout
      # transfer function 1/(m*s^2 + b*s + k)
      self.num = [1]
      self.den = [self.inout.mass, self.attenuator.b, self.spring.k]
      tf = TransferFunction(self.num, self.den)
      self.zeros, self.poles = list(tf.zeros), list(tf.poles)
    
      negative_poles = all(np.real(p) <= 0 for p in self.poles)
      
      if not negative_poles:
          self.correct_phase = True
      else:
          self.correct_phase = False

      #Zero is bad at log scale
      zeros_and_poles = [abs(f) for f in self.zeros + self.poles if abs(f) > 1e-6]

     # preparing range of the plot
      if zeros_and_poles:
        min_value = 0.1 * min(zeros_and_poles)
        max_value = 10 * max(zeros_and_poles)
      else:
        min_value = 0.1
        max_value = 100
      
      #the necessary range of the plot
      self.range = [min_value, max_value]

    def plotting_bode(self):
        fig = plt.figure(figsize=(10, 5))
        plt.suptitle("Bode Plot", fontsize=16)
        
        # w to put in tf
        w = np.logspace(np.log10(self.range[0]), np.log10(self.range[1]), 1000) # 1000 points to count
        
        # counts tf(j*w)
        w, plot_line = freqs(self.num, self.den,w)

        #change to dB and angle respectively
        magnitude = 20 * np.log10(abs(plot_line))
        
        phase =np.unwrap(np.angle(plot_line, deg=True))
        if self.correct_phase: phase -= 360
        
        #STABILITY    
        stable_poles = all(np.real(p) <= 0 for p in self.poles)
        
        if not stable_poles:
            self.stable = False
        else:
            to_zero = np.abs(phase + 180) #we're looking for phase at -180 so minimazing it to 0
            gain_margin_freq = np.argmin(to_zero)
            if to_zero[gain_margin_freq] < 2: 
                gain_margin = -magnitude[gain_margin_freq]
            else:
                gain_margin = np.inf    #inf when the phase doesn't dip below 180

            phase_margin_freq  = np.argmin(np.abs(magnitude))
            phase_0 = phase[phase_margin_freq]
            phase_margin = 180 + phase_0 
                        
            if gain_margin > 0 and phase_margin > 0:
                self.stable = True
            else:
                self.stable = False      

        plt.subplot(2, 1, 1)
        plt.plot(w, magnitude, label="Magnitude [dB]")
        plt.xscale("log")
        plt.title("Magnitude Bode Plot")
        plt.xlabel("Frequency [rad/s]")
        plt.ylabel("Magnitude [dB]")
        plt.grid(True)
        if stable_poles:
            plt.axvline(w[gain_margin_freq], color='red', linestyle='--',
                        label=f"Gain Margin: {gain_margin:.1f} dB")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(w, phase, label="Phase")
        plt.xscale("log")
        plt.title("Phase Bode Plot")
        plt.xlabel("Frequency [rad/s]")
        plt.ylabel("Phase [°]")
        plt.grid(True)
        if stable_poles:
            plt.axvline(w[phase_margin_freq], color='blue', linestyle='--',
                        label=f"Phase Margin: {phase_margin:.1f}°")
        plt.legend()

        stability_text = "Stable system: True" if self.stable else "Stable system: False"
        color = "green" if self.stable else "red"
        fig.text(0.5, 0.90, stability_text, ha='center', va='top',
                 fontsize=12, color=color, fontweight='bold')  
        plt.tight_layout(rect=[0, 0.05, 1, 0.90])
        plt.show()
        
class InputOutputFunction:
    def __init__(self, input_type, wheel, spring, attenuator):
        self.wheel = wheel
        self.spring = spring
        self.attenuator = attenuator
        self.input_type = input_type
        self.dt = 0.01

        self.frequency = 1.0
        self.amplitude = 50.0
        self.phase = 0.0
        self.pulse_width = 1.0
        self.mass = 1.0

    #Types of input to choose
    def input_function(self, t):
        if self.input_type == "sine":
            return self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)
        elif self.input_type == "sawtooth":
            T = 1/ self.frequency
            return (2 * self.amplitude / T) * (t % T) - self.amplitude
        elif self.input_type == "square":
            temp=np.sin(2 * np.pi * self.frequency * t + self.phase)
            return self.amplitude * np.where(temp >= 0, 1, -1)
        elif self.input_type == "rectangle impulse":
            return self.amplitude * np.where((t>0) & (t<self.pulse_width), 1, 0)
        elif self.input_type == "triangle":
            T = 1/ self.frequency
            t_mod = np.mod(t,T)
            return self.amplitude * (1 - 4 * np.abs(t_mod / T - 0.5))
        elif self.input_type == "impulse":
            u = np.zeros_like(t)
            dt = t[1] - t[0]
            idx = np.argmin(np.abs(t - 0.05))
            u[idx] = self.amplitude / dt  
            return u
        elif self.input_type == "step":
            return self.amplitude * np.ones_like(t)
        
    #Computes output - Euler differentation method
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

        for k in range(2, N):
            dy2[k] = (-k*y[k-1] - b*dy1[k-1] + u[k]) / m
            dy1[k] = dy1[k-1] + dt * dy2[k]
            y[k] = y[k-1] + dt * dy1[k]
        return y
    
    def simulate_system(self, t_start, t_end):
        dt=self.dt
        t = np.arange(t_start, t_end + dt, dt) 
        u = self.input_function(t) 
        y = self.euler_output(t)  
        return t, u, y

    def input_output_plot(self):
        t, u, y = self.simulate_system(0.0, 10.0)
        plt.figure(figsize=(10, 5))
        plt.suptitle("Input and Output Plot", fontsize=16)

        plt.figtext(
        0.5, 0.91,  # position in window
        rf"$G(s) = \frac{{1}}{{{self.mass:.1f}s^2 + {self.attenuator.b:.1f}s + {self.spring.k:.1f}}}$",
        ha='center', va='top',
        fontsize=14, color="black"
    )

        plt.subplot(2, 1, 1)
        plt.plot(t, u, label="u(t)")
        plt.xlabel("Time [s]")
        plt.ylabel("u(t)")
        plt.title("Input plot")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t, y, label="y(t)")
        plt.xlabel("Time [s]")
        plt.ylabel("y(t)")
        plt.title("Output plot")
        plt.grid(True)
        plt.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        plt.show()

 #Window to control parameters of car suspension       
class ParameterControl:
    def __init__(self):
        self.width = 800
        self.height = 1100
        self.fps = 60
        self.spring = Spring()
        self.attenuator = Attenuator()
        self.wheel = Wheel()
        self.input_type = "sine"
        self.in_out = InputOutputFunction(self.input_type, self.wheel, self.spring, self.attenuator)
        self.bode = BodePlot(self.attenuator, self.spring, self.in_out)
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
            new_energy = float(self.en_entry.get())
            new_type = self.input_type.get().lower()

            if not (10.0 <= new_k <= 1000.0):
                raise ValueError("Error - k must be in range [10, 1000]")
            if not (0.0 <= new_b <= 100.0):
                raise ValueError("Error - b must be in range [0, 100]")
            if not (50.0 <= new_a <= 100.0):
                raise ValueError("Error - A must be in range [50, 100]")
            if not (0.1 <= new_f <= 10.0):
                raise ValueError("Error - f must be in range [0.1, 10]")
            if not (-3.14 <= new_phase <= 3.14):
                raise ValueError("Error - phase must be in range [-π, π]")
            if not (0.0 <= new_pulse_width <= 10.0):
                raise ValueError("Error - pulse width must be in range [0,10]")
            if not (1.0 <= new_m <= 1000.0):
                raise ValueError("Error - mass must be in range [1,1000]")
            if not (5000.0 <= new_energy <= 10000.0):
                raise ValueError("Error - energy must be in range [5000,10000]")

            #Save correct values
            self.spring.k = new_k
            self.attenuator.b = new_b
            self.in_out.frequency = new_f
            self.in_out.amplitude = new_a
            self.in_out.phase = new_phase
            self.in_out.pulse_width = new_pulse_width
            self.in_out.mass = new_m
            self.in_out.input_type = new_type
            type = self.input_type.get().lower()
            if type == "impulse":
                self.in_out.amplitude = new_energy * self.in_out.dt

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

    def draw_bode_plot(self):
        if self.update_parameters():  
            self.bode.plotting_bode()


    def update_visibility(self, *args):
        type = self.input_type.get().lower()
        if type in ("sine", "square"):
            self.freq_label.grid()
            self.freq_entry.grid()
            self.amp_label.grid()
            self.amp_entry.grid()
            self.phase_label.grid()
            self.phase_entry.grid()
            self.en_label.grid_remove()
            self.en_entry.grid_remove()
            self.pulse_width_label.grid_remove()
            self.pulse_width_entry.grid_remove()
        elif type in ("triangle", "sawtooth"):
            self.freq_label.grid()
            self.freq_entry.grid()
            self.amp_label.grid()
            self.amp_entry.grid()
            self.phase_label.grid_remove()
            self.phase_entry.grid_remove()
            self.en_label.grid_remove()
            self.en_entry.grid_remove()
            self.pulse_width_label.grid_remove()
            self.pulse_width_entry.grid_remove()
        elif type == "step":
            self.freq_label.grid_remove()
            self.freq_entry.grid_remove()
            self.amp_label.grid()
            self.amp_entry.grid()
            self.phase_label.grid_remove()
            self.phase_entry.grid_remove()
            self.en_label.grid_remove()
            self.en_entry.grid_remove()
            self.pulse_width_label.grid_remove()
            self.pulse_width_entry.grid_remove()
        elif type == "impulse":
            self.freq_label.grid_remove()
            self.freq_entry.grid_remove()
            self.amp_label.grid_remove()
            self.amp_entry.grid_remove()
            self.en_label.grid()
            self.en_entry.grid()
            self.phase_label.grid_remove()
            self.phase_entry.grid_remove()
            self.pulse_width_label.grid_remove()
            self.pulse_width_entry.grid_remove()
        elif type == "rectangle impulse":
            self.freq_label.grid_remove()
            self.freq_entry.grid_remove()
            self.amp_label.grid()
            self.amp_entry.grid()
            self.en_label.grid_remove()
            self.en_entry.grid_remove()
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

        self.en_label = tk.Label(self.param_frame, text="Energy:", bg="white", font=font_settings)
        self.en_label.grid(row=0, column=0, padx=10)
        self.en_entry = tk.Entry(self.param_frame, width=7)
        self.en_entry.insert(0, str(self.in_out.amplitude/self.in_out.dt))
        self.en_entry.grid(row=0, column=1, padx=10)

        self.simulate_button = tk.Button(window, text="Simulate", command=self.simulate, bg="#3e8ef7", fg="white", padx=15, pady=5)
        self.simulate_button.pack(pady=(30, 5))
        tk.Button(window, text="Input and output", command=self.simulate_and_plot, bg="#3e8ef7", fg="white", padx=15, pady=5).pack(pady=(0,5))
        tk.Button(window, text="Draw Bode plot", command=self.draw_bode_plot, bg="#3e8ef7", fg="white", padx=15, pady=5).pack(pady=(0,20))

        self.update_visibility()

        self.error_label = tk.Label(window, text="", bg="white", foreground="red")
        self.error_label.pack(pady=5)
        window.mainloop()

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

    control = ParameterControl()

    # GUI in separate thread
    gui_thread = threading.Thread(target=control.open_control_window)
    gui_thread.daemon = True
    gui_thread.start()

    # Nie wywołuj control.simulate() tutaj

    camera_angle = 0.0
    rotate_left = rotate_right = False
    rotation_speed = 60
    radius = 20
    clock = pygame.time.Clock()
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    rotate_left = True
                elif event.key == pygame.K_d:
                    rotate_right = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_a:
                    rotate_left = False
                elif event.key == pygame.K_d:
                    rotate_right = False

        if rotate_left:
            camera_angle += rotation_speed * dt
        if rotate_right:
            camera_angle -= rotation_speed * dt

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        cam_x = radius * math.sin(math.radians(camera_angle))
        cam_z = radius * math.cos(math.radians(camera_angle))
        gluLookAt(cam_x, 5, cam_z, 0, 5, 0, 0, 1, 0)

        if control.simulation_data:
                 
            t, u, y = control.simulation_data
            control.frame = (control.frame + 1) % len(y)
            input_value = u[control.frame]
            output_value = y[control.frame]

            start_y = (3 + input_value)/20
            end_y = (3+output_value)*2.2 #just for scale 

            Spring().draw_3d_spring(start_y=start_y, end_y=end_y)
            Attenuator().damper(start_y, end_y, damper_radius=0.3, rod_radius=0.05)
            Other_elements().draw_platform(end_y, 2.5, 3.0, 1.2, 0, 0.1, 0.4, 0.7)
            Other_elements().draw_rod(start_y, rod_length=4)
            Wheel().draw_wheel(start_y, 1.5, 1.0, 0.7, 0.7, 0.7)
            Wheel().draw_wheel(start_y, 2.5, 0.9, 0.3, 0.3, 0.3)
            Other_elements().draw_platform(start_y - 3, 3.5, 3.0, 0.5, 2.0, 0.5, 0.8, 0.1)

        pygame.display.flip()

    pygame.quit()
    
if __name__ == "__main__":
    main()