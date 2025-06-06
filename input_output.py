import math
import numpy as np
import matplotlib.pyplot as plt

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
        self.pulse_width = 1.0
        self.mass = 1.0


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
            idx = np.argmin(np.abs(t - 0.01))
            u[idx] = self.amplitude
            return u
        elif self.input_type == "step":
            return self.amplitude * np.ones_like(t)
        
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
