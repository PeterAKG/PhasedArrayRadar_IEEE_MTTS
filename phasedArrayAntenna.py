import numpy as np
import matplotlib.pyplot as plt
import math as math

# Parameters
amplitude = 1.0      # Amplitude of the sine wave
frequency = 5        # Frequency in Hz
sampling_rate = 1000 # Sampling rate in Hz
duration = 1         # Duration in seconds
phase_shift = np.pi / 3
phase_shift2 = np.pi/6
c = 300000000        # Speed of light

# Time array
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Sine wave equation
y = amplitude * np.sin(2 * np.pi * frequency * t)
y1 = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)
y2 = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift2)

k = 1
L = 50.0 # distance from object 
Bx = phase_shift
By = phase_shift2
dx = 5.0
dy = 5.0

elevation = math.asin(math.sqrt(((Bx/dx)**2) + ((By/dy)**2)))
azimuth = math.atan((By*dx)/(Bx*dy))
print("Azumuth: ",np.degrees(azimuth))
print("Elevation: ",np.degrees(elevation))

# Plotting the sine wave
plt.plot(t, y)
plt.title('Sine Wave')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

plt.plot(t, y1, color = 'red')
plt.title('Sine Wave: Phase Shift 45 degrees')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

plt.plot(t, y2, color = 'green')
plt.title('Sine Wave: Phase Shift 90 degrees')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


