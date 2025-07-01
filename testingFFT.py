import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
fs = 1_000        # sample rate [Hz]
f0 = 50           # tone to remove [Hz]
N  = 2048         # number of samples
cutoff = 25       # LPF cutoff [Hz]

# --- Original real sinusoid ---
t = np.arange(N) / fs
x = np.sin(2 * np.pi * f0 * t)

# --- FFT of original (for reference) ---
X = np.fft.fft(x)
freqs = np.fft.fftfreq(N, 1/fs)

# --- Complex mixing (shift down by f0) ---
mixer = np.exp(-1j * 2 * np.pi * f0 * t)   # e^{-j 2πf0 t}
mixed  = x * mixer                         # complex-valued

# --- Low-pass filter in frequency domain ---
Y = np.fft.fft(mixed)
mask = np.abs(freqs) < cutoff              # simple brick-wall LPF
Y_lp = Y * mask
baseband = np.fft.ifft(Y_lp)               # complex baseband

# ----------------- PLOTS -----------------
# 1) Spectra before and after
plt.figure()
plt.stem(freqs[:N//2], 2*np.abs(X[:N//2])/N, basefmt=" ", label="original")
plt.stem(freqs[:N//2], 2*np.abs(Y_lp[:N//2])/N, basefmt=" ", label="shifted & LPF")
plt.title("Magnitude spectrum (single-sided)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.legend()
plt.tight_layout()

# 2) Baseband time-domain (first 0.1 s)
plt.figure()
show_len = int(0.1 * fs)       # 0.1 seconds
plt.plot(t[:show_len], baseband.real[:show_len], label="real part")
plt.plot(t[:show_len], baseband.imag[:show_len], linestyle="--", label="imag part")
plt.title("Baseband signal after mixing + LPF (first 0.1 s)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()

plt.show()
