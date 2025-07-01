#!/usr/bin/env python3
"""Monopulse tracker demo with calibration‑based alias‑frequency estimation.

During a short initial calibration period the script measures the aliased
carrier that results from severe undersampling. The estimated alias frequency
is then reused for complex base‑band conversion (I/Q extraction) during the
tracking phase – mimicking how the real hardware will work in the field.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq

# ──────────── USER PARAMETERS ───────────────────────────────────────────────
c: float = 3.0e8          # speed of light (m/s)
fc: float = 100e6         # true RF carrier (Hz) — unknown to receiver
fs: float = 0.643e6       # ADC sampling rate (Hz)
N: int   = 4              # ULA elements
n_samples: int = 8080     # total samples to generate
BW: float = 4_000         # desired complex BB bandwidth (Hz)

# Calibration settings
n_calib: int = 4096       # samples used for alias‑frequency estimation

# Target motion (held steady during calibration, then slews 1°)
theta_calib = np.deg2rad(10)                             # fixed during calib
theta_track = np.linspace(np.deg2rad(10), np.deg2rad(10),  # afterwards
                          n_samples - n_calib)
theta_true  = np.concatenate([np.full(n_calib, theta_calib),
                              theta_track])                # shape (n_samples,)

# Monopulse loop parameters
theta_err0 = 0.0                  # initial pointing error (rad)
K_gain     = 0.8                  # loop gain
EPS        = 1e-12                # protect against div‑by‑zero

# ──────────── GEOMETRY HELPER FUNCTIONS ─────────────────────────────────────
lam  = c / fc                      # RF wavelength (m)
d    = lam / 2                     # λ/2 element spacing
elem_idx = np.arange(N)

def array_phase(elements: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Spatial phase for each element at angles *theta* (vectorised)."""
    return -2 * np.pi * (elements[None, :] * d * np.sin(theta[:, None])) / lam

# ──────────── RF DATA GENERATION ────────────────────────────────────────────
fa = np.absolute(fc - np.round(fc / fs) * fs)   # analytical alias frequency (for truth plots)

t = np.arange(n_samples) / fs
phi_spatial = array_phase(elem_idx, theta_true)            # (n_samples, N)
# v_rf: real passband voltage at each array element (rows=time, cols=ant)
v_rf = np.cos(2 * np.pi * fa * t[:, None] + phi_spatial)

# ──────────── CALIBRATION — ESTIMATE ALIAS FREQUENCY ────────────────────────
print("\n— Calibration —")
print(f"True alias frequency      : {fa / 1e6:.2f} MHz")

calib_signal = v_rf[:n_calib, 0]       # use first element (any will do)
Z = fft(calib_signal)
calib_freqs = fftfreq(n_calib, 1 / fs)
fa_est = np.absolute(calib_freqs[np.argmax(np.abs(Z))])  # signed frequency (Hz)
print(f"Estimated alias frequency : {fa_est / 1e6:.2f} MHz")

# ──────────── HELPER: ANALYTIC FROM ALIAS FREQ ──────────────────────────────

def analytic_from_alias(x: np.ndarray, fa_alias: float,
                        fs: float, BW: float) -> np.ndarray:
    """Return complex BB analytic signal from severely undersampled *x*.

    Mixes with the *measured* alias frequency and ideal‑LPFs to ±BW/2 Hz.
    """
    N = x.size
    t = np.arange(N) / fs
    mix = x * np.exp(-1j * 2 * np.pi * fa_alias * t)      # shift to DC
    Z = fft(mix)
    freqs = fftfreq(N, 1 / fs)

    plt.plot(freqs, Z, "b*")
    #plt.ylim(0, 5)
    plt.show()

    Z[np.abs(freqs) > BW / 2] = 0                         # brick‑wall LPF
    #print(Z)

    plt.plot(freqs, Z, "b*")
    #plt.ylim(0, 5)
    plt.show()
    return ifft(Z)

# Build complex envelopes for every channel using the *measured* alias freq
x_complex = np.vstack([
    analytic_from_alias(v_rf[:, n], fa_est, fs, BW) for n in range(N)
])
print(x_complex[:, 0])
print(x_complex[:, 1]) 
print(x_complex[:, 2])
print(x_complex[:, 3])
print(x_complex[:, 4])
print(x_complex[:, 5])
print(x_complex[:, 6])
print(x_complex[:, 7])

# ──────────── MONOPULSE TRACKER ─────────────────────────────────────────────
print("— Tracking —")
mask_lr    = np.r_[np.ones(N // 2), -np.ones(N - N // 2)]  # l vs r weights
theta_steer = theta_true[0] + theta_err0
theta_hist  = np.empty(n_samples)

for k in range(n_samples):
    theta_hist[k] = theta_steer
    x   = x_complex[:, k]
    w   = np.exp(-1j * array_phase(elem_idx, np.array([theta_steer]))).ravel()
    sum_c  = np.sum(w * x)
    diff_c = np.sum(mask_lr * w * x)
    eps = (np.imag(diff_c / (sum_c + EPS)) *
    lam / (2 * np.pi * d * np.cos(theta_steer)))   # pointing error
    theta_steer += K_gain * eps

# ──────────── PLOTS ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3))

t_sec = np.arange(n_samples) / fs
ax.plot(t_sec, np.rad2deg(theta_hist), '.', ms=3, label='estimate')
ax.plot(t_sec, np.rad2deg(theta_true), 'g', label='true')
ax.set(xlabel='Time (s)', ylabel='Angle (deg)',
       title=(f"Monopulse; fs = {fs/1e6:.2f} MHz\n"
              f"true alias = {fa/1e6:.2f} MHz | est. = {fa_est/1e6:.2f} MHz"))
ax.grid(alpha=0.3)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

plt.plot(np.abs(x_complex[0,:]))
plt.show()