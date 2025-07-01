#!/usr/bin/env python3
"""
phase_monopulse_tracker_analog.py
─────────────────────────────────
4-element phase-comparison monopulse tracker driven by **real** RF
samples.  Per-element voltages:

    v_n[k] = cos(2π fc t_k + φ_spatial[n])

Those real waveforms are converted to analytic (I/Q) form with a
Hilbert transform; the complex envelopes are then fed to the usual
Σ/Δ monopulse processor.

© 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

# ───────────── Simulation parameters ───────────────────────────────
c           = 3.0e8          # speed of light  [m/s]
fc          = 1.0e9          # carrier         [Hz]
lam         = c / fc         # wavelength      [m]
d           = lam / 2        # inter-element spacing (λ/2)
N           = 4              # elements (4 × 1 ULA)

fs          = 4 * fc         # RF sampling rate (≥ 2·fc).  Here 4 GHz
n_samples   = 2000           # # of samples to simulate  (t_total ≈ 0.5 µs)

theta_true  = np.deg2rad(5)  # true AoA
theta_err0  = np.deg2rad(3)  # initial pointing error
K_gain      = 0.8            # monopulse loop gain
EPS         = 1e-12          # divide-by-zero guard

# ───────────── Helper functions ────────────────────────────────────
def array_phase(n, theta):
    """Spatial phase (rad) at element n for arrival angle θ."""
    return -2 * np.pi * (n * d * np.sin(theta)) / lam

def analytic_signal(x):
    """
    Return the analytic signal via the FFT Hilbert method.
    x : 1-D real array  →  complex array of same length.
    """
    X = fft(x)
    N = x.size
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2
    return ifft(X * h)

# LR mask  [+1 +1 -1 -1]  for the Δ channel
mask_lr = np.r_[np.ones(N//2), -np.ones(N - N//2)]

# ───────────── Generate analogue RF signals ────────────────────────
t            = np.arange(n_samples) / fs              # time vector
elem_idx     = np.arange(N)
phi_spatial  = array_phase(elem_idx, theta_true)      # constant per element

# Real voltages v_n[k] for all n, k  →  shape (N, n_samples)
v_rf = np.cos(2 * np.pi * fc * t[None, :] + phi_spatial[:, None])

# Convert every element to analytic (I/Q) baseband
x_complex = np.vstack([analytic_signal(v_rf[n]) for n in range(N)])

# ───────────── Tracking loop ───────────────────────────────────────
theta_steer   = theta_true + theta_err0
theta_hist    = np.empty(n_samples)

for k in range(n_samples):
    # 1) sample from every element (complex envelope)
    x = x_complex[:, k]

    # 2) steering weights
    w    = np.exp(-1j * array_phase(elem_idx, theta_steer))
    x_bf = w * x

    # 3) Σ / Δ
    sum_chan  = np.sum(x_bf)
    diff_chan = np.sum(mask_lr * x_bf)

    # 4) monopulse error estimate (phase-comparison)
    ratio = diff_chan / (sum_chan + EPS)
    eps   = np.imag(ratio) * lam / (2*np.pi*d*np.cos(theta_steer))

    # 5) update steering
    theta_steer += K_gain * eps
    theta_hist[k] = theta_steer

# ───────────── Plot result ─────────────────────────────────────────
plt.plot(t, np.rad2deg(theta_hist), label="Estimated θ")
plt.hlines(np.rad2deg(theta_true), 0, t[-1], colors="k",
           linestyles="--", label="True θ")
plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.title("Phase-Comparison Monopulse Tracking (RF → I/Q)")
plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.show()