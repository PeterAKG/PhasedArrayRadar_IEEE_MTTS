#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt
from numpy.fft import fft, ifft

# ────────── user / array parameters ────────────────────────────────
c   = 3.0e8
fc  = 100e6                 # 100 MHz carrier
fs  = 0.643e6              # arbitrary sample-rate (60 MHz here)
N   = 4                     # ULA elements
n_samples = 8080
BW  = 4_000                 # desired BB BW (4 kHz)

theta_true = np.linspace(np.deg2rad(90), np.deg2rad(91), n_samples)
theta_err0 = np.deg2rad(0)
K_gain, EPS = .8, 1e-12

# ────────── helpers ────────────────────────────────────────────────
lam  = c / fc
d    = lam / 2
elem_idx = np.arange(N)
def array_phase(n, th):    # spatial phase at element n
    return -2*np.pi*(n[None, :]*d*np.sin(th[:, None]))/lam

def array_phase_scalar(n, th):    # spatial phase at element n
    return -2*np.pi*(n[None, :]*d*np.sin(th))/lam

def analytic_signal_mix(x, fs, fc, BW):
    """
    Analytic signal from real x[n] by complex mixing & FFT low-pass.
    Works for any fs (incl. severe undersampling).
    """
    N = x.size
    t = np.arange(N) / fs

    fa = fc - np.round(fc/fs)*fs       # alias frequency (Hz)
    mix = x * np.exp(-1j*2*np.pi*fa*t) # shift alias to 0 Hz

    Z = fft(mix)
    freqs = np.fft.fftfreq(N, 1/fs)
    Z[np.abs(freqs) > BW/2] = 0        # ideal LPF
    z = ifft(Z)
    return z

# ────────── build real RF data (row = element) ─────────────────────

phi_spatial = array_phase(elem_idx, theta_true)

fa = fc - np.round(fc/fs)*fs            # alias (−20 MHz for 60 MHz fs)
t  = np.arange(n_samples) / fs
v_rf = np.cos(2*np.pi*fa*t[:, None] + phi_spatial)  # shape (N, n_samples)

# ────────── convert every channel to I/Q envelope ──────────────────
x_complex = np.vstack([analytic_signal_mix(v_rf[:,n], fs, fc, BW)
                       for n in range(N)])

# ────────── monopulse tracker ──────────────────────────────────────
theta_steer = theta_true[0] + theta_err0
mask_lr     = np.r_[np.ones(N//2), -np.ones(N-N//2)]
theta_hist  = np.empty(n_samples)

for k in range(n_samples):
    theta_hist[k] = theta_steer
    x   = x_complex[:, k]
    w   = np.exp(-1j*array_phase_scalar(elem_idx, theta_steer))
    sum_c  = np.sum(w*x)
    diff_c = np.sum(mask_lr*w*x)
    eps = np.imag(diff_c/(sum_c+EPS)) * lam /(2*np.pi*d*np.cos(theta_steer))
    theta_steer += K_gain*eps

# ────────── plot result ────────────────────────────────────────────
t_sec = np.arange(n_samples)/fs
plt.plot(t_sec, np.rad2deg(theta_hist), '.', ms=3, label='estimate')
#plt.hlines(np.rad2deg(theta_true), 0, t_sec[-1], colors='k', ls='--')
plt.plot(t_sec, np.rad2deg(theta_true), 'g', label='estimate')
plt.xlabel('Time (s)'); plt.ylabel('Angle (deg)')
#plt.ylim(20,60); 
plt.grid(alpha=.3)
plt.title(f'Monopulse (fs = {fs/1e6:.2f} MHz, alias = {fa/1e6:.2f} MHz)')
plt.tight_layout(); plt.show()