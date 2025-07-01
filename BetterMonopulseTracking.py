#!/usr/bin/env python3
"""
Streaming phase-comparison monopulse tracker
with an N-pole causal IIR low-pass (cascade of identical 1st-order sections).
"""

import numpy as np
import matplotlib.pyplot as plt

# ────────── array / RF parameters ──────────────────────────────────
c   = 3.0e8
fc  = 1.0e8                 # RF carrier 1 MHz
lam = c / fc
d   = lam / 2
N   = 4                     # ULA elements

fs  = 4 * 1000000              # severe undersampling (k = ~91 617 alias)
BW  = 4_000                # desired base-band BW  (4 kHz,  -3 dB)

N_POLES = 6                # <<<  set to 2 or 3 as requested >>>

n_samples  = 3000
theta_true = np.deg2rad(5)
theta_err0 = np.deg2rad(3)
K_gain, EPS = 0.8, 1e-12

# ────────── helpers ────────────────────────────────────────────────
elem_idx    = np.arange(N)
def array_phase(n, theta):
    return -2*np.pi * (n*d*np.sin(theta)) / lam
phi_spatial = array_phase(elem_idx, theta_true)

# alias frequency |fa| ≤ fs/2
fa      = ((fc + fs/2) % fs) - fs/2
lo      = 1 + 0j
lo_step = np.exp(1j*2*np.pi*fa/fs)

# 1-pole coefficient in exact exponential form (always 0<α<1)
alpha = 1 - np.exp(-2*np.pi*BW / fs)

# allocate IIR state arrays  (shape = (N, N_POLES))
I_prev = np.zeros((N, N_POLES))
Q_prev = np.zeros((N, N_POLES))

theta_steer = theta_true + theta_err0
theta_hist  = np.empty(n_samples)
mask_lr     = np.r_[np.ones(N//2), -np.ones(N-N//2)]

# ────────── streaming loop ─────────────────────────────────────────
for k in range(n_samples):
    # 1) real RF samples for each element (use alias freq!)
    v_rf = np.cos(2*np.pi*fa*k/fs + phi_spatial)            # (N,)

    # 2) quadrature mix to BB
    cos_lo =  2.0 * lo.real
    sin_lo = -2.0 * lo.imag
    i_raw  = v_rf * cos_lo
    q_raw  = v_rf * sin_lo

    # 3) N-pole causal IIR low-pass (cascade of 1-pole sections)
    I_sec = i_raw.copy()
    Q_sec = q_raw.copy()
    for p in range(N_POLES):
        I_sec = I_prev[:, p] + alpha * (I_sec - I_prev[:, p])
        Q_sec = Q_prev[:, p] + alpha * (Q_sec - Q_prev[:, p])
        I_prev[:, p] = I_sec          # update states
        Q_prev[:, p] = Q_sec
    x = I_sec + 1j * Q_sec            # complex envelope at pole N

    # 4) Σ / Δ monopulse
    w      = np.exp(-1j*array_phase(elem_idx, theta_steer))
    sum_c  = np.sum(w * x)
    diff_c = np.sum(mask_lr * w * x)
    ratio  = diff_c / (sum_c + EPS)
    eps    = np.imag(ratio) * lam / (2*np.pi*d*np.cos(theta_steer))
    theta_steer += K_gain * eps
    theta_hist[k] = theta_steer

    # 5) advance NCO
    lo *= lo_step

# ────────── plot ───────────────────────────────────────────────────
t = np.arange(n_samples) / fs
plt.plot(t, np.rad2deg(theta_hist), '.', ms=3,
         label=f'Estimated θ  ({N_POLES}-pole IIR)')
plt.hlines(np.rad2deg(theta_true), 0, t[-1], colors='k', linestyles='--',
           label='True θ')
plt.xlabel('Time (s)');  plt.ylabel('Angle (deg)')
plt.ylim(0, 10)
plt.title(f'Monopulse (fs = {fs:g} Hz,  fa = {fa:g} Hz,  {N_POLES}-pole LPF)')
plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.show()