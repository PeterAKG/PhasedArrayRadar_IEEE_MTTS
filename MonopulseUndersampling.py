#!/usr/bin/env python3
"""
phase_monopulse_tracker_stream_decim.py
---------------------------------------
Streaming phase-comparison monopulse tracker
that **undersamples** by keeping only one in every M RF samples.

    M = 20  ⇒  f_s,ADC = f_RFsample / 20

© 2025
"""
import numpy as np
import matplotlib.pyplot as plt

# ───────────  array / RF parameters  ───────────────────────────────
c   = 3.0e8
fc  = 1.0e8                   # Hz  (RF carrier)
lam = c / fc
d   = lam / 2                 # λ/2 spacing
N   = 4                       # elements

fs_high = 4 * fc              # Hz  (4-× Nyquist @ RF)  → 4 GHz
M      = 1                   # keep 1 sample every M ⇒ fs_ADC = fs_high / M
fs_adc = fs_high / M          # 200 MHz in this example

n_adc   = 3000                # how many ADC (decimated) samples to simulate
theta_true = np.deg2rad(5)
theta_err0 = np.deg2rad(3)
K_gain  = 0.8
EPS     = 1e-12

# ───────────  helper: spatial phase per element  ───────────────────
elem_idx     = np.arange(N)
phi_spatial  = -2*np.pi * (elem_idx * d * np.sin(theta_true)) / lam  # (N,)

# ───────────  causal low-pass (1-pole IIR) for base-band  ──────────
BW   = 5e6                               # desired BB bandwidth (5 MHz)
alpha = 2*np.pi*BW / fs_adc              # 0<α≪1
I_prev = np.zeros(N)
Q_prev = np.zeros(N)

# ───────────  LO stepping  (advance by M high-rate ticks each pass) ─
lo          = 1 + 0j
lo_step_M   = np.exp(1j * 2*np.pi * fc * M / fs_high)   # e^{j2πfc·M/fs_high}

# ───────────  monopulse state  ─────────────────────────────────────
theta_steer = theta_true + theta_err0
theta_hist  = np.empty(n_adc)
mask_lr     = np.r_[np.ones(N//2), -np.ones(N - N//2)]   # [+ + − −]

# ───────────  main streaming (decimated) loop  ─────────────────────
for k_adc in range(n_adc):
    k_high = k_adc * M                        # corresponding RF index
    t_k    = k_high / fs_high

    # 1) real RF voltages, one per element
    v_rf = np.cos(2*np.pi*fc*t_k + phi_spatial)           # shape (N,)

    # 2) quadrature mixing with current LO phase
    cos_lo =  2.0 * lo.real
    sin_lo = -2.0 * lo.imag
    i_raw  = v_rf * cos_lo
    q_raw  = v_rf * sin_lo

    # 3) causal 1-pole LPF → I/Q envelope
    I_lp = I_prev + alpha * (i_raw - I_prev)
    Q_lp = Q_prev + alpha * (q_raw - Q_prev)
    I_prev, Q_prev = I_lp, Q_lp
    x = I_lp + 1j * Q_lp                                  # shape (N,)

    # 4) steering weights (corrected)
    w    = np.exp(-1j * (-2*np.pi * elem_idx * d * np.sin(theta_steer) / lam))
    x_bf = w * x

    # 5) Σ / Δ channels
    sum_chan  = np.sum(x_bf)
    diff_chan = np.sum(mask_lr * x_bf)

    # 6) monopulse error + update
    ratio = diff_chan / (sum_chan + EPS)
    eps   = np.imag(ratio) * lam / (2*np.pi*d*np.cos(theta_steer))
    theta_steer += K_gain * eps
    theta_hist[k_adc] = theta_steer

    # 7) advance LO by M RF ticks
    lo *= lo_step_M

# ───────────  plot result  ─────────────────────────────────────────
t_adc = np.arange(n_adc) / fs_adc
plt.plot(t_adc, np.rad2deg(theta_hist), label="Estimated θ")
plt.hlines(np.rad2deg(theta_true), 0, t_adc[-1], colors="k",
           linestyles="--", label="True θ")
plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.title(f"Monopulse Tracking with Undersampling  (keep 1 / {M})")
plt.grid(alpha=.3)
plt.legend()
plt.tight_layout()
plt.show()
