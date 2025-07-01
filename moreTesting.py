import numpy as np
import matplotlib.pyplot as plt

#def calc_phase_shift(buf_ref: np.ndarray, buf_shifted: np.ndarray):
#    """
#    Calculate the phase shift Φ between two same-length sinusoidal buffers.
#   Returns Φ in radians and degrees.
#   """
#    N = buf_ref.size
#    F_ref = np.fft.fft(buf_ref)
#    F_sh  = np.fft.fft(buf_shifted)
#    
#    F_ref[0] = 0
#    k = np.argmax(np.abs(F_ref))
#    phi_rad = (np.angle(F_sh[k]) - np.angle(F_ref[k])) % (2*np.pi)
#    phi_deg = np.degrees(phi_rad)
#    return phi_rad, phi_deg

import numpy as np
import matplotlib.pyplot as plt
def calc_phase_shift(buf_ref: np.ndarray,
                     buf_shifted: np.ndarray,
                     debug_plot: bool = True):
    """
    Calculate the phase shift Φ between two same-length sinusoidal buffers.
    Optionally displays both FFT magnitudes for debugging.
    Only the positive-frequency bins (f>0) are considered when finding the peak.
    """
    N = buf_ref.size

    # Compute FFTs
    F_ref = np.fft.fft(buf_ref)
    F_sh  = np.fft.fft(buf_shifted)

    # Build frequency axis (cycles per sample)
    freq = np.fft.fftfreq(N, d=1/N)

    # Zero out DC so it can't win
    F_ref[0] = 0

    # Mask out non-positive frequencies
    mag = np.abs(F_ref)
    mag[freq <= 0] = 0

    # Find the fundamental in the positive half
    k = np.argmax(mag)

    # Compute phase difference
    phi_rad = (np.angle(F_sh[k]) - np.angle(F_ref[k])) % (2*np.pi)
    phi_deg = np.degrees(phi_rad)

    if debug_plot:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        fig.suptitle(f'Debug FFT Magnitudes (bin {k}, f={freq[k]:.3f})')

        axs[0].plot(freq, np.abs(F_ref))
        axs[0].set_ylabel('|FFT(ref)|')
        axs[0].axvline(freq[k], color='gray', linestyle='--')

        axs[1].plot(freq, np.abs(F_sh))
        axs[1].set_ylabel('|FFT(shifted)|')
        axs[1].set_xlabel('Frequency (cycles per sample)')
        axs[1].axvline(freq[k], color='gray', linestyle='--')

        # Zoom in ±10 bins around the detected peak
        bin_width = freq[1] - freq[0]
        lim = 10 * bin_width
        axs[0].set_xlim(freq[k]-lim, freq[k]+lim)
        axs[1].set_xlim(freq[k]-lim, freq[k]+lim)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    return phi_rad, phi_deg

# Generate test buffers
N = 512111                    #Number of samples and the sampling rate in Hz
phi = 2 * np.pi / 7        # Example phase shift (radians)
freq = 100000000                   # Number of sinusoid cycles over the buffer

t = np.arange(N) / N #Time normalized to end at one second
buffer1 = np.sin(2 * np.pi * freq * t)
buffer2 = np.sin(2 * np.pi * freq * t + phi)
buffer3 = np.sin(2 * np.pi * freq * t + 2 * phi)
buffer4 = np.sin(2 * np.pi * freq * t + 3 * phi)

# Test phase calculations
pairs = [(buffer1, buffer2, "1→2"),
         (buffer2, buffer3, "2→3"),
         (buffer3, buffer4, "3→4")]

print(f"Expected Φ: {phi:.4f} rad = {np.degrees(phi):.2f}°\n")
for ref, sh, label in pairs:
    rad, deg = calc_phase_shift(ref, sh)
    print(f"Calculated Φ for buffer{label}: {rad:.4f} rad = {deg:.2f}°")