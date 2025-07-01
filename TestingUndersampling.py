import numpy as np
import matplotlib.pyplot as plt

def undersampled_fft(f0, phi, fs, N, noise_std):
    """
    Generate a noisy sinusoid of frequency f0 [Hz] and phase phi [rad],
    sampled at fs [Hz] for N points, compute its FFT, and return:
      t       : time vector
      x       : noisy signal samples
      freqs   : FFT frequency bins (Hz)
      X       : complex FFT output
      I, Q    : real and imag parts of X
    """
    # time vector
    t = np.arange(N) / fs

    # pure tone + noise
    x = np.sin(2*np.pi*f0*t + phi) + noise_std * np.random.randn(N)

    # FFT
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, 1/fs)

    # separate I and Q
    I = np.real(X)
    Q = np.imag(X)

    return t, x, freqs, X, I, Q

def plot_results(t, x, freqs, X, I, Q):
    
    # Time-domain
    plt.figure(figsize=(10, 4))
    plt.plot(t, x, '.', markersize=3)
    plt.xlabel('Time [s]'); plt.ylabel('Amplitude')
    plt.title('Noisy Undersampled Sinusoid')
    plt.grid(True)

    # FFT magnitude
    plt.figure(figsize=(10, 4))
    # shift zero‐freq to center for display
    idx = np.argsort(freqs)
    plt.plot(freqs[idx], np.abs(X)[idx])
    plt.xlabel('Frequency [Hz]'); plt.ylabel('|X|')
    plt.title('FFT Magnitude')
    plt.grid(True)

    # I and Q components
    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    plt.plot(freqs[idx], I[idx], '.-')
    #plt.xlim(90000000, 110000000)
    plt.ylabel('I (Re{X})')
    plt.grid(True)
    plt.title('FFT In-phase (I) & Quadrature (Q)')
    plt.subplot(2,1,2)
    plt.plot(freqs[idx], Q[idx], '.-')
    #plt.xlim(90000000, 110000000)
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Q (Im{X})')
    plt.grid(True)

    plt.tight_layout()

if __name__ == '__main__':
    # WAVE 1
    WAVE1_f0       = 100000000       # signal frequency [Hz]
    WAVE1_phi      = np.pi/4        # signal phase [rad]
    WAVE1_fs       = 5000       # sampling rate [Hz] (undersampled if < 2*f0)
    WAVE1_N        = 40000        # number of samples
    WAVE1_noise_std = 0.2       # noise standard deviation

    # WAVE 2
    WAVE2_f0       = 100000000       # signal frequency [Hz]
    WAVE2_phi      = 1.3 * np.pi/4        # signal phase [rad]
    WAVE2_fs       = 210000000       # sampling rate [Hz] (undersampled if < 2*f0)
    WAVE2_N        = 40000        # number of samples
    WAVE2_noise_std = 0.2       # noise standard deviation

    # generate and process
    WAVE1_t, WAVE1_x, WAVE1_freqs, WAVE1_X, WAVE1_I, WAVE1_Q = undersampled_fft(WAVE1_f0, WAVE1_phi, WAVE1_fs, WAVE1_N, WAVE1_noise_std)
    WAVE2_t, WAVE2_x, WAVE2_freqs, WAVE2_X, WAVE2_I, WAVE2_Q = undersampled_fft(WAVE2_f0, WAVE2_phi, WAVE2_fs, WAVE2_N, WAVE2_noise_std)

    # plot everything
    plot_results(WAVE1_t, WAVE1_x, WAVE1_freqs, WAVE1_X, WAVE1_I, WAVE1_Q)
    plot_results(WAVE2_t, WAVE2_x, WAVE2_freqs, WAVE2_X, WAVE2_I, WAVE2_Q)
    plt.show()