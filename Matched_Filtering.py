import numpy as np
import matplotlib.pyplot as plt
import math as math


#Parameters
f_start = 325e6           # Min frequency: 325 MHz
f_end = 500e6           # Max frequency: 3800 MHz
duration = 1.5e-6         # Duration: 100 microseconds
sampling_rate = 30.72e6   # Sampling rate: 30.72 MHz (max resolution)
time_between_samples = 1/sampling_rate

#t = np.linspace(0, duration, num = int(sampling_rate * duration)) I am going to leave this commented out. This was the original way which we were calculating the cos wave. This is inaccurate though, so we changed it
t = np.zeros(1)
while (t[-1] < duration):
    t = np.append(t, t[-1] + time_between_samples)
t = t[0:-2]

k = (f_end - f_start)/duration
theta = 2 * np.pi * (f_start * t + (k/2) * t * t)

chirp_signal_transmitted = np.cos(theta)
chirp_signal_transmitted_complex = np.sin(theta) * complex(0, 1)
chirp_signal_transmitted = chirp_signal_transmitted + chirp_signal_transmitted_complex

chirp_signal_received = np.flip(chirp_signal_transmitted)

chirp_signal_matched_filter = chirp_signal_transmitted - 2 * chirp_signal_transmitted_complex #The matched filter is the time-reversed conjugate of the recieved signal


time_added = 12e-6
bound1 = 59 * time_added/60
bound2 = 1 * time_added/60

sample = 0
time = 0

time_before_sample = [0]

while (time_before_sample[-1] < bound1):
    sample += 1
    time = sample * time_between_samples
    time_before_sample.append(time)
time_before_sample = time_before_sample[0:-2]

time_during_sample = [time_before_sample[-1] + time_between_samples]

for(x) in range(int(sampling_rate * duration) - 1):
    time_during_sample.append(time_during_sample[-1] + time_between_samples)

time_after_sample = [time_during_sample[-1] + time_between_samples]

while(time_after_sample[-1] < time_during_sample[-1] + bound2):
    time_after_sample.append(time_after_sample[-1] + time_between_samples)

time_after_sample = time_after_sample[0:-2]

samples_added_before = len(time_before_sample)
samples_added_after = len(time_after_sample)
numsamples_added = samples_added_before + samples_added_after

signal_time = np.concatenate((time_before_sample, time_during_sample, time_after_sample))
signal_data = np.concatenate((np.zeros(samples_added_before), chirp_signal_received, np.zeros(samples_added_after)))

variance = 0

noise = np.random.normal(0, variance/2, len(signal_data))
complex_noise = np.random.normal(0, variance/2, len(signal_data)) * complex(0, 1)

signal_data = signal_data + noise + complex_noise


plt.figure(figsize=(10, 6))
plt.scatter(signal_time, signal_data.real, color='b', label = "real points")
plt.scatter(signal_time, signal_data.imag, color='g', label = "imaginary points")
plt.axvline(x = signal_time[samples_added_before], color = 'r')
plt.axvline(x = signal_time[len(signal_data) - samples_added_after], color = 'r')
plt.legend()
plt.title("Recieved Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

filter_size = len(chirp_signal_matched_filter)
minimum_block_length = 2 * filter_size
logBase2 = math.log2(minimum_block_length)
logBase2 = math.ceil(logBase2)
block_size = 2**(logBase2)

sample_size = block_size - filter_size + 1
if (sample_size % 2 == 1):
    sample_size = sample_size - 1

zero_padding_signal = np.zeros(block_size - sample_size)
zero_padding_filter = np.zeros(block_size - filter_size)

matched_filter = np.append(chirp_signal_matched_filter, zero_padding_filter)
matched_filter_fft = np.fft.fft(matched_filter)

window = np.hanning(sample_size)

convolution_function = np.zeros(int(len(chirp_signal_received) + numsamples_added + block_size - sample_size))

#Here we need to shorten the signal so it is divisible by the sample size. We will deal with leftover elements later
if((len(chirp_signal_received) + numsamples_added) % sample_size > 0):
    number_of_blocks = int((len(chirp_signal_received) + numsamples_added) / sample_size) #This is not really the number of blocks. This is the number of blocks given that we are doing whole steps instead of half steps
    truncated_signal = signal_data[:number_of_blocks * sample_size]
    leftover_signal =  signal_data[-((len(chirp_signal_received) + numsamples_added) % sample_size):]

for(x) in range(int(2 * (len(truncated_signal) / (sample_size)) - 1)):
    block_signal = truncated_signal[int(x * (sample_size/2)):int(x * (sample_size/2) + sample_size)]
    
    block_signal = block_signal * window

    block_signal = np.concatenate((block_signal, zero_padding_signal))
    
    signal_fft = np.fft.fft(block_signal)

    multiplied_fft = signal_fft * matched_filter_fft

    signal_convolution = np.fft.ifft(multiplied_fft)

    time_shifted_signal_convolution = np.concatenate((np.zeros(int(x * (sample_size/2))), signal_convolution, np.zeros(int(numsamples_added + len(chirp_signal_received) - x * (sample_size/2) - sample_size)))) #Used algebra to simplify last part
    
    convolution_function = convolution_function + time_shifted_signal_convolution

#Basically, because we did a constant overlap add with a hanning function with half sample size hops, what we multiplied the function with to do the FFT looks like
# /‾‾‾‾‾‾‾‾‾‾‾\. This works great for the middle sections, but for the beginning and ending section (/ and \), this produces inaccurate convolution results. We will
#account for these innacuracies in this section
#We need to divide the beginning part and ending part of the convolution function by the window itself, here's a link: https://gauss256.github.io/blog/cola.html
beginning_window = np.concatenate(([1], window[1:int((len(window) / 2))]))
convolution_function[:int((sample_size/2))] /= beginning_window

#end_window = np.concatenate((window[int(len(window) / 2):-1], [1]))
#convolution_function[-int((sample_size)/2):] /= end_window


#Now we have leftover elements. We need to multiply these elements by a windowing function, take the FFT, multiply by the matched filter FFT, take the IFFT, then divide
#by the window to get the correct convolution

leftover_signal_length = len(leftover_signal)

leftover_signal_window = np.hamming(leftover_signal_length) #We are using hamming window here because it doesn't taper to zero

leftover_signal = leftover_signal * leftover_signal_window
leftover_signal = np.concatenate((leftover_signal, zero_padding_signal))
signal_fft = np.fft.fft(leftover_signal)

matched_filter = chirp_signal_matched_filter
matched_filter = np.concatenate((matched_filter, np.zeros(len(leftover_signal) - len(matched_filter))))
matched_filter_fft = np.fft.fft(matched_filter)

multiplied_fft = signal_fft * matched_filter_fft
signal_convolution = np.fft.ifft(multiplied_fft)

signal_convolution[:leftover_signal_length] /= leftover_signal_window

time_shifted_signal_convolution = np.concatenate((np.zeros(len(truncated_signal)), signal_convolution))
convolution_function = convolution_function + time_shifted_signal_convolution

convolution_function = abs(convolution_function)
max_matched_filter_location = np.argmax(convolution_function)

#Now we need to make a new convolution time array because the convolution adds more elements to the array
convolution_samples_added = block_size - sample_size
convolution_time_added = convolution_samples_added / sampling_rate

convolution_time = np.linspace(0, signal_time[-1] + convolution_time_added, num = int(sampling_rate * duration) + numsamples_added + convolution_samples_added)

lines = True
plt.plot(convolution_time, convolution_function, color = 'r')
if lines == True:
    plt.axvline(x = signal_time[samples_added_before], color = 'g', dashes = (1, 5, 1, 5), label = 'Lower signal bound')
    plt.axvline(x = signal_time[len(signal_data) - samples_added_after], color = 'g', dashes = (1, 5, 1, 5), label = 'Upper signal bound')
    plt.axvline(x = signal_time[max_matched_filter_location], color = 'y', dashes = (2, 1, 2, 1), label = 'Maximum \n convolution \n value with \n matched \n filter')
    plt.axvline(x = signal_time[int(sample_size/2)])
    plt.axvline(x = signal_time[int(len(truncated_signal) - sample_size/2)])
plt.text(0, convolution_function[max_matched_filter_location].real/2, "Max matched filter result: " + str(round(convolution_function[max_matched_filter_location].real,2)) + "\n at time t = " + str(round(signal_time[max_matched_filter_location],10)), fontsize = 12)
plt.show()

