import numpy as np
import matplotlib.pyplot as plt
import math as math


#Parameters
f_start = 325e6           # Min frequency: 325 MHz
f_end = 500e6           # Max frequency: 3800 MHz
duration = 30e-6         # Duration: 100 microseconds
sampling_rate = 30.72e6   # Sampling rate: 30.72 MHz (max resolution)
time_between_samples = 1/sampling_rate

#t = np.linspace(0, duration, num = int(sampling_rate * duration)) I am going to leave this commented out. This was the original way which we were calculating the cos wave. This is inaccurate though, so we changed it
#Create a time array which goes from 0 to the closest multiple of the sampling_rate under duration.
t = np.zeros(1)
while (t[-1] < duration):
    t = np.append(t, t[-1] + time_between_samples)
t = t[0:-1]

#https://en.wikipedia.org/wiki/Chirp
#"The corresponding time-domain function for a sinusoidal linear chirp
# is the sine of the phase in radians:
# x(t) = sin[Φ_0 + 2π((c/2)t^2 + (f_0)t)]
c = (f_end - f_start)/duration #The "chirp constant," or the rate of change of the frequency
theta = 2 * np.pi * (f_start * t + (c/2) * t * t) #obtained by integrating the angular frequency

chirp_signal_transmitted = np.cos(theta)
chirp_signal_transmitted = chirp_signal_transmitted
chirp_signal_received = np.flip(chirp_signal_transmitted)


time_added = 12e-6
bound1 = 2 * time_added/5
bound2 = 3 * time_added/5

sample = 0
time = 0
time_between_samples = 1/sampling_rate

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

time_after_sample = time_after_sample[0:-1]

bound1 = len(time_before_sample)
bound2 = len(time_after_sample)
numsamples_added = bound1 + bound2

signal_time = np.concatenate((time_before_sample, time_during_sample, time_after_sample))
signal_data = np.concatenate((np.zeros(bound1), chirp_signal_received, np.zeros(bound2)))

variance = 0

noise = np.random.normal(0, variance/2, len(signal_data))

signal_data = signal_data + noise


plt.figure(figsize=(10, 6))
plt.scatter(signal_time, signal_data, color='b')
plt.axvline(x = signal_time[bound1], color = 'r')
plt.axvline(x = signal_time[len(signal_data) - bound2], color = 'r')
plt.title("Recieved Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

filter_size = len(chirp_signal_transmitted)
minimum_block_length = 2 * filter_size
logBase2 = math.log2(minimum_block_length)
logBase2 = math.ceil(logBase2)
block_size = 2**(logBase2)

sample_size = block_size - filter_size + 1
if (sample_size % 2 == 1):
    sample_size = sample_size - 1

zero_padding_signal = np.zeros(block_size - sample_size)
zero_padding_filter = np.zeros(block_size - filter_size)

matched_filter = np.append(chirp_signal_transmitted, zero_padding_filter)
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

#Deal with leftover elements
#Omg... I need to first zero pad the signal to the sample size, multiply by the window function, then pad so the convolution function goes up to filter_size - 1
#No. All I need to do is reduce the size of the window function, multiply by the window function, then bring it so convolution function goes up to filter_size - 1, then alter matched filter to be the same length, then multiply, then IFFT
leftover_signal_window = window[:len(leftover_signal)]
leftover_signal = leftover_signal * leftover_signal_window
leftover_signal = np.concatenate((leftover_signal, zero_padding_signal))
signal_fft = np.fft.fft(leftover_signal)

matched_filter = chirp_signal_transmitted
matched_filter = np.concatenate((matched_filter, np.zeros(len(leftover_signal) - len(matched_filter))))
matched_filter_fft = np.fft.fft(matched_filter)

multiplied_fft = signal_fft * matched_filter_fft
signal_convolution = np.fft.ifft(multiplied_fft)
time_shifted_signal_convolution = np.concatenate((np.zeros(len(truncated_signal)), signal_convolution))
convolution_function = convolution_function + time_shifted_signal_convolution

convolution_function = abs(convolution_function)
max_matched_filter_location = np.argmax(convolution_function)

#Now we need to make a new convolution time array because the convolution adds more elements to the array
convolution_samples_added = block_size - sample_size
convolution_time_added = convolution_samples_added / sampling_rate
convolution_time = np.linspace(0, duration + time_added + convolution_time_added, num = int(sampling_rate * duration) + numsamples_added + convolution_samples_added)

lines = True
plt.plot(convolution_time, convolution_function, color = 'r')
if lines == True:
    plt.axvline(x = signal_time[bound1], color = 'g', dashes = (1, 5, 1, 5), label = 'Lower signal bound')
    plt.axvline(x = signal_time[len(signal_data) - bound2], color = 'g', dashes = (1, 5, 1, 5), label = 'Upper signal bound')
    plt.axvline(x = signal_time[max_matched_filter_location], color = 'y', dashes = (2, 1, 2, 1), label = 'Maximum \n convolution \n value with \n matched \n filter')
plt.text(0, convolution_function[max_matched_filter_location].real/2, "Max matched filter result: " + str(round(convolution_function[max_matched_filter_location].real,2)) + "\n at time t = " + str(round(signal_time[max_matched_filter_location],10)), fontsize = 12)
plt.show()