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
#Create a time array which goes from 0 to the closest multiple of the sampling_rate under duration.
t = np.zeros(1)
while (t[-1] < duration):
    t = np.append(t, t[-1] + time_between_samples)
t = t[0:-2]

#https://en.wikipedia.org/wiki/Chirp
#"The corresponding time-domain function for a sinusoidal linear chirp
# is the sine of the phase in radians:
# x(t) = sin[Φ_0 + 2π((c/2)t^2 + (f_0)t)]
c = (f_end - f_start)/duration #The "chirp constant," or the rate of change of the frequency
theta = 2 * np.pi * (f_start * t + (c/2) * t * t) #obtained by integrating the angular frequency

#Creates the transmitted signal based on the phase of the signal which we calculated earier (theta)
chirp_signal_transmitted = np.cos(theta)
chirp_signal_transmitted_complex = np.sin(theta) * complex(0, 1)
chirp_signal_transmitted = chirp_signal_transmitted + chirp_signal_transmitted_complex

chirp_signal_received = np.flip(chirp_signal_transmitted) #The signal we receive is going to be time-reversed to the signal we emitted
#I want to look further into this, but I think we need to take into account that the signal we receieve could be phase shifted compared to the
#signal we emitted. This is because it's very unlikely that our sampler will just so happen to sample the same exact points on both waves.
#We should add a phase shift constant to the received wave to account for this. We can make the phase shift random.

chirp_signal_matched_filter = chirp_signal_transmitted - 2 * chirp_signal_transmitted_complex #The matched filter is the time-reversed conjugate of the recieved signal

#We already produced the signal we recieved, but we also need to add time around that signal in order to test if the program
#can find where the signal recieved is.
time_added = 12e-6 
bound1 = 25 * time_added/60
bound2 = 35 * time_added/60

#Loop variables. Sample is a counter for the number of samples. Time is what we use to keep track of time as more samples are added.
sample = 0
time = 0

time_before_sample = [0]

#While the time before the sample is less than bound1 (the amount of time which happens before the chirp is recieved), keep adding time until this condition is broken.
while (time_before_sample[-1] < bound1):
    sample += 1
    time = sample * time_between_samples
    time_before_sample.append(time)
time_before_sample = time_before_sample[0:-1]

#The time_during_sample should start the sample right after the time_before_sample
time_during_sample = [time_before_sample[-1] + time_between_samples]

#The (sampling rate * the duration of the chirp) is the number of samples that occur during the chirp. Add a new time point for every sample during the chirp.
for(x) in range(int(sampling_rate * duration) - 1):
    time_during_sample.append(time_during_sample[-1] + time_between_samples)

#The time_after_sample starts the sample after the chirp
time_after_sample = [time_during_sample[-1] + time_between_samples]

#Keep adding samples until the time added is more than the time added specified previously
while(time_after_sample[-1] < time_during_sample[-1] + bound2):
    time_after_sample.append(time_after_sample[-1] + time_between_samples)
time_after_sample = time_after_sample[0:-1]


samples_added_before = len(time_before_sample)
samples_added_after = len(time_after_sample)
numsamples_added = samples_added_before + samples_added_after #Get the number of time samples we added to the original number of samples (the chirp samples)

signal_time = np.concatenate((time_before_sample, time_during_sample, time_after_sample)) #The signal time is all the time samples we created
signal_data = np.concatenate((np.zeros(samples_added_before), chirp_signal_received, np.zeros(samples_added_after))) #The signal data is some empty samples before, the chirp, and some empty signal after

variance = 2 #Controls the variance in the guassian noise we add to the complex and imaginary signals

noise = np.random.normal(0, variance/2, len(signal_data)) #Guassian noise we add to real points
complex_noise = np.random.normal(0, variance/2, len(signal_data)) * complex(0, 1) #Guassian noise we add to imaginary points

signal_data = signal_data + noise + complex_noise #Add the noise to the theoretical sample we calculated

#Plot the sample recieved
plt.figure(figsize=(10, 6))
plt.scatter(signal_time, signal_data.real, color='b', label = "real points")
plt.scatter(signal_time, signal_data.imag, color='g', label = "imaginary points")
plt.axvline(x = signal_time[samples_added_before], color = 'r') #These lines are to show where our signal is
plt.axvline(x = signal_time[len(signal_data) - samples_added_after], color = 'r')
plt.legend()
plt.title("Recieved Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()


# In the previous part, we were simply simulating what antenna data we might recieve assuming we emitted a chirp (sinusodial wave with
# linearly increasing frequency). Now we have to convolute (https://en.wikipedia.org/wiki/Convolution) the data recieved with our
# matched filter. The point at which the convolution is the highest is the right part of our signal. Our matched filter is the 
# conjugate of the signal that we transmitted.
# [Taking the convolution of the signal with the time reversed conjugate is the same as taking the autocorrelation with the normal
# time conjugate; the reason we take a convolution with the time reversed conjugate (the conjugate of the signal transmitted) is because
# it is much computationally simpler and easier to take a convolution of a time reversed conjugate than to take the cross-correlation with
# a normal-time conjugate.]
# We are going to take the convolution of the signal by the matched filter by taking the FFT of both signals, multiplying them, and
# taking the inverse. Now, taking the FFT of the whole signal recieved is very computationally expensive. Luckily, there is a way to
# break it down. What we do instead is break down the signal recieved into blocks. Take the FFT of each block, multiply it by the
# FFT of the matched filter, and take the IFFT of that. Then, add that convolution data to the previous convolution data shifting it
# by the appropriate amount of time. <---- This is somewhat simplified, and does not account for the fact that you have to take a 
#                                          windowing function of each block and add the convolutions together such that they satisfy
#                                          the COLA condition (constant overlap add condition), but it gives you the general idea
#
#   Here is a visual to what we are essentially doing
#
#  (Signal Amplitude)
#   |        .              .     .   .   . . .                .  .  .            . 
#   |     .    .   .  .   .   .  .   .   . .   .   .    .    .    .    .  . . . .   . 
#   |  .    . .  .  .   .  .  . .  . .  .    .   .   . .   .  . .   .    .  .    .   . 
#   | .  . .  . .  .   . .  . .   . . .    .   .   .  .  .  .    .    .     .  .   . . .
#   |______________________________________________________________________________________ 
#                                                                                           (t)
#   |       |      |      |      |      |      |      |      |      |      |            |
#   \_______|______/\_____|______/\_____|______/\_____|______/\_____|______/\_____|_____/
#       samp|le 1     samp|ple 3    samp|ple 5    samp|ple 3    samp|ple 9    samp|le 11
#           \_____________/\____________/\____________/\____________/\____________/
#           |   sample 2      sample 4      sample 6      sample 8      sample 10 
#           |      |       |      |      |      |      |      |      |      |      |
#           |      |       |      |      |      |      |      |      |      |      |
#           V      V       V      V      V      V      V      V      V      V      V
#   | Convolution 1     |
#           |Convolution 2      |
#                   |Convolution 3     |
#                           |Convolution 4      |
#                                  |Convolution 5        |
#                                        |Convolution 6        |
#                                                 |Convolution 7        |
#                                                       |Convolution 8      |
#                                                               |Convolution 9      |
#                                                                     |Convolution 10       |
#                                                                            |Convolution 11        |
#  __________________________________________________________________________________________________ (t)
#  Every convolution starts at a different point in time, so add them into one total convolution with respect to the same starting time
#
#  Convolution
#  |
#  |
#  |                                            /\
#  |                                           /  \
#  |                                          /    \                        /\_____
#  |            /\          __          /\   /      \   /\              _/\/       \_        /\__
#  | /----\____/  \_/\_/-\_/  \__/\_/\_/  \_/        \_/  \_/\_/\__--__/             \___/\_/    \_____
#  |___________________________________________________________________________________________________ (t)
#  Now we have our convolution! Take the max to find where the signal is (or where the signal probably is).

# The size of our matched filter is the size of the signal we emitted (the number of samples in our chirp), doubled, and 
# rounded up to the nearest power of two. (We double it to make sure we are taking a linear convolution instead of a circular convolution)
filter_size = len(chirp_signal_matched_filter)
minimum_block_length = 2 * filter_size
logBase2 = math.log2(minimum_block_length)
logBase2 = math.ceil(logBase2)
block_size = 2**(logBase2)

#Here is a diagram of the block size, the sample size, and the filter size
# |      Sample Size          ||       Filter Size     |
# |                 Block Size                        |  <- One sample less than sample size + filter size
sample_size = block_size - filter_size + 1

#Make the sample size divisible by two because we will be taking steps half the size of the sample size
if (sample_size % 2 == 1):
    sample_size = sample_size - 1

#Zero padding so we can make the sample and the filter the same size (the block size). This is done to make sure the FFT of the
#sample and the filter have the same number of samples so we can multiply them.
zero_padding_signal = np.zeros(block_size - sample_size)
zero_padding_filter = np.zeros(block_size - filter_size)

#Zero padding the matched filter to the block size
matched_filter = np.append(chirp_signal_matched_filter, zero_padding_filter)
matched_filter_fft = np.fft.fft(matched_filter)

window = np.hanning(sample_size)

#This is the number of samples in the convolution function, but all values set to zero. We will add and build on this array, eventually producing a whole convolution function.
convolution_function = np.zeros(int(len(chirp_signal_received) + numsamples_added + block_size - sample_size))

#Here we need to shorten the signal so it is divisible by the sample size. We will deal with leftover elements later
if((len(chirp_signal_received) + numsamples_added) % sample_size > 0):
    number_of_blocks = int((len(chirp_signal_received) + numsamples_added) / sample_size) #This is not really the number of blocks. This is the number of blocks given that we are doing whole steps instead of half steps
    truncated_signal = signal_data[:number_of_blocks * sample_size]  #The signal which we can sample with sample_size sized blocks
    leftover_signal =  signal_data[-((len(chirp_signal_received) + numsamples_added) % sample_size):] #The signal at the end we can't sample with sample_size sized blocks


for(x) in range(int(2 * (len(truncated_signal) / (sample_size)) - 1)): #We are repeating this for loop for each sample block we take

    #Go x half-sample-size steps forward, and take a sample the size of sample_size
    block_signal = truncated_signal[int(x * (sample_size/2)):int(x * (sample_size/2) + sample_size)]
    
    #Window the block_signal with the hanning function to get rid of spectral discontinuities
    block_signal = block_signal * window

    #Zero pad the signal to the size of the block
    block_signal = np.concatenate((block_signal, zero_padding_signal))
    
    #Take the FFT of the zero padded signal
    signal_fft = np.fft.fft(block_signal)

    multiplied_fft = signal_fft * matched_filter_fft

    signal_convolution = np.fft.ifft(multiplied_fft)

    #Shift the time of the signal convolution we just calculated it so we can add it to the wider convolution function
    time_shifted_signal_convolution = np.concatenate((np.zeros(int(x * (sample_size/2))), signal_convolution, np.zeros(int(numsamples_added + len(chirp_signal_received) - x * (sample_size/2) - sample_size)))) #Used algebra to simplify last part
    
    convolution_function = convolution_function + time_shifted_signal_convolution

#Basically, because we did a constant overlap add with a hanning function with half sample size hops, what we multiplied the function with to do the FFT looks like
# /‾‾‾‾‾‾‾‾‾‾‾\. This works great for the middle sections, but for the beginning and ending section (/ and \), this produces inaccurate convolution results. We will
#account for these innacuracies in this section
#We need to divide the beginning part and ending part of the convolution function by the window itself, here's a link: https://gauss256.github.io/blog/cola.html
beginning_window = np.concatenate(([1], window[1:int((len(window) / 2))]))
convolution_function[:int((sample_size/2))] /= beginning_window * 0.1 #This is clearly broken, and you can tell since when you multiply by a constant, it doesnt affect half of the convolution.


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

