import numpy as np

phase_cal = 0
signal_start = 0
signal_end = 0 #THESE NEED TO BE CHANGED!!!

def monopulse_angle(sum, delta):
    sum_delta_corr = np.correlate(sum[signal_start:signal_end], delta[signal_start:signal_end], 'valid')
    angle_diff = np.angle(sum_delta_corr)
    return angle_diff

def scan_for_DOA():
    # go through all the possible phase shifts and find the peak, that will be the DOA (direction of arrival) aka steer_angle
    Rx_0= list()
    Rx_1= list()
    Rx_2 = list()
    Rx_3 = list()
    peak_sum = []
    peak_delta = []
    monopulse_phase = []
    delay_phases = np.arange(-180, 180, 2)    # phase delay in degrees
    for phase_delay in delay_phases:   
        delayed_Rx_1 = Rx_1 * np.exp(1j*np.deg2rad(phase_delay+phase_cal))
        delayed_sum = Rx_0 + delayed_Rx_1
        delayed_delta = Rx_0 - delayed_Rx_1
        
        delayed_sum_fft, delayed_sum_dbfs = dbfs(delayed_sum)
        delayed_delta_fft, delayed_delta_dbfs = dbfs(delayed_delta)
        mono_angle = monopulse_angle(delayed_sum_fft, delayed_delta_fft)
        
        peak_sum.append(np.max(delayed_sum_dbfs))
        peak_delta.append(np.max(delayed_delta_dbfs))
        monopulse_phase.append(np.sign(mono_angle))
        
        #Repeat this stuff but for the y axis? 

    peak_dbfs = np.max(peak_sum)
    peak_delay_index = np.where(peak_sum==peak_dbfs)
    peak_delay = delay_phases[peak_delay_index[0][0]]
    steer_angle = int(calcTheta(peak_delay))
    
    return delay_phases, peak_dbfs, peak_delay, steer_angle, peak_sum, peak_delta, monopulse_phase