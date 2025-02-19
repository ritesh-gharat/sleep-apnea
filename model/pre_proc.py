import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from tqdm import tqdm
import pickle
import os

FS = 100.0

def moving_median(data, window_size=5):
    """
    Calculate the moving median of the input data using a sliding window
    Args:
        data: Input signal array
        window_size: Size of the sliding window (default=5)
    Returns:
        result: Filtered signal array
    """
    result = np.zeros(len(data))
    for i in range(len(data)):
        start_idx = max(0, i - window_size//2)
        end_idx = min(len(data), i + window_size//2 + 1)
        result[i] = np.median(data[start_idx:end_idx])
    return result

# From https://github.com/rhenanbartels/hrv/blob/develop/hrv/classical.py
def create_time_info(rri):
    """
    Convert RR intervals to cumulative time series starting at zero
    Args:
        rri: Array of RR intervals in milliseconds
    Returns:
        Time points in seconds
    """
    rri_time = np.cumsum(rri) / 1000.0  # Convert to seconds
    return rri_time - rri_time[0]   # Start at zero

def create_interp_time(rri, fs):
    """
    Create uniformly spaced time points for interpolation
    Args:
        rri: Array of RR intervals
        fs: Target sampling frequency
    Returns:
        Array of uniform time points
    """
    time_rri = create_time_info(rri)
    return np.arange(0, time_rri[-1], 1 / float(fs))

def interp_cubic_spline(rri, fs):
    """
    Interpolate RR intervals using cubic spline
    Args:
        rri: Array of RR intervals
        fs: Target sampling frequency
    Returns:
        time_rri_interp: Interpolated time points
        rri_interp: Interpolated RR intervals
    """
    time_rri = create_time_info(rri)
    time_rri_interp = create_interp_time(rri, fs)
    tck = interpolate.splrep(time_rri, rri, s=0)
    rri_interp = interpolate.splev(time_rri_interp, tck, der=0)
    return time_rri_interp, rri_interp

def interp_cubic_spline_qrs(qrs_index, qrs_amp, fs):
    """
    Interpolate QRS amplitudes using cubic spline
    Args:
        qrs_index: Sample indices of QRS peaks
        qrs_amp: QRS amplitudes
        fs: Target sampling frequency
    Returns:
        time_qrs_interp: Interpolated time points
        qrs_interp: Interpolated QRS amplitudes
    """
    time_qrs = qrs_index / float(FS)
    time_qrs = time_qrs - time_qrs[0]
    time_qrs_interp = np.arange(0, time_qrs[-1], 1/float(fs))
    tck = interpolate.splrep(time_qrs, qrs_amp, s=0)
    qrs_interp = interpolate.splev(time_qrs_interp, tck, der=0)
    return time_qrs_interp, qrs_interp

data_path = './data/'
train_data_name = ['a02', 'a03', 'a04', 'a05',
             'a06', 'a07', 'a08', 'a09', 'a10',
             'a11', 'a12', 'a13', 'a14', 'a15',
             'a16', 'a17', 'a18', 'a19',
             'b02', 'b03', 'b04',
             'c02', 'c03', 'c04', 'c05',
             'c06', 'c07', 'c08', 'c09',
             ]
val_data_name = ['a01', 'b01', 'c01']
test_data_name = ['a20','b05','c10']
age = [51, 38, 54, 52, 58,
       63, 44, 51, 52, 58,
       58, 52, 51, 51, 60,
       44, 40, 52, 55, 58,
       44, 53, 53, 42, 52,
       31, 37, 39, 41, 28,
       28, 30, 42, 37, 27]

sex = [1, 1, 1, 1, 1,
       1, 1, 1, 1, 1,
       1, 1, 1, 1, 1,
       1, 1, 1, 1, 1,
       0, 1, 1, 1, 1,
       1, 1, 1, 0, 0,
       0, 0, 1, 1, 1]


def get_qrs_amp(ecg, qrs):
    """
    Extract QRS wave amplitudes from ECG signal
    Args:
        ecg: ECG signal array
        qrs: QRS peak locations
    Returns:
        qrs_amp: Array of QRS amplitudes
    """
    interval = int(FS * 0.250)
    qrs_amp = []
    for index in range(len(qrs)):
        curr_qrs = qrs[index]
        amp = np.max(ecg[curr_qrs-interval:curr_qrs+interval])
        qrs_amp.append(amp)

    return qrs_amp

MARGIN = 10
FS_INTP = 4
MAX_HR = 300.0
MIN_HR = 20.0
MIN_RRI = 1.0 / (MAX_HR / 60.0) * 1000
MAX_RRI = 1.0 / (MIN_HR / 60.0) * 1000
train_input_array = []
train_label_array = []

for data_index in range(len(train_data_name)):
    print (train_data_name[data_index])
    win_num = len(wfdb.rdann(os.path.join(data_path,train_data_name[data_index]), 'apn').symbol)
    signals, fields = wfdb.rdsamp(os.path.join(data_path,train_data_name[data_index]))
    for index in tqdm(range(1, win_num)):
        samp_from = index * 60 * FS # 60 seconds
        samp_to = samp_from + 60 * FS  # 60 seconds

        qrs_ann = wfdb.rdann(data_path + train_data_name[data_index], 'qrs', sampfrom=samp_from - (MARGIN*100), sampto=samp_to + (MARGIN*100)).sample
        apn_ann = wfdb.rdann(data_path + train_data_name[data_index], 'apn', sampfrom=samp_from, sampto=samp_to-1).symbol

        qrs_amp = get_qrs_amp(signals, qrs_ann)

        rri = np.diff(qrs_ann)
        rri_ms = rri.astype('float') / FS * 1000.0
        try:
            rri_filt = moving_median(rri_ms)

            if len(rri_filt) > 5 and (np.min(rri_filt) >= MIN_RRI and np.max(rri_filt) <= MAX_RRI):
                time_intp, rri_intp = interp_cubic_spline(rri_filt, FS_INTP)
                qrs_time_intp, qrs_intp = interp_cubic_spline_qrs(qrs_ann, qrs_amp, FS_INTP)
                rri_intp = rri_intp[(time_intp >= MARGIN) & (time_intp < (60+MARGIN))]
                qrs_intp = qrs_intp[(qrs_time_intp >= MARGIN) & (qrs_time_intp < (60 + MARGIN))]
                #time_intp = time_intp[(time_intp >= MARGIN) & (time_intp < (60+MARGIN))]

                if len(rri_intp) != (FS_INTP * 60):
                    skip = 1
                else:
                    skip = 0

                if skip == 0:
                    rri_intp = rri_intp - np.mean(rri_intp)
                    qrs_intp = qrs_intp - np.mean(qrs_intp)
                    if apn_ann[0] == 'N': # Normal
                        label = 0.0
                    elif apn_ann[0] == 'A': # Apnea
                        label = 1.0
                    else:
                        label = 2.0

                    train_input_array.append([rri_intp, qrs_intp, age[data_index], sex[data_index]])
                    train_label_array.append(label)
        except Exception as e:
            print(f"Error processing data: {e}")

with open('train_input.pickle','wb') as f: 
    pickle.dump(train_input_array, f)
with open('train_label.pickle','wb') as f: 
    pickle.dump(train_label_array, f)


val_input_array = []
val_label_array = []
for data_index in range(len(val_data_name)):
    print (val_data_name[data_index])
    win_num = len(wfdb.rdann(os.path.join(data_path,val_data_name[data_index]), 'apn').symbol)
    signals, fields = wfdb.rdsamp(os.path.join(data_path,val_data_name[data_index]))
    for index in tqdm(range(1, win_num)):
        samp_from = index * 60 * FS # 60 seconds
        samp_to = samp_from + 60 * FS  # 60 seconds

        qrs_ann = wfdb.rdann(data_path + val_data_name[data_index], 'qrs', sampfrom=samp_from - (MARGIN*100), sampto=samp_to + (MARGIN*100)).sample
        apn_ann = wfdb.rdann(data_path + val_data_name[data_index], 'apn', sampfrom=samp_from, sampto=samp_to-1).symbol

        qrs_amp = get_qrs_amp(signals, qrs_ann)

        rri = np.diff(qrs_ann)
        rri_ms = rri.astype('float') / FS * 1000.0
        try:
            rri_filt = moving_median(rri_ms)

            if len(rri_filt) > 5 and (np.min(rri_filt) >= MIN_RRI and np.max(rri_filt) <= MAX_RRI):
                time_intp, rri_intp = interp_cubic_spline(rri_filt, FS_INTP)
                qrs_time_intp, qrs_intp = interp_cubic_spline_qrs(qrs_ann, qrs_amp, FS_INTP)
                rri_intp = rri_intp[(time_intp >= MARGIN) & (time_intp < (60+MARGIN))]
                qrs_intp = qrs_intp[(qrs_time_intp >= MARGIN) & (qrs_time_intp < (60 + MARGIN))]
                #time_intp = time_intp[(time_intp >= MARGIN) & (time_intp < (60+MARGIN))]

                if len(rri_intp) != (FS_INTP * 60):
                    skip = 1
                else:
                    skip = 0

                if skip == 0:
                    rri_intp = rri_intp - np.mean(rri_intp)
                    qrs_intp = qrs_intp - np.mean(qrs_intp)
                    if apn_ann[0] == 'N': # Normal
                        label = 0.0
                    elif apn_ann[0] == 'A': # Apnea
                        label = 1.0
                    else:
                        label = 2.0

                    val_input_array.append([rri_intp, qrs_intp, age[data_index], sex[data_index]])
                    val_label_array.append(label)
        except Exception as e:
            print(f"Error processing data: {e}")

with open('val_input.pickle','wb') as f: 
    pickle.dump(val_input_array, f)
with open('val_label.pickle','wb') as f: 
    pickle.dump(val_label_array, f)

test_input_array = []
test_label_array = []
for data_index in range(len(test_data_name)):
    print (test_data_name[data_index])
    win_num = len(wfdb.rdann(os.path.join(data_path,test_data_name[data_index]), 'apn').symbol)
    signals, fields = wfdb.rdsamp(os.path.join(data_path,test_data_name[data_index]))
    for index in tqdm(range(1, win_num)):
        samp_from = index * 60 * FS # 60 seconds
        samp_to = samp_from + 60 * FS  # 60 seconds

        qrs_ann = wfdb.rdann(data_path + test_data_name[data_index], 'qrs', sampfrom=samp_from - (MARGIN*100), sampto=samp_to + (MARGIN*100)).sample
        apn_ann = wfdb.rdann(data_path + test_data_name[data_index], 'apn', sampfrom=samp_from, sampto=samp_to-1).symbol

        qrs_amp = get_qrs_amp(signals, qrs_ann)

        rri = np.diff(qrs_ann)
        rri_ms = rri.astype('float') / FS * 1000.0
        try:
            rri_filt = moving_median(rri_ms)

            if len(rri_filt) > 5 and (np.min(rri_filt) >= MIN_RRI and np.max(rri_filt) <= MAX_RRI):
                time_intp, rri_intp = interp_cubic_spline(rri_filt, FS_INTP)
                qrs_time_intp, qrs_intp = interp_cubic_spline_qrs(qrs_ann, qrs_amp, FS_INTP)
                rri_intp = rri_intp[(time_intp >= MARGIN) & (time_intp < (60+MARGIN))]
                qrs_intp = qrs_intp[(qrs_time_intp >= MARGIN) & (qrs_time_intp < (60 + MARGIN))]
                #time_intp = time_intp[(time_intp >= MARGIN) & (time_intp < (60+MARGIN))]

                if len(rri_intp) != (FS_INTP * 60):
                    skip = 1
                else:
                    skip = 0

                if skip == 0:
                    rri_intp = rri_intp - np.mean(rri_intp)
                    qrs_intp = qrs_intp - np.mean(qrs_intp)
                    if apn_ann[0] == 'N': # Normal
                        label = 0.0
                    elif apn_ann[0] == 'A': # Apnea
                        label = 1.0
                    else:
                        label = 2.0

                    test_input_array.append([rri_intp, qrs_intp, age[data_index], sex[data_index]])
                    test_label_array.append(label)
        except Exception as e:
            print(f"Error processing data: {e}")

with open('test_input.pickle','wb') as f: 
    pickle.dump(test_input_array, f)
with open('test_label.pickle','wb') as f: 
    pickle.dump(test_label_array, f)

def visualize_processing_steps(data_name, window_index=1):
    """
    Visualize each step of signal processing for a given data file and window
    """
    # Constants from pre_proc.py
    FS = 100.0
    MARGIN = 10
    FS_INTP = 4

    # Setup the figure - removed seaborn style
    # plt.style.use('seaborn')  # Remove or comment this line
    fig, axs = plt.subplots(5, 1, figsize=(15, 20))
    fig.suptitle(f'Signal Processing Steps for {data_name}, Window {window_index}', fontsize=16)

    # 1. Raw ECG Signal
    signals, fields = wfdb.rdsamp(os.path.join('./data/', data_name))
    samp_from = window_index * 60 * FS
    samp_to = samp_from + 60 * FS
    
    time = np.arange(samp_from, samp_to) / FS
    axs[0].plot(time, signals[int(samp_from):int(samp_to), 0])
    axs[0].set_title('1. Raw ECG Signal')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')

    # 2. QRS Detection
    qrs_ann = wfdb.rdann(f'./data/{data_name}', 'qrs', 
                         sampfrom=int(samp_from - (MARGIN*100)), 
                         sampto=int(samp_to + (MARGIN*100))).sample
    
    # Plot QRS points on ECG
    qrs_in_window = [q for q in qrs_ann if samp_from <= q <= samp_to]
    qrs_times = np.array(qrs_in_window) / FS
    qrs_amp = get_qrs_amp(signals[:, 0], qrs_in_window)
    
    axs[1].plot(time, signals[int(samp_from):int(samp_to), 0])
    axs[1].scatter(qrs_times, [signals[q, 0] for q in qrs_in_window], 
                   color='red', label='QRS peaks')
    axs[1].set_title('2. QRS Detection')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')
    axs[1].legend()

    # 3. RR Intervals
    rri = np.diff(qrs_ann)
    rri_ms = rri.astype('float') / FS * 1000.0
    rri_times = qrs_times[:-1]  # Remove last point for RR intervals
    
    axs[2].plot(rri_times, rri_ms, 'o-')
    axs[2].set_title('3. RR Intervals')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('RR Interval (ms)')

    # 4. Filtered RR Intervals
    rri_filt = moving_median(rri_ms)
    axs[3].plot(rri_times, rri_ms, 'o-', label='Original')
    axs[3].plot(rri_times, rri_filt, 'r-', label='Filtered')
    axs[3].set_title('4. Median Filtered RR Intervals')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('RR Interval (ms)')
    axs[3].legend()

    # 5. Interpolated Signals
    time_intp, rri_intp = interp_cubic_spline(rri_filt, FS_INTP)
    qrs_time_intp, qrs_intp = interp_cubic_spline_qrs(qrs_ann, qrs_amp, FS_INTP)
    
    # Normalize signals
    rri_intp = rri_intp - np.mean(rri_intp)
    qrs_intp = qrs_intp - np.mean(qrs_intp)
    
    axs[4].plot(time_intp, rri_intp, label='RR Intervals')
    axs[4].plot(qrs_time_intp, qrs_intp, label='QRS Amplitudes')
    axs[4].set_title('5. Interpolated and Normalized Signals')
    axs[4].set_xlabel('Time (s)')
    axs[4].set_ylabel('Normalized Amplitude')
    axs[4].legend()

    plt.tight_layout()
    plt.show()

    # Print apnea annotation for this window
    apn_ann = wfdb.rdann(f'./data/{data_name}', 'apn', 
                         sampfrom=int(samp_from), 
                         sampto=int(samp_to-1)).symbol
    print(f"Apnea annotation for this window: {apn_ann[0]}")

# Example usage
if __name__ == "__main__":
    # Visualize first window of first training file
    visualize_processing_steps('a02', window_index=1)