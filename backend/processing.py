import neurokit2 as nk
import numpy as np
from scipy import interpolate

def process_ecg(ecg_signal, fs=100):
    """Simulate hardware processing using software methods"""
    # QRS Detection
    signals, info = nk.ecg_peaks(ecg_signal, sampling_rate=fs)
    qrs_peaks = np.where(signals["ECG_R_Peaks"] == 1)[0]
    
    # Check if we have enough peaks
    if len(qrs_peaks) < 4:
        raise ValueError(f"Only {len(qrs_peaks)} QRS peaks detected. Need at least 4 for processing. Try a longer ECG recording.")
    
    # RR Interval Calculation
    rri = np.diff(qrs_peaks) / fs * 1000  # Convert to milliseconds
    
    # Moving Median Filter
    def moving_median(data, window_size=5):
        return np.array([np.median(data[max(0,i-window_size//2):min(len(data),i+window_size//2+1)])
                        for i in range(len(data))])
    
    rri_filt = moving_median(rri)
    
    # Cubic Spline Interpolation with adaptive order
    def interp_cubic(rri, fs=4):
        # Use data length to determine spline order (k)
        k = min(3, len(rri)-1)  # Use cubic (k=3) if possible, fall back to lower order if needed
        
        time_points = np.cumsum(rri)/1000
        time_points = time_points - time_points[0]  # Start at zero
        
        if len(time_points) <= k:
            raise ValueError(f"Not enough data points ({len(time_points)}) for spline interpolation of order {k}.")
            
        new_time = np.arange(0, time_points[-1], 1/fs)
        
        # If too few points for a cubic spline, use linear interpolation instead
        if k < 3:
            return np.interp(new_time, time_points, rri)
        else:
            tck = interpolate.splrep(time_points, rri, k=k, s=0)
            return interpolate.splev(new_time, tck)
    
    try:
        rri_intp = interp_cubic(rri_filt)
        
        # If we don't have enough interpolated points, pad with zeros
        if len(rri_intp) < 240:
            rri_intp = np.pad(rri_intp, (0, 240 - len(rri_intp)), 'constant')
        
        return rri_intp[:240]  # Ensure 240 points
    except Exception as e:
        # Provide more helpful error message
        raise ValueError(f"Interpolation failed: {str(e)}. This may be due to insufficient or irregular ECG data.")
