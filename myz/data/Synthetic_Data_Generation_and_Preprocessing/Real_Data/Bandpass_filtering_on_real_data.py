import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import h5py


file_paths = [
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW150914_4K_bbh.hdf5',
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW170104_4K_bbh.hdf5',
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW170817_4K_bns.hdf5',
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW190425_4K_bns.hdf5',
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW200105_4K_nsbh.hdf5',
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW200115_4K_nsbh.hdf5'
]


sampling_frequency = 4096  # Hz
duration = 32  # seconds
time = np.linspace(0, duration, int(sampling_frequency * duration))


for file_path in file_paths:
    with h5py.File(file_path, 'r') as hdf:
        metadata = {key: hdf['meta'][key][()] for key in hdf['meta']}    
        strain = hdf['strain']['Strain'][:]

    # Bandpass filter 
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        '''A bandpass filter is a tool that isolates a specific range of 
        frequencies from the input signal while removing (attenuating) frequencies 
        outside this range. It doesn't remove all noise but focuses on frequencies
        where the signal of interest (in this case, the gravitational wave) is most likely to be found.
        '''
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
    
        return y


    # Define frequency range (usually 20 Hz - 300 Hz)
    lowcut = 20
    highcut = 300

    filtered_strain = bandpass_filter(strain, lowcut, highcut, sampling_frequency)
    
    
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time, strain, label="Original Strain Data", alpha=0.7)
    plt.title("Original BBH Strain Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")

    plt.subplot(2, 1, 2)
    plt.plot(time, filtered_strain, label="Filtered Strain Data", color='r', alpha=0.7)
    plt.title(f"{file_path}")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")

    plt.tight_layout()
    plt.legend()
    plt.show()
