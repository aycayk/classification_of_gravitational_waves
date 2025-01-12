import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
from pycbc import distributions
from pycbc.waveform import get_td_waveform, td_approximants
from pycbc.detector import Detector
import matplotlib.pyplot as plt
import numpy as np
import gwpy
import pylab
from tqdm.notebook import tqdm
from gwpy.timeseries import TimeSeries
import pandas as pd
import os
import csv
import pycbc.noise
import pycbc.psd
from pycbc.filter import matched_filter
from scipy.signal import butter, filtfilt


signal_sampling_frequency = 4096 
signal_duration = 32  
duration = signal_duration


def plot_single_signal(signal, desc="Graviational Wave"):
    plt.plot(signal)
    plt.title(f"{desc}")
    plt.show()


def plot_signals(signals, labels):
    for signal, label in zip(signals, labels):
        plt.figure(figsize=(10, 4))  
        plt.plot(signal)
        plt.title(label)
        plt.grid(True)
        plt.tight_layout()
    plt.show()


def normalize_to_real_world(signal, min_value=-1., max_value=1.):
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    normalized_signal = (signal - signal_min) / (signal_max - signal_min)  
    normalized_signal = normalized_signal * (max_value - min_value) + min_value  
    return normalized_signal


def add_gauss_noise_to_signal(signal, flow=30.0, delta_f=1.0/16, delta_t=1.0/4096, duration=32):

    flow = np.random.uniform(20.0, 40.0)
    delta_f = np.random.uniform(1.0/32, 1.0/8)
    flen = int(2048 / delta_f) + 1
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

    if psd is None or len(psd) == 0:
        raise ValueError("Invalid PSD generated. Check 'flow', 'delta_f', or 'flen'.")

    delta_t = 1.0 / 4096
    tsamples = int(duration / delta_t)

    if tsamples < 1:
        raise ValueError(f"Invalid number of samples ({tsamples}). Check 'duration' or 'delta_t'.")

    noise = pycbc.noise.noise_from_psd(tsamples, delta_t, psd)

    noise *= np.random.uniform(0.8e21, 1.2e21)
    noise *= np.random.uniform(0.3, 0.6)
    noise = TimeSeries.from_pycbc(noise)

    if not isinstance(signal, TimeSeries):
        signal = TimeSeries(signal, sample_rate=1.0/delta_t)
    
    noise_signal = noise.inject(signal)    
        
    return noise_signal.value


def main():
    bbh_dataset = pd.read_csv('/Users/aycayk/Desktop/myz_project/myz/data/1000_example/BBH_1000_4K.csv')

    labels = bbh_dataset.iloc[:, -1].to_numpy()
    signals = bbh_dataset.iloc[:, :-1].to_numpy()
    for i in range(20):
        signal_example  = signals[i]
        print(f"signal_example: {signal_example} - shape: {signal_example.shape},  type: {type(signal_example)} ")

        if np.all(signal_example == 0):
            print(f"Skipping Signal {i} because it contains only zeros.")
            continue

        if np.allclose(signal_example, 0, atol=1e-6):
            print(f"Signal {i} has very low values. Adding small random noise.")
            signal_example = np.random.normal(0, 1e-6, len(signal_example))

        try:
            noisy_signal_gauss = add_gauss_noise_to_signal(signal_example, duration=32)
        except Exception as e:
            print(f"Error generating noise for Signal {i}: {e}")
            continue

        signals_to_plot = [signal_example, noisy_signal_gauss]
        labels_to_plot = [f"{i}. BBH Signal", f"{i}. BBH Signal + Noise"]

        plot_signals(signals_to_plot, labels_to_plot)
    return



if __name__ == "__main__":
    main()