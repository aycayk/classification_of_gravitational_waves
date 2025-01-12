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
import gwpy
import pylab
from tqdm import tqdm
from gwpy.timeseries import TimeSeries
import os
import csv
import pycbc.noise
import pycbc.psd
from pycbc.filter import matched_filter
from scipy.signal import butter, filtfilt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

############### ------------------------------------------------------- MODEL TEST ACCURACY ON GENERATED DATA --------------------------------#######################################

def load_data():
    # Veri setlerini yükleme
    file_paths = {
        "BBH": "/Users/aycayk/Desktop/myz_project/myz/data/10_example/BBH_10_4K.csv",
        "BNS": "/Users/aycayk/Desktop/myz_project/myz/data/10_example/BNS_10_4K.csv",
        "NSBH": "/Users/aycayk/Desktop/myz_project/myz/data/10_example/NSBH_10_4K.csv",
        }

    datasets = {key: pd.read_csv(path) for key, path in file_paths.items()}
    return datasets


def plot_signals(signals, labels):
    for signal, label in zip(signals, labels):
        plt.figure(figsize=(10, 4))  # Create a new figure for each signal
        plt.plot(signal)
        plt.title(label)
        #plt.xlabel("Sample Index")
        #plt.ylabel("Amplitude")
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
    flow = 30.0
    delta_f = 1.0 / 16
    flen = int(2048 / delta_f) + 1
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

    delta_t = 1.0 / 4096
    tsamples = int(duration / delta_t)
    noise = pycbc.noise.noise_from_psd(tsamples, delta_t, psd)

    noise *= 1e21
    noise *= 0.4
    noise = TimeSeries.from_pycbc(noise)

    if not isinstance(signal, TimeSeries):
        signal = TimeSeries(signal, sample_rate=1.0/delta_t)
        
        noise_signal = noise.inject(signal)    
        #noise_signal *= 1e-17
            
        return noise_signal.value
    
def preprocess_data(datasets, sequence_length, add_noise=False): 
    print(f"Preprocess Data - Noise: {add_noise}")
    X, y = [], []
    for label, df in tqdm(datasets.items()):
        for _, row in df.iterrows():
            features = row.drop("label").values
            
            # Check if feature divisible by sequence length
            if len(features) % sequence_length != 0:
                continue 

            #noise insert !
            if add_noise: 
                features_noised = add_gauss_noise_to_signal(features) 
                #if row['label'] == 0:
                    #plot_signals([features, features_noised], [f"NOISE FREE SIGNAL - {row['label']}" , f"NOISED SIGNAL - {row['label']}"])
                features = features_noised

            # normalize data
            features = normalize_to_real_world(features)  

            sequences = features.reshape(1, -1, sequence_length)

            X.append(sequences)
            y.append(row["label"])
    return np.vstack(X), np.array(y)

class StackedBiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.3, activation_function="relu"):
        super(StackedBiLSTMClassifier, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True, dropout=dropout_prob, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.activation = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}[activation_function]
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.batch_norm(out[:, -1, :])
        out = self.dropout(out)
        out = self.activation(out)
        return self.fc(out)

def main():
    datasets = load_data()
    print("Data are loaded!")

    # preprocess
    sequence_length = 64
    X, y = preprocess_data(datasets, sequence_length, add_noise=True)
    print(f"X.shape: {X.shape}, y.shape: {y.shape}")

    # model initialization   
    input_size = sequence_length
    hidden_size = 128
    num_layers = 2
    num_classes = 3
    dropout_prob = 0.4
    activation_function = "relu"

    model_path = "/Users/aycayk/Desktop/myz_project/myz/models/model_2_97.67.pt"    
    model = StackedBiLSTMClassifier(input_size, hidden_size, num_layers, num_classes, dropout_prob, activation_function)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device).eval()
    print("Model Initializd!")

    batch_size, workers = 16, 4
    torch_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    data_loader = DataLoader(torch_dataset, batch_size=1, num_workers=1,shuffle=True)
    print("Dataloader Initializd!")

    correct = 0
    total = 0
    for X_batch, y_batch in tqdm(data_loader, desc=f"Testing"):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.no_grad():
            outputs = model(X_batch)
        
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        print(f"outputs: {outputs}, pred: {predicted}, y_batch: {y_batch}")
        total += y_batch.size(0)
        
    accuracy = 100 * correct / total
    print(f"Test Acc: {accuracy:.2f}%")

    return


if __name__ == "__main__":
    main()
