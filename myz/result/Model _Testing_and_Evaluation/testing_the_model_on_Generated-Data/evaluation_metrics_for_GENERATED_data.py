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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score


###############--------------------------------------------- EVALUATION METRICS for GENERATED DATA ----------------------------------------------################################

sequence_length = 64

# Model Definition
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


# Initialize Model
input_size = sequence_length
hidden_size = 128
num_layers = 2
num_classes = 3
dropout_prob = 0.4
activation_function = "relu"

model_path = "/Users/aycayk/Desktop/myz_project/myz/models/model_95.33.pt"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = StackedBiLSTMClassifier(input_size, hidden_size, num_layers, num_classes, dropout_prob, activation_function)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.to(device)
model.eval()


# Normalization Function
def normalize_to_real_world(signal, min_value=-1., max_value=1.):
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    normalized_signal = (signal - signal_min) / (signal_max - signal_min)
    normalized_signal = normalized_signal * (max_value - min_value) + min_value
    return normalized_signal
#

# Visualize Data
def visualize_normalized_data(original, normalized, time, file_name):
    plt.figure(figsize=(12, 6))

    # Plot original data
    plt.subplot(2, 1, 1)
    plt.plot(time, original, label="Original Strain Data", alpha=0.7)
    plt.title(f"{file_name} - Original Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.legend()
    plt.grid()

    # Plot normalized data
    plt.subplot(2, 1, 2)
    plt.plot(time, normalized, label="Normalized Data", color='orange', alpha=0.7)
    plt.title(f"{file_name} - Normalized Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

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

csv_file_paths_with_labels = [
    ('/Users/aycayk/Desktop/myz_project/myz/data/10_example/BBH_10_4K.csv'),
    ('/Users/aycayk/Desktop/myz_project/myz/data/10_example/BNS_10_4K.csv'),
    ('/Users/aycayk/Desktop/myz_project/myz/data/10_example/NSBH_10_4K.csv'),
]

def test_on_csv_data_with_labels(model, csv_file_paths, device):
    true_labels = []
    predicted_labels = []
    class_labels = ["BBH", "BNS", "NSBH"]  

    for cls_id, file_path in  enumerate(csv_file_paths):
        data = pd.read_csv(file_path, header=None, low_memory=False)
        
        labels = data.iloc[1:11:, -1].values 
        strain_data = data.iloc[1:11, :-1].values 
        
        for i, (label, strain) in tqdm(enumerate(zip(labels, strain_data))):
            strain = np.array(strain, dtype=np.float32)

            #noise = add_gauss_noise_to_signal(strain)
            normalized = normalize_to_real_world(strain)
            reshaped_data = normalized.reshape(1, -1, sequence_length)

            with torch.no_grad():
                segment_tensor = torch.tensor(reshaped_data, dtype=torch.float32).to(device)
                output = model(segment_tensor)
                _, predicted = torch.max(output, 1)
                print(f"{i}. {class_labels[cls_id]}, output => {output}, pred: {predicted}")

            predicted_labels.append(predicted.item())
            true_labels.append(int(label))  
    print(f"true_labels: {true_labels}, \npredicted_labels: {predicted_labels}, \nclass_labels: {class_labels}")
    evaluate_model_performance(true_labels, predicted_labels, class_labels)



def evaluate_model_performance(true_labels, predicted_labels, class_labels):
    print(f"Accuracy: {accuracy_score(true_labels, predicted_labels) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=class_labels, zero_division=0))
    print(f"Balanced Accuracy: {balanced_accuracy_score(true_labels, predicted_labels) * 100:.2f}%")

    cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(class_labels)))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels).plot(cmap='viridis')
    plt.show()



test_on_csv_data_with_labels(model, csv_file_paths_with_labels, device)


def test_on_csv_data_with_labels(model, csv_file_paths, device, output_file="predictions.csv"):
    true_labels = []
    predicted_labels = []
    class_labels = ["BBH", "BNS", "NSBH"]
    prediction_rows = []  # To store predictions for saving

    for cls_id, file_path in enumerate(csv_file_paths):
        data = pd.read_csv(file_path, header=None, low_memory=False)
        
        labels = data.iloc[1:11, -1].values
        strain_data = data.iloc[1:11, :-1].values
        
        for i, (label, strain) in tqdm(enumerate(zip(labels, strain_data))):
            strain = np.array(strain, dtype=np.float32)

            normalized = normalize_to_real_world(strain)
            reshaped_data = normalized.reshape(1, -1, sequence_length)

            with torch.no_grad():
                segment_tensor = torch.tensor(reshaped_data, dtype=torch.float32).to(device)
                output = model(segment_tensor)
                _, predicted = torch.max(output, 1)

            predicted_label = predicted.item()
            predicted_labels.append(predicted_label)
            true_labels.append(int(label))

            # Save row for predictions
            prediction_rows.append({
                "True Label": class_labels[int(label)],
                "Predicted Label": class_labels[predicted_label],
                "Confidence Scores": output.cpu().numpy().tolist()
            })

    # Save predictions to a CSV file
    keys = ["True Label", "Predicted Label", "Confidence Scores"]
    with open(output_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"Predictions saved to {output_file}")
    print(f"true_labels: {true_labels}, \npredicted_labels: {predicted_labels}, \nclass_labels: {class_labels}")
    evaluate_model_performance(true_labels, predicted_labels, class_labels)
