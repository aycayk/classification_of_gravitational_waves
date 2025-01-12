import torch
import torch.nn as nn
import numpy as np
import h5py
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import numpy as np
import h5py
import matplotlib.pyplot as plt


###################-------------------------------------------- PREDICTED LABELS FOR GWOSC's REAL DATA's ----------------------------------------------------######################################


sequence_length = 64


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



input_size = sequence_length
hidden_size = 128
num_layers = 2
num_classes = 3
dropout_prob = 0.4
activation_function = "relu"



model_path = "/Users/aycayk/Desktop/myz_project/myz/models/model_2_97.67.pt"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = StackedBiLSTMClassifier(input_size, hidden_size, num_layers, num_classes, dropout_prob, activation_function)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()



# File paths for GW data
file_paths = [
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW150914_4K_bbh.hdf5',
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW170104_4K_bbh.hdf5',
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW170817_4K_bns.hdf5',
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW190425_4K_bns.hdf5',
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW200105_4K_nsbh.hdf5',
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW200115_4K_nsbh.hdf5'
]



# Normalization function
def normalize_to_real_world(signal, min_value=-1., max_value=1.):
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    normalized_signal = (signal - signal_min) / (signal_max - signal_min)
    normalized_signal = normalized_signal * (max_value - min_value) + min_value
    return normalized_signal

sampling_frequency = 4096



def visualize_normalized_data(original, normalized, time, file_name):
    if len(original) != len(normalized) or len(time) != len(original):
        raise ValueError("All input arrays (original, normalized, time) must have the same length.")

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, original, label="Original Strain Data", alpha=0.7, color='blue')
    plt.title(f"{file_name} - Original Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(time, normalized, label="Normalized Data", color='orange', alpha=0.7)
    plt.title(f"{file_name} - Normalized Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



class_labels = ["BBH", "BNS", "NSBH"]
for file_path in file_paths:
    with h5py.File(file_path, 'r') as hdf:
        strain_data = hdf['strain']['Strain'][:]
        normalized_strain_data = normalize_to_real_world(strain_data)
        reshaped_data = normalized_strain_data.reshape(1, -1, input_size)

        with torch.no_grad():
            segment_tensor = torch.tensor(reshaped_data, dtype=torch.float32).to(device)
            output = model(segment_tensor)
            _, predicted = torch.max(output, 1)
            print(f"output: {output}, pred: {predicted}")
        predicted_label = class_labels[predicted.item()]
        print(f"Predicted Labels for {file_path.split('/')[-1]}: {predicted_label}")
        
 









