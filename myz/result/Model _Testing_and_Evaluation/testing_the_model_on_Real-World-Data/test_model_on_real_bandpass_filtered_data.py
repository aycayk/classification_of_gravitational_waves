import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import butter, filtfilt
import torch
import matplotlib
import warnings
import torch.nn as nn
import numpy as np
import h5py
import matplotlib.pyplot as plt

####################-------------------------------------- Testing the model on real data with bandpass filtering ---------------------------------------#################################


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

model_path = "/Users/aycayk/Desktop/myz_project/myz/models/model_94.00.pt"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = StackedBiLSTMClassifier(input_size, hidden_size, num_layers, num_classes, dropout_prob, activation_function)
torch.save(model.state_dict(), model_path)
model.to(device)
model.eval()


file_paths = [
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW150914_4K_bbh.hdf5',
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW170104_4K_bbh.hdf5',
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW170809_4K_bbh.hdf5',
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW170817_4K_bns.hdf5',
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW190425_4K_bns.hdf5',
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW200105_4K_nsbh.hdf5',
    '/Users/aycayk/Desktop/myz_project/hdf5-4K/GW200115_4K_nsbh.hdf5'
]

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def normalize_to_real_world(signal, min_value=-1., max_value=1.):
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    normalized_signal = (signal - signal_min) / (signal_max - signal_min)
    normalized_signal = normalized_signal * (max_value - min_value) + min_value
    return normalized_signal

def visualize_data(original, normalized, filtered, time, file_name):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, original, label="Original Strain Data", alpha=0.7)
    plt.title(f"{file_name} - Original Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(time, filtered, label="Bandpass Filtered Data", color='green', alpha=0.7)
    plt.title(f"{file_name} - Bandpass Filtered Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(time, normalized, label="Normalized Data", color='orange', alpha=0.7)
    plt.title(f"{file_name} - Normalized Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def visualize_raw_logits(logits, class_labels):
    plt.figure(figsize=(8, 4))
    plt.bar(class_labels, logits, color=['blue', 'orange', 'green'], alpha=0.7)
    plt.title("Model Output (Raw Logits)")
    plt.ylabel("Logit Value")
    plt.xlabel("Class")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--') 
    plt.grid(axis='y')
    plt.show()


# Test on real data
def test_on_gwosc_data(model, segment, class_labels, device):
    model.eval()
    with torch.no_grad():
        segment_tensor = torch.tensor(segment, dtype=torch.float32).to(device)
        output = model(segment_tensor)
        print(f"output: {output}")
        
        visualize_raw_logits(output.cpu().numpy()[0], class_labels)


        _, predicted = torch.max(output, 1)
    return class_labels[predicted.item()]


lowcut = 20
highcut = 600
sampling_frequency = 4096


class_labels = ["BBH", "BNS", "NSBH"]
for i, file_path in enumerate(file_paths):
    with h5py.File(file_path, 'r') as hdf:
        strain_data = hdf['strain']['Strain'][:]

        filtered_strain = bandpass_filter(strain_data, lowcut, highcut, sampling_frequency)

        normalized_strain_data = normalize_to_real_world(filtered_strain)

        reshaped_data = normalized_strain_data.reshape(1, -1, input_size)

        predicted_labels = test_on_gwosc_data(model, reshaped_data, class_labels, device)
        print(f"Predicted Labels for {file_path.split('/')[-1]}: {predicted_labels}")

        time = np.linspace(0, len(strain_data) / sampling_frequency, len(strain_data))
        visualize_data(strain_data, filtered_strain, normalized_strain_data, time, file_path.split('/')[-1])










