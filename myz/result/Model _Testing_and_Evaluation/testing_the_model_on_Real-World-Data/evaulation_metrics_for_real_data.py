import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score


#####################-----------------------------------------------EVALUATION METRICS FOR GENERATED DATA-----------------------------------------------------########################


sampling_frequency = 4096
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


model_path = "/Users/aycayk/Desktop/myz_project/myz/models/model_95.33.pt"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = StackedBiLSTMClassifier(input_size, hidden_size, num_layers, num_classes, dropout_prob, activation_function)
torch.save(model.state_dict(), model_path)
model.to(device)
model.eval()


# File paths for GW data
file_paths_with_labels = [
    ('/Users/aycayk/Desktop/myz_project/hdf5-4K/GW150914_4K_bbh.hdf5', 0),  # BBH -> 0
    ('/Users/aycayk/Desktop/myz_project/hdf5-4K/GW170104_4K_bbh.hdf5', 0),  # BBH -> 0
    ('/Users/aycayk/Desktop/myz_project/hdf5-4K/GW170817_4K_bns.hdf5', 1),  # BNS -> 1
    ('/Users/aycayk/Desktop/myz_project/hdf5-4K/GW190425_4K_bns.hdf5', 1),  # BNS -> 1
    ('/Users/aycayk/Desktop/myz_project/hdf5-4K/GW200105_4K_nsbh.hdf5', 2), # NSBH -> 2
    ('/Users/aycayk/Desktop/myz_project/hdf5-4K/GW200115_4K_nsbh.hdf5', 2)  # NSBH -> 2
]


# Normalization 
def normalize_to_real_world(signal, min_value=-1., max_value=1.):
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    normalized_signal = (signal - signal_min) / (signal_max - signal_min)
    normalized_signal = normalized_signal * (max_value - min_value) + min_value
    return normalized_signal


def visualize_normalized_data(original, normalized, time, file_name):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time, original, label="Original Strain Data", alpha=0.7)
    plt.title(f"{file_name} - Original Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time, normalized, label="Normalized Data", color='orange', alpha=0.7)
    plt.title(f"{file_name} - Normalized Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def test_on_gwosc_data(model, segments, class_labels, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for segment in segments:
            segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(device)
            output = model(segment_tensor)
            _, predicted = torch.max(output, 1)
            predictions.append(class_labels[predicted.item()])
    return predictions


def test_model_and_evaluate(file_paths_with_labels, model, device):
    true_labels = []
    predicted_labels = []
    
    class_labels = ["BBH", "BNS", "NSBH"]  # 0 -> BBH, 1 -> BNS, 2 -> NSBH
    sequence_length = 64 
    
    for file_path, true_label in file_paths_with_labels:
        with h5py.File(file_path, 'r') as hdf:
            # strain data
            strain_data = hdf['strain']['Strain'][:]

            # normalized data
            normalized_strain_data = normalize_to_real_world(strain_data)
            
            # reshape the data for the model
            reshaped_data = normalized_strain_data.reshape(-1, sequence_length, sequence_length)
            
            # predict 
            predictions = []
            with torch.no_grad():
                for segment in reshaped_data:
                    segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(device)
                    output = model(segment_tensor)
                    _, predicted = torch.max(output, 1)
                    predictions.append(predicted.item())
            
            predicted_label = np.argmax(np.bincount(predictions))
            
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
    
    evaluate_model_performance(true_labels, predicted_labels, class_labels)



def evaluate_model_performance(true_labels, predicted_labels, class_labels):

    for i, (true, predicted) in enumerate(zip(true_labels, predicted_labels)):
        print(f"Sample {i + 1}: True Label = {true}, Predicted Label = {predicted}")

    # accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=class_labels, zero_division=0))
    
    balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
    print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")

    # confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(class_labels)))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels).plot(cmap='viridis')
    plt.show()




test_model_and_evaluate(file_paths_with_labels, model, device)

