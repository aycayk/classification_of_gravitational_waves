import pandas as pd
import numpy as np
import seaborn as sns
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
from gwpy.timeseries import TimeSeries
import os
import csv
import pycbc.noise
import pycbc.psd
from pycbc.filter import matched_filter
from scipy.signal import butter, filtfilt
from itertools import cycle
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc



def load_data():
    # Loading datasets
    file_paths = {
        "BBH": "/Users/aycayk/Desktop/myz_project/myz/data/1000_example/BBH_1000_4K.csv",
        "BNS": "/Users/aycayk/Desktop/myz_project/myz/data/1000_example/BNS_1000_4K.csv",
        "NSBH": "/Users/aycayk/Desktop/myz_project/myz/data/1000_example/NSBH_1000_4K.csv",
        }

    datasets = {key: pd.read_csv(path) for key, path in file_paths.items()}
    return datasets



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

            #noise insert 
            if add_noise: 
                features_noised = add_gauss_noise_to_signal(features) 
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



def im_danger():
    # bandpass filtering
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    # Sampling frequency and sequence length
    sampling_frequency = 4096  # Hz
    sequence_length = sampling_frequency * 2  # 2 seconds (example)

    # Class labels for your dataset (adjust as needed)
    class_labels = [0, 0, 1, 1, 2, 2]  # Example: 0 = BBH, 1 = BNS, 2 = NSBH

    # Process each file
    for file_path, class_label in zip(file_paths, class_labels):
        with h5py.File(file_path, 'r') as hdf:
            # Extract strain data
            strain = hdf['strain']['Strain'][:]

        # Apply bandpass filter
        lowcut = 20
        highcut = 300
        filtered_strain = bandpass_filter(strain, lowcut, highcut, sampling_frequency)

        # Reshape strain data into sequences
        num_sequences = len(filtered_strain) // sequence_length
        reshaped_strain = filtered_strain[:num_sequences * sequence_length].reshape((-1, 1, sequence_length))  # Shape: (num_sequences, 1, sequence_length)



def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels



def plot_roc_curve(y_true, y_scores, num_classes):
    y_true_binarized = np.eye(num_classes)[y_true]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    macro_fpr = all_fpr
    macro_tpr = mean_tpr
    macro_auc = auc(macro_fpr, macro_tpr)


    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'green', 'red'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot(macro_fpr, macro_tpr, color='darkorange', linestyle='--', lw=2,
             label=f"Macro Avg (AUC = {macro_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()



def evaluate_model_with_roc(model, test_loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    return accuracy, all_preds, all_labels, np.array(all_probs)



def main():
    datasets = load_data()
    print("Data are loaded!")

    # preprocess
    sequence_length = 64
    X, y = preprocess_data(datasets, sequence_length, add_noise=True)
    print(f"X.shape: {X.shape}, y.shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
    print("Train and Test data are generated!") 
  
    # model initialization   
    input_size = sequence_length
    hidden_size = 128
    num_layers = 2
    num_classes = 3
    dropout_prob = 0.4
    activation_function = "relu"

    model = StackedBiLSTMClassifier(input_size, hidden_size, num_layers, num_classes, dropout_prob, activation_function)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)
    print("Model Initializd!")

    batch_size, workers = 16, 4
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)
    print("Dataloader Initializd!")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # model trainig
    best_accuracy = 0
    num_epochs = 20

    # train Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for X_batch, y_batch in progress_bar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            progress_bar.set_postfix(loss=loss.item(), accuracy=f"{100 * correct / total:.2f}%")
        
        train_accuracy = 100 * correct / total
        test_accuracy, _, _ = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

        # save model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_path = f"/Users/aycayk/Desktop/myz_project/myz/models/model_2_{best_accuracy:.2f}.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation accuracy: {best_accuracy:.2f}%")

        print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print("Train is Completed!")
       
        
    test_accuracy, all_preds, all_labels, all_probs = evaluate_model_with_roc(model, test_loader, device, num_classes)
    print(f"Test Accuracy: {test_accuracy:.2f}%")



    # confusion matrix
    class_labels = ["BBH", "BNS", "NSBH"]
    _, all_preds, all_labels = evaluate_model(model, test_loader, device)
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_labels)))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    # classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_labels))

    plot_roc_curve(all_labels, all_probs, num_classes)

    return



if __name__ == "__main__":
    main()
