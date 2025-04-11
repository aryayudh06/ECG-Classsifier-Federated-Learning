import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import socket
import pickle
import time
import numpy as np
import wfdb
import os
from model import ECGResNet18
from utils import send_msg, recv_msg
from sklearn.model_selection import train_test_split


class FederatedClient:
    def __init__(self, client_id, train_data, test_dataset=None, server_host='localhost', server_port=5000):
        self.client_id = client_id
        self.server_host = server_host
        self.server_port = server_port
        
        self.train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) if test_dataset else None

        self.model = ECGResNet18(num_classes=6)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def connect_to_server(self):
        try:
            self.socket.connect((self.server_host, self.server_port))
            print(f"Client {self.client_id} connected to server at {self.server_host}:{self.server_port}")
            return True
        except socket.error as e:
            print(f"Connection failed for client {self.client_id}: {e}")
            return False

    def get_global_model(self):
        message = {
            'type': 'GET_MODEL',
            'client_id': self.client_id
        }
        send_msg(self.socket, message)
        response = recv_msg(self.socket)
        if response and response['type'] == 'MODEL':
            self.model.load_state_dict(response['model_state'])
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        else:
            raise RuntimeError("Invalid response from server or no model received")

    def local_train(self, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()

    def evaluate_model(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.train_loader:
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        accuracy = 100 * correct / total
        print(f"Client {self.client_id} - [TRAIN] Accuracy: {accuracy:.2f}%")
        return accuracy

    def evaluate_on_test(self):
        self.model.eval()
        if self.test_loader is None:
            print(f"Client {self.client_id} - No test data available.")
            return
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(self.test_loader)
        print(f"Client {self.client_id} - [TEST] Accuracy: {accuracy:.2f}% | Loss: {avg_loss:.4f}")
        return accuracy, avg_loss

    def participate_in_fl(self, rounds=1):
        for rnd in range(rounds):
            print(f"Client {self.client_id} - Round {rnd+1}/{rounds}")
            self.get_global_model()

            self.local_train()
            self.evaluate_model()

            message = {
                'type': 'UPDATE',
                'client_id': self.client_id,
                'model_state': self.model.state_dict()
            }
            send_msg(self.socket, message)
            print(f"Client {self.client_id} - Sent updated model to server")

        # Hanya tes sekali setelah semua training selesai
        self.evaluate_on_test()




def load_mitbih_record_rr(record_name, data_dir, fs=360):
    """Load MIT-BIH record and extract RR interval features"""
    record_path = os.path.join(data_dir, record_name)
    signals, fields = wfdb.rdsamp(record_path)
    annotations = wfdb.rdann(record_path, 'atr')
    
    rr_features = []
    rr_labels = []

    # Ambil hanya label yang digunakan
    beat_classes = ['N', 'L', 'R', 'V', 'A', 'F']
    beat_class_map = {'N':0, 'L':1, 'R':2, 'V':3, 'A':4, 'F':5}
    
    beat_indices = []
    beat_symbols = []
    for i, symbol in enumerate(annotations.symbol):
        if symbol in beat_classes:
            sample = annotations.sample[i]
            beat_indices.append(sample)
            beat_symbols.append(symbol)
    
    # Hitung RR interval
    rr_intervals = np.diff(beat_indices)  # dalam sample
    rr_intervals_sec = rr_intervals / fs  # dalam detik
    
    for i in range(1, len(rr_intervals) - 1):
        # Gunakan 3 RR interval: RR-1, RR0, RR+1
        rr_window = [
            rr_intervals_sec[i-1],
            rr_intervals_sec[i],
            rr_intervals_sec[i+1]
        ]
        rr_features.append(rr_window)
        rr_labels.append(beat_class_map[beat_symbols[i]])

    return np.array(rr_features), np.array(rr_labels)

def prepare_client_data_rr(client_id, data_dir='mitdb', records_per_client=5):
    """Prepare MIT-BIH data with RR interval features for specific client"""
    all_records = [f'{i:03}' for i in range(100, 124)] + [f'{i:03}' for i in range(200, 224)]
    
    start_idx = (client_id - 1) * records_per_client
    client_records = all_records[start_idx:start_idx + records_per_client]
    
    all_rr = []
    all_labels = []
    
    for record in client_records:
        try:
            rr, labels = load_mitbih_record_rr(record, data_dir)
            all_rr.append(rr)
            all_labels.append(labels)
        except Exception as e:
            print(f"Error loading record {record}: {e}")
            continue
    
    if len(all_rr) == 0:
        raise ValueError(f"No valid data loaded for client {client_id}")
    
    features = np.vstack(all_rr)
    labels = np.concatenate(all_labels)
    
    features_tensor = torch.FloatTensor(features)
    labels_tensor = torch.LongTensor(labels)
    
    return TensorDataset(features_tensor, labels_tensor)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python client.py <client_id> [data_directory]")
        sys.exit(1)
    
    client_id = int(sys.argv[1])
    data_dir = 'data/mit-bih-arrhythmia-database-1.0.0/'
    
    try:
        dataset = prepare_client_data_rr(client_id, data_dir)
        features, labels = dataset.tensors  # ambil tensors-nya

        # Split 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            features.numpy(), labels.numpy(), test_size=0.2, random_state=42
        )

        # Buat dataset baru
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

        # Create and run client pakai train_dataset
        client = FederatedClient(client_id, train_dataset, test_dataset=test_dataset)

        if client.connect_to_server():
            client.participate_in_fl(rounds=10)
        else:
            print(f"Client {client_id} failed to connect to server")

    except Exception as e:
        print(f"Client {client_id} error: {e}")