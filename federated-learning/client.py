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


class FederatedClient:
    def __init__(self, client_id, train_data, server_host='localhost', server_port=5000):
        self.client_id = client_id
        self.server_host = server_host
        self.server_port = server_port
        
        # Local data
        self.train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        
        # Model and optimizer
        self.model = ECGResNet18(num_classes=6)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    def connect_to_server(self):
        """Connect to the federated learning server"""
        try:
            self.socket.connect((self.server_host, self.server_port))
            print(f"Client {self.client_id} connected to server at {self.server_host}:{self.server_port}")
            return True
        except socket.error as e:
            print(f"Connection failed for client {self.client_id}: {e}")
            return False
    
    def get_global_model(self):
        """Request and receive global model from server"""
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
        """Train model on local data"""
        self.model.train()
        for epoch in range(epochs):
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()
    
    def participate_in_fl(self, rounds=10):
        """Participate in federated learning rounds"""
        for round in range(rounds):
            print(f"Client {self.client_id} starting round {round + 1}")
            
            # Get global model from server
            self.get_global_model()
            
            # Train locally
            local_state = self.local_train(epochs=1)
            
            # Send updated model to server
            message = {
                'type': 'SEND_MODEL',
                'client_id': self.client_id,
                'model_state': local_state
            }
            send_msg(self.socket, message)
            
            # Wait for acknowledgment
            response = recv_msg(self.socket)
            if not response or response['type'] != 'ACK':
                print(f"Client {self.client_id} failed to receive ACK from server")
                break
            
            print(f"Client {self.client_id} completed round {round + 1}")
            time.sleep(1)  # Small delay between rounds

def load_mitbih_record(record_name, data_dir):
    """Load a single MIT-BIH record"""
    record_path = os.path.join(data_dir, record_name)
    signals, fields = wfdb.rdsamp(record_path)
    annotations = wfdb.rdann(record_path, 'atr')
    
    # Get beats and their types (only using MLII lead)
    beats = []
    beat_types = []
    
    for i, symbol in enumerate(annotations.symbol):
        if symbol in ['N', 'L', 'R', 'V', 'A', 'F']:  # Normal and common arrhythmia types
            sample = annotations.sample[i]
            if sample > 180 and sample < len(signals) - 180:  # Ensure we have enough samples
                beat = signals[sample-180:sample+180, 0]  # Using MLII lead
                beats.append(beat)
                
                # Map annotation to class index
                if symbol == 'N':   # Normal beat
                    beat_types.append(0)
                elif symbol == 'L': # Left bundle branch block beat
                    beat_types.append(1)
                elif symbol == 'R':  # Right bundle branch block beat
                    beat_types.append(2)
                elif symbol == 'V':  # Premature ventricular contraction
                    beat_types.append(3)
                elif symbol == 'A':  # Atrial premature contraction
                    beat_types.append(4)
                elif symbol == 'F': # Fusion of ventricular and normal
                    beat_types.append(5)
    
    return np.array(beats), np.array(beat_types)

def prepare_client_data(client_id, data_dir='mitdb', records_per_client=5):
    """Prepare MIT-BIH data for specific client"""
    # List of available MIT-BIH records (first 48 records)
    all_records = [f'{i:03}' for i in range(100, 124)] + [f'{i:03}' for i in range(200, 224)]
    
    # Select records for this client
    start_idx = (client_id - 1) * records_per_client
    client_records = all_records[start_idx:start_idx + records_per_client]
    
    # Load and combine selected records
    all_beats = []
    all_labels = []
    
    for record in client_records:
        try:
            beats, labels = load_mitbih_record(record, data_dir)
            all_beats.append(beats)
            all_labels.append(labels)
        except Exception as e:
            print(f"Error loading record {record}: {e}")
            continue
    
    if len(all_beats) == 0:
        raise ValueError(f"No valid data loaded for client {client_id}")
    
    # Combine all beats and labels
    beats = np.vstack(all_beats)
    labels = np.concatenate(all_labels)
    
    # Convert to PyTorch tensors
    beats_tensor = torch.FloatTensor(beats).unsqueeze(1)  # Add channel dimension
    labels_tensor = torch.LongTensor(labels)
    
    return TensorDataset(beats_tensor, labels_tensor)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python client.py <client_id> [data_directory]")
        sys.exit(1)
    
    client_id = int(sys.argv[1])
    data_dir = 'data/mit-bih-arrhythmia-database-1.0.0/'
    
    try:
        # Prepare client-specific ECG data from MIT-BIH
        train_data = prepare_client_data(client_id, data_dir)
        
        # Create and run client
        client = FederatedClient(client_id, train_data)
        if client.connect_to_server():
            client.participate_in_fl(rounds=10)
        else:
            print(f"Client {client_id} failed to connect to server")
    except Exception as e:
        print(f"Client {client_id} error: {e}")