import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import socket
import pickle
import time
from model import ECGResNet18  # Changed import
from utils import send_msg, recv_msg


class FederatedClient:
    def __init__(self, client_id, train_data, server_host='localhost', server_port=5000, client_port=5001):
        self.client_id = client_id
        self.server_host = server_host
        self.server_port = server_port
        self.client_port = client_port
        
        # Local data
        self.train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        
        # Model and optimizer
        self.model = ECGResNet18(num_classes=6)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # Consistent optimizer
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('localhost', client_port))
    
    # [Rest of the methods remain exactly the same]
    
    def get_global_model(self):
        """Request and receive global model from server"""
        message = {'type': 'GET_MODEL'}
        send_msg(self.socket, message)

        response = recv_msg(self.socket)
        
        if response and response['type'] == 'MODEL':
            self.model.load_state_dict(response['model_state'])
            # Keep the same optimizer type (Adam) and just update its parameters
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        else:
            raise RuntimeError("Invalid response from server or no model received")

def prepare_client_data(client_id):
    """Prepare ECG data for specific client"""
    # This should be replaced with actual ECG data loading
    # For now creating dummy ECG-like data (batch_size, 1, 360) for 1-second ECG at 360Hz
    class DummyECGDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=1000):
            self.x = torch.randn(num_samples, 1, 360)  # Random ECG-like data
            self.y = torch.randint(0, 6, (num_samples,))  # Random labels (6 classes)
            
        def __len__(self):
            return len(self.x)
            
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]
    
    return DummyECGDataset()

if __name__ == '__main__':
    import sys
    client_id = int(sys.argv[1])
    client_port = 5000 + client_id
    
    # Prepare client-specific ECG data
    train_data = prepare_client_data(client_id)
    
    # Create and run client
    client = FederatedClient(client_id, train_data, client_port=client_port)
    client.connect_to_server()
    client.participate_in_fl(rounds=10)