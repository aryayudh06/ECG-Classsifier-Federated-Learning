import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import socket
import pickle
import time
from model import CNN
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
        self.model = CNN()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('localhost', client_port))
        
    def connect_to_server(self):
        """Connect to the server"""
        self.socket.connect((self.server_host, self.server_port))
        print(f"Client {self.client_id} connected to server")
        
    def get_global_model(self):
        """Request and receive global model from server"""
        message = {'type': 'GET_MODEL'}
        send_msg(self.socket, message)

        response = recv_msg(self.socket)
        
        if response and response['type'] == 'MODEL':
            self.model.load_state_dict(response['model_state'])
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        else:
            raise RuntimeError("Invalid response from server or no model received")

            
    def send_local_model(self):
        """Send locally trained model to server"""
        print(f"Client {self.client_id} sending model to server...")
        message = {
            'type': 'SEND_MODEL',
            'model_state': self.model.state_dict(),
            'client_id': self.client_id
        }
        send_msg(self.socket, message)

        # Wait for acknowledgement
        ack = recv_msg(self.socket)
        print(f"Client {self.client_id} received ack from server")
        return ack
        
    def local_train(self, epochs=1):
        """Train model on local data"""
        self.model.train()
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                # Log the loss to monitor training progress
                if batch_idx % 10 == 0:
                    print(f"Client {self.client_id} - Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}")

                
    def participate_in_fl(self, rounds=10):
        """Participate in federated learning rounds"""
        try:
            for round in range(rounds):
                print(f"\nClient {self.client_id} - Round {round + 1}")
                
                # Get global model from server
                self.get_global_model()
                
                # Local training
                print("Training locally...")
                self.local_train(epochs=1)
                
                # Send updated model to server
                print("Sending model to server...")
                self.send_local_model()
                
                time.sleep(1)
                
        except (ConnectionResetError, RuntimeError, socket.error) as e:
            print(f"[ERROR] Client {self.client_id} disconnected unexpectedly: {e}")
            
        finally:
            self.socket.close()
            print(f"Client {self.client_id} disconnected")

    def recv_all(self, sock, length, timeout=10):
        """Receive all data from socket with timeout"""
        sock.settimeout(timeout)
        data = b''
        while len(data) < length:
            try:
                packet = sock.recv(length - len(data))
                if not packet:
                    raise RuntimeError("Connection closed unexpectedly")
                data += packet
            except socket.timeout:
                raise RuntimeError("Timeout occurred while waiting for data")
        return data

def prepare_client_data(client_id):
    """Prepare MNIST data for specific client"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Split data between clients (client 1 gets digits 0-4, client 2 gets 5-9)
    if client_id == 1:
        indices = [i for i in range(len(train_data)) if train_data[i][1] < 5]
    else:
        indices = [i for i in range(len(train_data)) if train_data[i][1] >= 5]
        
    return Subset(train_data, indices)

if __name__ == '__main__':
    import sys
    client_id = int(sys.argv[1])
    client_port = 5000 + client_id
    
    # Prepare client-specific data
    train_data = prepare_client_data(client_id)
    
    # Create and run client
    client = FederatedClient(client_id, train_data, client_port=client_port)
    client.connect_to_server()
    client.participate_in_fl(rounds=10)
