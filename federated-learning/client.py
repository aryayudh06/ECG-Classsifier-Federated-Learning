import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy

class Client:
    def __init__(self, client_id, train_data):
        self.id = client_id
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Dataset lokal
        self.train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        
    def receive_model(self, model):
        """Menerima model dari server"""
        self.model = copy.deepcopy(model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
    def local_train(self, epochs=1):
        """Melakukan training lokal"""
        self.model.train()
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
        return self.model

def prepare_datasets():
    """Mempersiapkan dataset MNIST dan membaginya ke 2 client"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download dataset MNIST
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Bagi dataset ke 2 client (contoh: client 1 dapat digit 0-4, client 2 dapat digit 5-9)
    client1_indices = [i for i in range(len(train_data)) if train_data[i][1] < 5]
    client2_indices = [i for i in range(len(train_data)) if train_data[i][1] >= 5]
    
    client1_data = Subset(train_data, client1_indices)
    client2_data = Subset(train_data, client2_indices)
    
    return client1_data, client2_data