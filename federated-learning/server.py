import torch
import torch.nn as nn
import copy
import numpy as np
from collections import OrderedDict

class Server:
    def __init__(self):
        # Inisialisasi model global
        self.global_model = CNN()
        self.global_optimizer = torch.optim.SGD(self.global_model.parameters(), lr=0.01)
        self.client_models = []
        
    def aggregate(self):
        """Melakukan agregasi Federated Averaging"""
        global_dict = self.global_model.state_dict()
        
        # Akumulasi weight dari semua client
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_model.state_dict()[k].float() 
                                        for client_model in self.client_models], 0).mean(0)
        
        # Update model global
        self.global_model.load_state_dict(global_dict)
        
        # Kosongkan daftar model client
        self.client_models = []
        
    def distribute_model(self):
        """Mengirim model global ke client"""
        return copy.deepcopy(self.global_model)

class CNN(nn.Module):
    """Model CNN untuk klasifikasi MNIST"""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,               
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)          
        output = self.out(x)
        return output