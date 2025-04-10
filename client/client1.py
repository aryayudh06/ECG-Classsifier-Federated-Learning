import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import CNN
import requests
import numpy as np

# Load dataset lokal (contoh: MNIST)
transform = transforms.Compose([transforms.ToTensor()])
local_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(local_data, batch_size=32, shuffle=True)

# Inisialisasi model lokal
local_model = CNN()
optimizer = optim.SGD(local_model.parameters(), lr=0.01)

# Fungsi untuk mengirim update ke server
def send_update(weights, sample_size):
    url = 'http://localhost:5000/update'
    data = {
        'weights': [w.tolist() for w in weights],
        'sample_size': sample_size
    }
    response = requests.post(url, json=data)
    return response.json()

# Pelatihan lokal
def train_local(epochs=5):
    local_model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = local_model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    
    # Kirim bobot ke server
    weights = list(local_model.state_dict().values())
    sample_size = len(train_loader.dataset)
    global_weights = send_update(weights, sample_size)
    
    # Update model lokal dengan bobot global
    if 'weights' in global_weights:
        new_weights = [torch.tensor(np.array(w)) for w in global_weights['weights']]
        local_model.load_state_dict({k: v for k, v in zip(local_model.state_dict().keys(), new_weights)})

if __name__ == '__main__':
    train_local()