from torch.utils.data import Subset

# Bagi dataset MNIST secara non-IID
def split_mnist_non_iid():
    full_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    idx_cliente1 = [i for i, (_, label) in enumerate(full_data) if label < 5]  # Digit 0-4
    idx_cliente2 = [i for i, (_, label) in enumerate(full_data) if label >= 5] # Digit 5-9
    return Subset(full_data, idx_cliente1), Subset(full_data, idx_cliente2)

# Gunakan di klien1.py
data_cliente1, _ = split_mnist_non_iid()
train_loader = torch.utils.data.DataLoader(data_cliente1, batch_size=32, shuffle=True)

# Gunakan di klien2.py
_, data_cliente2 = split_mnist_non_iid()
train_loader = torch.utils.data.DataLoader(data_cliente2, batch_size=32, shuffle=True)