import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)  # Untuk dataset MNIST (1 channel)
        self.fc1 = nn.Linear(32 * 26 * 26, 10)  # 10 kelas untuk MNIST

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = x.view(-1, 32 * 26 * 26)
        x = self.fc1(x)
        return x