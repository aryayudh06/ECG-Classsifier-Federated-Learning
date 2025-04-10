from server import Server
from client import Client, prepare_datasets
import torch

def main():
    # Persiapan
    client1_data, client2_data = prepare_datasets()
    server = Server()
    client1 = Client(1, client1_data)
    client2 = Client(2, client2_data)
    
    # Jumlah putaran federated learning
    num_rounds = 10
    
    for round in range(num_rounds):
        print(f"\nRound {round + 1}/{num_rounds}")
        
        # Distribusi model global ke client
        global_model = server.distribute_model()
        client1.receive_model(global_model)
        client2.receive_model(global_model)
        
        # Training lokal di masing-masing client
        print("Client 1 training...")
        client1_model = client1.local_train(epochs=1)
        
        print("Client 2 training...")
        client2_model = client2.local_train(epochs=1)
        
        # Kumpulkan model dari client
        server.client_models.append(client1_model)
        server.client_models.append(client2_model)
        
        # Agregasi model dengan Federated Averaging
        server.aggregate()
        
        # Evaluasi model global
        if (round + 1) % 2 == 0:
            test_model(server.global_model)

def test_model(model, test_loader=None):
    """Evaluasi model pada test set"""
    if test_loader is None:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=True)
    
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

if __name__ == "__main__":
    main()