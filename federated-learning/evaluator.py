import torch
from torchvision import datasets, transforms
from model import CNN
import pickle
import socket

def evaluate_model(model, test_loader=None):
    """Evaluate model on test set"""
    if test_loader is None:
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
    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

def request_model_evaluation(host='localhost', port=5000):
    """Request current model from server for evaluation"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            message = {'type': 'GET_MODEL'}
            s.sendall(pickle.dumps(message))
            
            response = pickle.loads(s.recv(4096 * 16))
            if response['type'] == 'MODEL':
                model = CNN()
                model.load_state_dict(response['model_state'])
                accuracy = evaluate_model(model)
                return accuracy
    except Exception as e:
        print(f"Evaluation error: {e}")
        return None

if __name__ == '__main__':
    # Run periodic evaluation
    import time
    while True:
        print("\nEvaluating global model...")
        accuracy = request_model_evaluation()
        if accuracy is not None:
            print(f"Current model accuracy: {accuracy:.2f}%")
        time.sleep(10)  # Evaluate every 10 seconds