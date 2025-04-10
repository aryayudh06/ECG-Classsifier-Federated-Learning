import torch
import torch.nn as nn
import socket
import pickle
import threading
from model import ResNet18
from utils import send_msg, recv_msg

class FederatedServer:
    def __init__(self, host='localhost', port=5000):
        self.global_model = ResNet18()
        self.client_models = []
        self.clients_connected = 0
        self.rounds_completed = 0
        self.max_clients = 2
        
        # Setup socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))
        self.socket.listen(5)
        
        print(f"Server listening on {host}:{port}")
        
    def handle_client(self, conn, addr):
        print(f"Connection established with {addr}")
        self.clients_connected += 1

        try:
            while True:
                message = recv_msg(conn)
                if message is None:
                    print(f"No message from {addr}, client may have disconnected.")
                    break

                print(f"Server received message: {message['type']} from {addr}")
                
                if message['type'] == 'GET_MODEL':
                    response = {
                        'type': 'MODEL',
                        'model_state': self.global_model.state_dict()
                    }
                    send_msg(conn, response)

                elif message['type'] == 'SEND_MODEL':
                    print(f"Server received model from client {addr}")
                    client_model = CNN()
                    client_model.load_state_dict(message['model_state'])
                    self.client_models.append(client_model)

                    if len(self.client_models) == self.max_clients:
                        self.aggregate()
                        self.rounds_completed += 1
                        print(f"Round {self.rounds_completed} completed")
                        self.client_models = []

                    send_msg(conn, {'type': 'ACK'})

        except Exception as e:
            print(f"Error with client {addr}: {e}")

        finally:
            conn.close()
            self.clients_connected -= 1
            print(f"Client {addr} disconnected")

        
    def aggregate(self):
        """Federated Averaging aggregation"""
        print("Aggregating models...")
        global_dict = self.global_model.state_dict()

        for k in global_dict.keys():
            global_dict[k] = torch.stack(
                [client.state_dict()[k].float() for client in self.client_models], 0
            ).mean(0)

        self.global_model.load_state_dict(global_dict)
        print("Model aggregation complete.")

        # Simpan model setelah setiap agregasi
        model_path = f"data/local_model/global_model_round_{self.rounds_completed + 1}.pt"
        torch.save(self.global_model.state_dict(), model_path)
        print(f"Global model saved to {model_path}")

        
    def start(self):
        """Start the server to accept connections"""
        try:
            while True:
                conn, addr = self.socket.accept()
                thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                thread.start()
        except KeyboardInterrupt:
            print("Shutting down server...")
        finally:
            self.socket.close()

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

if __name__ == '__main__':
    server = FederatedServer()
    server.start()
