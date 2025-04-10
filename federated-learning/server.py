import torch
import socket
import threading
import time
from model import ECGResNet18
from utils import send_msg, recv_msg

class FederatedServer:
    def __init__(self, host='localhost', port=5000):
        self.global_model = ECGResNet18(num_classes=6)
        self.client_models = []
        self.clients_connected = 0
        self.rounds_completed = 0
        self.max_clients = 2
        self.shutdown_flag = False
        
        # Setup socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.settimeout(2.0)  # Add timeout for accept()
        try:
            self.socket.bind((host, port))
            self.socket.listen(5)
            print(f"Server initialized on {host}:{port}")
        except socket.error as e:
            print(f"Socket error: {e}")
            raise

    def handle_client(self, conn, addr):
        print(f"Connection established with {addr}")
        self.clients_connected += 1
        conn.settimeout(30.0)  # Set timeout for client operations

        try:
            while not self.shutdown_flag:
                try:
                    message = recv_msg(conn)
                    if message is None:
                        print(f"Client {addr} disconnected")
                        break

                    print(f"Received {message['type']} from {addr}")
                    
                    if message['type'] == 'GET_MODEL':
                        response = {
                            'type': 'MODEL',
                            'model_state': self.global_model.state_dict()
                        }
                        send_msg(conn, response)

                    elif message['type'] == 'SEND_MODEL':
                        client_model = ECGResNet18(num_classes=6)
                        client_model.load_state_dict(message['model_state'])
                        self.client_models.append(client_model)
                        print(f"Received model from client {message['client_id']}")

                        if len(self.client_models) == self.max_clients:
                            self.aggregate()
                            self.rounds_completed += 1
                            print(f"Round {self.rounds_completed} completed")
                            self.client_models = []

                        send_msg(conn, {'type': 'ACK'})

                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Error handling client {addr}: {e}")
                    break

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
        print("Aggregation complete")

    def start_server(self):
        """Start the server with proper shutdown handling"""
        print("Server starting...")
        try:
            while not self.shutdown_flag:
                try:
                    conn, addr = self.socket.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(conn, addr),
                        daemon=True
                    )
                    client_thread.start()
                    print(f"Active connections: {threading.active_count() - 1}")
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Server accept error: {e}")
                    break

        except KeyboardInterrupt:
            print("\nServer shutdown requested")
        finally:
            self.shutdown_flag = True
            self.socket.close()
            print("Server shutdown complete")

if __name__ == '__main__':
    server = FederatedServer()
    server.start_server()