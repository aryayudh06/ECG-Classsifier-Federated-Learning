from flask import Flask, request, jsonify
import torch
import numpy as np
from model import CNN

app = Flask(__name__)
global_model = CNN()  # Model global
clients_weights = []  # Penyimpanan sementara bobot klien
sample_sizes = []     # Jumlah sampel dari setiap klien

@app.route('/update', methods=['POST'])
def receive_update():
    data = request.json
    weights = [torch.tensor(np.array(w)) for w in data['weights']]
    clients_weights.append(weights)
    sample_sizes.append(data['sample_size'])
    
    if len(clients_weights) == 2:  # Jika 2 klien sudah mengirim update
        # Agregasi FedAvg
        averaged_weights = []
        for i in range(len(clients_weights[0])):
            weighted_sum = torch.zeros_like(clients_weights[0][i])
            for w, n in zip(clients_weights, sample_sizes):
                weighted_sum += w[i] * (n / sum(sample_sizes))
            averaged_weights.append(weighted_sum)
        
        # Update model global
        global_model.load_state_dict({k: v for k, v in zip(global_model.state_dict().keys(), averaged_weights)})
        clients_weights.clear()
        sample_sizes.clear()
        
        # Kirim bobot baru ke klien
        global_weights = [w.tolist() for w in averaged_weights]
        return jsonify({'weights': global_weights})
    return jsonify({'message': 'Menunggu klien lain...'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)