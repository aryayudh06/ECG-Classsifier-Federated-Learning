import wfdb
import os
import numpy as np
import torch

class MITBIHDataset(torch.utils.data.Dataset):
    def _init_(self, record_path, segment_length=360):
        record = wfdb.rdrecord(record_path)
        self.signal = record.p_signal[:, 0]  # Ambil 1 lead saja
        self.segment_length = segment_length

        self.samples = []
        for i in range(0, len(self.signal) - segment_length, segment_length):
            segment = self.signal[i:i+segment_length]
            self.samples.append(segment)

        # Dummy label, ganti sesuai anotasi jika ada
        self.labels = np.random.randint(0, 6, len(self.samples))

    def _len_(self):
        return len(self.samples)

    def _getitem_(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32).unsqueeze(0)  # [1, 360]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def prepare_client_data(client_id):
    """Load MIT-BIH ECG data per client"""
    # Daftar nama file rekaman MIT-BIH yang kamu miliki (tanpa ekstensi .dat/.hea)
    records = ['100', '101', '102', '103', '104']
    
    # Pilih salah satu berdasarkan client_id
    record_name = records[client_id % len(records)]
    
    # Ganti ini ke path folder di mana data MIT-BIH kamu disimpan
    mitbih_data_dir = "data/mit-bih-arrhythmia-database-1.0.0"  # <- ubah ini
    
    record_path = os.path.join(mitbih_data_dir, record_name)
    
    return MITBIHDataset(record_path)