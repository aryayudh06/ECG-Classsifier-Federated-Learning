import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

def load_mitbih_data():
    # Daftar record names dari MIT-BIH Arrhythmia Database
    records = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
               111, 112, 113, 114, 115, 116, 117, 118, 119, 
               121, 122, 123, 124, 200, 201, 202, 203, 205, 
               207, 208, 209, 210, 212, 213, 214, 215, 217, 219]

    signals = []
    labels = []
    beat_annotations = {
        'N': 0,  # Normal
        'L': 1,  # Left bundle branch block
        'R': 2,  # Right bundle branch block
        'A': 3,  # Atrial premature
        'V': 4,  # Premature ventricular contraction
        '/': 5   # Paced
    }

    for record in records:
        # Baca sinyal dan anotasi
        record_path = f'https://physionet.org/files/mitdb/1.0.0/{record}'
        signal, fields = wfdb.rdsamp(record_path)
        annotation = wfdb.rdann(record_path, 'atr')

        # Ekstrak heartbeat segments
        fs = fields['fs']  # Frekuensi sampling (biasanya 360 Hz)
        for i in range(1, len(annotation.sample)-1):
            if annotation.symbol[i] in beat_annotations:
                start = annotation.sample[i] - 180  # 0.5 detik sebelum R-peak
                end = annotation.sample[i] + 180   # 0.5 detik setelah R-peak
                
                if start > 0 and end < len(signal):
                    segment = signal[start:end, 0]  # Ambil lead MLII saja
                    signals.append(segment)
                    labels.append(beat_annotations[annotation.symbol[i]])

    # Konversi ke numpy array
    X = np.array(signals)
    y = np.array(labels)

    return X, y

def preprocess_data(X, y):
    # Normalisasi
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshape untuk CNN (samples, timesteps, channels)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # One-hot encoding label
    y = to_categorical(y)
    
    return X, y

# Contoh penggunaan
if __name__ == "__main__":
    X, y = load_mitbih_data()
    X_processed, y_processed = preprocess_data(X, y)
    
    print(f"Shape of X: {X_processed.shape}")
    print(f"Shape of y: {y_processed.shape}")
    print(f"Class distribution: {np.sum(y_processed, axis=0)}")

def load_and_preprocess_data(train_path, test_path):
    # Load dataset
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    
    # Split features and labels
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values
    
    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reshape for CNN (1D)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Encode labels
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    
    # Convert to one-hot encoding
    y_train_categorical = to_categorical(y_train_encoded)
    y_test_categorical = to_categorical(y_test_encoded)
    
    return X_train, X_test, y_train_categorical, y_test_categorical, encoder