from scripts.load_data import load_and_preprocess_data
from scripts.reshape_2d import reshape_to_2d
from scripts.model import create_cnn2d_model
from scripts.train import train_model
from scripts.evaluate import evaluate_model

# 1. Load and preprocess data
X_train, X_test, y_train, y_test, encoder = load_and_preprocess_data(
    'mitbih_train.csv', 
    'mitbih_test.csv'
)

# 2. Reshape to 2D
X_train_2d = reshape_to_2d(X_train)
X_test_2d = reshape_to_2d(X_test)

# 3. Create model
input_shape = (X_train_2d.shape[1], X_train_2d.shape[2], 1)
num_classes = y_train.shape[1]
model = create_cnn2d_model(input_shape, num_classes)

# 4. Train model
history = train_model(model, X_train_2d, y_train)

# 5. Evaluate
evaluate_model(model, X_test_2d, y_test, encoder)

# (Optional) Save model
model.save('ecg_cnn2d_model.h5')