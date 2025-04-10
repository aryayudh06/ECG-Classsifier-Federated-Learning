import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, encoder):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    print("Classification Report:")
    print(classification_report(
        y_test_classes, y_pred_classes, 
        target_names=encoder.classes_
    ))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_classes, y_pred_classes))
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")