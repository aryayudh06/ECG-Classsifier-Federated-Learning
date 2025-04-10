import numpy as np

def reshape_to_2d(data, width=14, height=14):
    """
    Reshape 1D ECG signal to 2D image-like format.
    Args:
        data: Input data of shape (n_samples, n_timesteps, 1)
        width, height: Target dimensions
    Returns:
        Reshaped data (n_samples, height, width, 1)
    """
    target_size = width * height
    if data.shape[1] < target_size:
        pad_size = target_size - data.shape[1]
        data = np.pad(data, ((0, 0), (0, pad_size), (0, 0)), mode='constant')
    elif data.shape[1] > target_size:
        data = data[:, :target_size, :]
    
    return data.reshape(data.shape[0], height, width, 1)