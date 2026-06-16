import numpy as np

def angle_features(angles):
    """Returns: np.ndarray of shape (3, n), rows are sin, cos, tan"""
    angles = np.array(angles, dtype=np.float64)
    sint = np.sin(angles)
    cost = np.cos(angles)
    tant = np.tan(angles)

    return np.stack([sint, cost, tant])