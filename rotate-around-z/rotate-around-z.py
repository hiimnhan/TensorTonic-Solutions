import numpy as np

def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).
    """
    points = np.asarray(points, dtype=float)
    single = (points.ndim == 1)
    if single:
        points = points.reshape(1, 3)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0], 
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]])
    rotated = points @ R.T
    return rotated[0] if single else rotated