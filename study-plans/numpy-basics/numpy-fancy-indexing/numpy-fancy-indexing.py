import numpy as np

def select_by_index(arr, indices, axis):
    """
    Returns: 2D ndarray of float64
    """
    arr = np.array(arr, dtype=np.float64)
    out = []

    match axis:
        case 0:
            for idx in indices:
                row = arr[idx, :]
                out.append(row)
            return np.array(out, dtype=np.float64)
        case 1:
            for idx in indices:
                col = arr[:, idx]
                out.append(col)
            return np.array(out, dtype=np.float64).transpose()
                