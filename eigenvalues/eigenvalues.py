import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try:
        matrix = np.asarray(matrix, dtype=float)
    except (ValueError, TypeError):
        return None

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if matrix.shape[0] == 0:
        return np.array([], dtype=complex)

    eigvals = np.linalg.eigvals(matrix)
    eigvals = eigvals[np.lexsort((eigvals.imag, eigvals.real))]
    return eigvals
