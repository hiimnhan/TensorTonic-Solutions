import numpy as np

def hidden_update(h_prev: np.ndarray, h_tilde: np.ndarray,
                  z_t: np.ndarray) -> np.ndarray:
    """
    Compute final state: h_t = z*h_prev + (1-z)*h_tilde
    """
    keep_old = z_t * h_prev
    replace_new = (1 - z_t) * h_tilde

    return keep_old + replace_new