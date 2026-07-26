import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def reset_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_r: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    """
    Compute reset gate: r_t = sigmoid(W_r @ [h, x] + b_r)
    """
    h_x = np.concat([h_prev, x_t], axis=-1) # (N, H + D)
    r_t = sigmoid(h_x @ W_r.T + b_r) 
    # output (N, H) so need to put h_x first 
    # (N, H + D) @ (H, H + D).T

    return r_t
    
    