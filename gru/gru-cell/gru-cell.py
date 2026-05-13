import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def gru_cell(x_t: np.ndarray, h_prev: np.ndarray,
             W_r: np.ndarray, W_z: np.ndarray, W_h: np.ndarray,
             b_r: np.ndarray, b_z: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Complete GRU cell forward pass.
    """
    def reset_gate(x_t, h_prev, W_r, b_r):
        concat = np.concatenate([h_prev, x_t], axis=-1)
        return sigmoid(concat @ W_r.T + b_r)

    def update_gate(x_t, h_prev, W_z, b_z):
        concat = np.concatenate([h_prev, x_t], axis=-1)
        return sigmoid(concat @ W_z.T + b_z)

    def candidate_gate(x_t, h_prev, r_t, W_h, b_h):
        gated_h = r_t * h_prev
        concat = np.concatenate([gated_h, x_t], axis=-1)
        return np.tanh(concat @ W_h.T + b_h)

    def hidden_state(z_t, h_prev, h_tilde):
        return (z_t * h_prev) + (1 - z_t) * h_tilde

    r_t = reset_gate(x_t, h_prev, W_r, b_r)
    z_t = update_gate(x_t, h_prev, W_z, b_z)
    h_tilde = candidate_gate(x_t, h_prev, r_t, W_h, b_h)
    h_t = hidden_state(z_t, h_prev, h_tilde)

    return h_t