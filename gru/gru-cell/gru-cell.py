import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def gru_cell(x_t: np.ndarray, h_prev: np.ndarray,
             W_r: np.ndarray, W_z: np.ndarray, W_h: np.ndarray,
             b_r: np.ndarray, b_z: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Complete GRU cell forward pass.
    """
    # reset gate: determine how much of prev hidden state to forget
    # r_ approx 0 -> erased
    h_x = np.concat([h_prev, x_t], axis=-1)
    r_t = sigmoid(h_x @ W_r.T + b_r)

    # update gate: control how much of old hidden state to carry forward 
    # instead of replacing with new content
    # z_t approx 0 -> replace entirely with the candidate
    # z_t approx 1 -> copy the old hidden state
    z_t = sigmoid(h_x @ W_z.T + b_z)

    # candidate hidden state: propose new content
    # let prev hidden state + input go thru reset gate -> create a new version of prev state
    gated_r = r_t * h_prev
    concat_h = np.concat([gated_r, x_t], axis=-1)
    h_tilde = np.tanh(concat_h @ W_h.T + b_h)

    # new hidden state
    h_t = z_t * h_prev + (1 - z_t) * h_tilde

    return h_t