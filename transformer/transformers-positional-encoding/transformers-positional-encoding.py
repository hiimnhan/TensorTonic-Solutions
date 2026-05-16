import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pos = np.arange(seq_length).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10_000.0) / d_model))

    pos_vec = np.zeros((seq_length, d_model))
    pos_vec[:, 0::2] = np.sin(pos * div_term)
    pos_vec[:, 1::2] = np.cos(pos * div_term)

    return pos_vec