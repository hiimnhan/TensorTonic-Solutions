import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    
    # Compute the divisors only for the even indices (0, 2, 4...)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(base) / d_model))
    
    # Assign sine to even indices and cosine to odd indices
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[:d_model // 2]) # Handles odd/even safety
    return pe