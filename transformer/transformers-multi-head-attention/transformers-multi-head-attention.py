import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    Q = np.matmul(Q, W_q).reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = np.matmul(K, W_k).reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = np.matmul(V, W_v).reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attn_weights = softmax(scores, axis=-1)
    attn_output = np.matmul(attn_weights, V)

    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    return np.dot(attn_output, W_o)
        