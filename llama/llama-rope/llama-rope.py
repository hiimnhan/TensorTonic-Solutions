import torch

def apply_rope(q, k, freqs_cos, freqs_sin):
    """
    Returns: tuple of (q_rotated, k_rotated) same shapes as input
    """
    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]
    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]

    cos = freqs_cos.unsqueeze(0).unsqueeze(0)
    sin = freqs_sin.unsqueeze(0).unsqueeze(0)

    q_rot, k_rot = torch.zeros_like(q), torch.zeros_like(k)

    q_rot[..., 0::2] = q_even * cos - q_odd * sin
    q_rot[..., 1::2] = q_even * sin + q_odd * cos
    k_rot[..., 0::2] = k_even * cos - k_odd * sin
    k_rot[..., 1::2] = k_even * sin + k_odd * cos

    return q_rot, k_rot

    