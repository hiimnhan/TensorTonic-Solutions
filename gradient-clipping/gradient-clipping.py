import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g = np.asarray(g, dtype=float)
    if max_norm <= 0:
        return g.copy()
    g_norm = np.linalg.norm(g)
    if g_norm == 0:
        return g.copy()
    if g_norm <= max_norm:
        return g.copy()
    scale = max_norm / g_norm
    return g * scale