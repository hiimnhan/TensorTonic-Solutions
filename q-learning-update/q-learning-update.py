import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    Q = np.asarray(Q, dtype=float)
    best_next = np.max(Q[s_next])
    Q[s, a] = Q[s, a] + alpha * (r + gamma * best_next - Q[s, a])

    return Q
    