def td_zero_update(V, transitions, alpha, gamma):
    """
    Returns: list of length S, updated V[s] rounded to 4 decimals
    """
    V = [float(v) for v in V]
    for s, r, sp in transitions:
        td_target = float(r) + gamma * V[sp]
        td_error = td_target - V[s]
        V[s] += alpha * td_error

    return V
    
