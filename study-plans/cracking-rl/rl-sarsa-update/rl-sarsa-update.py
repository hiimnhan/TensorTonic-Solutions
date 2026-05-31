def sarsa_update(Q, transitions, alpha, gamma):
    """
    Returns: 2D list of shape (S, A), updated Q values rounded to 4 decimals
    """
    Q = [[float(q) for q in row] for row in Q]
    S = len(Q)
    A = len(Q[0])

    for s, a, r, sp, ap in transitions:
        td_target = r + gamma * Q[sp][ap]
        td_error = td_target - Q[s][a]
        Q[s][a] += alpha * td_error

    return [[round(q, 4) for q in row] for row in Q]
