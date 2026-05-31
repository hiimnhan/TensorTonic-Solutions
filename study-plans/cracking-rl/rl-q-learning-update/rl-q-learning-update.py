def q_learning_update(Q, transitions, alpha, gamma):
    """
    Returns: 2D list of shape (S, A), updated Q values rounded to 4 decimals
    """
    S = len(Q)
    A = len(Q[0])
    Q = [[float(q) for q in row] for row in Q]
    for s, a, r, sp in transitions:
        max_q = max(Q[sp])
        td_target = r + gamma * max_q
        td_error = td_target - Q[s][a]
        Q[s][a] += alpha * td_error

    return [[round(q, 4) for q in row] for row in Q]

    
        
