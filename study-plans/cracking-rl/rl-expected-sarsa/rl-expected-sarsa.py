def expected_sarsa_update(Q, transitions, policy, alpha, gamma):
    """
    Returns: 2D list of shape (S, A), updated Q values rounded to 4 decimals
    """
    A = len(Q[0]) if Q else 0
    Q = [[float(q) for q in row] for row in Q]
    for s, a, r, sp in transitions:
        expected_q = sum([policy[sp][ap] * Q[sp][ap] for ap in range(A)])
        td_target = float(r) + gamma * expected_q
        td_error = td_target - Q[s][a]
        Q[s][a] += alpha * td_error

    return [[round(q, 4) for q in row] for row in Q]
        