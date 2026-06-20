def gae_advantages(rewards, values, gamma, lam, last_value=0.0):
    """
    Returns: list of T advantages rounded to 4 decimals
    """
    T = len(rewards)
    A = [0.0] * T
    running = 0.0

    for t in range(T - 1, -1, -1):
        v_next = values[t + 1] if t + 1 < T else last_value
        delta = rewards[t] + gamma * v_next - values[t]
        running = delta + gamma * lam * running
        A[t] = round(running, 4)

    return A
