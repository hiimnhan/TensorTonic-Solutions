def discounted_returns(rewards, gamma):
    """
    Returns: list of G_t values, one per timestep, each rounded to 4 decimals
    """
    T = len(rewards)
    G = [0.0] * T
    running = 0.0

    for t in range(T - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        G[t] = round(running, 4)

    return G
        
