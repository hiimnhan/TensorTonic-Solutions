def gae(rewards, values, gamma, lam):
    """
    Compute Generalized Advantage Estimation.
    """
    T = len(rewards)
    advantages = [0.0] * T
    running = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantages[t] = delta + gamma * lam * running
        running = advantages[t]

    return advantages
        