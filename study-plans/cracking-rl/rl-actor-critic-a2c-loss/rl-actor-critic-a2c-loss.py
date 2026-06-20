def a2c_loss(log_probs, advantages, values, returns, entropies, value_coef=0.5, entropy_coef=0.01):
    """
    Returns: float, total A2C loss rounded to 4 decimals
    """
    T = len(log_probs)
    policy_loss = 0.0
    value_loss = 0.0
    entropy = 0.0

    for t in range(T):
        policy_loss += log_probs[t] * advantages[t]
        value_loss += (returns[t] - values[t]) ** 2
        entropy += entropies[t]

    policy_mean = -policy_loss / T
    value_mean = value_loss / T
    entropy_mean = entropy / T

    total = policy_mean + value_coef * value_mean - entropy_coef * entropy_mean
    return round(total, 4)

