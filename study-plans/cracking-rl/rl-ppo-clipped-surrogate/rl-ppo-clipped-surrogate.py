import math

def ppo_clipped_loss(log_probs_new, log_probs_old, advantages, clip_eps):
    """
    Returns: float, PPO clipped surrogate loss rounded to 4 decimals
    """
    T = len(advantages)
    r = [0.0] * T

    for t in range(T):
        r[t] = math.exp(log_probs_new[t] - log_probs_old[t])

    loss = 0.0

    for t in range(T):
        clipped = min(r[t], 1 + clip_eps)
        clipped = max(clipped, 1 - clip_eps)

        loss += min(r[t] * advantages[t], clipped * advantages[t])

    return round(-loss / T, 4)
