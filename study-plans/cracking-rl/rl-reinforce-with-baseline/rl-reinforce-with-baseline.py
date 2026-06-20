def reinforce_baseline_loss(log_probs, returns, baselines):
    """
    Returns: float, REINFORCE-with-baseline loss rounded to 4 decimals
    """

    T = len(log_probs)
    s = sum(log_probs[t] * (returns[t] - baselines[t]) for t in range(T))
    return round(-s / T, 4)