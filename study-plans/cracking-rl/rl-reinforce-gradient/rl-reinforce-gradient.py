def reinforce_loss(log_probs, returns):
    """
    Returns: float, REINFORCE policy loss rounded to 4 decimals
    """

    T = len(log_probs)
    L = 0

    for i in range(T):
        L += log_probs[i] * returns[i]

    return -float(round(L / T, 4))