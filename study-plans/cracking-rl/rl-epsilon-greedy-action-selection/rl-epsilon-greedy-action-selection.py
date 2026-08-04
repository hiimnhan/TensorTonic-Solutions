def epsilon_greedy_probs(Q_values, epsilon):
    """
    Returns: list of length A, action probabilities under epsilon-greedy, rounded to 4 decimals
    """
    A = len(Q_values)
    best_a = Q_values.index(max(Q_values))

    policy = [(1 - epsilon) + (epsilon / A) if a == best_a else (epsilon / A) for a in range(A)]

    return policy

    
