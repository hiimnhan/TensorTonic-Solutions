def per_priorities_and_weights(td_errors, alpha, beta, epsilon=1e-6):
    """
    Returns: tuple (probs, is_weights), both lists of length N rounded to 4 decimals
    """
    N = len(td_errors)
    pi = [(abs(e) + epsilon) ** alpha for e in td_errors]
    Si = sum(pi)
    probs = [p / Si for p in pi]
    raw = [(N * p) ** (-beta) for p in probs]
    m = max(raw)
    is_w = [r / m for r in raw]
    return ([round(p, 4) for p in probs], [round(w, 4) for w in is_w])
    
