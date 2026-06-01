import math 

def ucb1_scores(Q, N, t, c):
    """
    Returns: list of K UCB1 scores, each rounded to 4 decimals
    """
    log_t = math.log(t)
    return [round(Q[a] + c * math.sqrt(log_t / N[a]), 4) for a in range(len(Q))]
