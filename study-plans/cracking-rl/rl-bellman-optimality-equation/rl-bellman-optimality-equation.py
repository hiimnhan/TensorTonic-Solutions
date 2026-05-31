def bellman_optimality_backup(P, R, gamma, V):
    """
    Returns: list of length S, V_new[s] rounded to 4 decimals
    """
    S = len(V)
    A = len(P[0])
    V_new = [0.0] * S

    for s in range(S):
        best = float("-inf")
        for a in range(A):
            q = 0.0
            for sp in range(S):
                q += P[s][a][sp] * (R[s][a][sp] + gamma * V[sp])
            if q > best:
                best = q
        V_new[s] = round(best, 4)

    return V_new
