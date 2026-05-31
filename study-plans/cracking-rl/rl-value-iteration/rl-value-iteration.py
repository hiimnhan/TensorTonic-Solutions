def value_iteration(P, R, gamma, tol=1e-6, max_iters=1000):
    """
    Returns: tuple (V, policy) where V is a list of S floats rounded to 4 decimals and policy is a list of S integer action indices
    """
    S = len(P)
    A = len(P[0])
    V = [0.0] * S

    for _ in range(max_iters):
        V_new = [0.0] * S
        for s in range(S):
            best = float("-inf")
            for a in range(A):
                q = 0.0
                for sp in range(S):
                    q += P[s][a][sp] * (R[s][a][sp] + gamma * V[sp])
                if q > best:
                    best = q
            V_new[s] = best
        delta = max(abs(V_new[s] - V[s]) for s in range(S))
        V = V_new
        if delta < tol:
            break

    pi = [0.0] * S
    for s in range(S):
        best_q = float("-inf")
        best_a = 0
        for a in range(A):
            q = 0.0
            for sp in range(S):
                q += P[s][a][sp] * (R[s][a][sp] + gamma * V[sp])

            if q > best_q:
                best_q = q
                best_a = a
        pi[s] = best_a

    V = [round(v, 4) for v in V]
    return (V, pi)
                    
        
