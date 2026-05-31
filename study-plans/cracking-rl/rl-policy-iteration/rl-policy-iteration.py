def policy_iteration(P, R, gamma, eval_tol=1e-8, max_iters=200):
    """
    Returns: tuple (V, policy) where V is a list of S floats rounded to 4 decimals and policy is a list of S integer action indices
    """
    S = len(P)
    A = len(P[0])
    policy = [0] * S
    V = [0.0] * S

    for _ in range(max_iters):
        # Policy Evaluation
        while True:
            V_new = [0.0] * S
            for s in range(S):
                a = policy[s]
                V_new[s] = sum(
                    P[s][a][sp] * (R[s][a][sp] + gamma * V[sp])
                    for sp in range(S)
                )
            delta = max(abs(V_new[s] - V[s]) for s in range(S))
            V = V_new
            if delta < eval_tol:
                break

        # Policy Improvement
        policy_stable = True
        new_policy = [0] * S
        for s in range(S):
            best_q = float('-inf')
            best_a = 0
            for a in range(A):
                q = sum(
                    P[s][a][sp] * (R[s][a][sp] + gamma * V[sp])
                    for sp in range(S)
                )
                if q > best_q + 1e-12:
                    best_q = q
                    best_a = a
            new_policy[s] = best_a
            if best_a != policy[s]:
                policy_stable = False
        policy = new_policy

        if policy_stable:
            break

    V_rounded = [round(v, 4) for v in V]
    return (V_rounded, policy)
