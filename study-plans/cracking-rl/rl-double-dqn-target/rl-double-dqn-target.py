def double_dqn_targets(Q_online_next, Q_target_next, rewards, dones, gamma):
    """
    Returns: list of length B, Double DQN targets rounded to 4 decimals
    """
    B = len(rewards)
    targets = []

    for i in range(B):
        a_star = Q_online_next[i].index(max(Q_online_next[i]))
        next = Q_target_next[i][a_star] * (1 - dones[i])
        target = rewards[i] + gamma * next
        targets.append(round(target, 4))

    return targets