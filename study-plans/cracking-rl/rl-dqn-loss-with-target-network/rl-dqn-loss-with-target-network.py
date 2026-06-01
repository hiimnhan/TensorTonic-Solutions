def dqn_loss(Q_online, Q_target_next, actions, rewards, dones, gamma):
    """
    Returns: float, mean squared TD error rounded to 4 decimals
    """
    B = len(rewards)
    A = len(Q_online[0])
    loss = 0.0

    for i in range(B):
        pred = Q_online[i][actions[i]]
        max_target_next = max(Q_target_next[i]) * (1 - dones[i])
        target = rewards[i] + gamma * max_target_next
        loss += (target - pred) ** 2

    return round(loss / B, 4)