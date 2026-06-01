def replay_buffer_sample(capacity, transitions, sample_indices):
    """
    Returns: tuple (states, actions, rewards, next_states, dones), each list of length len(sample_indices)
    """
    buf = [None] * capacity
    head = 0

    for t in transitions:
        buf[head] = t
        head = (head + 1) % capacity
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for i in sample_indices:
        s, a, r, sp, d = buf[i]
        states.append(s)
        actions.append(a)
        rewards.append(float(r))
        next_states.append(sp)
        dones.append(d)

    return (states, actions, rewards, next_states, dones)
        

    
