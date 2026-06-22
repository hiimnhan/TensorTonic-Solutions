def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    """
    H, W = len(X), len(X[0])
    H_out, W_out = H // pool_size, W // pool_size

    out = []

    for i in range(H_out):
        row = []
        for j in range(W_out):
            total = 0
            for pi in range(pool_size):
                for pj in range(pool_size):
                    total += X[i * pool_size + pi][j * pool_size + pj]
            mean = total / (pool_size ** 2)
            row.append(mean)
        out.append(row)

    return out