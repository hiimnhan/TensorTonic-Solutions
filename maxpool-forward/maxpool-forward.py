def maxpool_forward(X, pool_size, s):
    """
    Compute the forward pass of 2D max pooling.
    """
    H, W = len(X), len(X[0])

    out_h = (H - pool_size) // s + 1
    out_w = (W - pool_size) // s + 1

    out = []

    for i in range(out_h):
        row = []
        for j in range(out_w):
            max_val = float("-inf")
            for a in range(pool_size):
                for b in range(pool_size):
                    val = X[i * s + a][j * s + b]
                    max_val = max(max_val, val)
            row.append(max_val)
        out.append(row)

    return out