def max_pooling_2d(X, p):
    """
    Apply 2D max pooling with non-overlapping windows.
    """
    H, W = len(X), len(X[0])
    out_h, out_w = H // p, W // p
    out = []

    for i in range(out_h):
        row = []
        for j in range(out_w):
            max_val = float("-inf")
            for a in range(p):
                for b in range(p):
                    val = X[i * p + a][j * p + b]
                    max_val = max(max_val, val)
            row.append(max_val)
        out.append(row)

    return out