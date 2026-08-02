def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    dot = sum(a * b for a, b in zip(x1, x2))
    l_x1 = math.sqrt(sum(x**2 for x in x1))
    l_x2 = math.sqrt(sum(x**2 for x in x2))

    cos = dot / (l_x1 * l_x2)

    if label == 1:
        return 1 - cos
    return max(0, cos - margin)