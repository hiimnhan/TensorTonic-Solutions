def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top_k = recommended[:k]
    relevant = set(relevant)

    seen = 0
    for i in top_k:
        if i in relevant:
            seen += 1

    precision_at_k = seen / k
    recall_at_k = seen / len(relevant)

    return [precision_at_k, recall_at_k]