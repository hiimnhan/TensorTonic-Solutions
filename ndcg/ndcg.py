import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    def cal_dcg(relevance_scores, k, ideal=False):
        if ideal:
            relevance_scores = sorted(relevance_scores, reverse=True)
        relevance_scores = relevance_scores[:k]
        result = 0.0

        for i, rel in enumerate(relevance_scores):
            result += (2 ** (rel) - 1) / math.log2(i + 1 + 1)

        return result
    dcg = cal_dcg(relevance_scores, k)
    idcg = cal_dcg(relevance_scores, k, ideal=True)
    if dcg == 0.0 or idcg == 0.0:
        return 0.0
    return dcg / idcg