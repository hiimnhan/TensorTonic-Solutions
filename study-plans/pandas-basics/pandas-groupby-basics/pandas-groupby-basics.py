import pandas as pd

def groupby_basics(data, group_col, value_col):
    """
    Returns: dict with 'sum', 'mean', 'count' (each a dict)
    """
    df = pd.DataFrame(data)
    grouped = df.groupby(group_col)[value_col]
    sm = grouped.sum().to_dict()
    mn = grouped.mean().to_dict()
    cnt = grouped.count().to_dict()

    return {
        "sum": sm,
        "mean": mn,
        "count": cnt
    }