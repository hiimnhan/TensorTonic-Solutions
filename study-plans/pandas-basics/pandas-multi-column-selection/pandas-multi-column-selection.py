import pandas as pd

def select_columns(data, columns):
    """
    Returns: dict mapping selected column names to value lists
    """
    df = pd.DataFrame(data)
    selected = df[columns]

    return selected.to_dict("list")