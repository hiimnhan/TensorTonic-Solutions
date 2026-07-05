import pandas as pd

def select_column(data, column):
    """
    Returns: dict with 'values' (list) and 'length' (int)
    """
    df = pd.DataFrame(data)
    values = df[column]

    return {
        "values": values.tolist(),
        "length": int(values.size)
    }