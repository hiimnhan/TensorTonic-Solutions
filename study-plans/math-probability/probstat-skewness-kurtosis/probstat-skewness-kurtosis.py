import numpy as np

def skewness_kurtosis(data):
    """Returns: dict with 'skewness', 'kurtosis', and interpretation strings."""
    arr = np.array(data, dtype=float)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)

    skew = (n / ((n-1) * (n-2))) * np.sum(((arr - mean) / std) ** 3)
    kurt = ((n * (n+1)) / ((n-1) * (n-2) * (n-3))) * np.sum(((arr - mean) / std) ** 4) - (3 * (n-1)**2) / ((n-2) * (n-3))

    skew = round(float(skew), 4)
    kurt = round(float(kurt), 4)

    if skew > 0.5:
        skew_interp = "right-skewed"
    elif skew < -0.5:
        skew_interp = "left-skewed"
    else:
        skew_interp = "approximately symmetric"

    if kurt > 1:
        kurt_interp = "leptokurtic"
    elif kurt < -1:
        kurt_interp = "platykurtic"
    else:
        kurt_interp = "mesokurtic"

    return {
        "skewness": skew,
        "kurtosis": kurt,
        "skew_interpretation": skew_interp,
        "kurtosis_interpretation": kurt_interp,
    }
