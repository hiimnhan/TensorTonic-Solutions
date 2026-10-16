def pad_and_truncate(sequences: list, max_length: int, pad_value: int = 0) -> list:
    """
    Returns one integer list of length max_length per input sequence.
    """
    res = []

    for s in sequences:
        truncated = s[:max_length]
        padded = truncated + [pad_value] * (max_length - len(truncated))
        res.append(padded)

    return res

    