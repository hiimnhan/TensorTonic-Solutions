def tokenize(text: str) -> list:
    """
    Returns a list of token strings.
    """
    res = []
    N = len(text)

    i = 0
    

    while i < N:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c.isalnum() or c == '_':
            j = i
            while j < N and (text[j].isalnum() or text[j] == '_'):
                j += 1
            res.append(text[i:j])
            i = j
        else:
            res.append(c)
            i += 1
    return res