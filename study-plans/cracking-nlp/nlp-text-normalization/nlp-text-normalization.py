import re

def text_normalize(text: str, operations: list) -> str:
    """
    Returns the normalized string.
    """
    res = text
    for op in operations:
        if op == 'lowercase':
            res = res.lower()
        elif op == 'remove_punctuation':
            res = ''.join(c for c in res if c.isalnum() or c.isspace())
        elif op == 'remove_digits':
            res = re.sub(r'\d', '', res)
        elif op == 'collapse_whitespace':
            res = re.sub(r'\s+', ' ', res)
        elif op == 'strip':
            res = res.strip()
    return res