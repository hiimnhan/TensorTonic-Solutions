import numpy as np

def go_group_liberties(board, row, col):
    """
    Returns: the connected group and its distinct liberties in row-major order.
    """
    DIRECTION = [[-1, 0], [0, -1], [0, 1], [1, 0]]
    board = np.asarray(board)
    size = board.shape[0]
    color = board[row, col]
    stack = [(row, col)]
    group = set()
    liberties = set()

    while stack:
        point = stack.pop()
        if point in group:
            continue
        group.add(point)
        r, c = point
        for d in DIRECTION:
            dx, dy = d
            nr, nc = r + dx, c + dy
            if not (0 <= nr < size and 0 <= nc < size):
                continue 
            if board[nr, nc] == 0:
                liberties.add((nr, nc))
            elif board[nr, nc] == color and (nr, nc) not in group:
                stack.append((nr, nc))

    return sorted(group), sorted(liberties)
    
