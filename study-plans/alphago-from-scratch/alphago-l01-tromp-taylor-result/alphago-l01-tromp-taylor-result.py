import numpy as np
def tromp_taylor_result(board, komi, consecutive_passes, move_count):
    board = np.asarray(board)
    size = board.shape[0]

    if consecutive_passes < 2 and move_count < 2 * size ** 2:
        return False, None, None, None

    DIRECTIONS = [
        (-1, 0),
        (0, -1),
        (1, 0),
        (0, 1),
    ]

    black_territory = 0
    white_territory = 0
    visited = set()

    for r in range(size):
        for c in range(size):

            if board[r, c] != 0 or (r, c) in visited:
                continue

            # Find the entire connected empty region
            stack = [(r, c)]
            visited.add((r, c))
            region = []
            bordering_colours = set()

            while stack:
                pr, pc = stack.pop()
                region.append((pr, pc))

                for dr, dc in DIRECTIONS:
                    nr, nc = pr + dr, pc + dc

                    if not (0 <= nr < size and 0 <= nc < size):
                        continue

                    if board[nr, nc] == 0:
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            stack.append((nr, nc))

                    elif board[nr, nc] == 1:
                        bordering_colours.add(1)

                    elif board[nr, nc] == -1:
                        bordering_colours.add(-1)

            # Assign the whole region
            if bordering_colours == {1}:
                black_territory += len(region)

            elif bordering_colours == {-1}:
                white_territory += len(region)

            # {1, -1} means neutral territory

    black_score = np.float64(np.count_nonzero(board == 1))
    white_score = np.float64(np.count_nonzero(board == -1) + komi)

    black_score += black_territory
    white_score += white_territory

    winner = (
        1 if black_score > white_score
        else -1 if white_score > black_score
        else 0
    )

    return True, black_score, white_score, winner