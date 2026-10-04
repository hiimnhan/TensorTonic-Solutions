import math

def rotate_image(image: list, angle_degrees: float) -> list:
    """
    Returns the counterclockwise nearest-neighbor rotation.
    """
    H, W = len(image), len(image[0])
    res = []
    theta = angle_degrees * math.pi / 180

    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    cos_t, sin_t = math.cos(theta), math.sin(theta) 

    def is_in_range(x, y):
        if 0 <= x < H and 0 <= y < W:
            return True
        return False

    for i in range(H):
        rows = []
        for j in range(W):
            dy, dx = i - cy, j - cx
            sy = cy + dy * cos_t + dx * sin_t
            sx = cx - dy * sin_t + dx * cos_t
            sy, sx = round(sy), round(sx)
            rows.append(image[sy][sx] if is_in_range(sy, sx) else 0)
        res.append(rows)
    return res
            