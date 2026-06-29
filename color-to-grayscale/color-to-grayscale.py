def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    H, W = len(image), len(image[0])
    result = []

    for h in range(H):
        row = []
        for w in range(W):
            y = 0.299 * image[h][w][0] + 0.587 * image[h][w][1] + 0.114 * image[h][w][2]
            row.append(y)
        result.append(row)

    return result
                
            
    