import numpy as np

def forward_pass(x, weights, biases):
    """
    Returns: Dict with "activations" and "pre_activations", values rounded to 4 decimals.
    """
    a = np.array(x, dtype=float)
    activations = [[round(float(x), 4) for x in a]]
    pre_activations = []
    L = len(weights)

    for l in range(L):
        W = np.array(weights[l], dtype=float)
        b = np.array(biases[l], dtype=float)
        z = W @ a + b
        pre_activations.append([round(float(v), 4) for v in z])

        if l < L - 1:
            a = np.maximum(0, z)
        else:
            a = z

        activations.append([round(float(v), 4) for v in a])

    return {"activations": activations, "pre_activations": pre_activations}
        