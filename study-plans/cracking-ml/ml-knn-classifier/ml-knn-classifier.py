import numpy as np

def knn_classifier(X_train: list, y_train: list, X_test: list, k: int) -> list:
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    predictions = []
    for point in X_test:
        order = sorted(range(len(X_train)), key=lambda index: (float(np.sum((X_train[index] - point) ** 2)), index))
        counts = {}
        for index in order[:k]:
            label = y_train[index]
            counts[label] = counts.get(label, 0) + 1
        predictions.append(int(min(counts, key=lambda label: (-counts[label], label))))
    return predictions
