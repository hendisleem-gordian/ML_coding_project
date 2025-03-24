import math
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.x_train = []
        self.y_train = []

    def fit(self, x, y):
        self.x_train = np.array(x)
        self.y_train = np.array(y)

    def _euclidean_distance(self, x):
        return np.sqrt(np.sum((self.x_train - x) ** 2, axis=1)) 

    def predict(self, x):
        x = np.array(x)
        predictions = []
        for x in x:
            # Compute distances to ALLLLL training points
            distances = self._euclidean_distance(x)
            k_indices = distances.argsort()[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return predictions
