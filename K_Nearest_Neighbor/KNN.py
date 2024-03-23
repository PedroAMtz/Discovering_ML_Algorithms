# -------- KNN

"""_The objective of this script is to show KNN
    implementation from scratch using Python, showing step by step
    what is required in order to build up this algorithm and
    to succesfullly understand it_

    Author:
        Pedro Martínez - 22/03/2024
"""

import numpy as np
from metrics import euclidean_distance
from collections import Counter

class K_Nearest_Neighbor:

    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_labels).most_common()
        return most_common[0][0]