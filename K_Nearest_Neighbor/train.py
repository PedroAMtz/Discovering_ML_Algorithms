# -------- Train KNN

"""_The objective of this script is to show KNN
    training process_

    Author:
        Pedro Martínez - 22/03/2024
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import K_Nearest_Neighbor

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


if __name__ == "__main__":

    #plt.figure()
    #plt.scatter(X[:,2], X[:,3], c=y, cmap=cmap, edgecolors="black", s=20)
    #plt.show()
    clf = K_Nearest_Neighbor(k=5)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    acc = np.sum(predictions == y_test) / len(y_test)

    print(f"Test predictions: {predictions}")
    print(f"Test accuracy: {acc}")