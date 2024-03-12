# -------- Train Logistic Regression

"""_The objective of this script is to show Linear Regression
    training process_

    Author:
        Pedro Martínez - 12/03/2024
"""

import numpy as np
# sklear methods to split data and generate data
from sklearn.model_selection import train_test_split
from sklearn import datasets
# visualization
import matplotlib.pyplot as plt
import seaborn as sns
# import logistic regression model
from LogisticRegression import LogisticRegression


# Utility functions
def accuracy(y_pred, y_true):
    return np.sum(y_pred==y_true)/len(y_true)

dummy_data = datasets.load_breast_cancer()
X, y = dummy_data.data, dummy_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LogisticRegression(lr=0.1)

if __name__ == "__main__":

    lr_model.fit(X_train, y_train)
    y_predicted = lr_model.predict(X_test)
    accuracy_score = accuracy(y_predicted, y_test)
    print(accuracy_score)
    