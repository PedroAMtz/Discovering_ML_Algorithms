# -------- Train Linear Regression

"""_The objective of this script is to show Linear Regression
    training process using scikit-learn_

    Author:
        Pedro Martínez - 11/03/2024
"""

import numpy as np
# sklear methods to split data and generate data
from sklearn.model_selection import train_test_split
from sklearn import datasets
# visualization
import matplotlib.pyplot as plt
import seaborn as sns
# linear regression model from sklearn
from sklearn.linear_model import LinearRegression

#Utility functions
from train import plot_data, plot_regression, mean_squared_error

# Dummy data to represent regression and train our model
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize linear regression model
linear_reg_model = LinearRegression()

if __name__ == "__main__":
    #plot_data(X,y) # uncomment to see data distribution

    linear_reg_model.fit(X_train, y_train)
    # make predictions on test data
    predictions = linear_reg_model.predict(X_test)

    # compute mse
    mse = mean_squared_error(y_test, predictions)
    # obtain linear regression fir line
    y_pred_line = linear_reg_model.predict(X)
    # print final mse and plot results
    print(f"Final MSE: {mse}")
    plot_regression(y_pred_line, X, X_train, y_train, X_test, y_test)