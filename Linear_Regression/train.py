# -------- Train Linear Regression

"""_The objective of this script is to show Linear Regression
    training process_

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
# LinearRegression class from LinearRegression.py
from LinearRegression import LinearRegression

# define seaborn style
sns.set_style(style="darkgrid")

#Utility functions
def plot_data(X_data: np.ndarray, y_data: np.ndarray):
    """_plot distribution of data given two entry arrays_

    Parameters
    ----------
    X_data : np.ndarray
        _entry features_
    y_data : np.ndarray
        _entry targets_
    """
    fig = plt.figure(figsize=(8, 6))
    plt.title("Distribution of data")
    plt.scatter(X_data[:, 0], y_data, color="blue", marker="o", s=30)
    plt.show()

def plot_regression(y_pred_line:np.ndarray,
                    x_data: np.ndarray,
                    x_train: np.ndarray,
                    y_train: np.ndarray,
                    x_test: np.ndarray, 
                    y_test: np.ndarray):
    """_plot regression line and distribution of 
        training data & test data_

    Parameters
    ----------
    y_pred_line : np.ndarray
        _prediction regression line_
    x_data : np.ndarray
        _complete data before splitting_
    x_train : np.ndarray
        _training data features_
    y_train : np.ndarray
        _training data targets_
    x_test : np.ndarray
        _test data features_
    y_test : np.ndarray
        _test data targets_
    """
    fig = plt.figure(figsize=(8, 6))
    plt.title("Linear Regression model fit")
    m1 = plt.scatter(x_train, y_train, color="blue", s=10)
    m2 = plt.scatter(x_test, y_test, color="red", s=10)
    plt.plot(x_data, y_pred_line, color="k", linewidth=2, label="Prediction")
    plt.show()


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """_calculates mean squared error_

    Parameters
    ----------
    y_true : np.ndarray
        _ture target_
    y_pred : np.ndarray
        _predicted target_

    Returns
    -------
    _float_
        _float value of mse as a numpy object_
    """
    # again a problem if shapes of entry arrays mismatch
    try:
        return np.mean((y_true - y_pred)**2)
    except Exception as e:
        print("Encountered the following exception: {e}")

# Dummy data to represent regression and train our model
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    #plot_data(X, y) #uncomment if you want to visualize distribution of data
    
    linear_reg_model = LinearRegression(lr=0.01) # adjust learning rate 
    # fit the model with training data
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