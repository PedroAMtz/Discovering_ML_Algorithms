# -------- Linear Regression

"""_The objective of this script is to show Linear Regression
    implementation from scratch using Python, showing step by step
    what is required in order to build up this algorithm and
    to succesfullly understand it_

    Author:
        Pedro Martínez - 11/03/2024
"""
import numpy as np

class LinearRegression:

    def __init__(self, lr: float=0.001):
        """_Initialization of Linear Rgeression model_

        Parameters
        ----------
        lr : float, optional
            _learning rate which optimizes gradient descent algorithm_, by default 0.001
        """
        self.lr = lr
        # We start initializing weights & bias as None
        # Later in the fit method we are going to be otimizing
        # these parameters
        self.weigths = None
        self.bias = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int=1000):
        """_Supervised learning algorithm to fit Linear Regression model
            with samples of Features (X) and Targets (y), this method
            will run for n epochs_

        Parameters
        ----------
        X : np.ndarray
            _entry features_
        y : np.ndarray
            _entry targets_
        epochs : int, optional
            _number of iterations_, by default 1000
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # just for some protection to the method let´s thing about 
        # things that can be wrong and may break the code,
        # yes like if the shape of X and y aren´t the same

        try:
            for i in range(epochs):
                if i % 50 == 0:
                    print(f"Epoch: {i}")
                
                # calculate prediction
                # note that the below equation is the one defining the model behavior
                # having the slope equation to get y given x we only have to worry about
                # weights and bias term, then this is just a weighted sum of the input x adding the bias
                y_predicted =  np.dot(X, self.weights) + self.bias

                # --- Calculating error ---
                # We use MSE (Mean squared error) to calculate the error we want to optimize
                # using gradient descent -> MSE = 1/N sum(yi - (wxi + b))**2
                # from this equation we then solve the partial derivates to obtain the gradient
                # to update w and b.

                # partial derivate of MSE with respect to weights
                w_derivative = (1/n_samples) * np.dot(X.T, (y_predicted-y))
                # partial derivative of MSE with respect to bias
                b_derivative = (1/n_samples) * np.sum(y_predicted - y)

                # How much w and b have to change in order to find the optimal solution?
                # Now we use gradient descent to update new weights and new bias
                self.weights = self.weights - self.lr * w_derivative
                self.bias = self.bias - self.lr * b_derivative
        
        except Exception as e:
            if n_samples != y.shape[0]:
                print(f"X with shape {n_samples} and y with sahpe {y.shape[0]} not a match")
            print(f"Encountered the following exception {e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """_predicts for a given X_

        Parameters
        ----------
        X : np.ndarray
            _entry features_

        Returns
        -------
        _np.ndarray_
            _numpy array of predictions calculated_
        """
        new_pred = np.dot(X, self.weights) + self.bias
        return new_pred