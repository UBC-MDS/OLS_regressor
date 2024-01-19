import numpy as np
import pandas as pd


class LinearRegressor:
    """
    Ordinary Least Squares Linear Regressor

    LinearRegressor will fit a linear model with coefficients w = (w1, w2, ..., wn)
    to minimize Residual Sum of Squares (RSS) between the observed targets values
    in the dataset, and the targets predicted by the linear approximation for the
    examples in the dataset.
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Fits the linear model according to the OLS mechanism.

        Parameters
        ----------
        X : {array-like matrix} of shape (n_examples, n_features)
            Dataset that will be used as the feature values to train the model.

        y : array-like matrix of shape (n_examples, n_targets)
            Dataset that will be used as the target values to train the model.

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        pass

    def predict(self, X):
        """
        Predicts target values using the fitted linear model.
        
        Parameters
        ----------
        X : array-like matrix of shape (n_samples, n_features)
            Feature values that will be used to make predictions.
        
        Returns
        -------
        predictions : array-like matrix of shape (n_samples, n_targets)
            Predicted target values for the input feature values.
        """
        pass

    def score(self, y_true, y_pred):
        """
        Calculates the coefficient of determination R^2 for the prediction.

        Parameters
        ----------
        y_true : array-like matrix, shape (n_samples, n_targets)
            True target values.

        y_pred : array-like matrix, shape (n_samples, n_targets)
            Predicted target values.

        Returns
        -------
        r2_score : float
            Coefficient of determination R^2.
        """