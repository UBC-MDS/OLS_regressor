import numpy as np
import pandas as pd


class LinearRegressor:
    def __init__(self):
        pass

    def fit(self, X, y):
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
        pass