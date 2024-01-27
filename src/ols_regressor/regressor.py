import numpy as np
import pandas as pd


class LinearRegressor():
    """
    Ordinary Least Squares Linear Regressor

    LinearRegressor will fit a linear model with coefficients w = (w1, w2, ..., wn)
    to minimize Residual Sum of Squares (RSS) between the observed targets values
    in the dataset, and the targets predicted by the linear approximation for the
    examples in the dataset.
    """
    def __init__(self):
        self.coef = None
        pass

    def fit(self, X, y):
        X_np = np.array(X)
        y_np = np.array(y)

        # Check if dimensions of both matrices are correct
        if X_np.ndim != 2:
            raise ValueError("X should be a 2D array.")
        if y_np.ndim != 1:
            raise ValueError("y should be a 1D array.")

        # Check if the number of samples in X and y match
        if X_np.shape[0] != y_np.shape[0]:
            raise ValueError("The number of examples in X and y should be equal.")

        if X_np.shape[0] < X_np.shape[1]:
            raise ValueError("The number of examples in X should be greater than the number of features.")

        # Normalize input features
        X_normalized = (X_np - X_np.mean(axis=0)) / X_np.std(axis=0)

        # Add a column of ones for the intercept term
        X_normalized = np.hstack((np.ones((X_normalized.shape[0], 1)), X_normalized))
        # X_np = np.hstack((np.ones((X_np.shape[0], 1)), X_np))
        # Fit OLS with regularization (you can adjust the regularization parameter)
        lambda_reg = 0.1
        self.coef = np.linalg.inv(X_normalized.T @ X_normalized + lambda_reg * np.eye(X_normalized.shape[1])) @ X_normalized.T @ y_np

        return self.coef

    # def fit(self, X, y):
    #     """
    #     Fits the linear model according to the OLS mechanism.

    #     Parameters
    #     ----------
    #     X : {array-like matrix} of shape (n_examples, n_features)
    #         Dataset that will be used as the feature values to train the model.

    #     y : array-like matrix of shape (n_examples, n_targets)
    #         Dataset that will be used as the target values to train the model.

    #     Returns
    #     -------
    #     self : object
    #         Fitted Estimator.
    #     """
    #     # X_np = np.array(X)
    #     # self.coef = np.linalg.inv(X_np.T @ X_np) @ X_np.T @ y
    #     # return self
    #     X_np = np.array(X)
    #     y_np = np.array(y)

    #     # Check if dimensions of both matrices are correct
    #     if X_np.ndim != 2:
    #         raise ValueError("X should be a 2D array.")
    #     if y_np.ndim != 1:
    #         raise ValueError("y should be a 1D array.")

    #     # Check if the number of samples in X and y match
    #     if X_np.shape[0] != y_np.shape[0]:
    #         raise ValueError("The number of examples in X and y should be equal.")

    #     if X_np.shape[0] < X_np.shape[1]:
    #         raise ValueError("The number of examples in X should be greater than the number of features.")

        
    #     self.coef = np.linalg.inv(X_np.T @ X_np) @ X_np.T @ y_np
    #     return self.coef

    # def __init__(self, fit_intercept=True, copy_X=True, n_jobs=None, positive=False):
    #     self.fit_intercept = fit_intercept
    #     self.copy_X = copy_X
    #     self.n_jobs = n_jobs
    #     self.positive = positive
    #     self.coef_ = None
    #     self.intercept_ = None

    # def _preprocess_data(self, X, y, sample_weight=None):
    #     # Implement your data preprocessing logic here
    #     # You can refer to sklearn's _preprocess_data method for guidance
    #     pass

    # def fit(self, X, y, sample_weight=None):
    #     """
    #     Fit linear model.

    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix} of shape (n_samples, n_features)
    #         Training data.

    #     y : array-like of shape (n_samples,) or (n_samples, n_targets)
    #         Target values. Will be cast to X's dtype if necessary.

    #     sample_weight : array-like of shape (n_samples,), default=None
    #         Individual weights for each sample.

    #     Returns
    #     -------
    #     self : object
    #         Fitted Estimator.
    #     """
    #     n_jobs_ = self.n_jobs

    #     # Perform data preprocessing
    #     X, y, X_offset, y_offset, X_scale = self._preprocess_data(X, y, sample_weight)

    #     if self.positive:
    #         # Implement positive constraint logic
    #         pass
    #     elif sp.issparse(X):
    #         # Implement sparse matrix handling
    #         pass
    #     else:
    #         # Regular OLS fitting
    #         self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)
    #         self.coef_ = self.coef_.T

    #     if y.ndim == 1:
    #         self.coef_ = np.ravel(self.coef_)

    #     self.intercept_ = y_offset - X_offset.dot(self.coef_)

    #     return self

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
        if self.coef is None:
            raise ValueError("Model not fitted. Call fit first.")

        X = np.array(X)

        # Check if dimensions of X are correct
        if X.ndim != 2:
            raise ValueError("X should be a 2D array.")

        # Check if the number of features in X equals to the number of coefficients
        if X.shape[1] != len(self.coef)-1:
            raise ValueError("The number of features in X should be equal to the number of coefficients.")
        
        # Check if non-numeric values exist in input
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("Input contains non-numeric values.")
            
        # Check if NaN values exist in input
        if np.isnan(X).any():
            raise ValueError("Input contains NaN values.")
        
        # check if infinite values exist in input
        if not np.isfinite(X).all():
            raise ValueError("Input contains infinite values.")

        pred = X @ self.coef[1:]
        return pred


    def score(self, X, y):
        """
        Calculates the coefficient of determination R^2 for the prediction.

        Parameters
        ----------
        X : array-like matrix, shape (n_samples, n_features)
            Feature dataset.

        y : array-like matrix, shape (n_samples, )
            True target values.

        Returns
        -------
        r2_score : float
            Coefficient of determination R^2.
        """
        # Ensure y is a numpy array
        y_true = np.array(y)

        # Predict the y values using the model
        y_pred = self.predict(X)

        # Calculate the mean of the true y values
        y_true_mean = np.mean(y_true)

        # Calculate the Total Sum of Squares (SST)
        SST = np.sum((y_true - y_true_mean) ** 2)

        # Calculate the Sum of Squared Errors (SSE)
        SSE = np.sum((y_true - y_pred) ** 2)

        # Calculate R^2
        r2 = 1 - (SSE / SST)

        return r2