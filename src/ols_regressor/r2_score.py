import numpy as np
import pandas as pd


def r2_score(df, true_column, pred_column):
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

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if df.empty or len(df) < 2:
        raise ValueError("DataFrame must have at least two data points")

    y_true = df[true_column].values
    y_pred = df[pred_column].values

    #  Calculate mean value of y_true
    y_true_mean = np.mean(y_true)

    # Calculat（SST）
    SST = np.sum((y_true - y_true_mean) ** 2)

    # Calculate（SSE）
    SSE = np.sum((y_true - y_pred) ** 2)

    # Calculate R^2
    r2 = 1 - (SSE / SST)

    return r2
