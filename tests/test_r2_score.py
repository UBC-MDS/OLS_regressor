import numpy as np
import pandas as pd
import pytest
from ols_regressor.regressor import LinearRegressor


def test_score_method():
    # Setup Test Data
    X = np.array([[1, 2], [3, 4]])  
    y = np.array([1, 2])  

    model = LinearRegressor()

    model.predict = lambda x: np.array([0.9, 2.1])  

    # Calculate Expected R^2
    y_pred = model.predict(X)
    y_true_mean = np.mean(y)
    SST = np.sum((y - y_true_mean) ** 2)
    SSE = np.sum((y - y_pred) ** 2)
    expected_r2 = 1 - (SSE / SST)

    # Call the score method
    calculated_r2 = model.score(X, y)

    assert np.isclose(calculated_r2, expected_r2), f"Expected R^2: {expected_r2}, but got: {calculated_r2}"

# Test with Incorrect Input Shapes
def test_score_incorrect_input_shapes():
    model = LinearRegressor()
    X = np.array([[1, 2, 3]])  
    y = np.array([1])
    with pytest.raises(ValueError):
        model.score(X, y)

# Test with Missing Columns
def test_missing_columns():
    model = LinearRegressor()

    X_missing_columns = np.array([[1], [3]])
    y = np.array([1, 2])

    with pytest.raises(ValueError):  
        model.score(X_missing_columns, y)

# Test with Empty Data
def test_empty_data():
    model = LinearRegressor()
    X_empty = np.array([]).reshape(0, 0)
    y_empty = np.array([])

    with pytest.raises(ValueError):
        model.score(X_empty, y_empty)
