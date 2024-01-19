import pandas as pd
import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.regressor import LinearRegressor

# Test: X should be a 2D array
def test_fit_input_X_2D():
    model = LinearRegressor()
    X = np.array([1, 2, 3])
    y = np.array([4, 5])
    with pytest.raises(ValueError, match="X should be a 2D array."):
        model.fit(X, y)

# Test: y should be a 1D array
def test_fit_input_y_1D():
    model = LinearRegressor()
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[7, 8]])
    with pytest.raises(ValueError, match="y should be a 1D array."):
        model.fit(X, y)

# Test: The number of examples in X and y should be equal
def test_fit_input_examples_are_same():
    model = LinearRegressor()
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([7, 8, 9])
    with pytest.raises(ValueError, match="The number of examples in X and y should be equal."):
        model.fit(X, y)

# Test fit function with a square matrix
def test_fit_matrix_square():
    model = LinearRegressor()
    X = np.array([[1, 2], [3, 4]])
    y = np.array([5, 6])
    coefficients = model.fit(X, y)
    assert np.allclose(coefficients, np.array([-4., 4.5]))

# Test fit function with a rectangular matrix
def test_fit_matrix_rectanlge():
    model = LinearRegressor()
    X = np.array([[1, 2, 3], [4, 5, 8]])
    y = np.array([7, 8])
    coefficients = model.fit(X, y)
    assert np.allclose(coefficients, np.array([23.,  70., -42.]))
