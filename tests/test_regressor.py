import pandas as pd
import pytest
import numpy as np
from ols_regressor.regressor import LinearRegressor


# Test: X should be a 2D arrayß
def test_fit_input_X_2D():
    model = LinearRegressor()
    X = np.array([1, 2, 3])
    y = np.array([4, 5])
    with pytest.raises(ValueError, match="X should be a 2D array."):
        model.fit(X, y)


# Test: y should be a 1D array
def test_fit_input_y_1D():
    model = LinearRegressor()
    X = np.array([[1, 2], [4, 5]])
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
    assert np.allclose(coefficients, np.array([2.30769231, -0.76923077,  1.53846154]))


# Test fit function with a rectangular matrix
def test_fit_matrix_rectanlge():
    model = LinearRegressor()
    X = np.array([[1, 2, 3], [4, 5, 8]])
    y = np.array([7, 8])
    # coefficients = model.fit(X, y)
    with pytest.raises(ValueError, match="The number of examples in X should be greater than the number of features."):
        model.fit(X, y)
    # assert np.allclose(coefficients, np.array([23.,  70., -42.]))

def test_fit_matrix_rectangle_works():
    model = LinearRegressor()
    X = np.array([[1, 2, 3], [4, 5, 8], [11, 23, 34]])
    y = np.array([7, 8, 10])
    coefficients = model.fit(X, y)
    assert np.allclose(coefficients, np.array([5.9834636 ,  0.65147338, -0.25714111,  0.08315383]))