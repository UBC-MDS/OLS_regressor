import pandas as pd
import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.regressor import LinearRegressor

X_train = np.array([[1, 3], [5, 7], [9, 2]])
y_train = np.array([4, 6, 8])

# test predict function without calling fit 
def test_predict_without_fit():
    model = LinearRegressor()
    with pytest.raises(ValueError, match="Model not fitted. Call fit first."):
        X_test = np.array([[10, 14], [12, 16]])
        model.predict(X_test)

# test predict function outputs the correct shape of predicted values
def test_predict():
    model = LinearRegressor()
    model.fit(X_train, y_train)
    X_test = np.array([[10, 14], [12, 16]])
    pred = model.predict(X_test)
    assert pred.shape == (X_test.shape[0],)

# test with empty input
def test_predict_empty_input():
    model = LinearRegressor()
    model.fit(X_train, y_train)
    X_test = np.array([])
    with pytest.raises(ValueError, match="X should be a 2D array."):
        model.predict(X_test)

# test with input containing incorrect number of features
def test_predict_with_incorrect_num_of_features():
    model = LinearRegressor()
    model.fit(X_train, y_train)
    with pytest.raises(ValueError, match="The number of features in X should be equal to the number of coefficients."):
        X_test = np.array([[18, 11, 13, 20]])
        model.predict(X_test)

# test with non-numeric input
def test_predict_non_numeric_input():
    model = LinearRegressor()
    model.fit(X_train, y_train)
    X_test = [["123", "abc"], [0, "0"]]
    with pytest.raises(ValueError, match="Input contains non-numeric values."):
        model.predict(X_test)

# test with input containing NaN values
def test_predict_with_nan_values():
    model = LinearRegressor()
    model.fit(X_train, y_train)
    X_test = np.array([[10, np.nan], [12, 16]])
    with pytest.raises(ValueError, match="Input contains NaN values."):
        model.predict(X_test)

# test with input containing infinite values
def test_predict_with_infinite_values():
    model = LinearRegressor()
    model.fit(X_train, y_train)
    X_test = np.array([[10, np.inf], [12, 16]])
    with pytest.raises(ValueError, match="Input contains infinite values."):
        model.predict(X_test)