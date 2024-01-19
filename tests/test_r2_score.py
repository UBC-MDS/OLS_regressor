import numpy as np
import pandas as pd
import pytest
from ols_regressor.regressor import LinearRegressor


def test_r2_score():

    # create a data fram for pytest
    df = pd.DataFrame({
        'y_true': [1, 2, 3, 4],
        'y_pred': [1, 2, 3, 4]
    })

    # calculate the expected r^2 
    expected_r2_score = 1.0

    calculated_r2_score = LinearRegressor().score(df, 'y_true', 'y_pred')

    assert calculated_r2_score == expected_r2_score

def test_empty_data():
    df = pd.DataFrame({'y_true': [], 'y_pred': []})
    with pytest.raises(ValueError):  
        LinearRegressor().score(df, 'y_true', 'y_pred')

def test_single_point():
    df = pd.DataFrame({'y_true': [1], 'y_pred': [1]})
    with pytest.raises(ValueError):  
        LinearRegressor().score(df, 'y_true', 'y_pred')

def test_input_not_dataframe():
    with pytest.raises(TypeError):  
        LinearRegressor().score("not a dataframe", 'y_true', 'y_pred')

def test_missing_columns():
    df = pd.DataFrame({'y_true': [1, 2, 3]})
    with pytest.raises(KeyError):  
        LinearRegressor().score(df, 'y_true', 'y_pred')

