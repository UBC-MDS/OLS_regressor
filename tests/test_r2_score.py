import numpy as np
import pandas as pd
import pytest
from r2_score import r2_score


def test_r2_score():

    # create a data fram for pytest
    df = pd.DataFrame({
        'y_true': [1, 2, 3, 4],
        'y_pred': [1, 2, 3, 4]
    })

    # calculate the expected r^2 
    expected_r2_score = 1.0

    calculated_r2_score = r2_score(df, 'y_true', 'y_pred')

    assert calculated_r2_score == expected_r2_score

def test_empty_data():
    df = pd.DataFrame({'y_true': [], 'y_pred': []})
    with pytest.raises(ValueError):  
        r2_score(df, 'y_true', 'y_pred')

def test_single_point():
    df = pd.DataFrame({'y_true': [1], 'y_pred': [1]})
    with pytest.raises(ValueError):  
        r2_score(df, 'y_true', 'y_pred')

def test_input_not_dataframe():
    with pytest.raises(TypeError):  
        r2_score("not a dataframe", 'y_true', 'y_pred')

def test_missing_columns():
    df = pd.DataFrame({'y_true': [1, 2, 3]})
    with pytest.raises(KeyError):  
        r2_score(df, 'y_true', 'y_pred')


if __name__ == "__main__":
    pytest.main()
