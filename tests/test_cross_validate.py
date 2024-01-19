import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from ols_regressor.cross_validate import cross_validate


@pytest.fixture
def X():
    np.random.seed(123)
    X = np.random.rand(100, 3)
    return X


@pytest.fixture
def y(X):
    true_coefficients = np.array([1.5, -2.0, 3.0])
    np.random.seed(123)
    noise = np.random.randn(100) * 0.5 - 1
    y = X.dot(true_coefficients) + noise
    return y


def test_cross_validate_is_valid_dict(X, y):
    model = LinearRegression()
    cv_result = cross_validate(model, X, y)
    assert type(cv_result) is dict
    assert "train_score" in cv_result
    assert "test_score" in cv_result
    assert "fit_time" in cv_result
    assert "score_time" in cv_result


def test_cross_validate_contains_valid_data(X, y):
    model = LinearRegression()
    cv_result = cross_validate(model, X, y)
    assert type(cv_result["train_score"]) is list
    assert type(cv_result["test_score"]) is list
    assert type(cv_result["fit_time"]) is list
    assert type(cv_result["score_time"]) is list
    assert len(cv_result["train_score"]) == len(cv_result["test_score"])
    assert len(cv_result["train_score"]) == len(cv_result["fit_time"])
    assert len(cv_result["train_score"]) == len(cv_result["score_time"])


def test_cross_validate_return_correct_result(X, y):
    model = LinearRegression()
    cv_result = cross_validate(model, X, y)
    assert np.mean(cv_result["train_score"]) >= 0.6
    assert np.mean(cv_result["test_score"]) >= 0.6


def test_cross_validate_require_valid_model(X, y):
    with pytest.raises(TypeError):
        cross_validate(1, X, y)
    with pytest.raises(TypeError):
        cross_validate(None, X, y)
    with pytest.raises(TypeError):
        cross_validate([1, 2, 3], X, y)

    class MyClass:
        def __init__(self):
            pass

        def fit(self):
            pass

    with pytest.raises(TypeError):
        cross_validate(MyClass(), X, y)


def test_cross_validate_require_valid_X(X, y):
    model = LinearRegression()
    with pytest.raises(ValueError):
        cross_validate(model, None, y)
    with pytest.raises(ValueError):
        cross_validate(model, 1, y)
    with pytest.raises(ValueError):
        cross_validate(model, [1, 2, 3], y)


def test_cross_validate_require_valid_y(X, y):
    model = LinearRegression()
    with pytest.raises(ValueError):
        cross_validate(model, X, None)
    with pytest.raises(ValueError):
        cross_validate(model, X, 1)
    with pytest.raises(ValueError):
        cross_validate(model, X, [1])


def test_cross_validate_require_compatible_Xy():
    model = LinearRegression()
    np.random.seed(123)
    X = np.random.rand(100, 3)
    y = np.random.rand(10)
    with pytest.raises(ValueError):
        cross_validate(model, X, y)


def test_cross_validate_require_valid_cv(X, y):
    model = LinearRegression()
    with pytest.raises(ValueError):
        cross_validate(model, X, y, None)
    with pytest.raises(ValueError):
        cross_validate(model, X, y, -1)
    with pytest.raises(ValueError):
        cross_validate(model, X, y, [1, 2])
