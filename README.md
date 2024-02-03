# OLS_Regressor

[![Documentation Status](https://readthedocs.org/projects/olsregressor/badge/?version=latest)](https://olsregressor.readthedocs.io/en/latest/?badge=latest)

## About
The OLS Regression Package is a Python library designed to streamline the process of performing Ordinary Least Squares (OLS) regression analysis. Whether you're a data scientist, researcher, or analyst, this package aims to provide a simple and efficient tool for fitting linear models to your data. It will fit a linear model with coefficients w = (w1, w2, ..., wn) to minimize Residual Sum of Squares (RSS) between the observed targets values in the dataset, and the targets predicted by the linear approximation for the examples in the dataset.

## Installation
### Install the package from PyPi
Run this command to install the `ols_regressor` package from PyPi
```bash
$ pip install ols_regressor
```

### Install the package from GitHub
Run the following commands to install from GitHub if the installation is unsuccessful from PyPi.

**Clone the repository**
Open your terminal, navigate to where you would like the repository to be cloned and run the following command:
```bash
$ git clone git@github.com:UBC-MDS/OLS_regressor.git
```

**Create the conda environment and activate it**
Run the following command to create the conda environment which will include the necessary Python and Poetry versions and dependencies:
```bash
conda env create --name ols_regressor python=3.9 poetry==1.7.1 -y
```

Next, run the following command to activate the conda environment we created:
```bash
conda activate `ols_regressor`
```

**Install the package using Poetry**
Run the following command to install the package `ols_regressor`:
```bash
poetry install
```

## Running the tests for the package functions
Navigate to the root directory of the project and run the following command in your terminal to run the tests for the functions:
```bash
pytest tests/*
```

## Functions

- `fit`: Fits the linear model according to the OLS mechanism.
- `predict`: Predicts target values using the fitted linear model.
- `score`: Calculates the coefficient of determination R-squared value for the prediction.
- `cross_validate`: Performs cross-validated Ordinary Least Squares (OLS) regression.

## `OLS_Regressor` use in Python ecosystem
The OLS Regression Package seamlessly integrates into the rich Python ecosystem, offering a specialized solution for Ordinary Least Squares (OLS) regression analysis. While various Python libraries provide general-purpose machine learning and statistical functionalities, our package focuses specifically on the simplicity and efficiency of linear regression. scikit-learn is a widely used machine learning library that encompasses regression among its many capabilities [`scikit-learn`](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning). Our package distinguishes itself by providing a lightweight and user-friendly interface tailored for users seeking a straightforward solution for OLS regression without the overhead of extensive machine learning or statistical functionalities. If you find that your needs align more closely with a broader set of machine learning tools or comprehensive statistical modeling, scikit-learn or statsmodels may be suitable alternatives. As of [2024-01-12], no existing package caters specifically to OLS regression with our package's emphasis on simplicity and ease of use.

## Contributors
- Xia Yimeng (@YimengXia)
- Sifan Zhang (@Sifanz)
- Charles Xu (@charlesxch)
- Waleed Mahmood (@WaleedMahmood1)

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`OLS_Regressor` is licensed under the terms of the MIT license.

## Credits

`OLS_Regressor` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
