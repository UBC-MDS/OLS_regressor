import numpy as np
import pandas as pd


def cross_validate(model, X, y, cv, random_state=None):
    """
    Perform cross-validated Ordinary Least Squares (OLS) regression.

    Parameters
    ----------
    model : str
            Name of the model to run cross_validate with (it will be OLS in this case)
    
    X : array-like matrix of shape (n_examples, n_features)
            Dataset that will be used as the feature values to train the model.
    
    y : array-like matrix of shape (n_examples, n_targets)
            Dataset that will be used as the target values to train the model.
    
    cv : (int, optional)
            Number of cross-validation folds. Default is 5.
    
    random_state : (int or None, optional) 
                    Seed for reproducibility. Default is None.


    Returns
    -------
    list : List of dictionaries, each containing the results of an OLS regression fold.
    """
    pass