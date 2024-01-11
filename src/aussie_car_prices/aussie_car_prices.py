class OLSRegressor():
    """
    Ordinary Least Squares Linear Regressor

    OLSRegressor will fit a linear model with coefficients w = (w1, w2, ..., wn)
    to minimize Residual Sum of Squares (RSS) between the observed targets values
    in the dataset, and the targets predicted by the linear approximation for the
    examples in the dataset.
    """

    def fit(self, X, y):
        """
        Fits the linear model according to the OLS mechanism.

        Parameters
        ----------
        X : {array-like matrix} of shape (n_examples, n_features)
            Dataset that will be used as the feature values to train the model.

        y : array-like matrix of shape (n_examples, n_targets)
            Dataset that will be used as the target values to train the model.

        Returns
        -------
        self : object
            Fitted Estimator.
        """