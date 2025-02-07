import numpy as np

def compute_RMSE(y_true, y_pred):
    """Computes the Root Mean Squared Error (RMSE) given the ground truth
    values and the predicted values.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

class AnalyticalMethod:
    def __init__(self):
        """Class constructor for AnalyticalMethod"""
        self.W = None

    def feature_transform(self, X):
        """Appends a vector of ones for the bias term."""
        return np.c_[np.ones((X.shape[0], 1)), X]

    def compute_weights(self, X, y):
        """Compute the weights based on the analytical solution."""
        X = self.feature_transform(X)
        self.W = np.linalg.pinv(X.T @ X) @ X.T @ y
        return self.W

    def predict(self, X):
        """Predict values for test data using analytical solution."""
        X = self.feature_transform(X)
        return X @ self.W

class PolyFitMethod:
    def __init__(self):
        """Class constructor for PolyFitMethod"""
        self.W = None

    def compute_weights(self, X, y):
        """Compute the weights using np.polyfit()."""
        self.W = np.polyfit(X, y, deg=1)
        return self.W

    def predict(self, X):
        """Predict values for test data using np.poly1d()."""
        return np.poly1d(self.W)(X)
