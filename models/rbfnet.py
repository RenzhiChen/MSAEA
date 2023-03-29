import numpy as np


class RBFNet:
    def __init__(self, hidden_shape, sigma=1.0):
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _get_radial_basis(self, X):
        dist_matrix = np.zeros((X.shape[0], self.centers.shape[0]))
        for i, xi in enumerate(X):
            for j, c in enumerate(self.centers):
                dist_matrix[i, j] = np.exp(-self.sigma * np.linalg.norm(xi - c) ** 2)
        return dist_matrix

    def fit(self, X, y):
        # Randomly choose centers from training data
        random_idx = np.random.choice(X.shape[0], self.hidden_shape)
        self.centers = X[random_idx]

        # Calculate radial basis function output matrix
        rbf_output = self._get_radial_basis(X)

        # Solve for the weights using linear algebra
        self.weights = np.dot(np.linalg.pinv(rbf_output), y)

    def predict(self, X):
        rbf_output = self._get_radial_basis(X)
        y_pred = np.dot(rbf_output, self.weights)
        return y_pred
