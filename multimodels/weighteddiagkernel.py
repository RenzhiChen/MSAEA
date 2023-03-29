import numpy as np
import GPy


class WeightedDiagKernel(GPy.kern.Kern):
    def __init__(self, input_dim, weighted_function, kernel, active_dims=None):
        super(WeightedDiagKernel, self).__init__(input_dim, active_dims, name="weighted diag")
        self.kernel = kernel
        self.left = WeightedDiagKernelLeft(input_dim=input_dim, weighted_function=weighted_function)
        self.right = WeightedDiagKernelRight(input_dim = input_dim, weighted_function = weighted_function)

    def K(self, X, X2=None):
        return self.left.K(X,X2) @ self.kernel.K(X,X2) @ self.right.K(X,X2)

    def update_gradients_full(self, dL_dK, X, X2):
        self.kernel.update_gradients_full(dL_dK, X, X2)

    def Kdiag(self, X):
        return self.left.Kdiag(X) * self.kernel.Kdiag(X) * self.right.Kdiag(X)


class WeightedDiagKernelLeft(GPy.kern.Kern):
    def __init__(self, input_dim, weighted_function, active_dims=None):
        super(WeightedDiagKernelLeft, self).__init__(input_dim, active_dims, name="weighted diag left")
        self.weighted_function = weighted_function

    def K(self, X, X2=None):
        return np.diag(self.weighted_function(X))

    def Kdiag(self, X):
        return self.weighted_function(X)

    def update_gradients_full(self, dL_dK, X, X2):
        pass


class WeightedDiagKernelRight(GPy.kern.Kern):
    def __init__(self, input_dim, weighted_function, active_dims=None):
        super(WeightedDiagKernelRight, self).__init__(input_dim, active_dims, name="weighted diag right")
        self.weighted_function = weighted_function

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return np.diag(self.weighted_function(X2))

    def update_gradients_full(self, dL_dK, X, X2):
        pass

    def Kdiag(self, X):
        return self.weighted_function(X)


