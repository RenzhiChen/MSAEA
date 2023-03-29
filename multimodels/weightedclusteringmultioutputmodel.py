# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import GPy
import numpy.linalg
from functools import partial
from GPyOpt.models import BOModel
from sklearn.cluster import k_means
from sklearn.mixture import GaussianMixture
from multimodels.weighteddiagkernel import WeightedDiagKernel


class WeightedClusteringMultiOutputGPModel(BOModel):
    analytical_gradient_prediction = False

    def __init__(self, noise_var=None, exact_feval=False, optimizer='bfgs', max_iters=1000,
                 optimize_restarts=5, n_cluster=5, warping_function=None, warping_terms=3, verbose=False,
                 feasible_region=None):

        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.warping_function = warping_function
        self.warping_terms = warping_terms
        self.model = None
        self.n_cluster = n_cluster
        self.X = None
        self.Y = None
        self.feasible_region = feasible_region
        self.gmm = GaussianMixture(n_components=self.n_cluster, warm_start=True)

    def _create_model(self, X, Y):
        # --- define kernel
        self.input_dim = X.shape[1]

        self.gmm.fit(X=X)
        label = self.gmm.predict(X)
        prob = self.gmm.predict_proba(X)

        X_all = []
        Y_all = []
        kernels = []
        for i in range(self.n_cluster):
            # idx = np.where(prob[:,i]>1e-2)
            idx = np.where(label == i)
            X_all.append(X[idx])
            Y_all.append(Y[idx])
            ''' 
            def prob_func(X, i):
                return np.log(gmm.predict_proba(X)[:, i])


            kernels.append(WeightedDiagKernel(input_dim=self.input_dim,
                               weighted_function=partial(prob_func,i=i),
                               kernel=GPy.kern.RBF(self.input_dim)))
            '''
            kernels.append(GPy.kern.RBF(self.input_dim))
        kernel = GPy.util.multioutput.LCM(input_dim=X.shape[1], num_outputs=self.n_cluster, kernels_list=kernels)

        self.model = GPy.models.GPCoregionalizedRegression(X_list=X_all, Y_list=Y_all, kernel=kernel)

        if self.exact_feval:
            self.model.mixed_noise.constrain_fixed(1e-6, warning=False)
        else:
            self.model.mixed_noise.constrain_positive(warning=False)

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        self.X = X_all
        self.Y = X_all
        self._create_model(X_all, Y_all)
        self.model.optimize(optimizer=self.optimizer, messages=self.verbose, max_iters=self.max_iters)
        # self.debug()

    def predict(self, X):

        for i in range(self.n_cluster):
            idx = np.where(X[:, -1] == i)
        m, v = self.model.predict(X, full_cov=False, include_likelihood=False)
        v = np.clip(v, 1e-10, np.inf)
        v = np.sqrt(v)
        return m, v

    def get_fmin(self):
        return self.predict(self.X)[0].min()

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        return np.atleast_2d(self.model[:])

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        return self.model.parameter_names_flat().tolist()

    def debug(self):
        n_FE = len(self.X)

        np.savetxt('./debug/%d-trainx.txt' % (n_FE), self.model.X)
        np.savetxt('./debug/%d-trainy.txt' % (n_FE), self.model.Y)
        bounds = self.feasible_region.get_bounds()
        if self.input_dim == 1:
            x_test = np.expand_dims(np.linspace(bounds[0][0], bounds[0][1], 100), -1)
            for i in range(self.n_cluster):
                x_test_full = np.hstack([x_test, np.ones((len(x_test), 1)) * i])
                y_pred = self.model.predict(x_test_full)
                np.savetxt('./debug/%d-%d-predx.txt' % (n_FE, i), x_test_full)
                np.savetxt('./debug/%d-%d-predy.txt' % (n_FE, i), y_pred[0])
                np.savetxt('./debug/%d-%d-preds.txt' % (n_FE, i), y_pred[1])
        if self.input_dim == 2:
            x_grid, _ = np.meshgrid(np.linspace(bounds[0][0], bounds[0][1], 21),
                                    np.linspace(bounds[0][0], bounds[0][1], 21))
            x_test = np.vstack([x_grid.flatten(), x_grid.T.flatten()]).T
            for i in range(self.n_cluster):
                x_test_full = np.hstack([np.vstack([x_test, self.model.X[:, :-1]]),
                                         np.ones((len(x_test) + len(self.model.X[:, :-1]), 1)) * i])
                y_pred = self.model.predict(x_test_full, full_cov=False, include_likelihood=False)
                np.savetxt('./debug/%d-%d-predx.txt' % (n_FE, i), x_test_full)
                np.savetxt('./debug/%d-%d-predy.txt' % (n_FE, i), y_pred[0])
                np.savetxt('./debug/%d-%d-preds.txt' % (n_FE, i), y_pred[1])
