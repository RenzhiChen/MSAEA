# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import GPy

from GPyOpt.models import BOModel
from sklearn.cluster import k_means


class ClusteringGPModel(BOModel):
    analytical_gradient_prediction = False

    def __init__(self, kernels=None, noise_var=None, exact_feval=False, optimizer='bfgs', max_iters=1000,
                 optimize_restarts=5, n_cluster=5, warping_function=None, warping_terms=3, verbose=False,
                 feasible_region=None):

        self.kernels = kernels
        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.warping_function = warping_function
        self.warping_terms = warping_terms
        self.models = None
        self.n_cluster = n_cluster
        self.X = None
        self.Y = None
        self.feasible_region = feasible_region

    def _create_model(self, X, Y):
        # --- define kernel
        self.input_dim = X.shape[1]

        if self.kernels is None:
            self.kernels = []
            for i in range(self.n_cluster):
                self.kernels.append(GPy.kern.RBF(self.input_dim, variance=1.))

        _, label, _ = k_means(X=X, n_clusters=self.n_cluster)

        self.models = []
        for i in range(self.n_cluster):
            idx = np.where(label == i)
            model = GPy.models.gp_regression.GPRegression(X[idx], Y[idx], kernel=self.kernels[i],)
            if self.exact_feval:
                model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
            else:
                model.Gaussian_noise.constrain_positive(warning=False)
            self.models.append(model)

        # --- restrict variance if exact evaluations of the objective

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        self.X = X_all
        self.Y = X_all

        self._create_model(X_all, Y_all)

        for i in range(self.n_cluster):
            self.models[i].optimize(optimizer=self.optimizer, messages=self.verbose, max_iters=self.max_iters)
        # self.debug()

    def predict(self, X):
        if X.ndim == 1: X = X[None, :]
        ms = []
        vs = []
        for i in range(self.n_cluster):
            m, v = self.models[i].predict(X)
            v = np.clip(v, 1e-10, np.inf)
            v = np.sqrt(v)
            ms.append(ms)
            vs.append(vs)
        return ms[0], np.sqrt(vs[0])

    def get_fmin(self):
        return self.predict(self.X)[0].min()

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        params = []
        for model in self.models:
            params.append(model[:])
        return np.atleast_2d(np.hstack(params))

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        all_names = []
        for i, model in enumerate(self.models):
            names =  model.parameter_names_flat()
            for name in names:
                all_names.append("m%d-"%i+name)
        return all_names
    ''' 
    def debug(self):
        n_FE = len(self.X)
        for i in range(self.n_cluster):
            np.savetxt('./debug/%d-%d-trainx.txt'%(n_FE,i),self.models[i].X)
            np.savetxt('./debug/%d-%d-trainy.txt' % (n_FE, i), self.models[i].Y)
            bounds = self.feasible_region.get_bounds()
            if self.input_dim == 1:
                x_test = np.expand_dims(np.linspace(bounds[0][0],bounds[0][1],100),-1)
                y_pred = self.models[i].predict(x_test)
                np.savetxt('./debug/%d-%d-predx.txt' % (n_FE, i), x_test)
                np.savetxt('./debug/%d-%d-predy.txt' % (n_FE, i), y_pred[0])
                np.savetxt('./debug/%d-%d-preds.txt' % (n_FE, i), y_pred[1])
    '''