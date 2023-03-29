import GPy
from functools import partial
from . import Algorithm
from GPyOpt import Design_space
from GPyOpt.experiment_design import initial_design
import numpy as np
from scipy.optimize import differential_evolution
from models.rbfnet import RBFNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sko.PSO import PSO
import inspect

class MAMPSO(Algorithm):
    """
        Ji X, Zhang Y, Gong D, et al. Multisurrogate-assisted multitasking particle swarm optimization for expensive multimodal problems[J]. IEEE Transactions on Cybernetics, 2021.
    """

    def run(self):
        bounds = []
        for i in range(self.n_var):
            bounds.append(
                {'name': f'var_{i + 1}', 'type': 'continuous', 'domain': (self.bound[0], self.bound[1])})
        feasible_region = Design_space(space=bounds)
        n_cluster = 5
        popx = initial_design('random', feasible_region, self.n_init)
        popy, _ = self.benchmark.evaluate(popx)

        pbest = {'py': np.inf, 'rbf': np.inf, 'gp': np.inf}
        models = {'py': None, 'rbf': None, 'gp': None}
        c = {'py': 0, 'rbf': 0, 'gp': 0}
        counts = {'py': 0, 'rbf': 0, 'gp': 0}
        xi = {'py': 1.0, 'rbf': 1.0, 'gp': 1.0}
        tau = {'py': np.min(popy), 'rbf': np.min(popy), 'gp': np.min(popy)}
        X = {'py': initial_design('random', feasible_region, 40), 'rbf': initial_design('random', feasible_region, 40), 'gp': initial_design('random', feasible_region, 40)}
        while len(popx) < self.n_init + self.n_iter:
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(popx)
            model_py = LinearRegression()
            model_py.fit(X_poly, popy)
            model_rbf = RBFNet(n_cluster)
            model_rbf.fit(popx, popy)
            model_gp = GPy.models.GPRegression(popx, popy)
            model_gp.optimize()

            temp = []
            def obj_py(x):
                x_poly = poly.fit_transform(np.atleast_2d(x))
                return model_py.predict(x_poly).squeeze()
            def obj_rbf(x):
                return model_rbf.predict(np.atleast_2d(x)).squeeze()
            def obj_gp(x):
                return model_gp.predict(np.atleast_2d(x))[0].squeeze()
            objs ={'py':obj_py,'rbf':obj_rbf,'gp':obj_gp}
            lb, ub = [self.bound[0]] * self.n_var, [self.bound[1]] * self.n_var

            keys = ['py','rbf','gp']
            for key in keys:
                models[key] = PSO(func=objs[key], dim=self.n_var, lb=lb, ub=ub, c1=xi[key])
                models[key].X = X[key]

                if models[key].gbest_y < pbest[key]:
                    pbest[key] = models[key].gbest_y
                    c[key] = 0
                else:
                    c[key] += 1
                    if c[key] > 3:
                        xi[key] = xi[key] * 0.5
                        c[key] = 0
                        counts[key] += 1


            for key in keys:
                if counts[key] > 3:
                    if models[key].pbest_y != tau[key]:
                        X[key] = np.vstack([X[key],models[key].pbest_x])
                    else:
                        if models[key].pbest_x in models[key].X[:int(len(models[key].X)/2)]:
                            temp.append(models[key].pbest_x)
                            X[key] = initial_design('random', feasible_region, 40)
                c[key] = 0
                counts[key] = 0
                xi[key] = 1.0


                temp.append(models[key].pbest_x)
            temp.insert(0,popx)
            popx = np.vstack(temp)
            popy, _ = self.benchmark.evaluate(popx)
        return popx,popy