from . import Algorithm
from GPyOpt import Design_space
from GPyOpt.experiment_design import initial_design
import numpy as np
from scipy.optimize import differential_evolution
from models.rbfnet import RBFNet

class DREM(Algorithm):
    """
    Gao W, Wei Z, Gong M, et al. Solving expensive multimodal optimization problem by a decomposition differential evolution algorithm[J]. IEEE Transactions on Cybernetics, 2021.
    """
    def run(self):
        n_cluster = self.n_var
        k_neighbour = 5
        bounds = []
        for i in range(self.n_var):
            bounds.append(
                {'name': f'var_{i + 1}', 'type': 'continuous', 'domain': (self.bound[0], self.bound[1])})
        feasible_region = Design_space(space=bounds)
        popx = initial_design('random', feasible_region, self.n_init)
        popy, _ = self.benchmark.evaluate(popx)
        subpopx = []
        subpopy = []
        for i in range(1,n_cluster+1):
            subpopx.append(popx[:i*int(self.n_init/n_cluster),:])
            subpopy.append(popy[:i * int(self.n_init / n_cluster), :])

        while len(popx) < self.n_init + self.n_iter:
            tmp = []
            for i in range(n_cluster):
                for x in subpopx[i]:
                    distance = np.linalg.norm(x-subpopx[i],axis=1)
                    idx = np.argsort(distance)[:k_neighbour]
                    rbfnet = RBFNet(5)
                    rbfnet.fit(subpopx[i][idx],subpopy[i][idx])
                    lb = list(np.min(subpopx[i][idx], axis=0))
                    ub = list(np.max(subpopx[i][idx], axis=0))
                    def obj(x):
                        return rbfnet.predict(np.atleast_2d(x)).squeeze()

                    tmp.append(differential_evolution(obj, np.array((lb,ub)).T).x)
            tmp.insert(0,popx)
            popx = np.vstack(tmp)
            popy, _ = self.benchmark.evaluate(popx)
        return popx, popy
        ''' 
        We do not use local search here because we dont have enough FE for a full generation
        '''
