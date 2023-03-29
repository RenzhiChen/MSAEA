from . import Algorithm
from GPyOpt import Design_space
from GPyOpt.experiment_design import initial_design
import numpy as np
from scipy.optimize import differential_evolution
from models.rbfnet import RBFNet

class DRNESO(Algorithm):
    """
    Zhen H, Xiong S, Gong W, et al. Neighborhood evolutionary sampling with dynamic repulsion for expensive multimodal
    optimization[J]. Information Sciences, 2023, 630: 82-97.
    """
    def run(self):
        bounds = []
        for i in range(self.n_var):
            bounds.append(
                {'name': f'var_{i + 1}', 'type': 'continuous', 'domain': (self.bound[0], self.bound[1])})
        feasible_region = Design_space(space=bounds)

        #popsize = self.n_batch
        popsize = self.n_batch

        DBx = initial_design('random', feasible_region, self.n_init)
        DBy, _ = self.benchmark.evaluate(DBx)
        delta_1 = DBy.mean()
        A = []
        O = []
        phi = []
        while len(DBx) < self.n_init+self.n_iter:
            idx = np.argsort(DBy, axis=0).squeeze()
            popx = DBx[idx[:popsize], :]
            r =  min(20*self.n_var, popsize - 20, 150)
            K = (20 + r * np.random.random((len(popx),))).astype(int)
            U = self.neighborhood_evolutionary_sampling(popx,DBx,DBy, K, feasible_region.get_bounds())

            for u in U:

                if len(A) > 0:
                    restart = self.taboo_area_judgment(u,A,DBx,DBy,delta_1)
                    if restart:
                        u = self.elite_restart(u,DBx,DBy, feasible_region.get_bounds())
                DBx = np.vstack([DBx,u])
                DBy, _ = self.benchmark.evaluate(DBx)
                A,O,phi = self.update_archive(DBx[-1,:], DBy[-1,:], DBx, DBy, A,O,phi)

        return DBx, DBy

    def neighborhood_evolutionary_sampling(self, X, DBx, DBy, K, bounds):
        n_cluster = self.n_batch
        result = []
        for i in range(len(X)):
            idx = np.argsort(np.linalg.norm(DBx - X[i], axis=1))
            neighbor_x = DBx[idx[:K[i]]]
            neighbor_y = DBy[idx[:K[i]]]
            rbfnet = RBFNet(n_cluster)
            rbfnet.fit(neighbor_x, neighbor_y)
            def obj(x):
                return rbfnet.predict(np.atleast_2d(x)).squeeze()
            if np.random.random() > 0.5:
                result.append(differential_evolution(obj,bounds).x)
            else:
                lb = list(np.min(X,axis=0))
                ub = list(np.max(X,axis=0))
                result.append(differential_evolution(obj,np.array((lb,ub)).T).x)
        return np.vstack(result)

    def taboo_area_judgment(self,u,A,DBx,DBy,delta_1):
        delta_2 = DBy.mean()
        delta = (delta_1+delta_2) / 2
        r = np.zeros([len(A),])
        for j in range(len(A)):
            distances = np.linalg.norm(DBx - A[j], axis=1)
            idx = np.argsort(distances)
            for i in idx:
                if delta > DBy[i]:
                    r[j] = distances[i]
        dist = np.linalg.norm(u-np.array(A),axis=1)
        j = np.argmin(dist)
        if np.min(dist) < r[j]:
            return True
        return False

    def elite_restart(self,u,DBx,DBy, bounds):
        k =  self.n_batch
        n_cluster = self.n_batch
        idx = np.argsort(np.linalg.norm(DBx - u, axis=1))
        neighbor_x = DBx[idx[:k]]
        neighbor_y = DBy[idx[:k]]
        rbfnet = RBFNet(n_cluster)
        rbfnet.fit(neighbor_x, neighbor_y)

        def obj(x):
            return rbfnet.predict(np.atleast_2d(x)).squeeze()

        return differential_evolution(obj, bounds).x

    def update_archive(self, u, fu, DBx, DBy, A,O,phi):

        #print('1. O:%d phi:%d'%(len(O),len(phi)))
        #if len(O) != len(phi):
        #    print("ERROR")
        epsilon = 1e-6
        eta = 0.001 * len(u)
        best_id = np.argmin(DBy)
        if fu - DBy[best_id]<epsilon:
            if len(O) == 0:
                O.append(u)
                phi = np.append(phi,1)
                #print('2. O:%d phi:%d' % (len(O), len(phi)))
            else:
                dists = np.linalg.norm(O-u,axis=1)
                j = np.argmin(dists)
                if dists[j] < eta:
                    O[j] = u
                    phi[j] = phi[j] + 1
                else:
                    O.append(u)
                    phi = np.append(phi,1)
                    #print('3. O:%d phi:%d' % (len(O), len(phi)))
        if phi is not None:
            idx = np.where(np.array(phi)>5)
            if len(idx[0]) > 0:
                for i in idx[0]:
                    A.append(O[i])
        return A, O, phi


