from GPyOpt.experiment_design import initial_design
from GPyOpt.optimization import AcquisitionOptimizer
from GPyOpt.methods import ModularBayesianOptimization
from GPy.kern import RBF
from GPyOpt import Design_space

from algorithms import Algorithm
from multimodels.weightedclusteringmultioutputmodel import WeightedClusteringMultiOutputGPModel
from multimodels.m_batch import mRandom
from multimodels.mLCB import multioutputAcquisitionLCB
from multimodels.modular_bayesian_optimization_fix import ModularBayesianOptimizationFix


class MSAEA(Algorithm):
    def run(self):
        # number of clusters
        n_cluster = self.n_var

        # Determine the subset where we are allowed to sample
        bounds = []
        for i in range(self.n_var):
            bounds.append(
                {'name': f'var_{i + 1}', 'type': 'continuous', 'domain': (self.bound[0], self.bound[1])})
        feasible_region = Design_space(space=bounds)
        initial = initial_design('random', feasible_region, self.n_init)

        model = WeightedClusteringMultiOutputGPModel(exact_feval=True, optimize_restarts=10, verbose=False,
                                             feasible_region=feasible_region, n_cluster=n_cluster)

        # CHOOSE the acquisition optimizer
        acquisition_optimizer = AcquisitionOptimizer(feasible_region)

        # CHOOSE the type of acquisition
        acquisition = multioutputAcquisitionLCB(model, feasible_region, optimizer=acquisition_optimizer)

        # CHOOSE a collection method
        evaluator = mRandom(acquisition, self.n_batch)

        bo = ModularBayesianOptimizationFix(model, feasible_region, self.benchmark, acquisition, evaluator, initial)

        bo.run_optimization(max_iter=self.n_iter+self.n_init)

        return bo.X, bo.Y
