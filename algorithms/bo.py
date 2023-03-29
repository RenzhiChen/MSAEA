from GPyOpt.experiment_design import initial_design
from GPyOpt.models import GPModel
from GPyOpt import Design_space
from GPyOpt.optimization import AcquisitionOptimizer
from GPyOpt.acquisitions import AcquisitionLCB
from GPyOpt.methods import ModularBayesianOptimization
from GPyOpt.core.evaluators import RandomBatch
from GPy.kern import RBF

from algorithms import Algorithm
from multimodels.modular_bayesian_optimization_fix import ModularBayesianOptimizationFix

class BO(Algorithm):
    def run(self):
        # Determine the subset where we are allowed to sample
        bounds = []
        for i in range(self.n_var):
            bounds.append(
                {'name': f'var_{i + 1}', 'type': 'continuous', 'domain': (self.bound[0], self.bound[1])})
        feasible_region = Design_space(space=bounds)
        initial = initial_design('random', feasible_region, self.n_init)

        # CHOOSE the model type
        kernel = RBF(input_dim=self.n_var)
        model = GPModel(exact_feval=True, kernel=kernel, optimize_restarts=10, verbose=False)

        # CHOOSE the acquisition optimizer
        acquisition_optimizer = AcquisitionOptimizer(feasible_region)

        # CHOOSE the type of acquisition
        acquisition = AcquisitionLCB(model, feasible_region, optimizer=acquisition_optimizer)

        # CHOOSE a collection method
        evaluator = RandomBatch(acquisition,self.n_batch)

        bo = ModularBayesianOptimizationFix(model, feasible_region, self.benchmark, acquisition, evaluator, initial)

        bo.run_optimization(max_iter=self.n_iter+self.n_init)

        return bo.X, bo.Y
