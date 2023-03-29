import numpy as np

from utils.get_arguments import get_params
from experiment import get_benchmark
from utils.infomation_exporter import InformationExporter

if __name__ == "__main__":

    params = get_params()
    for algorithm in params.algorithms:
        for problem in params.problems:
            for seed in params.seeds:
                for n_var in params.n_vars:
                    params.algorithm = algorithm
                    params.problem = problem
                    params.seed = seed
                    params.n_var = n_var
                    algorithm_name = f'{params.algorithm}' if params.algorithm_extend is None \
                        else f'{params.algorithm}-{params.algorithm_extend}'
                    problem_name = f'{params.problem}' if params.problem_extend is None \
                        else f'{params.problem}-{params.problem_extend}'

                    folder_name = f'output/data/{problem_name}/x{params.n_var}/' \
                                          + '/' + algorithm_name + '/' + f'{params.seed}' + '/'

                    benchmark = get_benchmark(params.problem, params.n_var)

                    if params.n_var == 1:
                        x_true = np.expand_dims(np.linspace(benchmark.bound[0],benchmark.bound[1],1000),-1)
                        y_true, _ = benchmark.evaluate(x_true)
                        x_algo = np.genfromtxt(folder_name+'X.txt')
                        y_algo = np.genfromtxt(folder_name+'Y.txt')
                        ax = None
                        ax = InformationExporter.plot2D(x_true,y_true,marker='', ls=':',c='gray', ax=ax)
                        InformationExporter.plot2D(x_algo,y_algo,ls='',c='red', marker='x',ax =ax, show=True)

                    if params.n_var == 2:
                        x_true_grid, _ = np.meshgrid(np.linspace(benchmark.bound[0],benchmark.bound[1],50),
                                             np.linspace(benchmark.bound[0],benchmark.bound[1],50))
                        x_true = np.vstack([x_true_grid.flatten(), x_true_grid.T.flatten()]).T
                        y_true_grid = benchmark.evaluate(x_true)[0].reshape(x_true_grid.shape)
                        x_algo = np.genfromtxt(folder_name+'X.txt')
                        y_algo = np.genfromtxt(folder_name+'Y.txt')
                        ax = None
                        ax = InformationExporter.contour2D(x_true_grid,y_true_grid,ax=ax)
                        InformationExporter.plot2D(x_algo[:,0],x_algo[:,1],ls='',c='red',marker='x',show=True, ax=ax)