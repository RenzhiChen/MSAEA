# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import numpy as np
from argparse import Namespace

from utils.get_arguments import get_params
from utils.infomation_exporter import InformationExporter
import benchmarks
import algorithms
from algorithms import Algorithm
from benchmarks import MultiModalProblem


def get_benchmark(benchmark_name: str, n_var: int)-> MultiModalProblem:
    return getattr(benchmarks, benchmark_name)(n_var)


def get_algorithm(algorithm_name: str, benchmark: MultiModalProblem, n_init:int, n_iter: int, n_batch: int) -> Algorithm:
    return getattr(algorithms, algorithm_name)(benchmark, n_init, n_iter, n_batch)


def run_experiment(parameters: Namespace) -> None:
    """The actual experiment code."""
    for seed in parameters.seeds:
        for benchmark_name in parameters.problems:
            for algorithm_name in parameters.algorithms:
                for n_var in parameters.n_vars:
                    parameters.n_var = n_var
                    parameters.algorithm = algorithm_name
                    parameters.problem = benchmark_name
                    parameters.seed = seed
                    np.random.seed(parameters.seed)
                    n_init, n_iter, n_batch = parameters.n_init, parameters.n_iter, parameters.n_batch

                    exporter = InformationExporter(parameters)
                    benchmark = get_benchmark(benchmark_name, n_var)

                    algorithm = get_algorithm(algorithm_name,benchmark, n_init, n_iter, n_batch)

                    X, Y = algorithm.run()

                    xfmt = '%lf' + '\t%lf' * (parameters.n_var - 1)
                    yfmt = '%lf'
                    np.savetxt(exporter.folder_name['data'] + 'X.txt', X, xfmt)
                    np.savetxt(exporter.folder_name['data'] + 'Y.txt', Y, yfmt)


if __name__ == "__main__":
    params = get_params()
    run_experiment(params)
