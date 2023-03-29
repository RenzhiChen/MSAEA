from argparse import ArgumentParser
import numpy as np


def get_params():
    """

    """
    parser = ArgumentParser()
    parser.add_argument('--seeds', type=int, default=[0], nargs="+",
                        help='list of random seed')
    parser.add_argument('--n-vars', type=int, default=[1], nargs="+",
                        help='dimension of input space')
    parser.add_argument('--n-init', type=int, default=None,
                        help='number of initial sample')
    parser.add_argument('--n-iter', type=int, default=None,
                        help='number of FEs')
    parser.add_argument('--n-batch', type=int, default=None,
                        help='number of FEs per batch')
    parser.add_argument('--algorithms', type=str, default=['MSAEA'], nargs="+",
                        help='list of algorithm name')
    parser.add_argument('-p', '--problems', type=str, default=['Ackley'], nargs="+",
                        help='optimization problem name')
    parser.add_argument('--algorithm-extend', type=str, default=None,
                        help='custom algorithm name to distinguish between experiments on same '
                             'algorithm')
    parser.add_argument('--problem-extend', type=str, default=None,
                        help='custom problem name to distinguish between experiments on same '
                             'problem')
    params, _ = parser.parse_known_args(None)

    return params
