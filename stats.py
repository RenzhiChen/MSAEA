import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from scipy.stats import wilcoxon
import itertools
from utils.get_arguments import get_params
from experiment import get_benchmark
from utils.infomation_exporter import InformationExporter as IE
from scipy.stats import rankdata
from math import sqrt
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
import pandas as pd


if __name__ == "__main__":
    PLOT = False
    TIKIZ_TABLE = True
    params = get_params()

    trajs = {}
    epsilon_fs = {}
    epsilon_ts = {}
    for problem in params.problems:
        for n_var in params.n_vars:
            params.problem = problem
            params.n_var = n_var
            n_init = (params.n_var * 11 - 1) if params.n_init is None else params.n_init
            n_batch = max(params.n_var, 2) if params.n_batch is None else params.n_batch
            n_iter = (params.n_var * 11 - 1) if params.n_iter is None else params.n_iter
            n_FE = n_iter + n_init
            traj = [np.expand_dims(np.arange(n_iter+1),-1)]
            epsilon_f = []
            epsilon_t = []
            for algorithm in params.algorithms:
                losses = []
                for seed in params.seeds:

                    problem_name = f'{params.problem}' if params.problem_extend is None \
                        else f'{params.problem}-{params.problem_extend} '
                    algorithm_name = f'{algorithm}' if params.algorithm_extend is None \
                        else f'{algorithm}-{params.algorithm_extend}'
                    params.seed = seed
                    folder_name = f'output/data/{problem_name}/x{params.n_var}/' \
                                  + '/' + algorithm_name + '/' + f'{params.seed}' + '/'
                    trace = np.genfromtxt(folder_name + 'Y.txt')
                    loss = np.zeros_like(trace)
                    loss[0] = trace[0]
                    for i in range(1, len(trace)):
                        loss[i] = min(loss[i - 1], trace[i])
                    if len(loss) < n_FE:
                        print("Warning: %s has not enough data: %d (required %d)" % (
                        folder_name + 'Y.txt', len(loss), n_FE))
                        continue
                    losses.append(loss[:n_FE])
                losses = np.vstack(losses)
                epsilon_t.append(losses[:, -1])
                epsilon_f.append(losses.mean(axis=1))
                traj.append(np.vstack([np.mean(losses, axis=0)[n_init-1:],
                                       np.mean(losses, axis=0)[n_init-1:] + 0.1 * np.std(losses, axis=0)[n_init-1:],
                                       np.mean(losses, axis=0)[n_init-1:] - 0.1 * np.std(losses, axis=0)[n_init-1:]]).T)

            traj = np.hstack(traj)  # (n_FE) x (1+3*n_algo)
            epsilon_t = np.vstack(epsilon_t).T  # (n_seed) x (n_algo)
            epsilon_f = np.vstack(epsilon_f).T  # (n_seed) x (n_algo)
            trajs[problem + str(n_var)] = traj
            epsilon_ts[problem + str(n_var)] = epsilon_t
            epsilon_fs[problem + str(n_var)] = epsilon_f

            if PLOT:
                ax = None
                colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'gray']
                for i, algorithm in enumerate(params.algorithms):
                    c = colors[i]
                    ax = IE.plot2D(np.arange(len(traj[:, 3 * i])), traj[:, 3 * i], c=c, ls='-', marker='',
                                   label=algorithm,
                                   ax=ax)
                    ax = IE.fill2D(np.arange(len(traj[:, 3 * i])),
                                   traj[:, 3 * i + 1],
                                   traj[:, 3 * i + 2], c=c, ax=ax)
                    ax.set_yscale('log')
                IE.show()

            output_folder = f'output/data/{problem_name}/x{params.n_var}/'
            # Export traj
            np.savetxt(output_folder + problem_name + '_%dD' % params.n_var + '.traj', traj)

            # Export epsilon_f and epsilon_t
            np.savetxt(output_folder + problem_name + '_%dD' % params.n_var + '.epsilon_f', epsilon_f)
            np.savetxt(output_folder + problem_name + '_%dD' % params.n_var + '.epsilon_t', epsilon_t)

    # Export A12 result
    a12_all = []
    for i in range(1, len(params.algorithms)):
        a12 = np.zeros([4, ])
        for n_var in params.n_vars:
            for problem in params.problems:
                for k, indicator in enumerate([epsilon_ts, epsilon_fs]):
                    cnt = 0
                    cases = 0
                    data = indicator[problem + str(n_var)]
                    d0 = np.sort(data[:, 0])
                    d1 = np.sort(data[:, i])
                    for j in range(len(data[:, i])):
                        if d1[j] > d0[j]:
                            cnt += 1
                        cases += 1
                    rate = cnt / cases
                    if rate > 0.71:
                        a12[0] += 1
                        #print(problem + str(n_var) + 'D(%d): large' % k)
                    elif rate > 0.64:
                        a12[1] += 1
                        #print(problem + str(n_var) + 'D(%d): median' % k)
                    elif rate > 0.56:
                        a12[2] += 1
                        #print(problem + str(n_var) + 'D(%d): small' % k)
                    else:
                        a12[3] += 1
                        #print(problem + str(n_var) + 'D(%d): equal' % k)

        a12_all.append(a12/a12.sum())
    # Export a12
    a12_all = np.hstack([np.expand_dims(np.arange(1,len(params.algorithms)),-1),np.vstack(a12_all)])
    np.savetxt('output/data/a12.txt',a12_all,fmt='%d\t%.2lf\t%.2lf\t%.2lf\t%.2lf')

    # export scott-knott result
    ranking = []
    for k, indicator in enumerate([epsilon_ts, epsilon_fs]):
        for n_var in params.n_vars:
            for problem in params.problems:
                data = indicator[problem + str(n_var)]
                data_list = {}
                for i, algorithm in enumerate(params.algorithms):
                    data_list[algorithm] = data[:,i]
                data_list = pd.DataFrame(data_list)
                sk = importr('ScottKnottESD')
                r_sk = sk.sk_esd(data_list)
                column_order = list(r_sk[3] - 1)
                ranking.append(r_sk[1].astype("int"))
    ranking = np.vstack(ranking)
    scottknott_t = np.percentile(ranking[:int(len(ranking)/2)], [25, 50, 75], axis=0)
    scottknott_f = np.percentile(ranking[int(len(ranking)/2):], [25, 50, 75], axis=0)
    scottknott_t[0, :] = scottknott_t[1, :] - scottknott_t[0, :]
    scottknott_t[2, :] = scottknott_t[2, :] - scottknott_t[1, :]
    scottknott_f[0, :] = scottknott_f[1, :] - scottknott_f[0, :]
    scottknott_f[2, :] = scottknott_f[2, :] - scottknott_f[1, :]
    scottknott = np.vstack([scottknott_t,scottknott_f])
    np.savetxt('output/data/scottknott.txt',
               np.vstack([np.arange(len(params.algorithms)),scottknott]).T,
               fmt='%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f')


    if TIKIZ_TABLE:
        # Export line in table
        print("epsilon_t:")
        for problem in params.problems:
            for n_var in params.n_vars:
                epsilon_t = epsilon_ts[problem + str(n_var)]
                min_value = np.min(np.mean(epsilon_t, axis=0))
                min_id = np.argmin(np.mean(epsilon_t, axis=0))
                output_str = "%s\t%d" % (problem, n_var)
                for i, algorithm in enumerate(params.algorithms):
                    data_str = ('%.3E' % np.mean(epsilon_t[:, i])).replace('E+0', 'E+').replace('E-0', 'E-') + \
                               '(%s)' % (('%.2E' % np.std(epsilon_t[:, i])).replace('E+0', 'E+').replace('E-0', 'E-'))
                    if i == min_id:
                        data_str = "\\bb{%s}" % data_str
                    else:
                        _, p = wilcoxon(epsilon_t[:, i], epsilon_t[:, min_id])
                        if p < 0.05:
                            data_str += '$^\\dag$'
                    output_str += '\t%s' % data_str
                print(output_str)

        print("epsilon_f:")
        for problem in params.problems:
            for n_var in params.n_vars:
                epsilon_f = epsilon_fs[problem + str(n_var)]
                min_value = np.min(np.mean(epsilon_f, axis=0))
                min_id = np.argmin(np.mean(epsilon_f, axis=0))
                output_str = "%s\t%d" % (problem, n_var)
                for i, algorithm in enumerate(params.algorithms):
                    data_str = ('%.3E' % np.mean(epsilon_f[:, i])).replace('E+0', 'E+').replace('E-0', 'E-') + \
                               '(%s)' % (('%.2E' % np.std(epsilon_f[:, i])).replace('E+0', 'E+').replace('E-0', 'E-'))
                    if i == min_id:
                        data_str = "\\bb{%s}" % data_str
                    else:
                        _, p = wilcoxon(epsilon_f[:, i], epsilon_f[:, min_id])
                        if p < 0.05:
                            data_str += '$^\\dag$'
                    output_str += '\t%s' % data_str
                print(output_str)


