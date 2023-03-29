from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class MultiModalProblem(ABC):
    def __init__(self, n_var):
        self.n_var = n_var
        self.bound = [-1,1]

    @abstractmethod
    def evaluate(self, x):
        pass


class Rastrigin(MultiModalProblem):
    def __init__(self, n_var):
        super().__init__(n_var)
        self.bound = [-5.12,5.12]

    def evaluate(self, x):
        """
        :param x: n by d matrix
        :return: result n vector
        """
        a = 10
        out = np.expand_dims(a * self.n_var + np.sum(np.power(x, 2) - a * np.cos(2 * np.pi * x), axis=1),-1)
        return out, None


class Ackley(MultiModalProblem):
    def __init__(self, n_var):
        super().__init__(n_var)
        self.bound = [-5,5]

    def evaluate(self, x):
        _, n_var = x.shape
        sum_sq_term = -20 * np.exp(-0.2 * np.sqrt(np.sum(np.power(x, 2), axis=1)/n_var))
        cos_term = -np.exp(np.sum(np.cos(2 * np.pi * x), axis=1)/n_var)
        out = np.expand_dims(sum_sq_term+cos_term+20+np.exp(1),-1)
        return out, None


class Schwefel(MultiModalProblem):
    def __init__(self, n_var):
        super().__init__(n_var)
        self.bound = [-500,500]

    def evaluate(self, x):
        _, n_var = x.shape
        out = 418.9829 * n_var - np.sum(x*np.sin(np.sqrt(np.abs(x))), axis=1)
        return np.expand_dims(out,-1), None


class Griewank(MultiModalProblem):
    def __init__(self, n_var):
        super().__init__(n_var)
        self.bound = [-10,10]

    def evaluate(self, x):
        _, n_var = x.shape
        out = -1
        for i in range(n_var):
            out = out * np.cos(x[:,i]/np.sqrt(i+1))
        out += np.sum(x**2/4000,axis=1) + 1
        return np.expand_dims(out,-1), None



if __name__ == "__main__":
    print("DEBUG for Problem")
    n_sample = 50
    n_var = 1

    problem = Schwefel(n_var)
    xmin,xmax = problem.bound
    if n_var == 1:
        x0 = np.array([[0]])
        print(problem.evaluate(x0))
        x = np.linspace(xmin,xmax,n_sample*n_sample)
        x = np.expand_dims(x,-1)
        y, _ = problem.evaluate(x)
        plt.plot(x,y)

    if n_var == 2:
        x0 = np.array([[0,0]])
        print(problem.evaluate(x0))
        x_grid, _ = np.meshgrid(np.linspace(xmin, xmax, 20), np.linspace(xmin, xmax, 20))
        x = np.vstack([x_grid.flatten(), x_grid.T.flatten()]).T
        y, _ = problem.evaluate(x)
        y_grid = np.reshape(y, x_grid.shape)

        plt.contour(x_grid, x_grid.T, y_grid)
    plt.show()
