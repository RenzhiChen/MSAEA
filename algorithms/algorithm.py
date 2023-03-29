from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class Algorithm(ABC):
    def __init__(self, benchmark, n_init, n_iter, n_batch):
        self.benchmark = benchmark
        self.bound = benchmark.bound
        self.n_var = benchmark.n_var
        self.n_init = (self.n_var * 11 - 1) if n_init is None else n_init
        self.n_batch = min(max(self.n_var, 2), 5) if n_batch is None else n_batch
        self.n_iter = (self.n_var * 11 - 1) if n_iter is None else n_iter

    @abstractmethod
    def run(self):
        pass

