import numpy as np
from GPyOpt.core.evaluators import EvaluatorBase


class mRandom(EvaluatorBase):
    def compute_batch(self, duplicate_manager=None, context_manager=None):
        m_out = self.acquisition.optimize()
        x_batch = []

        if self.batch_size < len(m_out):
            idxs = np.random.permutation(np.arange(0,len(m_out)))[:self.batch_size]
            for i in idxs:
                x_batch.append(m_out[i][0])
        else:
            for out in m_out:
                x_batch.append(out[0])

        x_batch = np.unique(np.vstack(x_batch),axis=0)
        #if len(x_batch) < self.batch_size:
        #    print("Warning: duplicated solution in batch.")
        return x_batch
