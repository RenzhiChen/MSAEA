import numpy as np
from GPyOpt.acquisitions.base import AcquisitionBase
from functools import partial


class multimodelAcquisitionLCB(AcquisitionBase):
    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=2):
        self.optimizer = optimizer
        super(multimodelAcquisitionLCB, self).__init__(model, space, optimizer)
        self.exploration_weight = exploration_weight

        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')

    def _compute_acq_s(self, x, i):
        """
        Computes the GP-Lower Confidence Bound
        """
        m, s = self.model.models[i].predict(x)
        return -m + self.exploration_weight * s

    def _compute_acq_withGradients_s(self, x, i):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        """
        m, s, dmdx, dsdx = self.model.models[i].predict_withGradients(x)
        f_acqu = -m + self.exploration_weight * s
        df_acqu = -dmdx + self.exploration_weight * dsdx
        return f_acqu, df_acqu

    def optimize(self, duplicate_manager=None):
        """
        Optimizes the acquisition function (uses a flag from the model to use gradients or not).
        """
        out = []
        for i in range(self.model.n_cluster):
            if not self.analytical_gradient_acq:
                self._compute_acq = partial(self._compute_acq_s, i=i)
                self._compute_acq_withGradients = partial(self._compute_acq_withGradients_s, i=i)
                out.append(self.optimizer.optimize(f=self.acquisition_function, duplicate_manager=duplicate_manager))
            else:
                out.append(self.optimizer.optimize(f=self.acquisition_function,
                                                   f_df=self.acquisition_function_withGradients,
                                                   duplicate_manager=duplicate_manager))
        return out



class multioutputAcquisitionLCB(AcquisitionBase):
    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=2):
        self.optimizer = optimizer
        super(multioutputAcquisitionLCB, self).__init__(model, space, optimizer)
        self.exploration_weight = exploration_weight

        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')

    def _compute_acq_s(self, x, i):
        """
        Computes the GP-Lower Confidence Bound
        """
        x_with_task = np.hstack([x,np.ones([len(x),1])*i])
        m, s = self.model.predict(x_with_task)
        return -m + self.exploration_weight * s

    def _compute_acq_withGradients_s(self, x, i):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        """
        x_with_task = np.hstack([x, np.ones([len(x), 1]) * i])
        m, s, dmdx, dsdx = self.model.predict_withGradients(x_with_task)
        f_acqu = -m + self.exploration_weight * s
        df_acqu = -dmdx + self.exploration_weight * dsdx
        return f_acqu, df_acqu

    def optimize(self, duplicate_manager=None):
        """
        Optimizes the acquisition function (uses a flag from the model to use gradients or not).
        """
        out = []
        for i in range(self.model.n_cluster):
            if not self.analytical_gradient_acq:
                self._compute_acq = partial(self._compute_acq_s, i=i)
                self._compute_acq_withGradients = partial(self._compute_acq_withGradients_s, i=i)
                out.append(self.optimizer.optimize(f=self.acquisition_function, duplicate_manager=duplicate_manager))
            else:
                out.append(self.optimizer.optimize(f=self.acquisition_function,
                                                   f_df=self.acquisition_function_withGradients,
                                                   duplicate_manager=duplicate_manager))
        return out