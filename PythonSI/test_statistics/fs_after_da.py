import numpy as np
import numpy.typing as npt
from typing import Tuple, List
from scipy.linalg import block_diag

class SFS_DATestStatistic:
    def __init__(self, Xs: npt.NDArray[np.floating], ys: npt.NDArray[np.floating], Xt: npt.NDArray[np.floating], yt: npt.NDArray[np.floating]):
        self.Xs_node = Xs
        self.ys_node = ys
        self.Xt_node = Xt
        self.yt_node = yt
    def __call__(
        self,
        active_set: npt.NDArray[np.floating],
        feature_id: int,
        Sigmas: List[npt.NDArray[np.floating]]
    ) -> Tuple[list, npt.NDArray[np.floating], npt.NDArray[np.floating], float, float]:
        """
        Compute a simple feature selection statistic for a given feature.
        This is a placeholder function and should be replaced with a proper statistical test.
        """
        Xs = self.Xs_node()
        ys = self.ys_node()
        Xt = self.Xt_node()
        yt = self.yt_node()
        
        X = np.vstack((Xs, Xt))
        y = np.vstack((ys, yt))
        
        Sigma_s = Sigmas[0]
        Sigma_t = Sigmas[1]
        Sigma = block_diag(Sigma_s, Sigma_t)

        X_active = Xt[:, active_set]
        ej = np.zeros((len(active_set), 1))
        ej[feature_id, 0] = 1
        test_statistic_direction = np.vstack((np.zeros((Xs.shape[0], 1)), X_active.dot(np.linalg.inv(X_active.T.dot(X_active))).dot(ej)))
        
        b = Sigma.dot(test_statistic_direction).dot(np.linalg.inv(test_statistic_direction.T.dot(Sigma).dot(test_statistic_direction)))
        a = (np.identity(X_active.shape[0] + Xs.shape[0]) - b.dot(test_statistic_direction.T)).dot(y)
        
        test_statistic = test_statistic_direction.T.dot(y)[0, 0]
        variance = test_statistic_direction.T.dot(Sigma).dot(test_statistic_direction)[0, 0]
        deviation = np.sqrt(variance)
        
        self.Xs_node.parametrize(data=Xs)
        self.ys_node.parametrize(a=a[:Xs.shape[0], :], b=b[:Xs.shape[0], :])
        self.Xt_node.parametrize(data=Xt)
        self.yt_node.parametrize(a=a[Xs.shape[0]:, :], b=b[Xs.shape[0]:, :])
        return test_statistic_direction, a, b, test_statistic, variance, deviation

