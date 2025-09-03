import numpy as np
import numpy.typing as npt
from typing import Tuple

class FSTestStatistic:
    def __init__(self, X: npt.NDArray[np.floating], y: npt.NDArray[np.floating]):
        self.X_node = X
        self.y_node = y
        
    def __call__(
        self,
        active_set: npt.NDArray[np.floating],
        feature_id: int,
        Sigma: npt.NDArray[np.floating]
    ) -> Tuple[list, npt.NDArray[np.floating], npt.NDArray[np.floating], float, float]:
        """
        Compute a simple feature selection statistic for a given feature.
        This is a placeholder function and should be replaced with a proper statistical test.
        """
        X = self.X_node()
        y = self.y_node()
        
        X_active = X[:, active_set]
        ej = np.zeros((X_active.shape[1], 1))
        ej[feature_id, 0] = 1
        test_statistic_direction = X_active.dot(np.linalg.inv(X_active.T.dot(X_active))).dot(ej)
        
        b = Sigma.dot(test_statistic_direction).dot(np.linalg.inv(test_statistic_direction.T.dot(Sigma).dot(test_statistic_direction)))
        a = (np.identity(X_active.shape[0]) - b.dot(test_statistic_direction.T)).dot(y)
        
        test_statistic = test_statistic_direction.T.dot(y)[0, 0]
        variance = test_statistic_direction.T.dot(Sigma).dot(test_statistic_direction)[0, 0]
        deviation = np.sqrt(variance)
        
        self.X_node.parametrize(data=X)
        self.y_node.parametrize(a=a, b=b)
        return test_statistic_direction, a, b, test_statistic, variance, deviation

