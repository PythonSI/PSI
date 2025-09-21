import numpy as np
import numpy.typing as npt
from typing import Tuple


class FSTestStatistic:
    r"""Compute test statistic and other utilities for feature selection inference.

    This class computes test statistics for testing individual features
    after feature selection, implementing the post-selection inference
    framework for validating selected features.

    The test statistic is designed for testing:

    .. math::
        H_0: \beta_j = 0 \quad \text{vs} \quad H_1: \beta_j \neq 0

    for a specific feature :math:`j` in the active set, where :math:`\beta_j`
    is the coefficient of feature :math:`j` in the linear model.

    Parameters
    ----------
    x : array-like, shape (n, p)
        Design matrix containing all features
    y : array-like, shape (n, 1)
        Response vector

    Attributes
    ----------
    x_node : Data
        Node containing the design matrix
    y_node : Data
        Node containing the response vector
    """

    def __init__(self, x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]):
        self.x_node = x
        self.y_node = y

    def __call__(
        self,
        active_set: npt.NDArray[np.floating],
        feature_id: int,
        Sigma: npt.NDArray[np.floating],
    ) -> Tuple[
        list, npt.NDArray[np.floating], npt.NDArray[np.floating], float, float, list
    ]:
        r"""Compute test statistic for a selected feature.

        Computes the test statistic and parametrization for testing whether
        a specific feature in the active set has a non-zero coefficient.

        The test statistic follows the form:

        .. math::
            T = \eta_j^T \mathbf{y}

        where :math:`\eta_j^ = x` is the direction of the test statistic:

        .. math::
            \eta_j = x_{A}(x_{A}^T x_{A})^{-1} \bm{e}_j

        Parameters
        ----------
        active_set : array-like, shape (k,)
            Indices of features in the active set
        feature_id : int
            Index of the feature to test (within active_set)
        Sigma : array-like, shape (n, n)
            Covariance matrix of the noise

        Returns
        -------
        test_statistic_direction : array-like, shape (n, 1)
            Direction vector for the test statistic
        a : array-like, shape (n, 1)
            Parametrized intercept
        b : array-like, shape (n, 1)
            Parametrized coefficient
        test_statistic : float
            Observed value of the test statistic
        variance : float
            Variance of the test statistic under null hypothesis
        deviation : float
            Standard deviation of the test statistic
        """
        x = self.x_node()
        y = self.y_node()

        x_active = x[:, active_set]
        ej = np.zeros((x_active.shape[1], 1))
        ej[feature_id, 0] = 1
        test_statistic_direction = x_active.dot(
            np.linalg.inv(x_active.T.dot(x_active))
        ).dot(ej)

        b = Sigma.dot(test_statistic_direction).dot(
            np.linalg.inv(
                test_statistic_direction.T.dot(Sigma).dot(test_statistic_direction)
            )
        )
        a = (np.identity(x_active.shape[0]) - b.dot(test_statistic_direction.T)).dot(y)

        test_statistic = test_statistic_direction.T.dot(y)[0, 0]
        variance = test_statistic_direction.T.dot(Sigma).dot(test_statistic_direction)[
            0, 0
        ]
        deviation = np.sqrt(variance)

        self.x_node.parametrize(data=x)
        self.y_node.parametrize(a=a, b=b)
        return (
            test_statistic_direction,
            a,
            b,
            test_statistic,
            variance,
            deviation,
            [-np.inf, np.inf],
        )
