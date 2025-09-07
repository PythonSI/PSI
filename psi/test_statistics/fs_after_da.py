import numpy as np
import numpy.typing as npt
from typing import Tuple, List
from scipy.linalg import block_diag


class SFS_DATestStatistic:
    r"""Test statistic for feature selection inference after domain adaptation.

    This class computes test statistics for testing individual features
    after feature selection on domain-adapted data, implementing the 
    post-selection inference framework for cross-domain feature validation.

    The test statistic is designed for testing:

    .. math::
        H_0: \beta_j = 0 \quad \text{vs} \quad H_1: \beta_j \neq 0

    for a specific feature :math:`j` in the active set, where :math:`\beta_j`
    is the coefficient of feature :math:`j` in the target domain after
    domain adaptation via optimal transport.

    Parameters
    ----------
    xs : array-like, shape (ns, p)
        Source domain design matrix
    ys : array-like, shape (ns, 1)
        Source domain response vector
    xt : array-like, shape (nt, p)
        Target domain design matrix
    yt : array-like, shape (nt, 1)
        Target domain response vector

    Attributes
    ----------
    xs_node : Data
        Node containing the source domain design matrix
    ys_node : Data
        Node containing the source domain response vector
    xt_node : Data
        Node containing the target domain design matrix
    yt_node : Data
        Node containing the target domain response vector

    Notes
    -----
    The test statistic accounts for the domain adaptation step by focusing
    the inference on the target domain data while using the source domain
    for adaptation. This allows for valid inference on features selected
    after optimal transport domain adaptation.
    """

    def __init__(self, xs: npt.NDArray[np.floating], ys: npt.NDArray[np.floating], xt: npt.NDArray[np.floating], yt: npt.NDArray[np.floating]):
        self.xs_node = xs
        self.ys_node = ys
        self.xt_node = xt
        self.yt_node = yt

    def __call__(
        self,
        active_set: npt.NDArray[np.floating],
        feature_id: int,
        Sigmas: List[npt.NDArray[np.floating]]
    ) -> Tuple[list, npt.NDArray[np.floating], npt.NDArray[np.floating], float, float]:
        r"""Compute test statistic for a selected feature after domain adaptation.

        Computes the test statistic and parametrization for testing whether
        a specific feature in the active set has a non-zero coefficient in
        the target domain after domain adaptation.

        The test statistic focuses on the target domain:

        .. math::
            T = \eta_j^T \begin{bmatrix} \mathbf{y}_s \\ \mathbf{y}_t \end{bmatrix}

        where 

        .. math::
            \eta_j = \begin{bmatrix} \mathbf{0}_{ns} \\ \mathbf{x}_t^A(\mathbf{x}_t^{A^T}\mathbf{x}_t^A)^{-1}\mathbf{e}_j \end{bmatrix},

        :math:`n_s` is the number of source domain samples and :math:`\mathbf{x}_t^A` is the target domain active set design matrix.

        Parameters
        ----------
        active_set : array-like, shape (k,)
            Indices of features in the active set
        feature_id : int
            Index of the feature to test (within active_set)
        Sigmas : list of array-like
            List containing [Sigma_source, Sigma_target] covariance matrices

        Returns
        -------
        test_statistic_direction : array-like, shape (ns+nt, 1)
            Direction vector for the test statistic
        a : array-like, shape (ns+nt, 1)
            Parametrization intercept for combined data
        b : array-like, shape (ns+nt, 1)
            Parametrization coefficient for combined data
        test_statistic : float
            Observed value of the test statistic
        variance : float
            Variance of the test statistic under null hypothesis
        deviation : float
            Standard deviation of the test statistic
        """
        xs = self.xs_node()
        ys = self.ys_node()
        xt = self.xt_node()
        yt = self.yt_node()

        x = np.vstack((xs, xt))
        y = np.vstack((ys, yt))

        Sigma_s = Sigmas[0]
        Sigma_t = Sigmas[1]
        Sigma = block_diag(Sigma_s, Sigma_t)

        x_active = xt[:, active_set]
        ej = np.zeros((len(active_set), 1))
        ej[feature_id, 0] = 1
        test_statistic_direction = np.vstack((np.zeros((xs.shape[0], 1)), x_active.dot(
            np.linalg.inv(x_active.T.dot(x_active))).dot(ej)))

        b = Sigma.dot(test_statistic_direction).dot(np.linalg.inv(
            test_statistic_direction.T.dot(Sigma).dot(test_statistic_direction)))
        a = (np.identity(x_active.shape[0] + xs.shape[0]
                         ) - b.dot(test_statistic_direction.T)).dot(y)

        test_statistic = test_statistic_direction.T.dot(y)[0, 0]
        variance = test_statistic_direction.T.dot(
            Sigma).dot(test_statistic_direction)[0, 0]
        deviation = np.sqrt(variance)

        self.xs_node.parametrize(data=xs)
        self.ys_node.parametrize(a=a[:xs.shape[0], :], b=b[:xs.shape[0], :])
        self.xt_node.parametrize(data=xt)
        self.yt_node.parametrize(a=a[xs.shape[0]:, :], b=b[xs.shape[0]:, :])
        return test_statistic_direction, a, b, test_statistic, variance, deviation
