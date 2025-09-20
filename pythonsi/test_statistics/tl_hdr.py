import numpy as np
import numpy.typing as npt
from typing import Tuple, List
from scipy.linalg import block_diag


class TLHDRTestStatistic:
    r"""
    Test statistic for selection inference in high-dimensional regression
    after transfer learning with multiple source domains.

    This class computes test statistics for testing individual features
    after feature selection via a transfer learning procedure, implementing
    the post-selection inference framework for high-dimensional regression.

    The test statistic is designed for testing:

    .. math::
        H_0: \beta_j = 0 \quad \text{vs} \quad H_1: \beta_j \neq 0,

    where :math:`\beta_j` is the coefficient of feature :math:`j` in the
    target domain after transfer learning and feature selection.

    Parameters
    ----------
    XS_list : array-like, shape (K, nS, p)
        A 3D numpy array containing **source domain design matrices**.
        - ``K``: number of source domains
        - ``nS``: sample size per source domain
        - ``p``: number of features (shared across domains)

        The array is structured such that ``XS_list[k]`` corresponds to
        the design matrix of the :math:`k`-th source domain, with shape
        ``(nS, p)``.

    YS_list : array-like, shape (K * nS, 1)
        A 2D numpy array containing the **source domain response vectors**
        stacked vertically across all ``K`` source domains.
        - The first ``nS`` rows correspond to the first source domain,
          the next ``nS`` to the second, and so on.

    X0 : array-like, shape (nT, p)
        Target domain design matrix.
        - ``nT``: number of samples in the target domain
        - ``p``: number of features (same as in source domains)

    Y0 : array-like, shape (nT, 1)
        Target domain response vector.

    Attributes
    ----------
    XS_list_node : Data
        Node containing the collection of source domain design matrices.
    YS_list_node : Data
        Node containing the stacked source domain response vector.
    X0_node : Data
        Node containing the target domain design matrix.
    Y0_node : Data
        Node containing the target domain response vector.

    Notes
    -----
    The test statistic accounts for the transfer learning step by focusing
    the inference on the target domain while leveraging information from
    multiple source domains. This allows for valid inference on features selected after the transfer
    learning process.
    """

    def __init__(
        self,
        XS_list: npt.NDArray[np.floating],
        YS_list: npt.NDArray[np.floating],
        X0: npt.NDArray[np.floating],
        Y0: npt.NDArray[np.floating],
    ):
        self.XS_list_node = XS_list
        self.YS_list_node = YS_list
        self.X0_node = X0
        self.Y0_node = Y0

    def __call__(
        self,
        active_set: npt.NDArray[np.int64],
        feature_id: int,
        Sigmas: List[npt.NDArray[np.floating]],
    ) -> Tuple[list, npt.NDArray[np.floating], npt.NDArray[np.floating], float, float]:
        r"""Compute test statistic for a selected feature after transfer learning.

        Computes the test statistic and parametrization for testing
        whether a specific feature in the active set has a non-zero
        coefficient in the target domain after feature selection via transfer learning.

        The test statistic focuses on the target domain:

        .. math::
            \tau_j = \bm{\eta}_j^\top
            \begin{pmatrix} \bm{Y}^S \\ \bm{Y}^{(0)} \end{pmatrix},

        where:

        - :math:`\bm{Y}^S` is the stacked response vector from all
        source domains,
        - :math:`\bm{Y}^{(0)}` is the target domain response vector,
        - :math:`\bm{\eta}_j` is the direction vector associated with testing the
        coefficient of feature :math:`j` in the target domain active set.

        The direction vector is constructed as:

        .. math::
            \bm{\eta}_j = \begin{pmatrix}
            \bm{0}_{K n_S} \\
            X^{(0)}_\mathcal{M} \left({X^{(0)}_\mathcal{M}}^\top X^{(0)}_\mathcal{M}\right)^{-1} \bm{e}_j
            \end{pmatrix},

        where:
        - :math:`K` is the number of source domains,
        - :math:`n_S` is the number of samples per source domain,
        - :math:`X^{(0)}_\mathcal{M}` is the target domain design matrix restricted to the active set,
        - :math:`\bm{e}_j` is the canonical basis vector selecting feature :math:`j` within the active set.

        Parameters
        ----------
        active_set : array-like, shape (k,)
            Indices of features in the active set
        feature_id : int
            Index of the feature to test (within active_set)
        Sigmas : list of array-like
            List containing covariance matrices for both source and target
            domains:
            - ``Sigmas[0]``: list of source covariance matrices,
            - ``Sigmas[1]``: target domain covariance matrix.

        Returns
        -------
        test_statistic_direction : array-like, shape (K * nS + nT, 1)
            The direction vector :math:`\bm{\eta}_j` for the test statistic.
        a : array-like, shape (K * nS + nT, 1)
            Parametrization intercept for combined data
        b : array-like, shape (K * nS + nT, 1)
            Parametrization coefficient for combined data
        test_statistic : float
            Observed value of the test statistic
        variance : float
            Variance of the test statistic under null hypothesis

        deviation : float
            Standard deviation of the test statistic
        """
        XS_list = self.XS_list_node()
        YS_list = self.YS_list_node()
        X0 = self.X0_node()
        Y0 = self.Y0_node()

        K = len(XS_list)
        nS = XS_list[0].shape[0]
        n_sources = K * nS
        nT = Y0.shape[0]
        N = nS * K + nT
        Y = np.concatenate((YS_list, Y0))

        SigmaS_list = Sigmas[0]
        Sigma0 = Sigmas[1]
        Sigma = block_diag(*SigmaS_list, Sigma0)

        X0_active = X0[:, active_set]
        ej = np.zeros((len(active_set), 1))
        ej[feature_id, 0] = 1

        inv = np.linalg.pinv(X0_active.T @ X0_active)
        X0_active_inv = X0_active @ inv

        test_statistic_direction = np.vstack(
            (
                np.zeros((n_sources, 1)),
                X0_active_inv @ ej,
            )
        )
        e1 = test_statistic_direction.T @ Sigma @ test_statistic_direction
        b = (Sigma @ test_statistic_direction) / e1

        e2 = np.eye(N) - b @ test_statistic_direction.T
        a = e2 @ Y

        test_statistic = (test_statistic_direction.T @ Y)[0, 0]
        variance = (test_statistic_direction.T @ Sigma @ test_statistic_direction)[0, 0]
        deviation = np.sqrt(variance)

        self.XS_list_node.parametrize(data=XS_list)
        self.YS_list_node.parametrize(a=a[:n_sources, :], b=b[:n_sources, :])
        self.X0_node.parametrize(data=X0)
        self.Y0_node.parametrize(a=a[n_sources:, :], b=b[n_sources:, :])

        return test_statistic_direction, a, b, test_statistic, variance, deviation
