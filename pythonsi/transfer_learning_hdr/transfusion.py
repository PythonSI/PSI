import numpy as np
import numpy.typing as npt
from pythonsi.node import Data
from typing import List
from pythonsi.util import solve_linear_inequalities, intersect
from skglm import WeightedLasso, Lasso
from numpy.linalg import pinv


def construct_X(XS_list, X0, p, K):
    blocks = []
    for k in range(K):
        Xk = XS_list[k]
        row_k = np.hstack(
            [Xk if i == k else np.zeros((Xk.shape[0], p)) for i in range(K)] + [Xk]
        )
        blocks.append(row_k)
    row_0 = np.hstack([np.zeros((X0.shape[0], K * p)), X0])
    blocks.append(row_0)
    return np.vstack(blocks)


def construct_B(K, p, nS, nT):
    B = np.hstack([np.tile(nS * np.eye(p), K), (nS * K + nT) * np.eye(p)])
    return B


def construct_Q(nT, N):
    Q = np.zeros((nT, N))
    Q[:, N - nT :] = np.eye(nT)
    return Q


# Construct active set
def construct_active_set(coef_hat, X):
    p = X.shape[1]
    active_set = np.nonzero(coef_hat)[0]
    inactive_set = np.setdiff1d(np.arange(p), active_set)
    coef_active = coef_hat[active_set].reshape(-1, 1)
    sign_active = np.sign(coef_active).reshape(-1, 1)

    X_active = X[:, active_set] if active_set.size > 0 else np.zeros((X.shape[0], 0))
    X_inactive = (
        X[:, inactive_set] if inactive_set.size > 0 else np.zeros((X.shape[0], 0))
    )

    return coef_active, sign_active, active_set, X_active, inactive_set, X_inactive


class TLTransFusion:
    r"""Post-Transfer Learning Statistical Inference with TransFusion.

    This class implements the PTL-SI framework for high-dimensional
    regression with multiple source domains and one target domain. It
    performs feature selection via the TransFusion procedure and supports
    valid post-selection inference by modeling the selection event.

    The TransFusion algorithm proceeds in two steps:
    
    Step 1. Co-Training. In this step, both source and target data are used jointly to estimate
    intermediate regression coefficients for each domain:
    - :math:`\bm{\beta}^{(k)}` for each source domain, where :math:`k = 1, \dots, K`,
    - :math:`\bm{\beta}^{(0)}` for the target domain.

    .. math::
        \hat{\bm\beta} = \operatorname*{argmin}_{\bm\beta \in \mathbb{R}^{(K+1)p}} 
        \left\{ 
        \frac{1}{2N} \sum_{k=0}^{K} \| \bm{Y}^{(k)} - {X}^{(k)} \bm\beta^{(k)} \|_2^2 
        + \lambda_0 \left( \| \bm\beta^{(0)} \|_1 + \sum_{k=1}^{K} a_k \| \bm\beta^{(k)} - \bm\beta^{(0)} \|_1 \right) 
        \right\},

    where:
    - :math:`K` is the number of source domains,
    - :math:`n_S` is the number of samples per source domain,
    - :math:`n_T` is the number of samples in the target domain,
    - :math:`N = K n_S + n_T` is the total number of samples,
    - :math:`\lambda_0` is the regularization parameter,
    - :math:`a_k` is the non-negative weight for source domain :math:`k`.

    This is equivalent to solving a **weighted LASSO** problem:
    .. math::
        \hat{\bm{\theta}} = \underset{\bm{\theta} \in \mathbb{R}^{{(K+1)p}}}{\operatorname{argmin}} \left\{ \frac{1}{2N}\| \bm Y - {X}\bm{\theta}\|^2_2 + \lambda_0 
        \sum_{k=0}^Ka_{k}\|\bm{\theta}^{(k)}\|_1  \right\}

    where:
    .. math::
    a_0 = 1,
    \mathbf{Y} =
    \left(
        {\mathbf{Y}^{(1)}}^\top,
        {\mathbf{Y}^{(2)}}^\top,
        \ldots,
        {\mathbf{Y}^{(K)}}^\top,
        {\mathbf{Y}^{(0)}}^\top
    \right)^\top,

    .. math::
    X =
    \begin{pmatrix}
        X^{(1)} & 0 & \cdots & 0 & X^{(1)} \\
        0 & X^{(2)} & \cdots & 0 & X^{(2)} \\
        \vdots & \vdots & \ddots & \vdots & \vdots \\
        0 & 0 & \cdots & X^{(K)} & X^{(K)} \\
        0 & 0 & \cdots & 0 & X^{(0)}
    \end{pmatrix},
    \quad
    \bm{\theta} =
    \begin{pmatrix}
        \bm{\beta}^{(1)} - \bm{\beta}^{(0)} \\
        \bm{\beta}^{(2)} - \bm{\beta}^{(0)} \\
        \vdots \\
        \bm{\beta}^{(K)} - \bm{\beta}^{(0)} \\
        \bm{\beta}^{(0)}
    \end{pmatrix}.

    After :math:`\hat{\bm\beta}` is obtained, the algorithm computes the estimator:
    .. math::
        \hat{\bm{w}} = \frac{n_S}{N} \sum_{k=1}^K \hat{\bm{\beta}}^{(k)} + \frac{n_T}{N} \hat{\bm{\beta}}^{(0)}
    

    Step 2. Local Debias. This step aims to refine the initial estimator :math:`\hat{\bm w}` and compute the final estimated coefficients for the target task:
    .. math::
        \hat{\bm{\delta}} = \underset{\bm{\delta} \in \mathbb{R}^p}{\operatorname{argmin}} \left\{ \frac{1}{2n_T} \left\| \bm Y^{(0)} - {X}^{(0)} \hat{\bm{w}} - {X}^{(0)} \bm{\delta} \right\|_2^2 + \tilde{\lambda} \|\bm{\delta}\|_1\right\
        \hat{\bm{\beta}}^{(0)}_{\text{TransFusion}} = \hat{\bm{w}} + \hat{\bm{\delta}}
    where
    - :math:`\tilde{\lambda}` is the regularization parameter.

    Parameters
    ----------
    lambda_0 : float
        Regularization parameter for coupling target and source domain models in Step 1.
    lambda_tilde : float
        Lasso regularization parameter in Step 2.
    ak_weights : list of float
        Source domain weights :math:`ak` used in Step 1 to balance each
        domain's contribution.

    Attributes
    ----------
    XS_list_node : Data or None
        Source domain design matrices, shape ``(K, nS, p)``.
    YS_list_node : Data or None
        Stacked source domain response vectors, shape ``(K*nS, 1)``.
    X0_node : Data or None
        Target domain design matrix, shape ``(nT, p)``.
    Y0_node : Data or None
        Target domain response vector, shape ``(nT, 1)``.
    active_set_node : Data
        Output node containing selected feature indices
    interval : list or None
        Feasible interval from last inference call
    active_set_data : array-like or None
        Active set from last inference cal

    """

    def __init__(self, lambda_0: float, lambda_tilde: float, ak_weights: List[float]):
        self.lambda_0 = lambda_0
        self.lambda_tilde = lambda_tilde
        self.ak_weights = ak_weights

        self.XS_list_node = None
        self.YS_list_node = None
        self.X0_node = None
        self.Y0_node = None

        self.active_set_node = Data(self)
        self.interval = None
        self.active_set = None

    def run(
        self,
        XS_list: npt.NDArray[np.floating],
        YS_list: npt.NDArray[np.floating],
        X0: npt.NDArray[np.floating],
        Y0: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.int64]:
        r"""Configure TransFusion with input data and return active set node.

        Parameters
        ----------
        XS_list : array-like, shape (K, nS, p)
            Source domain design matrices.
        YS_list : array-like, shape (K*nS, 1)
            Stacked source domain response vectors.
        X0 : array-like, shape (nT, p)
            Target domain design matrix
        Y0 : array-like, shape (nT, 1)
            Target domain response vector.

        Returns
        -------
        active_set_node : Data
            Node containing indices of features selected in the target domain.
        """
        self.XS_list_node = XS_list
        self.YS_list_node = YS_list
        self.X0_node = X0
        self.Y0_node = Y0

        return self.active_set_node

    def forward(
        self,
        XS_list: npt.NDArray[np.floating],
        YS_list: npt.NDArray[np.floating],
        X0: npt.NDArray[np.floating],
        Y0: npt.NDArray[np.floating],
    ):
        r"""Run the TransFusion algorithm and extract active set information.

        Executes the TransFusion and returns the active set and intermediate quantities needed for
        selective inference.

        Parameters
        ----------
        XS_list : array-like, shape (K, nS, p)
            Source domain design matrices.
        YS_list : array-like, shape (K*nS, 1)
            Stacked source domain response vectors.
        X0 : array-like, shape (nT, p)
            Target domain design matrix.
        Y0 : array-like, shape (nT, 1)
            Target domain response vector.

        Returns
        -------
        active_set : array-like, shape (k,)
            Indices of selected features in the target domain.
        X : array-like, shape (N, (K+1)*p)
            Augmented design matrix combining source and target data.
        B : array-like, shape (p, (K+1)*p)
            Matrix used to express  :math:`\hat{w}` as a linear function of :math:`\theta` in Step 1:
            .. math::
            \hat\bm{w} = \frac{1}{N} B \hat\bm\theta,
        K : int
            Number of source domains.
        Y : array-like, shape (N, 1)
            Combined response vector from source and target domains.
        nS : int
            Number of samples per source domain.
        nT : int
            Number of samples in the target domain.
        N : int
            Total number of samples across all domains, ``N = K * nS + nT``.
        p : int
            Number of features.
        theta_hat : array-like, shape ((K+1)*p,)
            Estimated stacked coefficient vector :math:`\hat\bm{\theta}` from Step 1
        w_hat : array-like, shape (p,)
            Pooled estimator from Step 1.
        delta_hat : array-like, shape (p,)
            Estimated target-specific adjustment vector from Step 2
        beta0_hat : array-like, shape (p,)
            Final target domain coefficient estimator.
        """

        K = len(XS_list)
        nS = XS_list[0].shape[0]
        nT = Y0.shape[0]
        N = nS * K + nT
        p = X0.shape[1]
        X = construct_X(XS_list, X0, p, K)
        Y = np.concatenate((YS_list, Y0))
        B = construct_B(K, p, nS, nT)

        theta_hat, w_hat, delta_hat, beta0_hat = self._TransFusion(
            X,
            Y.ravel(),
            X0,
            Y0.ravel(),
            B,
            N,
            p,
            K,
            self.lambda_0,
            self.lambda_tilde,
            self.ak_weights,
        )
        active_set = np.nonzero(beta0_hat)[0]

        return (
            active_set,
            X,
            B,
            K,
            Y,
            nS,
            nT,
            N,
            p,
            theta_hat,
            w_hat,
            delta_hat,
            beta0_hat,
        )

    def __call__(self):
        r"""Execute the TransFusion algorithm on stored data.

        Returns
        -------
        active_set : array-like, shape (k,)
            Indices of selected target domain features.
        """
        XS_list_node = self.XS_list_node()
        YS_list_node = self.YS_list_node()
        X0_node = self.X0_node()
        Y0_node = self.Y0_node()

        active_set, _, _, _, _, _, _, _, _, _, _, _, _ = self.forward(
            XS_list_node, YS_list_node, X0_node, Y0_node
        )
        self.active_set_node.update(active_set)
        return active_set

    def inference(self, z: float) -> List:
        r"""Find feasible interval of the Transfusion for the parametrized data at z.

        ----------
        z : float
            Inference parameter value.

        Returns
        -------
        final_interval : list
            Feasible interval [lower, upper] for z
        """
        if self.interval is not None and self.interval[0] <= z <= self.interval[1]:
            self.active_set_node.parametrize(data=self.active_set)
            return self.interval

        XS_list, _, _, interval_XS_list = self.XS_list_node.inference(z)
        YS_list, a_YS_list, b_YS_list, interval_YS_list = self.YS_list_node.inference(z)
        X0, _, _, interval_X0 = self.X0_node.inference(z)
        Y0, a_Y0, b_Y0, interval_Y0 = self.Y0_node.inference(z)

        active_set, X, B, K, Y, nS, nT, N, p, theta_hat, w_hat, delta_hat, beta0_hat = (
            self.forward(XS_list, YS_list, X0, Y0)
        )
        a_tilde = np.concatenate(
            [self.ak_weights[k] * np.ones(p) for k in range(K)] + [np.ones(p)]
        ).reshape(-1, 1)
        a = np.vstack((a_YS_list, a_Y0))
        b = np.vstack((b_YS_list, b_Y0))

        Q = construct_Q(nT, N)
        thetaO, SO, O_set, XO, Oc, XOc = construct_active_set(theta_hat, X)
        deltaL, SL, L_set, X0L, Lc, X0Lc = construct_active_set(delta_hat, X0)
        betaM, SM, M_set, _, Mc, _ = construct_active_set(beta0_hat, X0)
        phi_u, iota_u, xi_uv, zeta_uv = self.calculate_phi_iota_xi_zeta(
            X,
            SO,
            O_set,
            XO,
            X0,
            SL,
            L_set,
            X0L,
            p,
            B,
            Q,
            self.lambda_0,
            self.lambda_tilde,
            a_tilde,
            N,
            nT,
        )

        theta_itv = self.interval_theta(
            thetaO, SO, O_set, XO, Oc, XOc, a, b, self.lambda_0, a_tilde, N
        )
        delta_itv = self.interval_delta(
            SL, L_set, X0L, Lc, X0Lc, phi_u, iota_u, a, b, self.lambda_tilde, nT
        )
        beta_itv = self.interval_beta(M_set, SM, Mc, xi_uv, zeta_uv, a, b)

        final_interval = intersect(intersect(theta_itv, delta_itv), beta_itv)
        final_interval = intersect(final_interval, interval_XS_list)
        final_interval = intersect(final_interval, interval_YS_list)
        final_interval = intersect(final_interval, interval_X0)
        final_interval = intersect(final_interval, interval_Y0)

        self.interval = final_interval
        self.active_set_node.parametrize(data=active_set)
        self.active_set = active_set
        return self.interval

    def _TransFusion(
        self, X, Y, X0, Y0, B, N, p, K, lambda_0, lambda_tilde, ak_weights
    ):
        r"""Run the TransFusion algorithm.

        Returns
        -------
        theta_hat : array-like, shape ((K+1)*p,)
            Estimated stacked coefficient vector :math:`\hat\bm{\theta}` from Step 1
        w_hat : array-like, shape (p,)
            Pooled estimator from Step 1.
        delta_hat : array-like, shape (p,)
            Estimated target-specific adjustment vector from Step 2
        beta0_hat : array-like, shape (p,)
            Final target domain coefficient estimator.
        """
        # Co-Training
        w_pen = np.ones((K + 1) * p)
        idx = 0
        for k in range(1, K + 1):
            w_pen[idx : idx + p] *= ak_weights[k - 1]
            idx += p

        co_training = WeightedLasso(
            alpha=lambda_0, fit_intercept=False, tol=1e-10, weights=w_pen
        )
        co_training.fit(X, Y)
        theta_hat = co_training.coef_

        w_hat = (1.0 / N) * (B @ theta_hat)

        # Debias
        debias = Lasso(alpha=lambda_tilde, fit_intercept=False, tol=1e-12)
        debias.fit(X0, Y0 - X0 @ w_hat)
        delta_hat = debias.coef_

        beta0_hat = w_hat + delta_hat

        return theta_hat, w_hat, delta_hat, beta0_hat

    def interval_theta(
        self, thetaO, SO, O_set, XO, Oc, XOc, a, b, lambda_0, a_tilde, N
    ):
        r"""Calculates the feasible interval for :math:`\hat\bm\theta`.

        Returns
        -------
        intervals : list
            The feasible interval.
        """

        a_tilde_O = a_tilde[O_set]
        a_tilde_Oc = a_tilde[Oc]

        psi0 = np.array([])
        gamma0 = np.array([])
        psi1 = np.array([])
        gamma1 = np.array([])

        if O_set.size > 0:
            inv = pinv(XO.T @ XO)
            XO_plus = inv @ XO.T

            # Calculate psi0
            XO_plus_b = XO_plus @ b
            psi0 = (-SO * XO_plus_b).ravel()

            # Calculate gamma0
            XO_plus_a = XO_plus @ a
            gamma0_term_inv = inv @ (a_tilde_O * SO)

            gamma0 = SO * XO_plus_a - N * lambda_0 * SO * gamma0_term_inv
            gamma0 = gamma0.ravel()

        if Oc.size > 0:
            if O_set.size == 0:
                proj = np.eye(N)
                temp2 = 0

            else:
                proj = np.eye(N) - XO @ XO_plus
                XO_O_plus = XO @ inv
                temp2 = (XOc.T @ XO_O_plus) @ (a_tilde_O * SO)
                temp2 = temp2 / a_tilde_Oc

            XOc_O_proj = XOc.T @ proj
            temp1 = (XOc_O_proj / a_tilde_Oc) / (lambda_0 * N)

            # Calculate psi1
            term_b = temp1 @ b
            psi1 = np.concatenate([term_b.ravel(), -term_b.ravel()])

            # Calculate gamma1
            term_a = temp1 @ a
            ones_vec = np.ones_like(term_a)

            gamma1 = np.concatenate(
                [
                    (ones_vec - temp2 - term_a).ravel(),
                    (ones_vec + temp2 + term_a).ravel(),
                ]
            )

        psi = np.concatenate((psi0, psi1))
        gamma = np.concatenate((gamma0, gamma1))
        interval = solve_linear_inequalities(-gamma, psi)

        return interval

    def interval_delta(
        self, SL, L_set, X0L, Lc, X0Lc, phi_u, iota_u, a, b, lambda_tilde, nT
    ):
        r"""Calculates the feasible interval for :math:`\hat\bm\delta`.

        Returns
        -------
        intervals : list
            The feasible interval.
        """

        nu0 = np.array([])
        kappa0 = np.array([])
        nu1 = np.array([])
        kappa1 = np.array([])

        phi_a_iota = (phi_u @ a) + iota_u
        phi_b = phi_u @ b

        if L_set.size > 0:
            inv = pinv(X0L.T @ X0L)
            X0L_plus = inv @ X0L.T

            # Calculate nu0
            X0L_plus_phi_b = X0L_plus @ phi_b
            nu0 = (-SL * X0L_plus_phi_b).ravel()

            # Calculate kappa0
            X0L_plus_a = X0L_plus @ phi_a_iota
            kappa0_term_inv = inv @ SL
            kappa0 = SL * X0L_plus_a - (nT * lambda_tilde) * SL * kappa0_term_inv
            kappa0 = kappa0.ravel()

        if L_set.size > 0:
            if L_set.size == 0:
                proj = np.eye(nT)
                temp2 = 0

            else:
                proj = np.eye(nT) - X0L @ X0L_plus

                X0L_T_plus = X0L @ inv
                temp2 = (X0Lc.T @ X0L_T_plus) @ SL

            X0Lc_T_proj = X0Lc.T @ proj
            temp1 = X0Lc_T_proj / (lambda_tilde * nT)

            # Calculate nu1
            term_b = temp1 @ phi_b
            nu1 = np.concatenate([term_b.ravel(), -term_b.ravel()])

            # Calculate kappa1
            term_a = temp1 @ phi_a_iota
            ones_vec = np.ones_like(term_a)
            kappa1 = np.concatenate(
                [
                    (ones_vec - temp2 - term_a).ravel(),
                    (ones_vec + temp2 + term_a).ravel(),
                ]
            )

        nu = np.concatenate((nu0, nu1))
        kappa = np.concatenate((kappa0, kappa1))

        interval = solve_linear_inequalities(-kappa, nu)

        return interval

    def interval_beta(self, M_set, SM, Mc, xi_uv, zeta_uv, a, b):
        r"""Calculates the feasible interval for :math:`\hat{\bm{\beta}}^{(0)}_{\text{TransFusion}}`.

        Returns
        -------
        intervals : list
            The feasible interval.
        """

        omega0 = np.array([])
        rho0 = np.array([])
        omega1 = np.array([])
        rho1 = np.array([])

        xi_a_zeta = (xi_uv @ a) + zeta_uv
        xi_b = xi_uv @ b

        if M_set.size > 0:
            Dt_xi_a_zeta = xi_a_zeta[M_set]
            Dt_xi_b = xi_b[M_set]

            # Calculate omega0, rho0
            omega0 = (-SM * Dt_xi_b).ravel()
            rho0 = (SM * Dt_xi_a_zeta).ravel()

        if M_set.size > 0:
            Dtc_xi_a_zeta = xi_a_zeta[Mc]
            Dtc_xi_b = xi_b[Mc]

            # Calculate omega1, rho1
            omega1 = np.concatenate([Dtc_xi_b.ravel(), -Dtc_xi_b.ravel()])
            rho1 = np.concatenate([-Dtc_xi_a_zeta.ravel(), Dtc_xi_a_zeta.ravel()])

        omega = np.concatenate((omega0, omega1))
        rho = np.concatenate((rho0, rho1))

        interval = solve_linear_inequalities(-rho, omega)

        return interval

    def calculate_phi_iota_xi_zeta(
        self,
        X,
        SO,
        O_set,
        XO,
        X0,
        SL,
        L_set,
        X0L,
        p,
        B,
        Q,
        lambda_0,
        lambda_tilde,
        a_tilde,
        N,
        nT,
    ):
        phi_u = Q.copy()
        iota_u = np.zeros((nT, 1))
        xi_uv = np.zeros((p, N))
        zeta_uv = np.zeros((p, 1))

        if len(O_set) > 0:
            a_tilde_O = a_tilde[O_set]
            Eu = np.eye(X.shape[1])[:, O_set]
            inv_XOT_XO = pinv(XO.T @ XO)
            XO_plus = inv_XOT_XO @ XO.T
            X0_B_Eu = X0 @ B @ Eu
            B_Eu_inv_XOT_XO = B @ Eu @ inv_XOT_XO

            phi_u -= (1.0 / N) * (X0_B_Eu @ XO_plus)
            iota_u = lambda_0 * (X0_B_Eu @ inv_XOT_XO) @ (a_tilde_O * SO)

            xi_uv += (1.0 / N) * (B_Eu_inv_XOT_XO @ XO.T)
            zeta_uv += -lambda_0 * B_Eu_inv_XOT_XO @ (a_tilde_O * SO)

        if len(L_set) > 0:
            Fv = np.eye(p)[:, L_set]
            inv_X0LT_X0L = pinv(X0L.T @ X0L)
            X0L_plus = inv_X0LT_X0L @ X0L.T

            xi_uv += Fv @ X0L_plus @ phi_u
            zeta_uv += Fv @ inv_X0LT_X0L @ (X0L.T @ iota_u - (nT * lambda_tilde) * SL)

        return phi_u, iota_u, xi_uv, zeta_uv
