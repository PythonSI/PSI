import numpy as np
import numpy.typing as npt
from pythonsi.node import Data
from typing import List
from pythonsi.util import solve_linear_inequalities, intersect
from skglm import WeightedLasso, Lasso
from numpy.linalg import pinv


def construct_Q(nT, N):
    Q = np.zeros((nT, N))
    Q[:, N - nT :] = np.eye(nT)
    return Q


from pythonsi.transfer_learning_hdr.transfusion import construct_Q


def construct_P(nT, nI):
    P = np.zeros((nI, nI + nT))
    P[:, :nI] = np.eye(nI)
    return P


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


class TLOracleTransLasso:
    r"""Post-Transfer Learning Statistical Inference with Oracle Trans-Lasso.

    This class implements the PTL-SI framework using the **Oracle Trans-Lasso**
    procedure for high-dimensional regression. It leverages **informative source domains** indexed by :math:`\mathcal{I} = \{1,\dots,K\}`,
    together with the target domain, to perform feature selection and
    valid post-selection inference.

    The Oracle Trans-Lasso algorithm proceeds in two steps:

    Step 1. The estimator :math:`\hat{\bm w}^\mathcal{I}` is obtained by solving a LASSO problem
    using the source domains in the informative set :math:`\mathcal{I}`:
    .. math::
    \begin{align*}
        \hat{\bm w}^{\mathcal{I}} &= 
        \underset{\bm{w} \in \mathbb{R}^{{p}}}{\operatorname{argmin}}
        \left\{ 
        \frac{1}{2n_{\mathcal{I}}} 
        \sum \limits_{k \in \mathcal{I}} 
        \| \bm{Y}^{(k)} -  X^{(k)} \bm{w} \|_2^2 + \lambda_{\bm w} \| \bm{w} \|_1 
        \right\} \\
        &= \underset{\bm{w} \in \mathbb{R}^{{p}}}{\operatorname{argmin}} \left\{ \frac{1}{2n_{\mathcal{I}}} \| \bm{Y}^{\mathcal{I}} -  X^{\mathcal{I}} \bm{w} \|_2^2 + \lambda_w \| \bm{w} \|_1 \right\}
    \end{align*}


    where:
    - :math:`\mathcal{I}` is the set of informative source domains,
    - :math:`n_\mathcal{I} = Kn_S` is the total number of samples across sources in :math:`\mathcal{I}`,
    - :math:`\lambda_w` is the regularization parameter.
    - :math:`\bm{Y}^{\mathcal{I}}` is the stacked response vector of sources in :math:`\mathcal{I}`,
    - :math:`X^{\mathcal{I}}` is the stacked design matrix of sources in :math:`\mathcal{I}`.

    Step 2. A correction term :math:`\hat{\bm{\delta}}^{\mathcal{I}}` is estimated from the target domain:

    .. math::
        \hat{\bm{\delta}}^{\mathcal{I}} = \underset{\bm{\delta} \in \mathbb{R}^p}{\operatorname{argmin}} 
        \left\{ \frac{1}{2n_T} \left\| \bm Y^{(0)} - {X}^{(0)} \hat{\bm{w}}^{\mathcal{I}} - {X}^{(0)} \bm{\delta} \right\|_2^2 + \lambda_{\bm \delta} \|\bm{\delta}\|_1\right\}

    and the final Oracle Trans-Lasso estimator is:

    .. math::
        \hat{\bm{\beta}}^{(0)}_{\rm OTL} = \hat{\bm{w}}^{\mathcal{I}} + \hat{\bm{\delta}}^{\mathcal{I}}

    where
    - :math: `\lambda_{\bm \delta}` is the regularization parameter.

    Parameters
    ----------
    lambda_w : float
        LASSO regularization parameter for Step 1.
    lambda_del : float
        LASSO regularization parameter for Step 2.

    Attributes
    ----------
    XI_list_node : Data or None
        Informative source domain design matrices, shape ``(K, nS, p)``.
    YI_list_node : Data or None
        Stacked informative source domain response vectors, shape ``(K*nS, 1)``.
    X0_node : Data or None
        Target domain design matrix, shape ``(nT, p)``.
    Y0_node : Data or None
        Target domain response vector, shape ``(nT, 1)``.
    active_set_node : Data
        Node containing selected feature indices.
    interval : list or None
        Feasible interval for the last inference call.
    active_set : array-like or None
        Active set of selected target domain features.
    """

    def __init__(self, lambda_w: float, lambda_del: float):
        self.lambda_w = lambda_w
        self.lambda_del = lambda_del
        self.XI_list_node = None
        self.YI_list_node = None
        self.X0_node = None
        self.Y0_node = None

        self.active_set_node = Data(self)
        self.interval = None
        self.active_set = None

    def run(
        self,
        XI_list: npt.NDArray[np.floating],
        YI_list: npt.NDArray[np.floating],
        X0: npt.NDArray[np.floating],
        Y0: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.int64]:
        r"""Configure Oracle Trans-Lasso with input data and return active set node.

        Parameters
        ----------
        XI_list : array-like, shape (K, nS, p)
            Design matrices of the informative source domains in set :math:`\mathcal{I}`.
        YI_list : array-like, shape (K*nS, 1)
            Stacked response vectors of the informative source domains.
        X0 : array-like, shape (nT, p)
            Target domain design matrix.
        Y0 : array-like, shape (nT, 1)
            Target domain response vector.

        Returns
        -------
        active_set_node : Data
            Node containing indices of features selected in the target domain.
        """
        self.XI_list_node = XI_list
        self.YI_list_node = YI_list
        self.X0_node = X0
        self.Y0_node = Y0

        return self.active_set_node

    def forward(
        self,
        XI_list: npt.NDArray[np.floating],
        YI_list: npt.NDArray[np.floating],
        X0: npt.NDArray[np.floating],
        Y0: npt.NDArray[np.floating],
    ):
        r"""Run the Oracle Trans-Lasso algorithm and extract active set information.

        Executes the Oracle Trans-Lasso procedure with informative sources :math:`\mathcal{I}`,
        and returns the active set together with intermediate quantities needed for
        selective inference.

        Parameters
        ----------
        XI_list : array-like, shape (K, nS, p)
            Design matrices of the informative source domains in :math:`\mathcal{I}`.
        YI_list : array-like, shape (K*nS, 1)
            Stacked response vectors of the informative source domains.
        X0 : array-like, shape (nT, p)
            Target domain design matrix.
        Y0 : array-like, shape (nT, 1)
            Target domain response vector.

        Returns
        -------
        active_set : array-like, shape (k,)
            Indices of selected features in the target domain.
        XI : array-like, shape (nI, p)
            Combined design matrix of informative source domains.
        YI : array-like, shape (nI, 1)
            Combined response vector of informative source domains.
        nI : int
            Total number of samples across informative sources.
        nT : int
            Number of samples in the target domain.
        p : int
            Number of features.
        w_hat : array-like, shape (p,)
            Estimator from Step 1.
        delta_hat : array-like, shape (p,)
            Estimated target-specific adjustment vector from Step 2.
        beta0_hat : array-like, shape (p,)
            Final target domain coefficient estimator.

        """
        nS = XI_list[0].shape[0]
        nT = Y0.shape[0]
        nI = nS * len(XI_list)
        N = nT + nI
        p = X0.shape[1]

        XI = np.concatenate(XI_list)
        YI = YI_list
        Y = np.concatenate((YI_list, Y0))
        w_hat, delta_hat, beta0_hat = self._OracleTransLasso(
            XI, YI.ravel(), X0, Y0.ravel(), self.lambda_w, self.lambda_del
        )
        active_set = np.nonzero(beta0_hat)[0]

        return active_set, XI, Y, nT, nI, N, p, w_hat, delta_hat, beta0_hat

    def __call__(self):
        r"""Execute the Oracle TransLasso algorithm on stored data.

        Returns
        -------
        active_set : array-like, shape (k,)
            Indices of selected target domain features.
        """
        XI_list_node = self.XI_list_node()
        YI_list_node = self.YI_list_node()
        X0_node = self.X0_node()
        Y0_node = self.Y0_node()

        active_set, _, _, _, _, _, _, _, _, _ = self.forward(
            XI_list_node, YI_list_node, X0_node, Y0_node
        )
        self.active_set_node.update(active_set)
        return active_set

    def inference(self, z: float) -> List:
        r"""Find feasible interval of the Oracle TransLasso for the parametrized data at z.

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

        XI_list, _, _, interval_XI_list = self.XI_list_node.inference(z)
        YI_list, a_YI_list, b_YI_list, interval_YI_list = self.YI_list_node.inference(z)
        X0, _, _, interval_X0 = self.X0_node.inference(z)
        Y0, a_Y0, b_Y0, interval_Y0 = self.Y0_node.inference(z)

        active_set, XI, Y, nT, nI, N, p, w_hat, delta_hat, beta0_hat = self.forward(
            XI_list, YI_list, X0, Y0
        )
        a = np.vstack((a_YI_list, a_Y0))
        b = np.vstack((b_YI_list, b_Y0))

        Q = construct_Q(nT, N)
        P = construct_P(nT, nI)

        wO, SO, O_set, XIO, Oc, XIOc = construct_active_set(w_hat, XI)
        deltaL, SL, L_set, X0L, Lc, X0Lc = construct_active_set(delta_hat, X0)
        betaM, SM, M_set, _, Mc, _ = construct_active_set(beta0_hat, X0)

        phi_u, iota_u, xi_uv, zeta_uv = self.calculate_phi_iota_xi_zeta(
            XI,
            SO,
            O_set,
            XIO,
            X0,
            SL,
            L_set,
            X0L,
            p,
            Q,
            P,
            self.lambda_w,
            self.lambda_del,
            nI,
            nT,
        )

        w_itv = self.interval_w(SO, O_set, XIO, Oc, XIOc, a, b, P, self.lambda_w, nI)
        delta_itv = self.interval_delta(
            SL, L_set, X0L, Lc, X0Lc, phi_u, iota_u, a, b, self.lambda_del, nT
        )
        beta_itv = self.interval_beta(M_set, SM, Mc, xi_uv, zeta_uv, a, b)

        final_interval = intersect(intersect(w_itv, delta_itv), beta_itv)
        final_interval = intersect(final_interval, interval_XI_list)
        final_interval = intersect(final_interval, interval_YI_list)
        final_interval = intersect(final_interval, interval_X0)
        final_interval = intersect(final_interval, interval_Y0)

        self.interval = final_interval
        self.active_set_node.parametrize(data=active_set)
        self.active_set = active_set
        return self.interval

    def _OracleTransLasso(self, XI, YI, X0, Y0, lambda_w, lambda_del):
        r"""Run the Oracle TransLasso algorithm.

        Returns
        -------
        w_hat : array-like, shape (p,)
            Estimator from Step 1.
        delta_hat : array-like, shape (p,)
            Estimated target-specific adjustment vector from Step 2.
        beta0_hat : array-like, shape (p,)
            Final target domain coefficient estimator.
        """
        w_hat = Lasso(alpha=lambda_w, fit_intercept=False, tol=1e-10).fit(XI, YI).coef_

        delta_hat = (
            Lasso(alpha=lambda_del, fit_intercept=False, tol=1e-12)
            .fit(X0, Y0 - X0 @ w_hat)
            .coef_
        )

        beta0_hat = w_hat + delta_hat

        return w_hat, delta_hat, beta0_hat

    def interval_w(self, SO, O_set, XIO, Oc, XIOc, a, b, P, lambda_w, nI):
        r"""Calculates the feasible interval for :math:`\hat{\bm{w}}`.

        Returns
        -------
        intervals : list
            The feasible interval.
        """
        psi0 = np.array([])
        gamma0 = np.array([])
        psi1 = np.array([])
        gamma1 = np.array([])

        if len(O_set) > 0:
            inv = pinv(XIO.T @ XIO)
            XIO_plus = inv @ XIO.T

            # Calculate psi0
            XIO_plus_Pb = XIO_plus @ P @ b
            psi0 = (-SO * XIO_plus_Pb).ravel()

            # Calculate gamma0
            XIO_plus_Pa = XIO_plus @ P @ a
            gamma0_term_inv = inv @ SO

            gamma0 = SO * XIO_plus_Pa - nI * lambda_w * SO * gamma0_term_inv
            gamma0 = gamma0.ravel()

        if len(Oc) > 0:
            if len(O_set) == 0:
                proj = np.eye(nI)
                temp2 = 0

            else:
                proj = np.eye(nI) - XIO @ XIO_plus
                XIO_T_plus = XIO @ inv
                temp2 = (XIOc.T @ XIO_T_plus) @ SO

            XIOc_T_proj = XIOc.T @ proj
            temp1 = XIOc_T_proj / (lambda_w * nI)

            # Calculate psi1
            term_Pb = temp1 @ P @ b
            psi1 = np.concatenate([term_Pb.ravel(), -term_Pb.ravel()])

            # Calculate gamma1
            term_Pa = temp1 @ P @ a
            ones_vec = np.ones_like(term_Pa)

            gamma1 = np.concatenate(
                [
                    (ones_vec - temp2 - term_Pa).ravel(),
                    (ones_vec + temp2 + term_Pa).ravel(),
                ]
            )

        psi = np.concatenate((psi0, psi1))
        gamma = np.concatenate((gamma0, gamma1))

        interval = solve_linear_inequalities(-gamma, psi)

        return interval

    def interval_delta(
        self, SL, L_set, X0L, Lc, X0Lc, phi_u, iota_u, a, b, lambda_del, nT
    ):
        r"""Calculates the feasible interval for :math:`\hat{\bm{\delta}}`.

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
            kappa0 = SL * X0L_plus_a - (nT * lambda_del) * SL * kappa0_term_inv
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
            temp1 = X0Lc_T_proj / (lambda_del * nT)

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
        r"""Calculates the feasible interval for :math:`\hat{\bm{\beta}}^{(0)}_{\text{OTL}}`.

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
        XI,
        SO,
        O_set,
        XIO,
        X0,
        SL,
        L_set,
        X0L,
        p,
        Q,
        P,
        lambda_w,
        lambda_del,
        nI,
        nT,
    ):
        phi_u = Q.copy()
        iota_u = np.zeros((nT, 1))
        xi_uv = np.zeros((p, nI + nT))
        zeta_uv = np.zeros((p, 1))

        if len(O_set) > 0:
            Eu = np.eye(XI.shape[1])[:, O_set]
            inv_XIO_T_XIO = pinv(XIO.T @ XIO)
            XIO_plus = inv_XIO_T_XIO @ XIO.T
            X0_Eu = X0 @ Eu

            phi_u -= X0_Eu @ XIO_plus @ P
            iota_u = (nI * lambda_w) * (X0_Eu @ inv_XIO_T_XIO @ SO)
            xi_uv += Eu @ XIO_plus @ P
            zeta_uv += -(nI * lambda_w) * (Eu @ inv_XIO_T_XIO @ SO)

        if len(L_set) > 0:
            Fv = np.eye(p)[:, L_set]
            inv_X0L_T_X0L = pinv(X0L.T @ X0L)
            X0L_plus = inv_X0L_T_X0L @ X0L.T

            xi_uv += Fv @ X0L_plus @ phi_u
            zeta_uv += Fv @ inv_X0L_T_X0L @ (X0L.T @ iota_u - (nT * lambda_del) * SL)

        return phi_u, iota_u, xi_uv, zeta_uv
