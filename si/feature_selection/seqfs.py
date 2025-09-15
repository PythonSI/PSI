import numpy as np
import numpy.typing as npt
from si.node import Data
from typing import Optional, Literal, Tuple
from si.util import intersect, solve_quadratic_inequality
import warnings


class SequentialFeatureSelection:
    r"""Sequential feature selection with selective inference support.

    This class performs sequential feature selection in a greedy manner,
    either by adding features (forward selection) or removing them
    (backward selection) to construct a subset of relevant features.

    The number of selected features can be determined automatically using
    model selection criteria such as AIC, BIC, or adjusted R².

    In addition, the class provides support for performing selective
    inference on the chosen feature set, allowing valid statistical
    inference after the selection process.

    The selection problem for each step is to find the feature that
    maximizes the fit improvement (e.g., minimizes the residual sum of squares).
    For forward selection, at step $k$, the next feature $j$ is chosen to
    solve:

    .. math::
        j = \mathop{\arg \min}_{j \notin \mathcal{A}_{k-1}} \quad \|\mathbf{y} - \mathbf{x}_{\mathcal{A}_{k-1} \cup \{j\}}\boldsymbol{\beta}_{k-1}\|_2^2

    where $\mathcal{A}_{k-1}$ is the active set from the previous step.

    Parameters
    ----------
    n_features_to_select : int, optional
        Number of features to select. If `None`, the number of features is
        determined automatically by the `criterion`. Default is `None`.
    direction : Literal["forward", "backward"], optional
        The direction of the selection process. 'forward' adds features
        and 'backward' removes them. Default is 'forward'.
    criterion : Literal["aic", "bic", "adj_r2"], optional
        The model selection criterion to use. If `None`, the selection
        stops when `n_features_to_select` is reached. Default is `None`.

    Attributes
    ----------
    x_node : Data or None
        Input feature matrix node
    y_node : Data or None
        Input response vector node
    n_features_to_select : int or None
        Number of features to select, as specified by the user.
    n_features_to_select_ : int or None
        The actual number of features selected. This may differ from
        `n_features_to_select` if a criterion is used.
    direction : Literal["forward", "backward"]
        The direction of feature selection.
    criterion : Literal["aic", "bic", "adj_r2"] or None
        The criterion used for automatic selection.
    active_set_node : Data
        Output node containing selected feature indices.
    active_set : array-like or None
        Indices of the selected features.
    interval : list or None
        Feasible interval from the last inference call.
    """

    def __init__(
        self,
        n_features_to_select: Optional[int] = None,
        direction: Literal["forward", "backward"] = "forward",
        criterion: Optional[Literal["aic", "bic", "adj_r2"]] = None,
    ):
        if criterion is not None and n_features_to_select is not None:
            warnings.warn(
                "'n_features_to_select' is ignored when 'criterion' is set. "
                "The selection process will stop based on the chosen criterion."
            )
        # # Input for Sequential Selection
        self.x_node = None
        self.y_node = None
        self.sigma = None
        self.n_features_to_select = n_features_to_select
        self.n_features_to_select_ = None
        self.direction = direction
        self.criterion = criterion

        # # Output for Sequential Selection
        self.active_set_node = Data(self)
        self.interval = None
        self.active_set = None

    def __call__(self):
        r"""Execute sequential feature selection on stored data.

        Returns
        -------
        active_set : array-like, shape (k,)
            Indices of selected features.
        """
        x = self.x_node()
        y = self.y_node()

        active_set = self.forward(x=x, y=y, sigma=self.covariance)

        self.active_set_node.update(active_set)
        return active_set

    def run(self, x: Data, y: Data, covariance: Optional[Data] = None) -> Data:
        r"""Configure sequential selection with input data and return active set node.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Feature matrix.
        y : array-like, shape (n, 1)
            Response vector.
        covariance : array-like, shape (n, n), optional
            Covariance matrix for the residuals.

        Returns
        -------
        active_set_node : Data
            Node containing selected feature indices.
        """
        self.covariance = covariance
        self.x_node = x
        self.y_node = y
        return self.active_set_node

    def forward(self, x, y, sigma=None):
        r"""Fit the sequential feature selection model.

        Performs the sequential selection process based on the configured
        `direction` and either the `n_features_to_select` or the `criterion`.

        Parameters
        ----------
        X : array-like, shape (n, p)
            Feature matrix.
        y : array-like, shape (n, 1)
            Response vector.
        sigma : array-like, shape (n, n), optional
            Covariance matrix of the residuals.
        """
        self.fit(x, y, sigma)
        return self.active_set

    def fit(
        self,
        X,
        y,
        sigma=None,
    ):
        r"""Fit the sequential feature selection model.

        Performs the sequential selection process based on the configured
        `direction` and either the `n_features_to_select` or the `criterion`.

        Parameters
        ----------
        X : array-like, shape (n, p)
            Feature matrix.
        y : array-like, shape (n, 1)
            Response vector.
        sigma : array-like, shape (n, n), optional
            Covariance matrix of the residuals.
        """
        n_samples, n_features = X.shape

        if self.criterion is not None:
            if sigma is None:
                sigma = np.eye(n_samples)

            if self.criterion == "aic":
                self.active_set = self._AIC(y, X, sigma, self.direction)
            elif self.criterion == "bic":
                self.active_set = self._BIC(y, X, sigma, self.direction)
            elif self.criterion == "adj_r2":
                self.active_set = self._adjR2(y, X, self.direction)
        else:
            if self.n_features_to_select is None:
                self.n_features_to_select = n_features // 2
            if self.n_features_to_select >= n_features:
                raise ValueError("n_features_to_select must be < n_features.")

            self.active_set, _ = self._stepwise_selection(
                y, X, self.n_features_to_select, self.direction
            )
            self.n_features_to_select_ = len(self.active_set)

        return self

    def inference(self, z: float) -> Tuple[list, npt.NDArray[np.floating]]:
        r"""Find feasible interval of the Sequential Feature Selection for the parametrized data at z.

        Calculates the feasible interval for the given parameter `z`,
        ensuring the selected feature set remains the same within this interval.
        The interval is determined by the stepwise selection process and, if
        applicable, the chosen model selection criterion.

        Parameters
        ----------
        z : float
            Inference parameter value.

        Returns
        -------
        final_interval : list
            Feasible interval [lower, upper] for z.
        """
        if self.interval is not None and self.interval[0] <= z <= self.interval[1]:
            self.active_set_node.parametrize(data=self.active_set_data)
            return self.interval
        x, _, _, interval_x = self.x_node.inference(z)
        y, a, b, interval_y = self.y_node.inference(z)
        active_set = self.forward(x, y)

        lst_SELECk, lst_P = self.list_residualvec(x, y)

        self.active_set_node.parametrize(data=active_set)

        itvCriterion = [-np.inf, np.inf]
        if self.criterion is not None:
            Sigmatilde = np.eye(x.shape[0])
            if self.criterion == "aic":
                itvCriterion = self.interval_AIC(
                    x, lst_P, len(self.active_set), a, b, Sigmatilde, z
                )
            elif self.criterion == "bic":
                itvCriterion = self.interval_BIC(
                    x, lst_P, len(self.active_set), a, b, Sigmatilde, z
                )
            elif self.criterion == "adj_r2":
                itvCriterion = self.interval_adjr2(
                    x, lst_P, len(self.active_set), a, b, z
                )

        itv_stepwise = self._interval_stepwisefs(
            x, len(self.active_set), lst_SELECk, lst_P, a, b, z
        )

        final_interval = intersect(interval_x, interval_y)
        final_interval = intersect(
            final_interval, intersect(itvCriterion, itv_stepwise)
        )

        self.active_set_node.parametrize(data=active_set)

        self.interval = final_interval
        self.active_set_data = active_set
        return final_interval

    def list_residualvec(self, X, Y) -> list:
        r"""Generates a list of residual vectors during the selection process.

        Returns
        -------
        lst_SELEC_k : list
            A list of active sets at each step.
        lst_Portho : list
            A list of orthogonal projection matrices at each step.
        """

        # Create 1 ... p matrixes which multiplies Y to get "residual vector of best k-subset"
        lst_Portho = []
        lst_SELEC_k = []
        i = np.identity(Y.shape[0])
        s = None

        if self.direction == "forward":
            s = 0
        elif self.direction == "backward":
            s = 1

        for k in range(s, X.shape[1] + 1):
            selec_k, _ = self._stepwise_selection(Y, X, k, self.direction)
            lst_SELEC_k.append(selec_k)
            X_Mk = X[:, selec_k].copy()
            lst_Portho.append(
                i - np.dot(np.dot(X_Mk, np.linalg.inv(np.dot(X_Mk.T, X_Mk))), X_Mk.T)
            )
        if self.direction == "backward":
            lst_SELEC_k.reverse()

        return lst_SELEC_k, lst_Portho

    def _stepwise_selection(self, Y, X, k, direc):
        r"""Performs a single stepwise feature selection process.

        Parameters
        ----------
        Y : array-like, shape (n, 1)
            Response vector.
        X : array-like, shape (n, p)
            Feature matrix.
        k : int
            Number of features to select.
        direc : Literal["forward", "backward"]
            Direction of selection.

        Returns
        -------
        selection : array-like, shape (k,)
            Indices of the selected features.
        rsdv : array-like
            Residual vector.
        """
        if direc == "forward":
            selection = []
            rest = list(range(X.shape[1]))
            rss = np.linalg.norm(Y) ** 2
            rsdv = None
            for i in range(1, k + 1):
                rss = np.inf
                sele = selection.copy()
                selection.append(None)
                for feature in rest:
                    if feature not in selection:
                        # select necessary data
                        X_temp = X[:, sorted(sele + [feature])].copy()
                        # create linear model

                        # calculate rss of model
                        rss_temp, rsdv_temp = SequentialFeatureSelection.RSS(Y, X_temp)

                        # choose feature having minimum rss and append to selection
                        if rss > rss_temp:
                            rss = rss_temp
                            rsdv = rsdv_temp
                            selection.pop()
                            selection.append(feature)
            return selection, rsdv
        elif direc == "backward":
            p = X.shape[1]
            selection = list(range(p))
            if k == p:
                rss, rsdv = SequentialFeatureSelection.RSS(Y, X)
                return selection, rsdv

            for i in range(p - k):
                rss = np.inf
                for j in selection:
                    sele = [x for x in selection if x != j]
                    X_temp = X[:, sele].copy()
                    rss_temp, rsdv_temp = SequentialFeatureSelection.RSS(Y, X_temp)
                    if rss > rss_temp:
                        rss = rss_temp
                        rsdv = rsdv_temp
                        jselec = j
                selection = [x for x in selection if x != jselec]
            return selection, rsdv
        else:
            raise TypeError("Direc must be in ['forward', 'backward']")

    def _AIC(self, Y, X, Sigma, direc):
        r"""Performs selection based on the AIC criterion.

        Returns
        -------
        bset : array-like
            Indices of the selected features.
        """
        AIC = np.inf
        n, p = X.shape

        for i in range(1, p + 1):
            sset, rsdv = self._stepwise_selection(Y, X, i, direc)
            aic = rsdv.T.dot(Sigma.dot(rsdv)) + 2 * i
            if aic < AIC:
                bset = sset
                AIC = aic
        return bset

    def _BIC(self, Y, X, Sigma, direc):
        r"""Performs selection based on the BIC criterion.

        Returns
        -------
        bset : array-like
            Indices of the selected features.
        """
        BIC = np.inf
        n, p = X.shape

        for i in range(1, p + 1):
            sset, rsdv = self._stepwise_selection(Y, X, i, direc)
            bic = rsdv.T.dot(Sigma.dot(rsdv)) + i * np.log(n)
            if bic < BIC:
                bset = sset
                BIC = bic
        return bset

    def _adjR2(self, Y, X, direc):
        r"""Performs selection based on the adjusted R-squared criterion.

        Returns
        -------
        bset : array-like
            Indices of the selected features.
        """
        AdjR2 = -np.inf
        n, p = X.shape
        TSS = np.linalg.norm(Y - np.mean(Y)) ** 2
        for i in range(1, p + 1):
            sset, rsdv = self._stepwise_selection(Y, X, i, direc)
            RSS = np.linalg.norm(rsdv) ** 2

            adjr2 = 1 - (RSS / (n - i - 1)) / (TSS / (n - 1))
            if adjr2 > AdjR2:
                bset = sset
                AdjR2 = adjr2
        return bset

    @staticmethod
    def RSS(Y, X):
        r"""Calculates the Residual Sum of Squares (RSS).

        Returns
        -------
        rss : float
            The RSS value.
        residual_vec : array-like
            The residual vector.
        """
        rss = 0
        coef = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
        yhat = np.dot(X, coef)
        residual_vec = Y - yhat
        rss = np.linalg.norm(residual_vec) ** 2
        return rss, residual_vec

    def _interval_stepwisefs(self, X, K, lst_SELEC_k, lst_Portho, a, b, z):
        r"""Calculates the feasible interval for the stepwise selection.

        Returns
        -------
        intervals : list
            The feasible interval.
        """
        n_sample, n_fea = X.shape
        intervals = [-np.inf, np.inf]

        if self.direction == "forward":
            i = np.identity(n_sample)
            for step in range(1, K + 1):
                jk = lst_SELEC_k[-1]
                # print('----',lst_SELEC_k[step])
                X_jk = X[:, sorted(lst_SELEC_k[step])].copy()
                Pjk = i - np.dot(
                    np.dot(X_jk, np.linalg.inv(np.dot(X_jk.T, X_jk))), X_jk.T
                )

                Pjk_a = Pjk.dot(a)
                Pjk_b = Pjk.dot(b)
                for j in range(n_fea):
                    if j not in lst_SELEC_k[step]:
                        Mj = lst_SELEC_k[step - 1] + [j]
                        X_j = X[:, sorted(Mj)].copy()
                        Pj = i - np.dot(
                            np.dot(X_j, np.linalg.inv(np.dot(X_j.T, X_j))), X_j.T
                        )
                        Pj_a = Pj.dot(a)
                        Pj_b = Pj.dot(b)

                        g1 = Pjk_a.T.dot(Pjk_a) - Pj_a.T.dot(Pj_a)
                        g2 = (
                            Pjk_a.T.dot(Pjk_b)
                            + Pjk_b.T.dot(Pjk_a)
                            - Pj_a.T.dot(Pj_b)
                            - Pj_b.T.dot(Pj_a)
                        )
                        g3 = Pjk_b.T.dot(Pjk_b) - Pj_b.T.dot(Pj_b)

                        g1, g2, g3 = g1.item(), g2.item(), g3.item()
                        itv = solve_quadratic_inequality(g3, g2, g1, z)
                        intervals = intersect(intervals, itv)
            return intervals
        elif self.direction == "backward":
            i = np.identity(n_sample)
            for step in range(1, n_fea - K + 1):
                jk = [x for x in lst_SELEC_k[step - 1] if x not in lst_SELEC_k[step]]
                jk = jk[0]
                X_jk = X[:, lst_SELEC_k[step]].copy()
                Pjk = i - np.dot(
                    np.dot(X_jk, np.linalg.inv(np.dot(X_jk.T, X_jk))), X_jk.T
                )

                Pjk_a = Pjk.dot(a)
                Pjk_b = Pjk.dot(b)
                for j in lst_SELEC_k[step - 1]:
                    if j != jk:
                        Mj = [i for i in lst_SELEC_k[step - 1] if i != j]
                        X_j = X[:, Mj].copy()
                        Pj = i - np.dot(
                            np.dot(X_j, np.linalg.inv(np.dot(X_j.T, X_j))), X_j.T
                        )
                        Pj_a = Pj.dot(a)
                        Pj_b = Pj.dot(b)

                        g1 = Pjk_a.T.dot(Pjk_a) - Pj_a.T.dot(Pj_a)
                        g2 = (
                            Pjk_a.T.dot(Pjk_b)
                            + Pjk_b.T.dot(Pjk_a)
                            - Pj_a.T.dot(Pj_b)
                            - Pj_b.T.dot(Pj_a)
                        )
                        g3 = Pjk_b.T.dot(Pjk_b) - Pj_b.T.dot(Pj_b)
                        g1, g2, g3 = g1.item(), g2.item(), g3.item()
                        itv = solve_quadratic_inequality(g3, g2, g1, z)

                        intervals = intersect(intervals, itv)
            return intervals

    def interval_AIC(self, X, Portho, K, a, b, Sigma, z):
        r"""Calculates the feasible interval for the AIC criterion.

        Returns
        -------
        intervals : list
            The feasible interval.
        """
        n_sample, n_fea = X.shape
        if self.direction == "forward":
            iteration = range(1, n_fea + 1)
            K = K
        elif self.direction == "backward":
            iteration = range(0, n_fea)
            K = K - 1

        Pka = Portho[K].dot(a)
        Pkb = Portho[K].dot(b)

        intervals = [-np.inf, np.inf]

        for step in iteration:
            if step != K:
                Pja = Portho[step].dot(a)
                Pjb = Portho[step].dot(b)
                g1 = (
                    Pka.T.dot(Sigma.dot(Pka))
                    - Pja.T.dot(Sigma.dot(Pja))
                    + 2 * (K - step)
                )
                g2 = (
                    Pka.T.dot(Sigma.dot(Pkb))
                    + Pkb.T.dot(Sigma.dot(Pka))
                    - Pja.T.dot(Sigma.dot(Pjb))
                    - Pjb.T.dot(Sigma.dot(Pja))
                )
                g3 = Pkb.T.dot(Sigma.dot(Pkb)) - Pjb.T.dot(Sigma.dot(Pjb))

                g1, g2, g3 = g1.item(), g2.item(), g3.item()

                itv = solve_quadratic_inequality(g3, g2, g1, z)

                intervals = intersect(intervals, itv)
        return intervals

    def interval_BIC(self, X, Portho, K, a, b, Sigma, z):
        r"""Calculates the feasible interval for the BIC criterion.

        Returns
        -------
        intervals : list
            The feasible interval.
        """
        n_sample, n_fea = X.shape
        if self.direction == "forward":
            iteration = range(1, n_fea + 1)
            K = K
        elif self.direction == "backward":
            iteration = range(0, n_fea)
            K = K - 1
        Pka = Portho[K].dot(a)
        Pkb = Portho[K].dot(b)

        intervals = [-np.inf, np.inf]

        for step in iteration:
            if step != K:
                Pja = Portho[step].dot(a)
                Pjb = Portho[step].dot(b)
                g1 = (
                    Pka.T.dot(Sigma.dot(Pka))
                    - Pja.T.dot(Sigma.dot(Pja))
                    + np.log(n_sample) * (K - step)
                )
                g2 = (
                    Pka.T.dot(Sigma.dot(Pkb))
                    + Pkb.T.dot(Sigma.dot(Pka))
                    - Pja.T.dot(Sigma.dot(Pjb))
                    - Pjb.T.dot(Sigma.dot(Pja))
                )
                g3 = Pkb.T.dot(Sigma.dot(Pkb)) - Pjb.T.dot(Sigma.dot(Pjb))

                g1, g2, g3 = g1.item(), g2.item(), g3.item()

                itv = solve_quadratic_inequality(g3, g2, g1, z)

                intervals = intersect(intervals, itv)
        return intervals

    def interval_adjr2(self, X, Portho, K, a, b, z):
        r"""Calculates the feasible interval for the adjusted R-squared criterion.

        Returns
        -------
        intervals : list
            The feasible interval.
        """
        n_sample, n_fea = X.shape
        ljk = 1 / (n_sample - K - 1)

        if self.direction == "forward":
            iteration = range(1, n_fea + 1)
            K = K
        elif self.direction == "backward":
            iteration = range(0, n_fea)
            K = K - 1

        Pka = Portho[K].dot(a)
        Pkb = Portho[K].dot(b)

        intervals = [-np.inf, np.inf]
        for step in iteration:
            if step != K:
                lj = 1 / (n_sample - step - 1 - 1)
                Pja = Portho[step].dot(a)
                Pjb = Portho[step].dot(b)
                g1 = ljk * Pka.T.dot(Pka) - lj * Pja.T.dot(Pja)
                g2 = (
                    ljk * Pka.T.dot(Pkb)
                    + ljk * Pkb.T.dot(Pka)
                    - lj * Pja.T.dot(Pjb)
                    - lj * Pjb.T.dot(Pja)
                )
                g3 = ljk * Pkb.T.dot(Pkb) - lj * Pjb.T.dot(Pjb)

                g1, g2, g3 = g1.item(), g2.item(), g3.item()

                itv = solve_quadratic_inequality(g3, g2, g1, z)

                intervals = intersect(intervals, itv)
        return intervals
