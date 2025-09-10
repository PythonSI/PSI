from sklearn.linear_model import Lasso
import numpy as np
import numpy.typing as npt
from si.node import Data
from typing import Tuple
from si.util import solve_linear_inequalities, intersect


class LassoFeatureSelection:
    r"""LASSO feature selection with selective inference support.

    This class implements LASSO feature selection with the capability to perform selective inference on the
    selected features. The LASSO optimization problem is:

    .. math::
        \hat{\boldsymbol{\beta}} = \mathop{\arg \min}_{\boldsymbol{\beta}} \quad 
        \frac{1}{2} \|\mathbf{y} - \mathbf{x}\boldsymbol{\beta}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_1

    where :math:`\lambda` is the regularization parameter that controls sparsity.

    Parameters
    ----------
    lambda_ : float, optional
        Regularization parameter for LASSO, default 10

    Attributes
    ----------
    x_node : Data or None
        Input feature matrix node
    y_node : Data or None  
        Input response vector node
    lambda_ : float
        Regularization parameter
    active_set_node : Data
        Output node containing selected feature indices
    self.interval : list or None
        Feasible interval from last inference call
    self.active_set_data : array-like or None
        Active set from last inference call
    """

    def __init__(self, lambda_: float = 10):
        # Input for Lasso regression
        self.x_node = None
        self.y_node = None
        self.lambda_ = lambda_

        # Output for Lasso regression
        self.active_set_node = Data(self)
        
        self.interval = None
        self.active_set_data = None

    def __call__(self) -> npt.NDArray[np.floating]:
        r"""Execute LASSO feature selection on stored data.

        Returns
        -------
        active_set : array-like, shape (k,)
            Indices of selected features
        """
        x = self.x_node()
        y = self.y_node()

        active_set, _, _ = self.forward(x=x, y=y)

        self.active_set_node.update(active_set)
        return active_set

    def run(self, x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        r"""Configure LASSO with input data and return active set node.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Feature matrix
        y : array-like, shape (n, 1)
            Response vector

        Returns
        -------
        active_set_node : Data
            Node containing selected feature indices
        """
        self.x_node = x
        self.y_node = y
        return self.active_set_node

    def forward(
        self,
        x: npt.NDArray[np.floating],
        y: npt.NDArray[np.floating]
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        r"""Solve LASSO optimization and extract active set information.

        Solves the LASSO problem and returns the active set (selected features),
        inactive set, and signs of active coefficients.

        Parameters
        ----------
        x : array-like, shape (n, p)
            Feature matrix
        y : array-like, shape (n, 1)
            Response vector

        Returns
        -------
        active_set : array-like, shape (k,)
            Indices of selected features
        inactive_set : array-like, shape (p-k,)
            Indices of unselected features  
        sign_active : array-like, shape (k, 1)
            Signs of coefficients for active features
        """
        num_of_dimension = x.shape[1]

        lasso = Lasso(alpha=self.lambda_ /
                      x.shape[0], fit_intercept=False, tol=1e-10, max_iter=100000000)
        lasso.fit(x, y)

        coefficients = lasso.coef_.reshape(num_of_dimension, 1)
        active_set = np.nonzero(coefficients)[0]
        inactive_set = np.setdiff1d(np.arange(num_of_dimension), active_set)
        sign_active = np.sign(coefficients[active_set]).reshape(-1, 1)

        # # Uncomment this to checkKKT for Lasso
        # self.checkKKT_Lasso(x, y, coefficients, self.lambda_)

        return active_set, inactive_set, sign_active

    def inference(self, z: float) -> Tuple[list, npt.NDArray[np.floating]]:
        r"""Find feasible interval of the Lasso Feature Selection for the parametrized data at z.
        
        ----------
        z : float
            Inference parameter value

        Returns
        -------
        final_interval : list
            Feasible interval [lower, upper] for z
        """
        if self.interval is not None and self.interval[0] <= z <= self.interval[1]:
            self.active_set_node.parametrize(data=self.active_set_data)
            return self.interval
        
        x, _, _, interval_x = self.x_node.inference(z)
        y, a, b, interval_y = self.y_node.inference(z)

        active_set, inactive_set, sign_active = self.forward(x, y)
        inactive_set = np.setdiff1d(np.arange(x.shape[1]), active_set)

        self.active_set_node.parametrize(data=active_set)

        # x_a: x with active features
        x_a = x[:, active_set]
        # x_i: x with inactive features
        x_i = x[:, inactive_set]

        x_a_plus = np.linalg.inv(x_a.T.dot(x_a)).dot(x_a.T)
        x_aT_plus = x_a.dot(np.linalg.inv(x_a.T.dot(x_a)))
        temp = x_i.T.dot(x_aT_plus).dot(sign_active)

        # A + Bz <= 0 (elemen-wise)
        A0 = (self.lambda_ * sign_active * np.linalg.inv(x_a.T.dot(x_a)).dot(sign_active)
              - sign_active * x_a_plus.dot(a))
        B0 = -1 * sign_active * x_a_plus.dot(b)

        temperal_variable = x_i.T.dot(
            np.identity(x.shape[0]) - x_a.dot(x_a_plus))

        A10 = -(np.ones((temp.shape[0], 1)) - temp
                - (temperal_variable.dot(a)) / self.lambda_)
        B10 = (temperal_variable.dot(b)) / self.lambda_

        A11 = -(np.ones((temp.shape[0], 1)) + temp
                + (temperal_variable.dot(a)) / self.lambda_)
        B11 = -(temperal_variable.dot(b)) / self.lambda_

        solve_linear_inequalities(A0, B0)
        solve_linear_inequalities(A10, B10)
        solve_linear_inequalities(A11, B11)

        A = np.vstack((A0, A10, A11))
        B = np.vstack((B0, B10, B11))

        final_interval = intersect(interval_x, interval_y)
        final_interval = intersect(
            final_interval, solve_linear_inequalities(A, B))

        self.active_set_node.parametrize(data=active_set)
        
        self.interval = final_interval
        self.active_set_data = active_set

        return final_interval

    def checkKKT_Lasso(self, x, Y, beta_hat, Lambda, tol=1e-10):
        r"""Validate KKT conditions for LASSO solution.

        Checks that the computed LASSO solution satisfies the Karush-Kuhn-Tucker
        optimality conditions:

        - **Active features**: :math:`\mathbf{x}_j^T(\mathbf{y} - \mathbf{x}\hat{\boldsymbol{\beta}}) = \lambda \cdot \text{sign}(\hat{\beta}_j)` for :math:`\hat{\beta}_j \neq 0`
        - **Inactive features**: :math:`|\mathbf{x}_j^T(\mathbf{y} - \mathbf{x}\hat{\boldsymbol{\beta}})| \leq \lambda` for :math:`\hat{\beta}_j = 0`

        Parameters
        ----------
        x : array-like, shape (n, d)
            Design matrix
        Y : array-like, shape (n, 1)
            Response vector
        beta_hat : array-like, shape (d, 1)
            Estimated LASSO coefficients
        Lambda : float
            Regularization parameter
        tol : float, optional
            Numerical tolerance for checks, default 1e-10

        Raises
        ------
        AssertionError
            If any KKT condition is violated

        Notes
        -----
        This is a helper function for debugging and validation.
        Prints detailed information about each constraint.
        """
        # Residuals
        print(x.shape, Y.shape, beta_hat.shape)
        r = Y - x @ beta_hat  # (n,1)

        # Gradient = x^T (Y - xβ)
        grad = x.T @ r  # (d,1)

        print("--------------- KKT Conditions for Lasso ---------------")
        n_active_ok, n_inactive_ok, n_viol = 0, 0, 0

        for j in range(beta_hat.shape[0]):
            if abs(beta_hat[j]) > tol:  # Active set
                cond = np.isclose(grad[j, 0], Lambda *
                                  np.sign(beta_hat[j, 0]), atol=tol)
                if cond:
                    print(
                        f"[Active]   j={j:2d}, β={beta_hat[j,0]:.4f}, grad={grad[j,0]:.4f} ✅ OK")
                    n_active_ok += 1
                else:
                    print(
                        f"[Active]   j={j:2d}, β={beta_hat[j,0]:.4f}, grad={grad[j,0]:.4f} ❌ VIOLATION")
                    n_viol += 1
                    assert cond, f"KKT violation at active index {j}"
            else:  # Inactive set
                cond = (-Lambda - tol <= grad[j, 0] <= Lambda + tol)
                if cond:
                    print(
                        f"[Inactive] j={j:2d}, β={beta_hat[j,0]:.4f}, grad={grad[j,0]:.4f} ✅ OK")
                    n_inactive_ok += 1
                else:
                    print(
                        f"[Inactive] j={j:2d}, β={beta_hat[j,0]:.4f}, grad={grad[j,0]:.4f} ❌ VIOLATION")
                    n_viol += 1
                    assert cond, f"KKT violation at inactive index {j}"

        print("---------------------------------------------------------")
        print(
            f"Summary: {n_active_ok} active OK, {n_inactive_ok} inactive OK, {n_viol} violations")
        print("---------------------------------------------------------")

        assert n_viol == 0, f"{n_viol} KKT conditions violated!"
        print("✅ All KKT conditions satisfied.")
