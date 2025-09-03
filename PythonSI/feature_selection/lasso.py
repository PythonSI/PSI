from sklearn.linear_model import Lasso
import numpy as np 
import numpy.typing as npt
from ..node import Data
from typing import Tuple
from ..util import solve_linear_inequalities, intersect

class LassoFeatureSelection:
    def __init__(self, lambda_: float = 10):
        # Input for Lasso regression
        self.X_node = None
        self.y_node = None
        self.lambda_ = lambda_
        
        # Output for Lasso regression
        self.active_set_node = Data(self)
    
    def __call__(self) -> npt.NDArray[np.floating]:
        X = self.X_node()
        y = self.y_node()
        
        active_set, _, _ = self.forward(X=X, y=y)
        
        self.active_set_node.update(active_set)
        return active_set 

    def run(self, X: npt.NDArray[np.floating], y: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        self.X_node = X
        self.y_node = y
        return self.active_set_node

    def forward(
        self, 
        X: npt.NDArray[np.floating], 
        y: npt.NDArray[np.floating]
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        num_of_dimension = X.shape[1]

        lasso = Lasso(alpha=self.lambda_ / X.shape[0], fit_intercept=False, tol=1e-10, max_iter=100000000)
        lasso.fit(X, y)

        coefficients = lasso.coef_.reshape(num_of_dimension, 1)
        active_set = np.nonzero(coefficients)[0]
        inactive_set = np.setdiff1d(np.arange(num_of_dimension), active_set)
        sign_active = np.sign(coefficients[active_set]).reshape(-1, 1)
        
        # # Uncomment this to checkKKT for Lasso
        # self.checkKKT_Lasso(X, y, coefficients, self.lambda_)
        
        return active_set, inactive_set, sign_active

    def inference(self, z: float) -> Tuple[list, npt.NDArray[np.floating]]:
        X, _, _, interval_X = self.X_node.inference(z)
        y, a, b, interval_y = self.y_node.inference(z)
        
        active_set, inactive_set, sign_active = self.forward(X, y)
        inactive_set = np.setdiff1d(np.arange(X.shape[1]), active_set)
        
        self.active_set_node.parametrize(data=active_set)

        # X_a: X with active features
        X_a = X[:, active_set]
        # X_i: X with inactive features
        X_i = X[:, inactive_set]
        
        X_a_plus = np.linalg.inv(X_a.T.dot(X_a)).dot(X_a.T)
        X_aT_plus = X_a.dot(np.linalg.inv(X_a.T.dot(X_a)))
        temp = X_i.T.dot(X_aT_plus).dot(sign_active)
        
        # A + Bz <= 0 (elemen-wise)
        A0 = (self.lambda_ * sign_active * np.linalg.inv(X_a.T.dot(X_a)).dot(sign_active)
             - sign_active * X_a_plus.dot(a))
        B0 = -1 * sign_active * X_a_plus.dot(b)

        temperal_variable = X_i.T.dot(np.identity(X.shape[0]) - X_a.dot(X_a_plus))

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
        
        final_interval = intersect(interval_X, interval_y)
        final_interval = intersect(final_interval, solve_linear_inequalities(A, B))
        
        self.active_set_node.parametrize(data=active_set)

        return final_interval
    
    def checkKKT_Lasso(self, X, Y, beta_hat, Lambda, tol=1e-10):
        """
        Helper function
        Check and assert KKT conditions for Lasso regression solution.

        Parameters
        ----------
        X : ndarray (n, d)
            Design matrix.
        Y : ndarray (n, 1)
            Response vector.
        beta_hat : ndarray (d, 1)
            Estimated coefficients.
        Lambda : float
            Regularization parameter.
        tol : float
            Numerical tolerance.
        """
        # Residuals
        print(X.shape, Y.shape, beta_hat.shape)
        r = Y - X @ beta_hat  # (n,1)

        # Gradient = X^T (Y - Xβ)
        grad = X.T @ r  # (d,1)

        print("--------------- KKT Conditions for Lasso ---------------")
        n_active_ok, n_inactive_ok, n_viol = 0, 0, 0

        for j in range(beta_hat.shape[0]):
            if abs(beta_hat[j]) > tol:  # Active set
                cond = np.isclose(grad[j,0], Lambda * np.sign(beta_hat[j,0]), atol=tol)
                if cond:
                    print(f"[Active]   j={j:2d}, β={beta_hat[j,0]:.4f}, grad={grad[j,0]:.4f} ✅ OK")
                    n_active_ok += 1
                else:
                    print(f"[Active]   j={j:2d}, β={beta_hat[j,0]:.4f}, grad={grad[j,0]:.4f} ❌ VIOLATION")
                    n_viol += 1
                    assert cond, f"KKT violation at active index {j}"
            else:  # Inactive set
                cond = (-Lambda - tol <= grad[j,0] <= Lambda + tol)
                if cond:
                    print(f"[Inactive] j={j:2d}, β={beta_hat[j,0]:.4f}, grad={grad[j,0]:.4f} ✅ OK")
                    n_inactive_ok += 1
                else:
                    print(f"[Inactive] j={j:2d}, β={beta_hat[j,0]:.4f}, grad={grad[j,0]:.4f} ❌ VIOLATION")
                    n_viol += 1
                    assert cond, f"KKT violation at inactive index {j}"

        print("---------------------------------------------------------")
        print(f"Summary: {n_active_ok} active OK, {n_inactive_ok} inactive OK, {n_viol} violations")
        print("---------------------------------------------------------")

        assert n_viol == 0, f"{n_viol} KKT conditions violated!"
        print("✅ All KKT conditions satisfied.")
