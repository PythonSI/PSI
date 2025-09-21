import numpy as np
import numpy.typing as npt
from pythonsi.node import Data
from typing import Tuple, Optional
from pythonsi.util import solve_quadratic_inequality, intersect
from pythonsi.dnn import InferenceModel
import torch


class AutoEncoderAD:
    def __init__(self, model: object, device: str = "cpu", alpha: float = 0.05):
        self.x_node = None
        self.xt_node = None

        self.anomaly_node = Data(self)

        self.interval = None
        self.anomaly_data = None

        self.model = model.to(device)
        self.inference_model = InferenceModel(model, device)
        self.alpha = alpha
        self.device = device

    def run(self, x: Data, target_data: Optional[Data]) -> Data:
        self.x_node = x

        if target_data is not None:
            self.xt_node = target_data
        return self.anomaly_node

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        x_hat = self.model(x_tensor)
        x_hat = x_hat.detach().cpu().numpy()
        reconstruction_loss = self._reconstruction_loss(x, x_hat)

        return self._get_anomalies(reconstruction_loss)

    def _get_anomalies(self, reconstruction_loss: npt.NDArray) -> list:
        """
        Get anomalies based on reconstruction loss using alpha percentile threshold.

        Args:
            reconstruction_loss: Array of reconstruction losses for each sample

        Returns:
            List of anomaly indices
        """
        sorted_indices = np.argsort(reconstruction_loss)[::-1]
        n_anomalies = int(self.alpha * len(reconstruction_loss))
        anomalies = sorted_indices[:n_anomalies]

        # Filter anomalies based on target data if available
        if self.xt_node is not None:
            x_t = self.xt_node()
            ns = reconstruction_loss.shape[0] - x_t.shape[0]
            anomalies = [i - ns for i in anomalies if i >= ns]

        return anomalies

    def _reconstruction_loss(self, x: npt.NDArray, x_hat: npt.NDArray) -> npt.NDArray:
        reconstruction_loss = np.sum(np.abs(x - x_hat), axis=1)
        return reconstruction_loss

    def __call__(self):
        x = self.x_node()
        anomalies = self.forward(x)
        self.anomaly_node.update(anomalies)
        return anomalies

    def inference(self, z: float) -> Tuple[list, npt.NDArray]:
        if self.interval is not None and self.interval[0] <= z <= self.interval[1]:
            return self.interval

        x, u, v, itv_x = self.x_node.inference(z)

        final_itv = [-np.inf, np.inf]

        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        x_hat = self.model(x_tensor)
        x_hat = x_hat.detach().cpu().numpy()
        p, q, itv = self.inference_model.forward(u, v, z)

        anomalies = self.forward(x)

        final_itv = intersect(final_itv, itv)
        final_itv = intersect(final_itv, itv_x)

        s = np.sign(x_hat - x)
        u_args = s * (u - p)
        v_args = s * (v - q)
        all_lower_bounds = np.full(u_args.shape, -np.inf)
        all_upper_bounds = np.full(u_args.shape, np.inf)
        mask_v_pos = v_args > 1e-16  # v_arg > 0  => z < -u_arg/v_arg
        mask_v_neg = v_args < -1e-16  # v_arg < 0  => z > -u_arg/v_arg
        np.divide(-u_args, v_args, out=all_upper_bounds, where=mask_v_pos)
        np.divide(-u_args, v_args, out=all_lower_bounds, where=mask_v_neg)
        mask_v_zero_u_bad = (np.abs(v_args) <= 1e-16) & (u_args > 0)
        all_lower_bounds[mask_v_zero_u_bad] = np.inf
        all_upper_bounds[mask_v_zero_u_bad] = -np.inf
        final_lower_bound = np.max(all_lower_bounds)
        final_upper_bound = np.min(all_upper_bounds)
        itv = intersect(itv, [final_lower_bound, final_upper_bound])

        reconstruction_loss = np.array(self._reconstruction_loss(x, x_hat))
        pivot = self._get_alpha_percent_greatest(reconstruction_loss, self.alpha)
        # 1. Vectorize the calculation of A and B using np.sum.
        # This replaces the first double loop. We compute the element-wise product
        # and then sum along the column axis (axis=1).
        A = np.sum(s * (p - u), axis=1)
        B = np.sum(s * (q - v), axis=1)

        # 2. Vectorize the final interval calculation.
        # This replaces the second loop.

        # Get the pivot values.
        A_pivot = A[pivot]
        B_pivot = B[pivot]

        # Create a boolean mask to handle the if/else logic for all points at once.
        mask_lt = reconstruction_loss < reconstruction_loss[pivot]

        # Use np.where to create the argument arrays for solve_linear_inequality.
        # This is a concise way to implement the if/else logic on entire arrays.
        # u_args = (A - A_pivot) if mask_lt is True, else (A_pivot - A)
        u_args = np.where(mask_lt, A - A_pivot, A_pivot - A)
        v_args = np.where(mask_lt, B - B_pivot, B_pivot - B)

        # 3. Vectorize the `solve_linear_inequality` logic for these new arguments.
        # This is the same robust pattern used in the previous optimization.
        all_lower_bounds = np.full(u_args.shape, -np.inf)
        all_upper_bounds = np.full(u_args.shape, np.inf)

        mask_v_pos = v_args > 1e-16
        mask_v_neg = v_args < -1e-16

        np.divide(-u_args, v_args, out=all_upper_bounds, where=mask_v_pos)
        np.divide(-u_args, v_args, out=all_lower_bounds, where=mask_v_neg)

        mask_v_zero_u_bad = (np.abs(v_args) <= 1e-16) & (u_args > 0)
        all_lower_bounds[mask_v_zero_u_bad] = np.inf
        all_upper_bounds[mask_v_zero_u_bad] = -np.inf

        # 4. Aggregate all constraints and intersect with the current interval `itv`.
        final_lower_bound = np.max(all_lower_bounds)
        final_upper_bound = np.min(all_upper_bounds)

        itv = intersect(itv, [final_lower_bound, final_upper_bound])

        self.anomaly_node.parametrize(data=anomalies)

        self.interval = final_itv
        self.anomaly_data = anomalies
        return final_itv

    def _get_alpha_percent_greatest(self, X, alpha):
        return np.argsort(X)[-int(alpha * len(X))]
