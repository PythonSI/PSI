import numpy as np
import numpy.typing as npt
from pythonsi.node import Data
from typing import Tuple
from pythonsi.util import solve_quadratic_inequality, intersect
from pythonsi.dnn import InferenceModel
import torch


class RepresentationLearningDA:
    def __init__(self, model: object, device: str = "cpu"):
        self.x_source_node = None
        self.x_target_node = None

        self.x_tilde_node = Data(self)

        self.interval = None
        self.x_tilde_data = None

        self.model = model.to(device)
        self.inference_model = InferenceModel(model, device)
        self.device = device

    def run(self, xs: Data, xt: Data) -> Data:
        self.x_source_node = xs
        self.x_target_node = xt

        return self.x_tilde_node

    def forward(self, xs: npt.NDArray, xt: npt.NDArray) -> npt.NDArray:
        x = np.vstack((xs, xt))
        x = x.astype(np.float32)  # Ensure proper dtype before tensor conversion
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x_tilde = self.model(x).detach().cpu().numpy()
        return x_tilde

    def __call__(self):
        xs = self.x_source_node()
        xt = self.x_target_node()

        x_tilde = self.forward(xs, xt)
        self.x_tilde_node.update(x_tilde)
        return x_tilde

    def inference(self, z: float) -> Tuple[list, npt.NDArray]:
        if self.interval is not None and self.interval[0] <= z <= self.interval[1]:
            return self.interval

        xs, a_xs, b_xs, itv_xs = self.x_source_node.inference(z)
        xt, a_xt, b_xt, itv_xt = self.x_target_node.inference(z)

        x = np.vstack((xs, xt))
        a = np.vstack((a_xs, a_xt))
        b = np.vstack((b_xs, b_xt))

        final_itv = [-np.inf, np.inf]

        x_tilde = self.forward(xs, xt)
        a_tilde, b_tilde, itv = self.inference_model.forward(a, b, z)

        final_itv = intersect(final_itv, itv)
        final_itv = intersect(final_itv, itv_xs)
        final_itv = intersect(final_itv, itv_xt)

        self.x_tilde_node.parametrize(a=a_tilde, b=b_tilde, data=x_tilde)

        self.interval = final_itv
        self.x_tilde_data = x_tilde
        return final_itv
