from .operations import Linear, ReLU
from . import util
import numpy as np


class CPUModel:
    def __init__(self, model):
        self.layers = util.parse_model(model)

    def forward(self, a, b, z):
        a = np.asarray(a)
        b = np.asarray(b)
        itv = np.array([-np.inf, np.inf])

        for name, params in self.layers:
            if name == "Linear":
                a, b = Linear(a, b, params)
            elif name == "ReLU":
                a, b, itv = ReLU(a, b, z, itv)
        return a, b, [itv[0], itv[1]]
