from .CPU import CPUModel
from .CUDA import CUDAModel


def InferenceModel(model, device):
    if device == "cuda":
        return CUDAModel(model)
    else:
        return CPUModel(model)
