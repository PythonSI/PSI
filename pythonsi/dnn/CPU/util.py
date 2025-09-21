import numpy as np
import torch
from torch import nn


def parse_torch_model(model: nn.Module):
    """
    Flatten model into a list of (layer_type, tensor_or_None)
    e.g. [("Weight", tensor), ("Bias", tensor), ("ReLU", None), ...]

    Args:
        model: PyTorch nn.Module
        to_numpy: If True, convert tensors to numpy arrays
    """
    results = []

    for module in model.modules():
        if isinstance(module, nn.Sequential) or module is model:
            continue  # skip containers

        # Handle parameterized modules (Linear, Conv, etc.)
        if hasattr(module, "weight") or hasattr(module, "bias"):
            w = (
                module.weight.detach()
                if hasattr(module, "weight") and module.weight is not None
                else None
            )
            b = (
                module.bias.detach()
                if hasattr(module, "bias") and module.bias is not None
                else None
            )

            w = w.cpu().numpy().T if w is not None else None
            b = b.cpu().numpy() if b is not None else None

            results.append((module.__class__.__name__, (w, b)))
            continue
        # Handle activation functions / parameterless modules
        if not any(p.requires_grad for p in module.parameters()):
            results.append((module.__class__.__name__, None))

    return results


def is_torch_model(model):
    return isinstance(model, nn.Module)


def parse_model(model):
    if is_torch_model(model):
        return parse_torch_model(model)
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")
