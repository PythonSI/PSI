import numpy as np
import numpy.typing as npt

class Data():
    """
    A class to present a Data node, handling a single matrix or number
    """
    def __init__(self, parent: any = None):
        self.data = None
        self.parent = parent
        self.a = None
        self.b = None
        self.inference_data = None
        
    def __call__(self) -> npt.NDArray[np.floating]:
        if self.parent is None:
            if self.data is None:
                raise ValueError("Data node has no data or parent to compute from.")
            return self.data
        self.parent()
        return self.data

    def update(self, data: npt.NDArray[np.floating]):
        self.data = data
        
    def parametrize(self, a: npt.NDArray[np.floating] = None, b: npt.NDArray[np.floating] = None, data: npt.NDArray[np.floating] = None):
        self.a = a
        self.b = b
        self.inference_data = data

    def inference(self, z: float):
        if self.parent is not None:
            interval = self.parent.inference(z)
        else:
            interval = [-np.inf, np.inf]
            
        if self.a is not None and self.b is not None:
            self.inference_data = self.a + self.b * z
            return self.inference_data, self.a, self.b, interval
        return self.inference_data, self.a, self.b, interval