# import numpy as np
# import numpy.typing as npt

# class Result:
#     """
#     Class to hold the result of pipelines and can be conditioned on.
#     """
#     def __init__(self, parent):
#         self.data = None
#         self.parent = parent
        
#     def __call__(self) -> npt.NDArray[np.float32 | np.float64]:
#         # If data is already computed, return it
#         if self.data is not None:
#             return self.data
        
#         # If no data is present, we can compute it from the parent
#         self.data = self.parent()
#         return self.data

#     def update(self, data: npt.NDArray[np.float32 | np.float64]):
#         self.data = data
