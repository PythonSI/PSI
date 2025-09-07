import numpy as np
import numpy.typing as npt

class Data():
    r"""Data object (node) which is used to store data and parametrized data.
    
    Parameters
    ----------
    parent : any, optional
        Parent node to compute data from. If None, data must be set directly.
        
    Attributes
    ----------
    data : array-like or None
        The observed data stored in this node
    parent : any or None
        Parent node for computation
    a : array-like or None
        Parametrized intercept
    b : array-like or None
        Parametrized coefficient for inference
    inference_data : array-like or None
        Data used in the inference process
    """
    def __init__(self, parent: any = None):
        self.data = None
        self.parent = parent
        self.a = None
        self.b = None
        self.inference_data = None
        
    def __call__(self) -> npt.NDArray[np.floating]:
        r"""Retrieve or compute the observed data from this node.
        
        If the node has a parent, it will compute the data from the parent.
        Otherwise, it returns the directly stored data.
        
        Returns
        -------
        data : array-like, shape (n, d)
            The data array from this node
            
        Raises
        ------
        ValueError
            If no data is available and no parent to compute from
        """
        if self.parent is None:
            if self.data is None:
                raise ValueError("Data node has no data or parent to compute from.")
            return self.data
        self.parent() # The parent should automatically update its data
        return self.data

    def update(self, data: npt.NDArray[np.floating]):
        r"""Update the observed data stored in this node.

        Parameters
        ----------
        data : array-like, shape (n, d)
            New data to store in the node
        """
        self.data = data
        
    def parametrize(self, a: npt.NDArray[np.floating] = None, b: npt.NDArray[np.floating] = None, data: npt.NDArray[np.floating] = None):
        r"""Set parameters for selective inference process.
        
        Parameters
        ----------
        a : array-like, shape (d,), optional
            Linear intercept parameter
        b : array-like, shape (d,), optional
            Linear coefficient parameter
        data : array-like, shape (n, d), optional
            Inference data to store
        """
        self.a = a
        self.b = b
        self.inference_data = data

    def inference(self, z: float):
        r"""Perform inference computation with given scalar z.
        
        Computes the linear relationship :math:`\mathbf{data} = \mathbf{a} + \mathbf{b} \cdot z`
        and retrieves the feasible interval from parent if available.

        Parameters
        ----------
        z : float
            Parameter value for inference computation
            
        Returns
        -------
        inference_data : array-like, shape (d,)
            Computed inference data
        a : array-like, shape (d,)
            Parametrized intercept
        b : array-like, shape (d,)
            Parametrized coefficient
        interval : list of float
            Feasible interval from parent or [-inf, inf] if no parent
        """
        if self.parent is not None:
            interval = self.parent.inference(z)
        else:
            interval = [-np.inf, np.inf]
            
        if self.a is not None and self.b is not None:
            self.inference_data = self.a + self.b * z
            return self.inference_data, self.a, self.b, interval
        return self.inference_data, self.a, self.b, interval