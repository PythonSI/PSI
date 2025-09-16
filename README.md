# PSI: Python Selective Inference

[![PyPI version](https://badge.fury.io/py/PSI.svg)](https://badge.fury.io/py/PSI)
[![License](https://anaconda.org/conda-forge/pot/badges/license.svg)](https://github.com/PythonSI/PSI/blob/master/LICENSE)

This open source Python library provides APIs for selective inference for problems in machine learning such as feature selection, anomaly detection and domain adaptation.

Website and documentation: [https://pythonsi.github.io/](https://pythonsi.github.io/)

Source code (MIT): [https://github.com/PythonSI/PSI](https://github.com/PythonSI/PSI)

## Implemented Features

PSI have provide selective inference support for methods:

* Feature Selection:
    * Lasso Feature Selection
    * Sequential Feature Selection
* Domain Adaptation:
    * Optimal Transport-based Domain Adaptation

## Installation

### Pip Installation

You can install the toolbox through PyPI with:

```console
pip install PSI
```

### Post installation check
After a correct installation, you should be able to import the module without errors:

```python
import si
```

Note that for easier access the module is named `si` instead of `psi`.

## Examples and Notebooks

The examples folder contain several examples and use case for the library. The full documentation with examples and output is available on [https://PythonSI.github.io/](https://PythonSI.github.io/).