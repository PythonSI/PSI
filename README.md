# PythonSI: Python Selective Inference

[![PyPI version](https://badge.fury.io/py/pyselectinf.svg)](https://badge.fury.io/py/pyselectinf)
[![License](https://anaconda.org/conda-forge/pot/badges/license.svg)](https://github.com/PythonSI/PySelectInf/blob/master/LICENSE)

This open source Python library provides APIs for selective inference for problems in machine learning such as feature selection, anomaly detection and domain adaptation.

Website and documentation: [https://pythonsi.github.io/](https://pythonsi.github.io/)

Source code (MIT): [https://github.com/PythonSI/PySelectInf](https://github.com/PythonSI/PySelectInf)

## Implemented Features

PythonSI have provide selective inference support for methods:

* Feature Selection:
    * Lasso Feature Selection
    * Sequential Feature Selection
* Domain Adaptation:
    * Optimal Transport-based Domain Adaptation

## Installation

The library has only been tested on Windows with Python 3.10. It requires some of the following modules:
- numpy (==2.2.6)
- mpmath (==1.3.0)
- POT (==0.9.5)
- scikit-learn (==1.7.1)
- scipy (==1.15.3)
- skglm (==0.5)


Note: Other versions of Python and dependencies shall be tested in the future.

### Pip Installation

You can install the toolbox through PyPI with:

```console
pip install pyselectinf
```

### Post installation check
After a correct installation, you should be able to import the module without errors:

```python
import pythonsi
```

Note that for easier access the module is named `pythonsi` instead of `pyselectinf`.

## Examples and Notebooks

The examples folder contain several examples and use case for the library. The full documentation with examples and output is available on [https://PythonSI.github.io/](https://PythonSI.github.io/).

## References

[1] Le Duy, V. N., & Takeuchi, I. (2021, March). Parametric programming approach for more powerful and general lasso selective inference. In International conference on artificial intelligence and statistics (pp. 901-909). PMLR.

[2] Tibshirani, R. J., Taylor, J., Lockhart, R., & Tibshirani, R. (2016). Exact post-selection inference for sequential regression procedures. Journal of the American Statistical Association, 111(514), 600-620.

[3] Loi, N. T., Loc, D. T., & Duy, V. N. L. (2025). "Statistical Inference for Feature Selection after Optimal Transport-based Domain Adaptation." In International Conference on Artificial Intelligence and Statistics, pp. 1747-1755. PMLR, 2025.

[4] Li, S., Cai, T. T., & Li, H. (2022). Transfer learning for high-dimensional linear regression: Prediction, estimation and minimax optimality. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(1), 149-173.

[5] Tam, N. V. K., My, C. H., & Duy, V. N. L. (2025). Post-Transfer Learning Statistical Inference in High-Dimensional Regression. arXiv preprint arXiv:2504.18212.

[6] He, Z., Sun, Y., & Li, R. (2024, April). Transfusion: Covariate-shift robust transfer learning for high-dimensional regression. In International Conference on Artificial Intelligence and Statistics (pp. 703-711). PMLR.

[7] Kiet, T. T., Loi, N. T., & Duy, V. N. L. (2025). Statistical inference for autoencoder-based anomaly detection after representation learning-based domain adaptation. arXiv preprint arXiv:2508.07049.