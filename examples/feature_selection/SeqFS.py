r"""
Selective inference for Sequential Feature Selection
====================================================
"""

# Author: Duong Tan Loc

from pythonsi import Pipeline
from pythonsi.feature_selection import SequentialFeatureSelection
from pythonsi import Data
from pythonsi.test_statistics import FSTestStatistic
import numpy as np
import matplotlib.pyplot as plt

# %%
# Define the pipeline
# -------------------


def SeqFS(k, sigma=None) -> Pipeline:
    x = Data()
    y = Data()

    seqfs = SequentialFeatureSelection(
        n_features_to_select=k, direction="forward", criterion=None
    )
    active_set = seqfs.run(x, y, sigma)
    return Pipeline(
        inputs=(x, y), output=active_set, test_statistic=FSTestStatistic(x=x, y=y)
    )


# %%
# Generate data
# --------------


def gen_data(n, p, true_beta):
    x = np.random.normal(loc=0, scale=1, size=(n, p))
    true_beta = true_beta.reshape(-1, 1)

    mu = x.dot(true_beta)
    Sigma = np.identity(n)
    Y = mu + np.random.normal(loc=0, scale=1, size=(n, 1))
    return x, Y, Sigma


x, y, sigma = gen_data(150, 5, np.asarray([0, 0, 0, 0, 0]))
k = 2
my_pipeline2 = SeqFS(k, sigma=sigma)

# %%
# Run the pipeline
# -----------------

selected_features, p_values = my_pipeline2([x, y], sigma)
print("Selected features: ", selected_features)
print("P-values: ", p_values)

# %%
# Plot the p-values
plt.figure()
plt.bar(range(len(p_values)), p_values)
plt.xlabel("Feature index")
plt.ylabel("P-value")
plt.ylim((0, 1.0))
plt.show()
