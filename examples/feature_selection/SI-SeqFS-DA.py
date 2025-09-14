r"""
SI for Sequential Feature Selection after Optimal Transport-based Domain Adaptation
========================================================================
"""

# Author: Duong Tan Loc

from si import Pipeline
from si.feature_selection import SequentialFeatureSelection
from si import Data
from si.test_statistics import SFS_DATestStatistic
from si.domain_adaptation import OptimalTransportDA
import numpy as np
import matplotlib.pyplot as plt

# %%
# Define the pipeline
# -------------------

def SI_SeqFS_DA(k) -> Pipeline:
    xs = Data()
    ys = Data() 
    
    xt = Data()
    yt = Data()
    
    OT = OptimalTransportDA()
    x_tilde, y_tilde = OT.run(xs=xs, ys=ys, xt=xt, yt=yt)

    seqfs = SequentialFeatureSelection(k,direction="forward")
    active_set = seqfs.run(x_tilde, y_tilde)
    return Pipeline(inputs=(xs, ys, xt, yt), output=active_set, test_statistic=SFS_DATestStatistic(xs=xs, ys=ys, xt=xt, yt=yt))

my_pipeline = SI_SeqFS_DA(3)

# %%
# Generate data
# -------------- 

def gen_data(n, p, true_beta):
    x = np.random.normal(loc = 0, scale = 1, size = (n, p))
    true_beta = true_beta.reshape(-1, 1)
    
    mu = x.dot(true_beta)
    Sigma = np.identity(n)
    Y = mu + np.random.normal(loc = 0, scale = 1, size = (n, 1))
    return x, Y, Sigma

xs, ys, sigma_s = gen_data(150, 5, np.asarray([0, 0, 0, 0, 0]))
xt, yt, sigma_t = gen_data(25, 5, np.asarray([0, 0, 0, 0, 0]))

# %%
# Run the pipeline
# -----------------

selected_features, p_values = my_pipeline(inputs=[xs, ys, xt, yt], covariances=[sigma_s, sigma_t])

print("Selected features: ", selected_features)
print("P-values: ", p_values)

# %%
# Plot the p-values
plt.figure()
plt.bar(range(len(p_values)), p_values)
plt.xlabel("Feature index")
plt.ylabel("P-value")
plt.show()