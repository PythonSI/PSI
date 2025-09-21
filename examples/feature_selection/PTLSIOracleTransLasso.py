r"""
Selective inference for Oracle TransLasso Feature Selection
================================================
"""

# Author: Nguyen Vu Khai Tam & Cao Huyen My

from pythonsi import Pipeline
from pythonsi.transfer_learning_hdr import TLOracleTransLasso
from pythonsi import Data
from pythonsi.test_statistics import TLHDRTestStatistic
import numpy as np
import matplotlib.pyplot as plt


# %%
# Generate data
def generate_coef(p, s, true_beta=0.25, num_info_aux=3, num_uninfo_aux=2, gamma=0.01):
    K = num_info_aux + num_uninfo_aux
    beta_0 = np.concatenate([np.full(s, true_beta), np.zeros(p - s)])

    Beta_S = np.tile(beta_0, (K, 1)).T
    if s >= 0:
        Beta_S[0, :] -= 2 * true_beta
        for m in range(K):
            if m < num_uninfo_aux:
                Beta_S[:50, m] += np.random.normal(0, true_beta * gamma * 10, 50)
            else:
                Beta_S[:25, m] += np.random.normal(0, true_beta * gamma, 25)
    return Beta_S, beta_0


def generate_data(
    p, s, nS, nT, true_beta=0.25, num_info_aux=3, num_uninfo_aux=2, gamma=0.01
):
    K = num_info_aux + num_uninfo_aux

    Beta_S, beta_0 = generate_coef(p, s, true_beta, num_info_aux, num_uninfo_aux, gamma)
    Beta = np.column_stack([Beta_S[:, i] for i in range(K)] + [beta_0])

    X_list = []
    Y_list = []

    cov = np.eye(p)
    N_vec = [nS] * K + [nT]

    for k in range(K + 1):
        Xk = np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=N_vec[k])
        true_Yk = Xk @ Beta[:, k]
        noise = np.random.normal(0, 1, N_vec[k])
        # noise = np.random.laplace(0, 1, N_vec[k])
        # noise = skewnorm.rvs(a=10, loc=0, scale=1, size=N_vec[k])
        # noise = np.random.standard_t(df=20, size=N_vec[k])
        Yk = true_Yk + noise
        X_list.append(Xk)
        Y_list.append(Yk.reshape(-1, 1))

    XS_list = np.array(X_list[:-1])
    YS_list = np.array(Y_list[:-1]).reshape(-1, 1)
    X0 = X_list[-1]
    Y0 = Y_list[-1]
    SigmaS_list = np.array([np.eye(nS) for _ in range(K)])
    Sigma0 = np.eye(nT)

    return XS_list, YS_list, X0, Y0, SigmaS_list, Sigma0


# %% Define hyper-parameters
p = 100
s = 10
true_beta = 1
gamma = 0.1
nS = 50
nT = 50
num_uninfo_aux = 0
num_info_aux = 5
K = num_info_aux + num_uninfo_aux
nI = nS * K
lambda_w = np.sqrt(np.log(p) / nI) * 4
lambda_del = np.sqrt(np.log(p) / nT) * 2


# %% Define pipeline
def PTL_SI_OTL() -> Pipeline:
    XI_list = Data()
    YI_list = Data()
    X0 = Data()
    Y0 = Data()
    SigmaI_list = Data()
    Sigma0 = Data()

    otl = TLOracleTransLasso(lambda_w, lambda_del)
    active_set = otl.run(XI_list, YI_list, X0, Y0)
    return Pipeline(
        inputs=(XI_list, YI_list, X0, Y0, SigmaI_list, Sigma0),
        output=active_set,
        test_statistic=TLHDRTestStatistic(
            XS_list=XI_list, YS_list=YI_list, X0=X0, Y0=Y0
        ),
    )


my_pipeline = PTL_SI_OTL()


# %% Run the pipeline
XI_list, YI_list, X0, Y0, SigmaI_list, Sigma0 = generate_data(
    p, s, nS, nT, true_beta, num_info_aux, num_uninfo_aux, gamma
)
selected_features, p_values = my_pipeline(
    inputs=[XI_list, YI_list, X0, Y0], covariances=[SigmaI_list, Sigma0], verbose=True
)
print("Selected features: ", selected_features)
print("P-values: ", p_values)

# %% Plot p-values
plt.figure()
plt.bar(range(len(p_values)), p_values)
plt.xlabel("Feature index")
plt.ylabel("P-value")
plt.show()
