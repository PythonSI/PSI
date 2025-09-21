from networkx import sigma
import numpy as np
import numpy.typing as npt
from typing import Tuple, List
from scipy.linalg import block_diag
from pythonsi.node.data import Data
from pythonsi.util import intersect, solve_linear_inequalities


class AD_DATestStatistic:
    def __init__(
        self,
        xs: Data,
        xt: Data,
    ):
        self.xs_node = xs
        self.xt_node = xt

    def __call__(
        self,
        anomalies: npt.NDArray[np.floating],
        anomaly_idx: int,
        Sigmas: List[npt.NDArray[np.floating]],
    ) -> Tuple[
        list, npt.NDArray[np.floating], npt.NDArray[np.floating], float, float, list
    ]:
        xs = self.xs_node()
        xt = self.xt_node()

        ns = xs.shape[0]
        nt = xt.shape[0]
        d = xs.shape[1]

        sigma_s = Sigmas[0]
        sigma_t = Sigmas[1]
        sigma = block_diag(sigma_s, sigma_t)

        x = np.vstack(
            (xs.flatten().reshape((ns * d, 1)), xt.flatten().reshape((nt * d, 1)))
        )
        yt_hat = np.zeros(xt.shape[0])
        yt_hat[anomalies] = 1
        Oc = list(np.where(yt_hat == 0)[0])

        etj = np.zeros((nt * d, 1))
        for i in range(d):
            etj[anomaly_idx * d + i] = 1
        etOc = np.zeros((nt * d, 1))
        for i in Oc:
            for k in range(d):
                etOc[i * d + k] = 1

        s = np.zeros((ns * d + nt * d, 1))
        for i in range(d):
            testj = xt[anomaly_idx, i]
            testOc = (1 / len(Oc)) * np.sum(xt[Oc[k], i] for k in range(len(Oc)))
            if np.sign(testj - testOc) == -1:
                etj[anomaly_idx * d + i] = -1
                for k in Oc:
                    etOc[k * d + i] = -1
        etaj = np.vstack((np.zeros((ns * d, 1)), etj - (1 / len(Oc)) * etOc))
        etajTx = etaj.T.dot(x)
        etajTsigmaetaj = etaj.T.dot(sigma).dot(etaj)

        b = sigma.dot(etaj).dot(np.linalg.inv(etajTsigmaetaj))
        a = (np.identity(ns * d + nt * d) - b.dot(etaj.T)).dot(x)

        itv = [-np.inf, np.inf]
        j = anomaly_idx
        for i in range(d):
            testj = xt[j, i]
            testOc = (1 / len(Oc)) * np.sum(xt[Oc[k], i] for k in range(len(Oc)))
            if (testj - testOc) < 1e-9 and (testj - testOc) > -1e-9:
                # print(f'Feature {i+1} is constant for anomaly index {j}, skipping...')
                continue
            if (testj - testOc) < 0:
                itv = intersect(
                    itv,
                    solve_linear_inequalities(
                        a[j * d + i + ns * d]
                        - (1 / len(Oc))
                        * np.sum(a[Oc[k] * d + i + ns * d] for k in range(len(Oc))),
                        b[j * d + i + ns * d]
                        - (1 / len(Oc))
                        * np.sum(b[Oc[k] * d + i + ns * d] for k in range(len(Oc))),
                    ),
                )
                # print(solve_linear_inequality(a[j * d + i + ns * d] - (1/len(Oc))*np.sum(a[Oc[k] * d + i + ns * d] for k in range(len(Oc))), b[j * d + i + ns * d] - (1/len(Oc))*np.sum(b[Oc[k] * d + i + ns * d] for k in range(len(Oc)))))
            else:
                itv = intersect(
                    itv,
                    solve_linear_inequalities(
                        -a[j * d + i + ns * d]
                        + (1 / len(Oc))
                        * np.sum(a[Oc[k] * d + i + ns * d] for k in range(len(Oc))),
                        -b[j * d + i + ns * d]
                        + (1 / len(Oc))
                        * np.sum(b[Oc[k] * d + i + ns * d] for k in range(len(Oc))),
                    ),
                )
        self.xs_node.parametrize(
            a=a[: ns * d].reshape(-1, d), b=b[: ns * d].reshape(-1, d)
        )
        self.xt_node.parametrize(
            a=a[ns * d :].reshape(-1, d), b=b[ns * d :].reshape(-1, d)
        )
        return (
            etaj,
            a,
            b,
            etajTx[0][0],
            etajTsigmaetaj[0][0],
            np.sqrt(etajTsigmaetaj[0][0]),
            itv,
        )
