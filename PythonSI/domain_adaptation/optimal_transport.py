import numpy as np 
import numpy.typing as npt
from ..node import Data
from typing import Tuple
from ..util import solve_linear_inequalities, solve_quadratic_inequality, intersect
from scipy.cluster.hierarchy import DisjointSet
import ot

def construct_Theta(ns, nt):
    return np.hstack((np.kron(np.identity(ns), np.ones((nt, 1))), np.kron(- np.ones((ns, 1)), np.identity(nt))))

def construct_cost(Xs, ys, Xt, yt):
    Xs_squared = np.sum(Xs**2, axis=1, keepdims=True)  # shape (n_s, 1)
    Xt_squared = np.sum(Xt**2, axis=1, keepdims=True).T  # shape (1, n_t)
    cross_term = Xs @ Xt.T  # shape (n_s, n_t)

    c_ = Xs_squared - 2 * cross_term + Xt_squared

    ys_squared = np.sum(ys**2, axis=1, keepdims=True)  # shape (n_s, 1)
    yt_squared = np.sum(yt**2, axis=1, keepdims=True).T  # shape (1, n_t)
    cross_term = ys @ yt.T  # shape (n_s, n_t)

    c__ = ys_squared - 2 * cross_term + yt_squared
    c = c_ + c__
    return c_.reshape(-1,1), c.reshape(-1,1)

def construct_H(ns, nt):
    Hr = np.zeros((ns, ns * nt))
    
    for i in range(ns):
        Hr[i:i+1, i*nt:(i+1)*nt] = np.ones((1, nt))
        
    Hc = np.identity(nt)
    for _ in range(ns - 1):
        Hc = np.hstack((Hc, np.identity(nt)))

    H = np.vstack((Hr, Hc))
    H = H[:-1,:]
    return H

def construct_h(ns, nt):
    h = np.vstack((np.ones((ns, 1)) / ns, np.ones((nt, 1)) / nt))
    h = h[:-1,:]
    return h

def construct_B(T, u, v, c):
    ns, nt = T.shape
    DJ = DisjointSet(range(ns + nt))
    B = []

    # Vectorized first loop - process elements where T > 0
    large_T_indices = np.where(T > 0)
    for i, j in zip(large_T_indices[0], large_T_indices[1]):
        DJ.merge(i, j + ns)
        B.append(i * nt + j)
    
    # Early exit if we already have enough elements
    if len(B) >= ns + nt - 1:
        return sorted(B[:ns + nt - 1])
    
    # Vectorized computation of reduced costs
    rc = c - u[:, np.newaxis] - v[np.newaxis, :]
    
    # Find candidates with smallest |rc|
    flat_rc = np.abs(rc).flatten()
    sorted_indices = np.argsort(flat_rc)
    
    # Process candidates in order of smallest reduced cost
    for idx in sorted_indices:
        i, j = divmod(idx, nt)
        if len(B) >= ns + nt - 1:
            break
        if not DJ.connected(i, j + ns):
            DJ.merge(i, j + ns)
            B.append(i * nt + j)
    
    return sorted(B)

class OptimalTransportDA:
    def __init__(self):
        self.X_source_node = None 
        self.y_source_node = None
        self.X_target_node = None
        self.y_target_node = None
        
        self.X_output_node = Data(self)
        self.y_output_node = Data(self)

    def run(self, Xs: npt.NDArray[np.floating], ys: npt.NDArray[np.floating], Xt: npt.NDArray[np.floating], yt: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        self.X_source_node = Xs
        self.y_source_node = ys
        self.X_target_node = Xt
        self.y_target_node = yt
        return self.X_output_node, self.y_output_node

    def forward(self, Xs: npt.NDArray[np.floating], ys: npt.NDArray[np.floating], Xt: npt.NDArray[np.floating], yt: npt.NDArray[np.floating]):
        X = np.vstack((Xs, Xt))
        y = np.vstack((ys, yt))
        
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        
        row_mass = np.ones(ns) / ns
        col_mass = np.ones(nt) / nt
        
        _c, c = construct_cost(Xs, ys, Xt, yt)
        T, log = ot.emd(a=row_mass, b=col_mass, M=c.reshape(ns, nt), log=True)
        B = np.where(T.reshape(-1) != 0)[0].tolist()

        if len(B) != ns + nt - 1:
            B = construct_B(T, log['u'], log['v'], c.reshape(ns, nt))
        T = T.reshape(ns, nt)
        Omega = np.hstack((np.zeros((ns + nt, ns)), np.vstack((ns * T, np.identity(nt)))))
        X_tilde = Omega.dot(X)
        y_tilde = Omega.dot(y)
        return X_tilde, y_tilde, B, _c, Omega

    def __call__(self) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        Xs = self.X_source_node()
        ys = self.y_source_node()
        Xt = self.X_target_node()
        yt = self.y_target_node()
        
        X, y, _, _, _ = self.forward(Xs, ys, Xt, yt)
        self.X_output_node.update(X)
        self.y_output_node.update(y)
        return X, y
    
    def inference(self, z: float) -> Tuple[list, npt.NDArray[np.floating]]:
        Xs, _, _, interval_Xs = self.X_source_node.inference(z)
        ys, a_ys, b_ys, interval_ys = self.y_source_node.inference(z)
        Xt, _, _, interval_Xt = self.X_target_node.inference(z)
        yt, a_yt, b_yt, interval_yt = self.y_target_node.inference(z)
        
        _, _, B, c_, Omega = self.forward(Xs, ys, Xt, yt)

        X = np.vstack((Xs, Xt))
        y = np.vstack((ys, yt))

        a = np.vstack((a_ys, a_yt))
        b = np.vstack((b_ys, b_yt))
        
        ns = Xs.shape[0]
        nt = Xt.shape[0]

        Bc = list(set(range(ns*nt))-set(B))
        
        H = construct_H(ns, nt)
        
        Theta = construct_Theta(ns, nt)
        Theta_a = Theta.dot(a)
        Theta_b = Theta.dot(b)
        
        p_tilde = c_ + Theta_a * Theta_a
        q_tilde = 2 * Theta_a * Theta_b
        r_tilde = Theta_b * Theta_b

        HB_invHBc = np.linalg.inv(H[:, B]).dot(H[:, Bc])

        p = (p_tilde[Bc, :].T - p_tilde[B, :].T.dot(HB_invHBc)).T
        q = (q_tilde[Bc, :].T - q_tilde[B, :].T.dot(HB_invHBc)).T
        r = (r_tilde[Bc, :].T - r_tilde[B, :].T.dot(HB_invHBc)).T
        
        final_interval = [-np.inf, np.inf]

        for i in range(p.shape[0]):
            fa = - r[i][0]
            sa = - q[i][0]
            ta = - p[i][0]

            temp = solve_quadratic_inequality(fa, sa, ta, z)
            final_interval = intersect(final_interval, temp)
            
        final_interval = intersect(final_interval, interval_Xs)
        final_interval = intersect(final_interval, interval_ys)
        final_interval = intersect(final_interval, interval_Xt)
        final_interval = intersect(final_interval, interval_yt)
        
        self.X_output_node.parametrize(data=Omega.dot(X))
        self.y_output_node.parametrize(a=Omega.dot(a), b=Omega.dot(b), data=Omega.dot(y))
        return final_interval
