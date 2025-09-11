import numpy as np
import numpy.typing as npt
from si.node import Data
from typing import Tuple
from si.util import solve_quadratic_inequality, intersect
from scipy.cluster.hierarchy import DisjointSet
import ot


def construct_Theta(ns, nt):
    return np.hstack(
        (
            np.kron(np.identity(ns), np.ones((nt, 1))),
            np.kron(-np.ones((ns, 1)), np.identity(nt)),
        )
    )


def construct_cost(xs, ys, xt, yt):
    xs_squared = np.sum(xs**2, axis=1, keepdims=True)  # shape (n_s, 1)
    xt_squared = np.sum(xt**2, axis=1, keepdims=True).T  # shape (1, n_t)
    cross_term = xs @ xt.T  # shape (n_s, n_t)

    c_ = xs_squared - 2 * cross_term + xt_squared

    ys_squared = np.sum(ys**2, axis=1, keepdims=True)  # shape (n_s, 1)
    yt_squared = np.sum(yt**2, axis=1, keepdims=True).T  # shape (1, n_t)
    cross_term = ys @ yt.T  # shape (n_s, n_t)

    c__ = ys_squared - 2 * cross_term + yt_squared
    c = c_ + c__
    return c_.reshape(-1, 1), c.reshape(-1, 1)


def construct_H(ns, nt):
    Hr = np.zeros((ns, ns * nt))

    for i in range(ns):
        Hr[i : i + 1, i * nt : (i + 1) * nt] = np.ones((1, nt))

    Hc = np.identity(nt)
    for _ in range(ns - 1):
        Hc = np.hstack((Hc, np.identity(nt)))

    H = np.vstack((Hr, Hc))
    H = H[:-1, :]
    return H


def construct_h(ns, nt):
    h = np.vstack((np.ones((ns, 1)) / ns, np.ones((nt, 1)) / nt))
    h = h[:-1, :]
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
        return sorted(B[: ns + nt - 1])

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
    r"""Optimal Transport Domain Adaptation with selective inference support.

    The optimal transport problem solved is:

    .. math::
        \min_{T \in \mathcal{P}} \langle C, T \rangle

    where :math:`\mathcal{P}` is the set of transport plans with given marginals
    and :math:`C` is the cost matrix between domains.

    Attributes
    ----------
    x_source_node : Data or None
        Source domain feature node
    y_source_node : Data or None
        Source domain label node
    x_target_node : Data or None
        Target domain feature node
    y_target_node : Data or None
        Target domain label node
    x_output_node : Data
        Adapted feature output node
    y_output_node : Data
        Adapted label output node
    interval : list or None
        Feasible interval for the last inference call
    x_output_data : array-like or None
        Stored adapted features from last inference call
    y_output_data : tuple or None
        Stored adapted labels from last inference call
    """

    def __init__(self):
        self.x_source_node = None
        self.y_source_node = None
        self.x_target_node = None
        self.y_target_node = None

        self.x_output_node = Data(self)
        self.y_output_node = Data(self)

        self.interval = None
        self.x_output_data = None
        self.y_output_data = None

    def run(
        self,
        xs: npt.NDArray[np.floating],
        ys: npt.NDArray[np.floating],
        xt: npt.NDArray[np.floating],
        yt: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        r"""Configure domain adaptation with input data.

        Parameters
        ----------
        xs : array-like, shape (ns, d)
            Source domain features
        ys : array-like, shape (ns, 1)
            Source domain labels
        xt : array-like, shape (nt, d)
            Target domain features
        yt : array-like, shape (nt, 1)
            Target domain labels

        Returns
        -------
        x_output_node : Data
            Node containing adapted features
        y_output_node : Data
            Node containing adapted labels

        Examples
        --------
        >>> ot_da = OptimalTransportDA()
        >>> x_out, y_out = ot_da.run(xs, ys, xt, yt)
        >>> adapted_x = x_out()
        """
        self.x_source_node = xs
        self.y_source_node = ys
        self.x_target_node = xt
        self.y_target_node = yt
        return self.x_output_node, self.y_output_node

    def forward(
        self,
        xs: npt.NDArray[np.floating],
        ys: npt.NDArray[np.floating],
        xt: npt.NDArray[np.floating],
        yt: npt.NDArray[np.floating],
    ):
        r"""Solve optimal transport and construct adapted dataset.

        Parameters
        ----------
        xs : array-like, shape (ns, d)
            Source domain features
        ys : array-like, shape (ns, 1)
            Source domain labels
        xt : array-like, shape (nt, d)
            Target domain features
        yt : array-like, shape (nt, 1)
            Target domain labels

        Returns
        -------
        x_tilde : array-like, shape (ns+nt, d)
            Adapted feature matrix
        y_tilde : array-like, shape (ns+nt, 1)
            Adapted label vector
        B : list of int
            Basic feasible solution indices
        c_features : array-like, shape (ns*nt, 1)
            Feature space cost matrix
        Omega : array-like, shape (ns+nt, ns+nt)
            Transformation matrix for adaptation

        Notes
        -----
        The adapted dataset is constructed as:

        .. math::
            \tilde{\mathbf{x}} = \Omega \begin{bmatrix} \mathbf{x}_s \\ \mathbf{x}_t \end{bmatrix}

            \tilde{\mathbf{y}} = \Omega \begin{bmatrix} \mathbf{y}_s \\ \mathbf{y}_t \end{bmatrix}

        where :math:`\Omega` incorporates the optimal transport plan.
        """
        x = np.vstack((xs, xt))
        y = np.vstack((ys, yt))

        ns = xs.shape[0]
        nt = xt.shape[0]

        row_mass = np.ones(ns) / ns
        col_mass = np.ones(nt) / nt

        _c, c = construct_cost(xs, ys, xt, yt)
        T, log = ot.emd(a=row_mass, b=col_mass, M=c.reshape(ns, nt), log=True)
        B = np.where(T.reshape(-1) != 0)[0].tolist()

        if len(B) != ns + nt - 1:
            B = construct_B(T, log["u"], log["v"], c.reshape(ns, nt))
        T = T.reshape(ns, nt)
        Omega = np.hstack(
            (np.zeros((ns + nt, ns)), np.vstack((ns * T, np.identity(nt))))
        )
        x_tilde = Omega.dot(x)
        y_tilde = Omega.dot(y)
        return x_tilde, y_tilde, B, _c, Omega

    def __call__(self) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        r"""Execute domain adaptation on stored data.

        Returns
        -------
        x_tilde : array-like, shape (ns+nt, d)
            Adapted feature matrix
        y_tilde : array-like, shape (ns+nt, 1)
            Adapted label vector

        Examples
        --------
        >>> ot_da = OptimalTransportDA()
        >>> # ... set up data nodes ...
        >>> x_adapted, y_adapted = ot_da()
        """
        xs = self.x_source_node()
        ys = self.y_source_node()
        xt = self.x_target_node()
        yt = self.y_target_node()

        x_tilde, y_tilde, _, _, _ = self.forward(xs, ys, xt, yt)
        self.x_output_node.update(x_tilde)
        self.y_output_node.update(y_tilde)
        return x_tilde, y_tilde

    def inference(self, z: float) -> Tuple[list, npt.NDArray[np.floating]]:
        r"""Find feasible interval of the Optimal Transport for the parametrized data at z .

        Parameters
        ----------
        z : float
            Scalar parameter

        Returns
        -------
        final_interval : list
            Feasible interval [lower, upper] for z
        """

        if self.interval is not None and self.interval[0] <= z <= self.interval[1]:
            self.x_output_node.parametrize(data=self.x_output_data)
            self.y_output_node.parametrize(
                a=self.y_output_data[0],
                b=self.y_output_data[1],
                data=self.y_output_data[2],
            )
            return self.interval

        xs, _, _, interval_xs = self.x_source_node.inference(z)
        ys, a_ys, b_ys, interval_ys = self.y_source_node.inference(z)
        xt, _, _, interval_xt = self.x_target_node.inference(z)
        yt, a_yt, b_yt, interval_yt = self.y_target_node.inference(z)

        _, _, B, c_, Omega = self.forward(xs, ys, xt, yt)

        x = np.vstack((xs, xt))
        y = np.vstack((ys, yt))

        a = np.vstack((a_ys, a_yt))
        b = np.vstack((b_ys, b_yt))

        ns = xs.shape[0]
        nt = xt.shape[0]

        Bc = list(set(range(ns * nt)) - set(B))

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
            fa = -r[i][0]
            sa = -q[i][0]
            ta = -p[i][0]

            temp = solve_quadratic_inequality(fa, sa, ta, z)
            final_interval = intersect(final_interval, temp)

        final_interval = intersect(final_interval, interval_xs)
        final_interval = intersect(final_interval, interval_ys)
        final_interval = intersect(final_interval, interval_xt)
        final_interval = intersect(final_interval, interval_yt)

        x_tilde = Omega.dot(x)
        y_tilde = Omega.dot(y)
        a_tilde = Omega.dot(a)
        b_tilde = Omega.dot(b)

        self.x_output_node.parametrize(data=x_tilde)
        self.y_output_node.parametrize(a=a_tilde, b=b_tilde, data=y_tilde)

        self.interval = final_interval
        self.x_output_data = x_tilde
        self.y_output_data = (a_tilde, b_tilde, y_tilde)

        return self.interval
