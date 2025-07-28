# from __future__ import division, print_function

import abc
from collections import namedtuple
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import six
from scipy.spatial import distance as scipy_distance
from scipy.spatial import distance_matrix
from tqdm import tqdm

import scipy.special as spsp
from scipy.spatial import cKDTree


############################################
##############Transformation################
import abc
import itertools

import numpy as np
try:
    import open3d as o3
except:
    class Utility():
        Vector3dVector = np.ndarray
    class geometry():
        PointCloud = np.ndarray
    class o3:
        utility = Utility()
        geometry = geometry()
import six

try:
    from dq3d import op
    from probreg import math_utils as mu
    _imp_dq = True
except:
    _imp_dq = False

@six.add_metaclass(abc.ABCMeta)
class Transformation:
    def __init__(self, ndim=3, xp=np):
        self.xp = xp
        self.ndim = ndim

    def transform(self, points, array_type=o3.utility.Vector3dVector):
        if isinstance(points, array_type):
            _array_after_transform = self._transform(self.xp.asarray(points))

            if self.xp.__name__ == "cupy":
                _array_after_transform = _array_after_transform.get()

            return array_type(_array_after_transform)
        return self.xp.c_[self._transform(points[:, :self.ndim]), points[:, self.ndim:]]

    @abc.abstractmethod
    def _transform(self, points):
        return points

class RigidTransformation(Transformation):
    """Rigid Transformation

    Args:
        rot (numpy.ndarray, optional): Rotation matrix.
        t (numpy.ndarray, optional): Translation vector.
        scale (Float, optional): Scale factor.
        xp (module, optional): Numpy or Cupy.
    """

    def __init__(self, rot=None, t=None, scale=None, ndim=3, xp=np):
        super(RigidTransformation, self).__init__(ndim=ndim, xp=xp)
        self.rot = np.identity(self.ndim)  if rot is  None else rot
        self.t = np.zeros(self.ndim) if t is None else t
        self.scale = 1.0 if scale is None else scale

    def _transform(self, points):
        return self.scale * self.xp.dot(points, self.rot.T) + self.t

    def inverse(self):
        return RigidTransformation(self.rot.T,
                                    -self.xp.dot(self.rot.T, self.t) / self.scale, 1.0 / self.scale,
                                    ndim=self.ndim, xp=self.xp)

    def __mul__(self, other):
        return RigidTransformation(
            self.xp.dot(self.rot, other.rot),
            self.t + self.scale * self.xp.dot(self.rot, other.t),
            self.scale * other.scale,
            ndim=self.ndim, xp=self.xp
        )

class AffineTransformation(Transformation):
    """Affine Transformation

    Args:
        b (numpy.ndarray, optional): Affine matrix.
        t (numpy.ndarray, optional): Translation vector.
        xp (module, optional): Numpy or Cupy.
    """

    def __init__(self, b=None, t=None, ndim=3, xp=np):
        super(AffineTransformation, self).__init__(ndim=ndim, xp=xp)
        self.b = np.identity(self.ndim)  if b is  None else b
        self.t = np.zeros(self.ndim) if t is None else t

    def _transform(self, points):
        return self.xp.dot(points, self.b.T) + self.t

class NonRigidTransformation(Transformation):
    """Nonrigid Transformation

    Args:
        w (numpy.array): Weights for kernel.
        points (numpy.array): Source point cloud data.
        beta (float, optional): Parameter for gaussian kernel.
        xp (module): Numpy or Cupy.
    """

    def __init__(self, w, points, beta=2.0, ndim=3, xp=np):
        super(NonRigidTransformation, self).__init__(ndim=ndim, xp=xp)
        if xp == np:
            self.g = mu.rbf_kernel(points, points, beta)
        else:
            from .. import cupy_utils

            self.g = cupy_utils.rbf_kernel(points, points, beta)
        self.w = w

    def _transform(self, points):
        return points + self.xp.dot(self.g, self.w)

class CombinedTransformation(Transformation):
    """Combined Transformation

    Args:
        rot (numpy.array, optional): Rotation matrix.
        t (numpy.array, optional): Translation vector.
        scale (float, optional): Scale factor.
        v (numpy.array, optional): Nonrigid term.
    """

    def __init__(self, rot=None, t=None, scale=None,  v=0.0, ndim=3, xp=np):
        super(CombinedTransformation, self).__init__(ndim=ndim, xp=xp)

        self.rot = np.identity(self.ndim)  if rot is  None else rot
        self.t = np.zeros(self.ndim) if t is None else t
        self.scale = 1.0 if scale is None else scale

        self.rigid_trans = RigidTransformation(rot, t, scale, ndim=self.ndim, xp=self.xp)
        self.v = v

    def _transform(self, points):
        return self.rigid_trans._transform(points + self.v)

class TPSTransformation(Transformation):
    """Thin Plate Spline transformaion.

    Args:
        a (numpy.array): Affine matrix.
        v (numpy.array): Translation vector.
        control_pts (numpy.array): Control points.
        kernel (function, optional): Kernel function.
    """

    def __init__(self, a, v, control_pts, kernel=None):
        super(TPSTransformation, self).__init__()
        self.a = a
        self.v = v
        self.control_pts = control_pts
        self._kernel = mu.tps_kernel if kernel is None else kernel

    def prepare(self, landmarks):
        control_pts = self.control_pts
        m, d = landmarks.shape
        n, _ = control_pts.shape
        pm = np.c_[np.ones((m, 1)), landmarks]
        pn = np.c_[np.ones((n, 1)), control_pts]
        u, _, _ = np.linalg.svd(pn)
        pp = u[:, d + 1 :]
        kk = self._kernel(control_pts, control_pts)
        uu = self._kernel(landmarks, control_pts)
        basis = np.c_[pm, np.dot(uu, pp)]
        kernel = np.dot(pp.T, np.dot(kk, pp))
        return basis, kernel

    def transform_basis(self, basis):
        return np.dot(basis, np.r_[self.a, self.v])

    def _transform(self, points):
        basis, _ = self.prepare(points)
        return self.transform_basis(basis)

class DeformableKinematicModel(Transformation):
    """Deformable Kinematic Transformation

    Args:
        dualquats (:obj:`list` of :obj:`dq3d.dualquat`): Transformations for each link.
        weights (DeformableKinematicModel.SkinningWeight): Skinning weight.
    """

    class SkinningWeight(np.ndarray):
        """SkinningWeight
                Transformations and weights for each point.

        .       tf = SkinningWeight['val'][0] * dualquats[SkinningWeight['pair'][0]] + SkinningWeight['val'][1] * dualquats[SkinningWeight['pair'][1]]
        """

        def __new__(cls, n_points):
            return super(DeformableKinematicModel.SkinningWeight, cls).__new__(
                cls, n_points, dtype=[("pair", "i4", 2), ("val", "f4", 2)]
            )

        @property
        def n_nodes(self):
            return self["pair"].max() + 1

        def pairs_set(self):
            return itertools.permutations(range(self.n_nodes), 2)

        def in_pair(self, pair):
            """
            Return indices of the pairs equal to the given pair.
            """
            return np.argwhere((self["pair"] == pair).all(1)).flatten()

    @classmethod
    def make_weight(cls, pairs, vals):
        weights = cls.SkinningWeight(pairs.shape[0])
        weights["pair"] = pairs
        weights["val"] = vals
        return weights

    def __init__(self, dualquats, weights):
        if not _imp_dq:
            raise RuntimeError("No dq3d python package, deformable kinematic model not available.")
        super(DeformableKinematicModel, self).__init__()
        self.weights = weights
        self.dualquats = dualquats
        self.trans = [op.dlb(w[1], [self.dualquats[i] for i in w[0]]) for w in self.weights]

    def _transform(self, points):
        return np.array([t.transform_point(p) for t, p in zip(self.trans, points)])


###########################################
#############BCPD##########################


EstepResult = namedtuple("EstepResult", ["nu_d", "nu", "n_p", "px", "x_hat"])
MstepResult = namedtuple("MstepResult", ["transformation", "u_hat", "sigma_mat", "alpha", "sigma2"])
MstepResult.__doc__ = """Result of Maximization step.

    Attributes:
        transformation (Transformation): Transformation from source to target.
        u_hat (numpy.ndarray): A parameter used in next Estep.
        sigma_mat (numpy.ndarray): A parameter used in next Estep.
        alpha (float): A parameter used in next Estep.
        sigma2 (float): Variance of Gaussian distribution.
"""


@six.add_metaclass(abc.ABCMeta)
class BayesianCoherentPointDrift:
    """Bayesian Coherent Point Drift algorithm.

    Args:
        source (numpy.ndarray, optional): Source point cloud data.
    """

    def __init__(self, source=None, ndim=None,xp = np):
        self._source = source
        self._tf_type = None
        self._callbacks = []
        self.ndim = ndim
        self.xp = xp

    def set_source(self, source):
        self._source = source

    def set_callbacks(self, callbacks):
        self._callbacks.extend(callbacks)

    @abc.abstractmethod
    def _initialize(self, target):
        return MstepResult(None, None, None, None, None)

    def expectation_step(self, t_source, target, scale, alpha, sigma_mat, sigma2, w=0.0):
        """Expectation step for BCPD"""
        assert t_source.ndim == 2 and target.ndim == 2, "source and target must have 2 dimensions."
        dim = self.ndim 
        pmat = np.stack([np.sum(np.square(target - ts), axis=1) for ts in t_source])
        pmat = np.exp(-pmat / (2.0 * sigma2))
        pmat /= (2.0 * np.pi * sigma2) ** (dim * 0.5)
        pmat = pmat.T
        pmat *= np.exp(-(scale**2) / (2 * sigma2) * np.diag(sigma_mat) * dim)
        pmat *= (1.0 - w) * alpha
        den = w / target.shape[0] + np.sum(pmat, axis=1)
        den[den == 0] = np.finfo(np.float32).eps
        pmat = np.divide(pmat.T, den)

        nu_d = np.sum(pmat, axis=0)
        nu = np.sum(pmat, axis=1)
        nu_inv = 1.0 / np.kron(nu, np.ones(dim))
        px = np.dot(np.kron(pmat, np.identity(dim)), target.ravel())
        x_hat = np.multiply(px, nu_inv).reshape(-1, dim)
        return EstepResult(nu_d, nu, np.sum(nu), px.reshape(-1, dim), x_hat)

    def maximization_step(self, target, estep_res, sigma2_p=None):
        return self._maximization_step(self._source, target, estep_res, sigma2_p)

    @staticmethod
    @abc.abstractmethod
    def _maximization_step(source, target, estep_res, sigma2_p=None):
        return None

    def registration(self, target, w=0.0, maxiter=50, tol=0.001):
        assert not self._tf_type is None, "transformation type is None."
        res = self._initialize(target)
        target_tree = cKDTree(target, leafsize=10)
        rmse = None
        for i in range(maxiter):
            t_source = res.transformation.transform(self._source)
            estep_res = self.expectation_step(
                t_source, target, res.transformation.rigid_trans.scale, res.alpha, res.sigma_mat, res.sigma2, w
            )
            res = self.maximization_step(target, res.transformation.rigid_trans, estep_res, res.sigma2)
            for c in self._callbacks:
                c(res.transformation)
            tmp_rmse = mu.compute_rmse(t_source, target_tree)
            print("Iteration: {}, Criteria: {}".format(i, tmp_rmse))
            if not rmse is None and abs(rmse - tmp_rmse) < tol:
                break
            rmse = tmp_rmse
        return res.transformation


class CombinedBCPD(BayesianCoherentPointDrift):
    def __init__(self, source=None, lmd=2.0, k=1.0e20, gamma=1.0, **kargs):
        super(CombinedBCPD, self).__init__(source, **kargs)
        self._tf_type = CombinedTransformation
        self.lmd = lmd
        self.k = k
        self.gamma = gamma

    def _initialize(self, target):
        m, dim = self._source.shape
        self.gmat = mu.inverse_multiquadric_kernel(self._source, self._source)
        self.gmat_inv = np.linalg.inv(self.gmat)
        sigma2 = self.gamma * mu.squared_kernel_sum(self._source, target)
        q = 1.0 + target.shape[0] * dim * 0.5 * np.log(sigma2)
        return MstepResult(self._tf_type(np.identity(dim), np.zeros(dim), ndim=dim), None, np.identity(m), 1.0 / m, sigma2)

    def maximization_step(self, target, rigid_trans, estep_res, sigma2_p=None):
        return self._maximization_step(
            self._source, target, rigid_trans, estep_res, self.gmat_inv, self.lmd, self.k, sigma2_p
        )

    def _maximization_step(self, source, target, rigid_trans, estep_res, gmat_inv, lmd, k, sigma2_p=None):
        nu_d, nu, n_p, px, x_hat = estep_res
        dim = self.ndim
        m = source.shape[0]
        s2s2 = rigid_trans.scale**2 / (sigma2_p**2)
        sigma_mat_inv = lmd * gmat_inv + s2s2 * np.diag(nu)
        sigma_mat = np.linalg.inv(sigma_mat_inv)
        residual = rigid_trans.inverse().transform(x_hat) - source
        v_hat = s2s2 * np.matmul(
            np.multiply(np.kron(sigma_mat, np.identity(dim)), np.kron(nu, np.ones(dim))), residual.ravel()
        ).reshape(-1, dim)
        u_hat = source + v_hat
        alpha = np.exp(spsp.psi(k + nu) - spsp.psi(k * m + n_p))
        x_m = np.sum(nu * x_hat.T, axis=1) / n_p
        sigma2_m = np.sum(nu * np.diag(sigma_mat), axis=0) / n_p
        u_m = np.sum(nu * u_hat.T, axis=1) / n_p
        u_hm = u_hat - u_m
        s_xu = np.matmul(np.multiply(nu, (x_hat - x_m).T), u_hm) / n_p
        s_uu = np.matmul(np.multiply(nu, u_hm.T), u_hm) / n_p + sigma2_m * np.identity(dim)
        phi, _, psih = np.linalg.svd(s_xu, full_matrices=True)
        c = np.ones(dim)
        c[-1] = np.linalg.det(np.dot(phi, psih))
        rot = np.matmul(phi * c, psih)
        tr_rsxu = np.trace(np.matmul(rot, s_xu))
        scale = tr_rsxu / np.trace(s_uu)
        t = x_m - scale * np.dot(rot, u_m)
        y_hat = rigid_trans.transform(source + v_hat)
        s1 = np.dot(target.ravel(), np.kron(nu_d, np.ones(dim)) * target.ravel())
        s2 = np.dot(px.ravel(), y_hat.ravel())
        s3 = np.dot(y_hat.ravel(), np.kron(nu, np.ones(dim)) * y_hat.ravel())
        sigma2 = (s1 - 2.0 * s2 + s3) / (n_p * dim) + scale**2 * sigma2_m
        return MstepResult(CombinedTransformation(rot, t, scale, v_hat, ndim=self.ndim, xp=self.xp), 
                           u_hat, sigma_mat, alpha, sigma2)


def registration_bcpd(
    source: Union[np.ndarray, o3.geometry.PointCloud],
    target: Union[np.ndarray, o3.geometry.PointCloud],
    w: float = 0.0,

    maxiter: int = 50,
    tol: float = 0.001,
    callbacks: List[Callable] = [],
    **kwargs: Any,
) -> Transformation:
    """BCPD Registraion.

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
        maxitr (int, optional): Maximum number of iterations to EM algorithm.
        tol (float, optional) : Tolerance for termination.
        callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
            `callback(probreg.Transformation)`

    Returns:
        probreg.Transformation: Estimated transformation.
    """
    cv = lambda x: np.asarray(x.points if isinstance(x, o3.geometry.PointCloud) else x)
    bcpd = CombinedBCPD(cv(source), **kwargs)
    bcpd.set_callbacks(callbacks)
    return bcpd.registration(cv(target), w, maxiter, tol)

