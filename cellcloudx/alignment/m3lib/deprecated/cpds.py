# from __future__ import division, print_function

import abc
from collections import namedtuple
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Union

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
from scipy.spatial import distance as scipy_distance
from scipy.spatial import distance_matrix
from tqdm import tqdm

from probreg import math_utils as mu


############################################
##############Transformation################
import abc
import itertools

import numpy as np
import six

try:
    from dq3d import op

    _imp_dq = True
except:
    _imp_dq = False


@six.add_metaclass(abc.ABCMeta)
class Transformation:
    def __init__(self, ndim=3, xp=np):
        self.xp = xp
        self.ndim = ndim

    def transform(self, points, array_type=o3.utility.Vector3dVector):
        # if isinstance(points, array_type):
        #     _array_after_transform = self._transform(self.xp.asarray(points))

        #     if self.xp.__name__ == "cupy":
        #         _array_after_transform = _array_after_transform.get()

        #     return array_type(_array_after_transform)

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
        return RigidTransformation(self.rot.T, -self.xp.dot(self.rot.T, self.t) / self.scale, 1.0 / self.scale)

    def __mul__(self, other):
        return RigidTransformation(
            self.xp.dot(self.rot, other.rot),
            self.t + self.scale * self.xp.dot(self.rot, other.t),
            self.scale * other.scale,
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
            from ... import cupy_utils

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

    def __init__(self, rot=None, t=None, scale=None,  ndim=3, xp=np, v=0.0):
        super(CombinedTransformation, self).__init__(ndim=ndim, xp=xp)

        self.rot = np.identity(self.ndim)  if rot is  None else rot
        self.t = np.zeros(self.ndim) if t is None else t
        self.scale = 1.0 if scale is None else scale

        self.rigid_trans = RigidTransformation(rot, t, scale)
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

    def __init__(self, a, v, control_pts, kernel=mu.tps_kernel):
        super(TPSTransformation, self).__init__()
        self.a = a
        self.v = v
        self.control_pts = control_pts
        self._kernel = kernel

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

###############################################
##################CPD##########################

EstepResult = namedtuple("EstepResult", ["pt1", "p1", "px", "n_p"])
MstepResult = namedtuple("MstepResult", ["transformation", "sigma2", "q"])
MstepResult.__doc__ = """Result of Maximization step.

    Attributes:
        transformation (Transformation): Transformation from source to target.
        sigma2 (float): Variance of Gaussian distribution.
        q (float): Result of likelihood.
"""


class DistModule:
    def __init__(self, xp):
        self.xp = xp

    def cdist(self, x1, x2, metric):
        return self.xp.stack([self.xp.sum(self.xp.square(x2 - ts), axis=1) for ts in x1])


@six.add_metaclass(abc.ABCMeta)
class CoherentPointDrift:
    """Coherent Point Drift algorithm.
    This is an abstract class.
    Based on this class, it is inherited by rigid, affine, nonrigid classes
    according to the type of transformation.
    In this class, Estimation step in EM algorithm is implemented and
    Maximazation step is implemented in the inherited classes.

    Args:
        source (numpy.ndarray, optional): Source point cloud data.
        use_color (bool, optional): Use color information (if available).
        use_cuda (bool, optional): Use CUDA.
    """

    def __init__(self, source: Optional[np.ndarray] = None,
                 ndim : int = 3,
                 use_color: bool = False, use_cuda: bool = False) -> None:

        F = source.shape[1]
        self._N_DIM = ndim
        self._N_COLOR = F - self._N_DIM
        self._source = source
        self._tf_type = None
        self._callbacks = []
        self._use_color = use_color
        if use_cuda:
            import cupy as cp

            from ... import cupy_utils

            self.xp = cp
            self.distance_module = DistModule(cp)
            self.cupy_utils = cupy_utils
            self._squared_kernel_sum = cupy_utils.squared_kernel_sum
        else:
            self.xp = np
            self.distance_module = scipy_distance
            self._squared_kernel_sum = mu.squared_kernel_sum

    def set_source(self, source: np.ndarray) -> None:
        self._source = source

    def set_callbacks(self, callbacks: List[Callable]) -> None:
        self._callbacks.extend(callbacks)

    @abc.abstractmethod
    def _initialize(self, target: np.ndarray) -> MstepResult:
        return MstepResult(None, None, None)

    def _compute_pmat_numerator(self, t_source: np.ndarray, target: np.ndarray, sigma2: float) -> np.ndarray:
        pmat = self.distance_module.cdist(t_source, target, "sqeuclidean")
        # pmat = distance_matrix(t_source, target, p=2)
        # pmat = pmat** 2
        pmat = self.xp.exp(-pmat / (2.0 * sigma2))
        return pmat

    def expectation_step(
        self,
        t_source: np.ndarray,
        target: np.ndarray,
        sigma2: float,
        sigma2_c: float,
        w: float = 0.0,
    ) -> EstepResult:
        """Expectation step for CPD"""
        assert t_source.ndim == 2 and target.ndim == 2, "source and target must have 2 dimensions."
        pmat = self._compute_pmat_numerator(t_source[:, : self._N_DIM], target[:, : self._N_DIM], sigma2)

        c = (2.0 * np.pi * sigma2) ** (self._N_DIM * 0.5)
        c *= w / (1.0 - w) * t_source.shape[0] / target.shape[0]
        den = self.xp.sum(pmat, axis=0)
        den[den == 0] = self.xp.finfo(np.float32).eps
        if self._use_color:
            pmat_c = self._compute_pmat_numerator(t_source[:, self._N_DIM :], target[:, self._N_DIM :], sigma2_c)
            den_c = self.xp.sum(pmat_c, axis=0)
            den_c[den_c == 0] = self.xp.finfo(np.float32).eps
            den = np.multiply(den, den_c)
            o_c = t_source.shape[0] * (2 * np.pi * sigma2_c) ** (0.5 * (self._N_DIM + self._N_COLOR - 1))
            o_c *= self.xp.exp(
                -1.0 / t_source.shape[0] * self.xp.square(self.xp.sum(pmat_c, axis=0)) / (2.0 * sigma2_c)
            )
            den += o_c
            c *= (2.0 * np.pi * sigma2_c) ** (self._N_COLOR * 0.5)
            pmat = self.xp.multiply(pmat, pmat_c)
        den += c

        pmat = self.xp.divide(pmat, den)
        pt1 = self.xp.sum(pmat, axis=0)
        p1 = self.xp.sum(pmat, axis=1)
        px = self.xp.dot(pmat, target[:, : self._N_DIM])

        self.Np = np.sum(pt1)
        self.w = 1-self.Np/ target.shape[0]
        self.w1 = 1-self.Np/ t_source.shape[0]
        return EstepResult(pt1, p1, px, np.sum(p1))

    def maximization_step(
        self, target: np.ndarray, estep_res: EstepResult, sigma2_p: Optional[float] = None
    ) -> Optional[MstepResult]:
        return self._maximization_step(
            self._source[:, : self._N_DIM], target[:, : self._N_DIM], estep_res, sigma2_p, xp=self.xp
        )

    @staticmethod
    @abc.abstractmethod
    def _maximization_step(
        source: np.ndarray,
        target: np.ndarray,
        estep_res: EstepResult,
        sigma2_p: Optional[float] = None,
        xp: ModuleType = np,
    ) -> Optional[MstepResult]:
        return None

    def registration(self, target: np.ndarray, w: float = 0.0, maxiter: int = 50, tol: float = 0.001) -> MstepResult:
        assert not self._tf_type is None, "transformation type is None."
        res = self._initialize(target[:, : self._N_DIM])
        sigma2_c = 0.0
        if self._use_color:
            sigma2_c = self._squared_kernel_sum(self._source[:, self._N_DIM :], target[:, self._N_DIM :])
        q = res.q
        # for i in range(maxiter):
        pbar = tqdm(range(maxiter), total=maxiter)
        for i in pbar:
            t_source = res.transformation.transform(self._source)
            estep_res = self.expectation_step(t_source, target, res.sigma2, sigma2_c, w)
            res = self.maximization_step(target, estep_res, res.sigma2)
            for c in self._callbacks:
                c(res.transformation)

            pbar.set_postfix({'Iteration': i, 'Criteria': res.q, 'Q':res.q - q, 
                              'sigma2': res.sigma2,
                              'w':self.w, 'w1':self.w1})
            # print("Iteration: {}, Criteria: {}".format(i, res.q))
            if abs(res.q - q) < tol:
                break
            q = res.q
        pbar.close()
        return res


class RigidCPD(CoherentPointDrift):
    """Coherent Point Drift for rigid transformation.

    Args:
        source (numpy.ndarray, optional): Source point cloud data.
        update_scale (bool, optional): If this flag is True, compute the scale parameter.
        tf_init_params (dict, optional): Parameters to initialize transformation.
        use_color (bool, optional): Use color information (if available).
        use_cuda (bool, optional): Use CUDA.
    """

    def __init__(
        self,
        source: Optional[np.ndarray] = None,
        ndim: int = 3,
        update_scale: bool = True,
        tf_init_params: Dict = {},
        use_color: bool = False,
        use_cuda: bool = False,
    ) -> None:
        super(RigidCPD, self).__init__(source, ndim, use_color, use_cuda)
        self._tf_type = RigidTransformation
        self._update_scale = update_scale
        self._tf_init_params = tf_init_params

    def _initialize(self, target: np.ndarray) -> MstepResult:
        dim = self._N_DIM
        sigma2 = self._squared_kernel_sum(self._source, target)
        q = 1.0 + target.shape[0] * dim * 0.5 * np.log(sigma2)
        if len(self._tf_init_params) == 0:
            self._tf_init_params = {"rot": self.xp.identity(dim), "t": self.xp.zeros(dim)}
        if not "xp" in self._tf_init_params:
            self._tf_init_params["xp"] = self.xp

        self._tf_init_params["ndim"] = self._N_DIM
        return MstepResult(self._tf_type(**self._tf_init_params), sigma2, q)

    def maximization_step(
        self, target: np.ndarray, estep_res: EstepResult, sigma2_p: Optional[float] = None
    ) -> MstepResult:
        return self._maximization_step(
            self._source[:, : self._N_DIM], target[:, : self._N_DIM], estep_res, sigma2_p, self._update_scale, self.xp
        )

    @staticmethod
    def _maximization_step(
        source: np.ndarray,
        target: np.ndarray,
        estep_res: EstepResult,
        sigma2_p: Optional[float] = None,
        update_scale: bool = True,
        xp: ModuleType = np,
    ) -> MstepResult:
        pt1, p1, px, n_p = estep_res
        # dim = CoherentPointDrift._N_DIM
        dim = source.shape[1]
        mu_x = xp.sum(px, axis=0) / n_p
        mu_y = xp.dot(source.T, p1) / n_p
        target_hat = target - mu_x
        source_hat = source - mu_y
        a = xp.dot(px.T, source_hat) - xp.outer(mu_x, xp.dot(p1.T, source_hat))
        u, _, vh = np.linalg.svd(a, full_matrices=True)
        c = xp.ones(dim)
        c[-1] = xp.linalg.det(xp.dot(u, vh))
        rot = xp.dot(u * c, vh)
        tr_atr = np.trace(xp.dot(a.T, rot))
        tr_yp1y = np.trace(xp.dot(source_hat.T * p1, source_hat))
        scale = tr_atr / tr_yp1y if update_scale else 1.0
        t = mu_x - scale * xp.dot(rot, mu_y)
        tr_xp1x = xp.trace(xp.dot(target_hat.T * pt1, target_hat))
        if update_scale:
            sigma2 = (tr_xp1x - scale * tr_atr) / (n_p * dim)
        else:
            sigma2 = (tr_xp1x + tr_yp1y - scale * tr_atr) / (n_p * dim)
        sigma2 = max(sigma2, np.finfo(np.float32).eps)
        q = (tr_xp1x - 2.0 * scale * tr_atr + (scale**2) * tr_yp1y) / (2.0 * sigma2)
        q += dim * n_p * 0.5 * np.log(sigma2)
        return MstepResult(RigidTransformation(rot, t, scale, ndim=dim, xp=xp), sigma2, q)


class AffineCPD(CoherentPointDrift):
    """Coherent Point Drift for affine transformation.

    Args:
        source (numpy.ndarray, optional): Source point cloud data.
        tf_init_params (dict, optional): Parameters to initialize transformation.
        use_color (bool, optional): Use color information (if available).
        use_cuda (bool, optional): Use CUDA.
    """

    def __init__(
        self,
        source: Optional[np.ndarray] = None,
        ndim: int = 3,
        tf_init_params: Dict = {},
        use_color: bool = False,
        use_cuda: bool = False,
    ) -> None:
        super(AffineCPD, self).__init__(source, ndim, use_color, use_cuda)
        self._tf_type = AffineTransformation
        self._tf_init_params = tf_init_params

    def _initialize(self, target: np.ndarray) -> MstepResult:
        dim = self._N_DIM
        sigma2 = self._squared_kernel_sum(self._source, target)
        q = 1.0 + target.shape[0] * dim * 0.5 * np.log(sigma2)
        if len(self._tf_init_params) == 0:
            self._tf_init_params = {"b": self.xp.identity(dim), "t": self.xp.zeros(dim)}
        if not "xp" in self._tf_init_params:
            self._tf_init_params["xp"] = self.xp
        self._tf_init_params["ndim"] = self._N_DIM
        return MstepResult(self._tf_type(**self._tf_init_params), sigma2, q)

    @staticmethod
    def _maximization_step(
        source: np.ndarray,
        target: np.ndarray,
        estep_res: EstepResult,
        sigma2_p: Optional[float] = None,
        xp: ModuleType = np,
    ) -> MstepResult:
        pt1, p1, px, n_p = estep_res
        # dim = CoherentPointDrift._N_DIM
        dim = source.shape[1]
        mu_x = xp.sum(px, axis=0) / n_p
        mu_y = xp.dot(source.T, p1) / n_p
        target_hat = target - mu_x
        source_hat = source - mu_y
        a = xp.dot(px.T, source_hat) - xp.outer(mu_x, xp.dot(p1.T, source_hat))
        yp1y = xp.dot(source_hat.T * p1, source_hat)
        b = xp.linalg.solve(yp1y.T, a.T).T
        t = mu_x - xp.dot(b, mu_y)
        tr_xp1x = xp.trace(xp.dot(target_hat.T * pt1, target_hat))
        tr_xpyb = xp.trace(xp.dot(a, b.T))
        sigma2 = (tr_xp1x - tr_xpyb) / (n_p * dim)
        tr_ab = xp.trace(xp.dot(a, b.T))
        sigma2 = max(sigma2, np.finfo(np.float32).eps)
        q = (tr_xp1x - 2 * tr_ab + tr_xpyb) / (2.0 * sigma2)
        q += dim * n_p * 0.5 * np.log(sigma2)
        return MstepResult(AffineTransformation(b, t, ndim=dim, xp=xp), sigma2, q)


class NonRigidCPD(CoherentPointDrift):
    """Coherent Point Drift for nonrigid transformation.

    Args:
        source (numpy.ndarray, optional): Source point cloud data.
        beta (float, optional): Parameter of RBF kernel.
        lmd (float, optional): Parameter for regularization term.
        use_color (bool, optional): Use color information (if available).
        use_cuda (bool, optional): Use CUDA.
    """

    def __init__(
        self,
        source: Optional[np.ndarray] = None,
        ndim: int = 3,
        beta: float = 2.0,
        lmd: float = 2.0,
        use_color: bool = False,
        use_cuda: bool = False,
    ) -> None:
        super(NonRigidCPD, self).__init__(source, ndim, use_color, use_cuda)
        self._tf_type = NonRigidTransformation
        self._beta = beta
        self._lmd = lmd
        self._tf_obj = None
        if not self._source is None:
            self._tf_obj = self._tf_type(None, self._source, self._beta,
                                         ndim=ndim, xp=self.xp)

    def set_source(self, source: np.ndarray) -> None:
        self._source = source
        self._tf_obj = self._tf_type(None, self._source, self._beta,
                                     ndim = self._N_DIM,
                                     xp=self.xp)

    def maximization_step(
        self, target: np.ndarray, estep_res: EstepResult, sigma2_p: Optional[float] = None
    ) -> MstepResult:
        return self._maximization_step(
            self._source[:, : self._N_DIM],
            target[:, : self._N_DIM],
            estep_res,
            sigma2_p,
            self._tf_obj,
            self._lmd,
            self.xp,
        )

    def _initialize(self, target: np.ndarray) -> MstepResult:
        dim = self._N_DIM
        sigma2 = self._squared_kernel_sum(self._source, target)
        q = 1.0 + target.shape[0] * dim * 0.5 * np.log(sigma2)
        self._tf_obj.w = self.xp.zeros_like(self._source)
        return MstepResult(self._tf_obj, sigma2, q)

    @staticmethod
    def _maximization_step(
        source: np.ndarray,
        target: np.ndarray,
        estep_res: EstepResult,
        sigma2_p: float,
        tf_obj: NonRigidTransformation,
        lmd: float,
        xp: ModuleType = np,
    ) -> MstepResult:
        pt1, p1, px, n_p = estep_res
        # dim = CoherentPointDrift._N_DIM
        dim = source.shape[1]
        w = xp.linalg.solve((p1 * tf_obj.g).T + lmd * sigma2_p * xp.identity(source.shape[0]), px - (source.T * p1).T)
        t = source + xp.dot(tf_obj.g, w)
        tr_xp1x = xp.trace(xp.dot(target.T * pt1, target))
        tr_pxt = xp.trace(xp.dot(px.T, t))
        tr_tpt = xp.trace(xp.dot(t.T * p1, t))
        sigma2 = (tr_xp1x - 2.0 * tr_pxt + tr_tpt) / (n_p * dim)
        tf_obj.w = w

        return MstepResult(tf_obj, sigma2, sigma2)


class ConstrainedNonRigidCPD(CoherentPointDrift):
    """
       Extended Coherent Point Drift for nonrigid transformation.
       Like CoherentPointDrift, but allows to add point correspondance constraints
       See: https://people.mpi-inf.mpg.de/~golyanik/04_DRAFTS/ECPD2016.pdf

    Args:
        source (numpy.ndarray, optional): Source point cloud data.
        beta (float, optional): Parameter of RBF kernel.
        lmd (float, optional): Parameter for regularization term.
        alpha (float): Degree of reliability of priors.
            Approximately between 1e-8 (highly reliable) and 1 (highly unreliable)
        use_cuda (bool, optional): Use CUDA.
        use_color (bool, optional): Use color information (if available).
        idx_source (numpy.ndarray of ints, optional): Indices in source matrix
            for which a correspondance is known
        idx_target (numpy.ndarray of ints, optional): Indices in target matrix
            for which a correspondance is known
    """

    def __init__(
        self,
        source: Optional[np.ndarray] = None,
        ndim: int = 3,
        beta: float = 2.0,
        lmd: float = 2.0,
        alpha: float = 1e-8,
        use_color: bool = False,
        use_cuda: bool = False,
        idx_source: Optional[np.ndarray] = None,
        idx_target: Optional[np.ndarray] = None,
    ):
        super(ConstrainedNonRigidCPD, self).__init__(source, ndim, use_color, use_cuda)
        self._tf_type = NonRigidTransformation
        self._beta = beta
        self._lmd = lmd
        self.alpha = alpha
        self._tf_obj = None
        self.idx_source, self.idx_target = idx_source, idx_target
        if not self._source is None:
            self._tf_obj = self._tf_type(None, self._source, self._beta, self.xp)

    def set_source(self, source: np.ndarray) -> None:
        self._source = source
        self._tf_obj = self._tf_type(None, self._source, self._beta)

    def maximization_step(
        self, target: np.ndarray, estep_res: EstepResult, sigma2_p: Optional[float] = None
    ) -> MstepResult:
        return self._maximization_step(
            self._source[:, : self._N_DIM],
            target[:, : self._N_DIM],
            estep_res,
            sigma2_p,
            self._tf_obj,
            self._lmd,
            self.alpha,
            self.p1_tilde,
            self.px_tilde,
            self.xp,
        )

    def _initialize(self, target: np.ndarray) -> MstepResult:
        dim = self._N_DIM
        sigma2 = self._squared_kernel_sum(self._source, target)
        q = 1.0 + target.shape[0] * dim * 0.5 * np.log(sigma2)
        self._tf_obj.w = self.xp.zeros_like(self._source)
        self.p_tilde = self.xp.zeros((self._source.shape[0], target.shape[0]))
        if self.idx_source is not None and self.idx_target is not None:
            self.p_tilde[self.idx_source, self.idx_target] = 1
        self.p1_tilde = self.xp.sum(self.p_tilde, axis=1)
        self.px_tilde = self.xp.dot(self.p_tilde, target)
        return MstepResult(self._tf_obj, sigma2, q)

    @staticmethod
    def _maximization_step(
        source: np.ndarray,
        target: np.ndarray,
        estep_res: EstepResult,
        sigma2_p: float,
        tf_obj: NonRigidTransformation,
        lmd: float,
        alpha: float,
        p1_tilde: float,
        px_tilde: float,
        xp: ModuleType = np,
    ) -> MstepResult:
        pt1, p1, px, n_p = estep_res
        # dim = CoherentPointDrift._N_DIM
        dim = source.shape[1]
        w = xp.linalg.solve(
            (p1 * tf_obj.g).T
            + sigma2_p / alpha * (p1_tilde * tf_obj.g).T
            + lmd * sigma2_p * xp.identity(source.shape[0]),
            px - (source.T * p1).T + sigma2_p / alpha * (px_tilde - (source.T * p1_tilde).T),
        )
        t = source + xp.dot(tf_obj.g, w)
        tr_xp1x = xp.trace(xp.dot(target.T * pt1, target))
        tr_pxt = xp.trace(xp.dot(px.T, t))
        tr_tpt = xp.trace(xp.dot(t.T * p1, t))
        sigma2 = (tr_xp1x - 2.0 * tr_pxt + tr_tpt) / (n_p * dim)
        tf_obj.w = w
        return MstepResult(tf_obj, sigma2, sigma2)


def registration_cpd(
    source: Union[np.ndarray, o3.geometry.PointCloud],
    target: Union[np.ndarray, o3.geometry.PointCloud],
    ndim: int = 3,
    tf_type_name: str = "rigid",
    dtype='nparray',
    w: float = 0.0,
    maxiter: int = 100,
    tol: float = 0.001,
    callbacks: List[Callable] = [],
    use_color: bool = False,
    use_cuda: bool = False,
    **kwargs: Any,
) -> MstepResult:
    """CPD Registraion.

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        tf_type_name (str, optional): Transformation type('rigid', 'affine', 'nonrigid', 'nonrigid_constrained')
        w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
        maxitr (int, optional): Maximum number of iterations to EM algorithm.
        tol (float, optional): Tolerance for termination.
        callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
            `callback(probreg.Transformation)`
        use_color (bool, optional): Use color information (if available).
        use_cuda (bool, optional): Use CUDA.

    Keyword Args:
        update_scale (bool, optional): If this flag is true and tf_type is rigid transformation,
            then the scale is treated. The default is true.
        tf_init_params (dict, optional): Parameters to initialize transformation (for rigid or affine).

    Returns:
        MstepResult: Result of the registration (transformation, sigma2, q)
    """
    xp = np
    if use_cuda:
        import cupy as cp

        xp = cp
    if use_color:
        cv = (
            lambda x: xp.c_[xp.asarray(x.points), xp.asarray(x.colors)]
            if isinstance(x, o3.geometry.PointCloud)
            else xp.asanyarray(x)[:, :]
        )
    else:
        # cv = lambda x: xp.asarray(x.points if isinstance(x, o3.geometry.PointCloud) else x)[
        #     :, : ndim
        # ]
        cv = lambda x: xp.asarray(x)[
            :, : ndim
        ]
    if tf_type_name == "rigid":
        cpd = RigidCPD(cv(source), ndim=ndim, use_color=use_color, use_cuda=use_cuda, **kwargs)
    elif tf_type_name == "affine":
        cpd = AffineCPD(cv(source), ndim=ndim, use_color=use_color, use_cuda=use_cuda, **kwargs)
    elif tf_type_name == "nonrigid":
        cpd = NonRigidCPD(cv(source), ndim=ndim,use_color=use_color, use_cuda=use_cuda, **kwargs)
    elif tf_type_name == "nonrigid_constrained":
        cpd = ConstrainedNonRigidCPD(cv(source), ndim=ndim, use_color=use_color, use_cuda=use_cuda, **kwargs)
    else:
        raise ValueError("Unknown transformation type %s" % tf_type_name)
    cpd.set_callbacks(callbacks)
    return cpd.registration(cv(target), w, maxiter, tol)





############################################
##############BCPD##########################
# import abc
# from collections import namedtuple
# from typing import Any, Callable, List, Union

# import numpy as np
# import open3d as o3
# import scipy.special as spsp
# import six
# from scipy.spatial import cKDTree


# EstepResult = namedtuple("EstepResult", ["nu_d", "nu", "n_p", "px", "x_hat"])
# MstepResult = namedtuple("MstepResult", ["transformation", "u_hat", "sigma_mat", "alpha", "sigma2"])
# MstepResult.__doc__ = """Result of Maximization step.

#     Attributes:
#         transformation (Transformation): Transformation from source to target.
#         u_hat (numpy.ndarray): A parameter used in next Estep.
#         sigma_mat (numpy.ndarray): A parameter used in next Estep.
#         alpha (float): A parameter used in next Estep.
#         sigma2 (float): Variance of Gaussian distribution.
# """


# @six.add_metaclass(abc.ABCMeta)
# class BayesianCoherentPointDrift:
#     """Bayesian Coherent Point Drift algorithm.

#     Args:
#         source (numpy.ndarray, optional): Source point cloud data.
#     """

#     def __init__(self, source=None):
#         self._source = source
#         self._tf_type = None
#         self._callbacks = []

#     def set_source(self, source):
#         self._source = source

#     def set_callbacks(self, callbacks):
#         self._callbacks.extend(callbacks)

#     @abc.abstractmethod
#     def _initialize(self, target):
#         return MstepResult(None, None, None, None, None)

#     def expectation_step(self, t_source, target, scale, alpha, sigma_mat, sigma2, w=0.0):
#         """Expectation step for BCPD"""
#         assert t_source.ndim == 2 and target.ndim == 2, "source and target must have 2 dimensions."
#         dim = t_source.shape[1]
#         pmat = np.stack([np.sum(np.square(target - ts), axis=1) for ts in t_source])
#         pmat = np.exp(-pmat / (2.0 * sigma2))
#         pmat /= (2.0 * np.pi * sigma2) ** (dim * 0.5)
#         pmat = pmat.T
#         pmat *= np.exp(-(scale**2) / (2 * sigma2) * np.diag(sigma_mat) * dim)
#         pmat *= (1.0 - w) * alpha
#         den = w / target.shape[0] + np.sum(pmat, axis=1)
#         den[den == 0] = np.finfo(np.float32).eps
#         pmat = np.divide(pmat.T, den)

#         nu_d = np.sum(pmat, axis=0)
#         nu = np.sum(pmat, axis=1)
#         nu_inv = 1.0 / np.kron(nu, np.ones(dim))
#         px = np.dot(np.kron(pmat, np.identity(dim)), target.ravel())
#         x_hat = np.multiply(px, nu_inv).reshape(-1, dim)
#         return EstepResult(nu_d, nu, np.sum(nu), px.reshape(-1, dim), x_hat)

#     def maximization_step(self, target, estep_res, sigma2_p=None):
#         return self._maximization_step(self._source, target, estep_res, sigma2_p)

#     @staticmethod
#     @abc.abstractmethod
#     def _maximization_step(source, target, estep_res, sigma2_p=None):
#         return None

#     def registration(self, target, w=0.0, maxiter=50, tol=0.001):
#         assert not self._tf_type is None, "transformation type is None."
#         res = self._initialize(target)
#         target_tree = cKDTree(target, leafsize=10)
#         rmse = None
#         for i in range(maxiter):
#             t_source = res.transformation.transform(self._source)
#             estep_res = self.expectation_step(
#                 t_source, target, res.transformation.rigid_trans.scale, res.alpha, res.sigma_mat, res.sigma2, w
#             )
#             res = self.maximization_step(target, res.transformation.rigid_trans, estep_res, res.sigma2)
#             for c in self._callbacks:
#                 c(res.transformation)
#             tmp_rmse = mu.compute_rmse(t_source, target_tree)
#             print("Iteration: {}, Criteria: {}".format(i, tmp_rmse))
#             if not rmse is None and abs(rmse - tmp_rmse) < tol:
#                 break
#             rmse = tmp_rmse
#         return res.transformation


# class CombinedBCPD(BayesianCoherentPointDrift):
#     def __init__(self, source=None, lmd=2.0, k=1.0e20, gamma=1.0):
#         super(CombinedBCPD, self).__init__(source)
#         self._tf_type = CombinedTransformation
#         self.lmd = lmd
#         self.k = k
#         self.gamma = gamma

#     def _initialize(self, target):
#         m, dim = self._source.shape
#         self.gmat = mu.inverse_multiquadric_kernel(self._source, self._source)
#         self.gmat_inv = np.linalg.inv(self.gmat)
#         sigma2 = self.gamma * mu.squared_kernel_sum(self._source, target)
#         q = 1.0 + target.shape[0] * dim * 0.5 * np.log(sigma2)
#         return MstepResult(self._tf_type(np.identity(dim), np.zeros(dim)), None, np.identity(m), 1.0 / m, sigma2)

#     def maximization_step(self, target, rigid_trans, estep_res, sigma2_p=None):
#         return self._maximization_step(
#             self._source, target, rigid_trans, estep_res, self.gmat_inv, self.lmd, self.k, sigma2_p
#         )

#     @staticmethod
#     def _maximization_step(source, target, rigid_trans, estep_res, gmat_inv, lmd, k, sigma2_p=None):
#         nu_d, nu, n_p, px, x_hat = estep_res
#         dim = source.shape[1]
#         m = source.shape[0]
#         s2s2 = rigid_trans.scale**2 / (sigma2_p**2)
#         sigma_mat_inv = lmd * gmat_inv + s2s2 * np.diag(nu)
#         sigma_mat = np.linalg.inv(sigma_mat_inv)
#         residual = rigid_trans.inverse().transform(x_hat) - source
#         v_hat = s2s2 * np.matmul(
#             np.multiply(np.kron(sigma_mat, np.identity(dim)), np.kron(nu, np.ones(dim))), residual.ravel()
#         ).reshape(-1, dim)
#         u_hat = source + v_hat
#         alpha = np.exp(spsp.psi(k + nu) - spsp.psi(k * m + n_p))
#         x_m = np.sum(nu * x_hat.T, axis=1) / n_p
#         sigma2_m = np.sum(nu * np.diag(sigma_mat), axis=0) / n_p
#         u_m = np.sum(nu * u_hat.T, axis=1) / n_p
#         u_hm = u_hat - u_m
#         s_xu = np.matmul(np.multiply(nu, (x_hat - x_m).T), u_hm) / n_p
#         s_uu = np.matmul(np.multiply(nu, u_hm.T), u_hm) / n_p + sigma2_m * np.identity(dim)
#         phi, _, psih = np.linalg.svd(s_xu, full_matrices=True)
#         c = np.ones(dim)
#         c[-1] = np.linalg.det(np.dot(phi, psih))
#         rot = np.matmul(phi * c, psih)
#         tr_rsxu = np.trace(np.matmul(rot, s_xu))
#         scale = tr_rsxu / np.trace(s_uu)
#         t = x_m - scale * np.dot(rot, u_m)
#         y_hat = rigid_trans.transform(source + v_hat)
#         s1 = np.dot(target.ravel(), np.kron(nu_d, np.ones(dim)) * target.ravel())
#         s2 = np.dot(px.ravel(), y_hat.ravel())
#         s3 = np.dot(y_hat.ravel(), np.kron(nu, np.ones(dim)) * y_hat.ravel())
#         sigma2 = (s1 - 2.0 * s2 + s3) / (n_p * dim) + scale**2 * sigma2_m
#         return MstepResult(CombinedTransformation(rot, t, scale, v_hat), u_hat, sigma_mat, alpha, sigma2)


# def registration_bcpd(
#     source: Union[np.ndarray, o3.geometry.PointCloud],
#     target: Union[np.ndarray, o3.geometry.PointCloud],
#     w: float = 0.0,
#     maxiter: int = 50,
#     tol: float = 0.001,
#     callbacks: List[Callable] = [],
#     **kwargs: Any,
# ) -> Transformation:
#     """BCPD Registraion.

#     Args:
#         source (numpy.ndarray): Source point cloud data.
#         target (numpy.ndarray): Target point cloud data.
#         w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
#         maxitr (int, optional): Maximum number of iterations to EM algorithm.
#         tol (float, optional) : Tolerance for termination.
#         callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
#             `callback(probreg.Transformation)`

#     Returns:
#         probreg.Transformation: Estimated transformation.
#     """
#     cv = lambda x: np.asarray(x.points if isinstance(x, o3.geometry.PointCloud) else x)
#     bcpd = CombinedBCPD(cv(source), **kwargs)
#     bcpd.set_callbacks(callbacks)
#     return bcpd.registration(cv(target), w, maxiter, tol)


def test_rigid_3d():
    import numpy as np
    import open3d as o3
    import transforms3d as t3d
    from probreg import cpd
    from probreg import callbacks
    import utils
    # import logging
    # log = logging.getLogger('probreg')
    # log.setLevel(logging.DEBUG)

    source, target = utils.prepare_source_and_target_rigid_3d('bunny.pcd')

    cbs = [callbacks.Open3dVisualizerCallback(source, target)]
    tf_param, _, _ = registration_cpd(source, target,
                                      ndim = 3,
                                    tf_type_name = "rigid", callbacks=cbs)

    print("result: ", np.rad2deg(t3d.euler.mat2euler(tf_param.rot)),
          tf_param.scale, tf_param.t)

def test_affine_3d():
    import numpy as np
    use_cuda = True
    use_cuda = False
    if use_cuda:
        import cupy as cp
        to_cpu = cp.asnumpy
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    else:
        cp = np
        to_cpu = lambda x: x
    from probreg import cpd
    import utils
    import time

    source, target = utils.prepare_source_and_target_nonrigid_3d('face-x.txt', 'face-y.txt', voxel_size=3.0)
    # source = cp.asarray(source.points, dtype=cp.float32)
    # target = cp.asarray(target.points, dtype=cp.float32)

    tf_param, _, _ = registration_cpd(cp.asarray(source.points, dtype=cp.float32),
                                      cp.asarray(target.points, dtype=cp.float32),
                                      ndim = 3,
                                        tf_type_name = "affine",
                                    )
    print("result: ", to_cpu(tf_param.b), to_cpu(tf_param.t))

    import open3d as o3

    result = tf_param.transform(cp.asarray(source.points, dtype=cp.float32))
    pc = o3.geometry.PointCloud()
    pc.points = o3.utility.Vector3dVector(to_cpu(result))
    pc.paint_uniform_color([0, 1, 0])
    target.paint_uniform_color([0, 0, 1])
    o3.visualization.draw_geometries([pc, target])

def test_deform_2d():
    from probreg import cpd
    from probreg import callbacks
    import matplotlib.pyplot as plt
    import utils

    source, target = utils.prepare_source_and_target_nonrigid_2d('fish_source.txt',
                                                                 'fish_target.txt')
    cbs = [callbacks.Plot2DCallback(source, target)]
    print(source.dtype)
    tf_param, _, _ = registration_cpd(source, target, ndim = 2,
                                             tf_type_name = "nonrigid",
                                            callbacks=cbs)
    plt.show()

def test_deform_3d():
    import numpy as np
    use_cuda = False
    if use_cuda:
        import cupy as cp
        to_cpu = cp.asnumpy
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    else:
        cp = np
        to_cpu = lambda x: x
    import open3d as o3
    from probreg import cpd
    import utils
    import time

    source, target = utils.prepare_source_and_target_nonrigid_3d('face-x.txt', 'face-y.txt', voxel_size=3.0)
    source_pt = cp.asarray(source.points, dtype=cp.float32)
    target_pt = cp.asarray(target.points, dtype=cp.float32)

    # acpd = cpd.NonRigidCPD(source_pt, use_cuda=use_cuda)
    # start = time.time()
    # tf_param, _, _ = acpd.registration(target_pt)
    # elapsed = time.time() - start
    # print("time: ", elapsed)

    tf_param, _, _ = registration_cpd(source_pt, target_pt, ndim = 3,
                                             tf_type_name = "nonrigid",
                                            w=0.3,
                                      beta=4,
                                      tol =1e-5,
                                      maxiter=100,
                                            )


    result = tf_param.transform(source_pt)
    pc = o3.geometry.PointCloud()
    pc.points = o3.utility.Vector3dVector(to_cpu(result))
    pc.paint_uniform_color([0, 1, 0])
    target.paint_uniform_color([0, 0, 1])
    o3.visualization.draw_geometries([pc, target])
    print( np.dot( tf_param.g,  tf_param.w))

