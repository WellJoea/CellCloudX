import numpy as np
from skimage.transform import  EuclideanTransform
from skimage.transform._geometric import _euler_rotation_matrix

class SimilarityTransform(EuclideanTransform):
    """Similarity transformation.

    Has the following form in 2D::

        X = a0 * x - b0 * y + a1 =
          = s * x * cos(rotation) - s * y * sin(rotation) + a1

        Y = b0 * x + a0 * y + b1 =
          = s * x * sin(rotation) + s * y * cos(rotation) + b1

    where ``s`` is a scale factor and the homogeneous transformation matrix is::

        [[a0  -b0  a1]
         [b0  a0  b1]
         [0   0    1]]

    The similarity transformation extends the Euclidean transformation with a
    single scaling factor in addition to the rotation and translation
    parameters.

    Parameters
    ----------
    matrix : (dim+1, dim+1) array_like, optional
        Homogeneous transformation matrix.
    isoscale : bool, optional
        If True, the transformation is constrained to isotropic scaling.
    scale : float, or (dim,) optional
        Scale factor. Implemented only for 2D and 3D.
    rotation : float, optional
        Rotation angle, clockwise, as radians.
        Implemented only for 2D and 3D. For 3D, this is given in ZYX Euler
        angles.
    translation : (dim,) array_like, optional
        x, y[, z] translation parameters. Implemented only for 2D and 3D.

    Attributes
    ----------
    params : (dim+1, dim+1) array
        Homogeneous transformation matrix.

    """

    def __init__(self, matrix=None,  isoscale=False, scale=None, rotation=None,
                 translation=None, *, dimensionality=2):
        self.params = None
        self.isoscale = isoscale
        params = any(param is not None
                     for param in (scale, rotation, translation))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            matrix = np.asarray(matrix)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            else:
                self.params = matrix
                dimensionality = matrix.shape[0] - 1
        if params:
            if dimensionality not in (2, 3):
                raise ValueError('Parameters only supported for 2D and 3D.')
            matrix = np.eye(dimensionality + 1, dtype=float)
            if scale is None:
                if self.isoscale:
                    scale = 1
                else:
                    scale = np.ones(dimensionality)

            if rotation is None:
                rotation = 0 if dimensionality == 2 else (0, 0, 0)
            if translation is None:
                translation = (0,) * dimensionality
            if dimensionality == 2:
                ax = (0, 1)
                c, s = np.cos(rotation), np.sin(rotation)
                matrix[ax, ax] = c
                matrix[ax, ax[::-1]] = -s, s
            else:  # 3D rotation
                matrix[:3, :3] = _euler_rotation_matrix(rotation)

            matrix[:dimensionality, :dimensionality] *= scale #check
            matrix[:dimensionality, dimensionality] = translation
            self.params = matrix
        elif self.params is None:
            # default to an identity transform
            self.params = np.eye(dimensionality + 1)

    def estimate(self, src, dst):
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """

        self.params = _umeyama(src, dst, estimate_scale=True, isoscale=self.isoscale)

        # _umeyama will return nan if the problem is not well-conditioned.
        return not np.any(np.isnan(self.params))

    @property
    def scale(self):
        # det = scale**(# of dimensions), therefore scale = det**(1/ndim)
        if self.isoscale:
            if self.dimensionality == 2:
                return np.sqrt(np.linalg.det(self.params))
            elif self.dimensionality == 3:
                return np.cbrt(np.linalg.det(self.params))
            else:
                raise NotImplementedError(
                    'Scale is only implemented for 2D and 3D.')
        else:
            U, S, V = np.linalg.svd(self.params[:self.dimensionality,:self.dimensionality])
            R = U @ V
            S = V.T @ np.diag(S) @ V
            return S.diagonal()

def _umeyama(src, dst, estimate_scale, isoscale ):
    """Estimate N-D similarity transformation with or without scaling.

    Parameters
    ----------
    src : (M, N) array_like
        Source coordinates.
    dst : (M, N) array_like
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    isoscale : bool
        Whether to keep the scale factors of isotropic scaling.
         Only relevant if estimate_scale is True.

    Returns
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.

    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`

    """
    src = np.asarray(src)
    dst = np.asarray(dst)

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    T = np.eye(dim + 1, dtype=np.float64)
    
    if isoscale:
        # Eq. (38).
        A = dst_demean.T @ src_demean / num
        # Eq. (39).
        d = np.ones((dim,), dtype=np.float64)
        if np.linalg.det(A) < 0:
            d[dim - 1] = -1
        U, S, V = np.linalg.svd(A)

        # Eq. (40) and (43).
        rank = np.linalg.matrix_rank(A)
        if rank == 0:
            return np.nan * T
        elif rank == dim - 1:
            if np.linalg.det(U) * np.linalg.det(V) > 0:
                T[:dim, :dim] = U @ V
            else:
                s = d[dim - 1]
                d[dim - 1] = -1
                R = U @ np.diag(d) @ V
                d[dim - 1] = s
        else:
            R = U @ np.diag(d) @ V
            # Eq. (41) and (42).
            scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
            # scale = (V.T @ np.diag(S) @ V).diagonal() / src_demean.var(axis=0) #pass
    else:
        A = np.linalg.lstsq(src_demean, dst_demean, rcond=None)[0].T
        scale = np.linalg.norm(A, axis=0)
        R_temp = A / scale
        U, _, Vt = np.linalg.svd(R_temp)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

    T[:dim, dim] = dst_mean - (R * scale ) @ src_mean.T # check
    T[:dim, :dim] = R * scale

    return T