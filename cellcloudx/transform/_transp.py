import numpy as np
import functools

from sklearn.metrics.pairwise import euclidean_distances
from ._transi import swap_tmat
from ._similarity_transform import SimilarityTransform

def _to_torch_dtype(np_dtype):
    import torch
    if np_dtype is None:
        return None
    return getattr(torch, np.dtype(np_dtype).name)

def _default_kargs(_func):
    import inspect
    signature = inspect.signature(_func)
    return { k: v.default
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty}
    
def npto2v_wrap(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rkargs = _default_kargs(func)
        rkargs.update(kwargs)
        xp = rkargs.get('xp')
        original_attrs = {}
        if xp is not None:
            func_names = ['asarray', 'ones', 'zeros']
            for name in func_names:
                if hasattr(xp, name):
                    original = getattr(xp, name)
                    original_attrs[name] = original
                    if xp.__name__ == 'numpy':
                        def make_wrapped(original):
                            def wrapped(*args, **rkargs):
                                rkargs.pop('device', None)
                                return original(*args, **rkargs)
                            return wrapped
                        wrapped = make_wrapped(original)
                        setattr(xp, name, wrapped)
                    elif xp.__name__ == 'torch_pass':
                        def make_wrapped(original):
                            def wrapped(*args, **rkargs):
                                device = rkargs.pop('device', None)
                                dtype = rkargs.get('dtype', None)
                                if dtype is not None:
                                    rkargs['dtype'] = _to_torch_dtype(dtype)
                                return original(*args, **rkargs, device=device)
                            return wrapped
                        wrapped = make_wrapped(original)
                        setattr(xp, name, wrapped)
        try:
            result = func(*args, **rkargs)
        finally:
            for name, attr in original_attrs.items():
                setattr(xp, name, attr)
        return result
    return wrapper

class XpWrapper:
    def __init__(self, xp):
        self._xp = xp
        self._wrapped_funcs = {}

    def __getattr__(self, name):
        return getattr(self._xp, name)

    def _wrap_function(self, name):
        original = getattr(self._xp, name)
        
        if self._xp.__name__ == 'numpy':
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                kwargs.pop('device', None)
                return original(*args, **kwargs)
        elif self._xp.__name__ == 'torch_pass':
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                device = kwargs.pop('device', None)
                dtype = kwargs.get('dtype', None)
                if dtype is not None:
                    kwargs['dtype'] = _to_torch_dtype(dtype)
                return original(*args, **kwargs, device=device)
        else:
            wrapped = original
        return wrapped

    def __getattribute__(self, name):
        if name in ['asarray', 'ones', 'zeros', 'eye']:
            if name not in self._wrapped_funcs:
                wrapped = self._wrap_function(name)
                self._wrapped_funcs[name] = wrapped
            return self._wrapped_funcs[name]
        else:
            return super().__getattribute__(name)
def npto2v(xp):
    return XpWrapper(xp)

def homotransform_estimate(src, dst, transformer='rigid', **kwargs):
    '''
    Y, TY
    '''
    import skimage.transform as skitf
    TRANS = {
        'rigid':skitf.EuclideanTransform, #3
        'euclidean':skitf.EuclideanTransform, #3
        'isosimilarity':SimilarityTransform, #4
        'similarity':SimilarityTransform, #5
        'affine':skitf.AffineTransform, #6
        'projective':skitf.ProjectiveTransform, # 8
        'homography':skitf.ProjectiveTransform,
        'piecewise-affine':skitf.PiecewiseAffineTransform,
        'fundamental': skitf.FundamentalMatrixTransform,
        'essential': skitf.EssentialMatrixTransform,
        'polynomial': skitf.PolynomialTransform,
    }

    assert src.shape == dst.shape
    D = src.shape[1]
    if transformer == 'similarity':
        kwargs.update(dict(isoscale = False))
    elif transformer == 'isosimilarity':
        kwargs.update(dict(isoscale = True))

    model= TRANS[transformer](dimensionality=D, **kwargs)
    model.estimate(src, dst)
    return model

def cscale_vertices(vertices, rescale):
    '''
    import matplotlib.pyplot as plt
    a = np.meshgrid(range(11), range(13))
    a = np.vstack([a[0].ravel(), a[1].ravel()]).T
    b = cscale_vertices(a, 2)
    plt.scatter(a[:,0], a[:,1], c='red')
    plt.scatter(b[:,0], b[:,1], c='blue')
    '''
    center = (np.max(vertices, axis=0) + np.min(vertices, axis=0))/2
    verticesN = np.copy(vertices)
    verticesN = (verticesN -center)*rescale
    verticesN = verticesN + center
    return verticesN

def rescale_mesh(mesh, rescale):
    extend_mesh = mesh.copy()
    centroid = extend_mesh.centroid
    extend_mesh.apply_translation(-centroid)
    extend_mesh.vertices *= rescale
    extend_mesh = extend_mesh.apply_translation(centroid)
    return extend_mesh

def rescale_point2d(locs, scale=None, inverse=False):
    if (scale is None) or (scale==1):
        return locs

    if type(scale) in [int, float]:
        scale = [scale, scale]

    locs = np.asarray(locs).copy()
    new_locs = locs.copy()[:,:2]
    if inverse:
        new_locs = new_locs / np.array(scale)
    else:
        new_locs = new_locs * np.array(scale)
    locs[:,:2] = new_locs[:,:2]
    return locs

def rescale_points(locs, scales, inverse=False):
    assert len(locs) == len(scales), 'the length between locs and scales must be same.'
    new_locs = [ rescale_point2d(locs[i], scales[i], inverse=inverse)
                    for i in range(len(locs))]
    return new_locs

def fieldtransform_point(locs, VU, method='linear'):
    from scipy.interpolate import interpn
    def feild_interp(F, point):
        h, w = F.shape[:2]
        x = np.arange(0, h)
        y = np.arange(0, w)
        points= (x,y)
        return interpn(points, F, point, method=method)
    locst = locs.copy()
    loci = locs.copy()

    Vp = feild_interp(VU[0], loci)
    Up = feild_interp(VU[1], loci)

    locst[:,0] = locst[:,0] - Vp
    locst[:,1] = locst[:,1] - Up
    #use_ants 
    # import ants
    # Ts = -np.array([V, U]).transpose(1,2,0)
    # Ff = ants.from_numpy(Ts)

    # import nibabel as nib
    # nib.save(Ff, 'aa.nii.gz')

    # mov_locs = ants.apply_transforms_to_points( 2,
    #                             pd.DataFrame(locs1[:,[1,0]], columns=['x', 'y']) ,
    #                             transformlist=['aa.nii.gz'],
    #                             whichtoinvert=None,
    #                             verbose=0).values
    return locst

def fieldtransform_points(locs, VUs,  method='linear' ):
    assert len(locs) == len(VUs), 'the length between locs and tmats must be same.'
    new_locs = [ fieldtransform_point(locs[i], VUs[i], method=method) for i in range(len(locs))]
    return new_locs

# @npto2v
def rbf_kernal(X, Y, sigma2=None, xp=None):
    assert X.shape[1] == Y.shape[1]
    (N, D) = X.shape
    M = Y.shape[0]
    if hasattr(X ,'detach'):
        import torch as th
        xp = th
        istorch = True
    else:
        xp = np if xp is None else xp
        xp = npto2v(xp)
        istorch = False

    if istorch:
        dist2 = xp.cdist(X, Y, p=2)
        dist2.pow_(2)
    else:
        # dist2 = scipy_distance.cdist(X, Y, "sqeuclidean")
        dist2 = euclidean_distances(X, Y, squared=True)
    if sigma2 is None:
        sigma2 = xp.sum(dist2) / (D*N*M)
    if istorch:
        dist2.div_(-1 * sigma2)
        dist2.exp_()
    else:
        xp.divide(dist2, -1 * sigma2, out=dist2)
        xp.exp(dist2, out=dist2)
    return dist2, sigma2

def ccf_transform_point(Y, paras, use_fgt=False, inverse=False, **kargs):
    TY = Y.copy()
    D = Y.shape[1]
    for para in paras:
        if 'tform' in para:
            tform = para['tform']
            assert tform.shape == (D+1, D+1)
            TY = homotransform_point(TY, tform, inverse=False, **kargs)
            #if inverse: #TODO
        else:
            if inverse:
                raise ValueError('inverse is not supported for non-homotransform.')
            TY = ccf_deformable_transform_point(TY, **para, use_fgt=use_fgt, **kargs)
    return TY

# @npto2v
def ccf_deformable_transform_point(Yt, W=None, Y=None, Ym=0, Ys=1, Xm=0, Xs=1, beta=1, 
                                     G=None, Q=None, S=None, xp=None, floatx=None,
                                     device=None,
                                     eps=1e-10, nsplit=None,
                                     is_large_data=None, use_strict=True, **kargs):
    assert not W is None
    if hasattr(Yt ,'detach'):
        import torch as th
        xp = th
    else:
        if xp is None:
            try:
                import torch as th
                xp = th
            except:
                xp = np
        elif type(xp) is str:
            xp = eval(f'{xp}')
        else:
            xp = xp

    if xp.__name__ == 'numpy':
        device = 'cpu'
        xp = npto2v(xp)
    Yt = xp.asarray(Yt)
    dtype = Yt.dtype if floatx is None else eval(f'xp.{floatx}')
    device = Yt.device if device is None else device

    (Yt, W, Y, Ym, Ys, Xm, Xs, beta, Q, S) = \
        [ xp.asarray(i, dtype=dtype, device=device) if not i is None else None
            for i in [Yt, W, Y, Ym, Ys, Xm, Xs, beta, Q, S]]
    Y_d = (Yt -Ym)/Ys
    if (Y is None) or ((Y.shape == Y_d.shape) and xp.allclose(Y, Y_d)):
        assert Yt.shape == W.shape, 'the shape of Y and W must be same.'
        if (G is None):
            tmat = Q @ S @ (Q.T @ W)
        else:
            tmat = G @ W
    else:
        from ..io._logger import logger
        line_warning = Y_d.shape[0]* Y.shape[0] > 1e10
        msg = '\n*Please use gauss tranfsform to compute the transform matrix for large data.\n'
        msg += '*This may take a long time (product size >1e10).\n'
        msg += "*You can set is_large_data=True, xp=th, floatx='float32', device='cuda'.\n"
        msg += "*You also can set use_strict=False to use the fast gauss transform.\n"
        msg += '*You can set is_large_data=False if you are sure that the data is not large.'

        if line_warning:
            logger.warning(msg)

        if is_large_data is None:
            is_large_data = (True if Y_d.shape[0]* Y.shape[0] > 1e10 else False)
        if is_large_data:
            from ..third_party._ifgt_warp import GaussTransform
            logger.info(f'Use gauss transform with use_strict={use_strict},')
            logger.info(f'xp={xp.__name__}, device={device}, floatx={floatx}')
            gt = GaussTransform(Y, beta**0.5, eps=eps, 
                                 xp=xp, device=device, floatx=dtype, 
                                 nsplit=nsplit,
                                 use_strict=use_strict)
            tmat = gt.compute(Y_d, W.T)
            if hasattr(tmat, 'detach'):
                tmat = tmat.detach().cpu()
            tmat = xp.asarray(tmat, dtype=dtype, device=device)
        else:
            G = rbf_kernal(Y_d, Y, sigma2=beta, xp=xp)[0]
            tmat = G @ W

    Y_n = (Y_d + tmat)* Xs + Xm
    return Y_n

# @npto2v
def homotransform_point(locs, tmat, inverse=False, swap_axes=None, 
                        xp=None, floatx=None, device=None, **kargs):
    if (locs is None) or len(locs) == 0:
        return locs

    if hasattr(locs ,'detach'):
        import torch as th
        xp = th
    else:
        xp = np if xp is None else xp

    if xp.__name__ == 'numpy':
        device = 'cpu'
        xp = npto2v(xp)
    locs = xp.asarray(locs)
    dtype = locs.dtype if floatx is None else eval(f'xp.{floatx}')
    device = locs.device if device is None else device

    new_locs, tmat = [ xp.asarray(i, dtype=dtype, device=device) for i in [locs, tmat] ]
    tmat = swap_tmat(tmat, swap_axes=swap_axes, xp=xp)
    ndim = tmat.shape[0]-1

    if hasattr(new_locs ,'copy'):
        new_locs = new_locs.copy()
    elif hasattr(new_locs ,'clone'):
        new_locs = new_locs.clone()
    homo_base =  xp.ones((new_locs.shape[0],1), dtype=dtype, device=device)
    t_locs = xp.hstack([new_locs[:,:ndim], homo_base])

    if inverse:
        t_locs =  t_locs @ xp.linalg.inv(tmat).transpose(1,0)
    else:
        t_locs =  t_locs @ tmat.transpose(1,0)

    new_locs[:,:ndim] = t_locs[:,:ndim]/t_locs[:,[ndim]]
    return new_locs

def homotransform_points(locs, tmats, inverse=False, swap_axes=None, **kargs):
    assert len(locs) == len(tmats), 'the length between locs and tmats must be same.'
    new_locs = [ homotransform_point(locs[i], tmats[i], inverse=inverse, 
                                     swap_axes=swap_axes, **kargs)
                    for i in range(len(locs))]
    return new_locs

def homotransform_decompose(tmat): #TODO BUG
    assert tmat.shape[0] == tmat.shape[1]
    N = tmat.shape[0] - 1
    if N == 2:
        a, b, e, c, d, f = tmat.flatten(order='C')
        B = tmat[:-1, :-1]
        S = np.linalg.det(B)**(1/(B.shape[0]))
        shift = np.float64([e, f])
        if (a != 0 or b != 0):
            r = np.sqrt(a*a + b*b)
            rotation = np.acos(a / r) if b > 0  else -np.acos(a / r)
            Rotation = np.float64([[np.cos(rotation), -np.sin(rotation)], 
                                    [np.sin(rotation), np.cos(rotation)] ])
            scale = np.float64([r, S / r])
            skew  = np.float64([np.atan((a * c + b * d) / (r * r)), 0])
            return Rotation, rotation, scale, skew, shift

        elif (c != 0 or d != 0):
            s = np.sqrt(c*c + d*d)
            rotation =  np.pi /2 - np.acos(-c/s) if d >0 else  np.pi /2 + np.acos(c/s)
            Rotation = np.float64([[np.cos(rotation), -np.sin(rotation)], 
                                    [np.sin(rotation), np.cos(rotation)] ])
            scale = np.float64([S/s, s])
            skew  = np.float64([0, np.atan((a * c + b * d) / (s * s))])
            return Rotation, rotation, scale, skew, shift

def trans_masklabel(mask, points, space=None,method='nearest', bounds_error=False, fill_value=np.nan, **kargs):
    from scipy.interpolate import interpn
    assert mask.ndim == points.shape[1]
    if space is None:
        space = np.ones(mask.ndim)
    points = points.copy()/np.array(space)

    w, h,d = mask.shape 
    grid_p = (np.arange(w), np.arange(h), np.arange(d))
    label = interpn(grid_p, mask, points, method=method, bounds_error=bounds_error, fill_value=fill_value, **kargs)
    return label
