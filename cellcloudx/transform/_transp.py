import numpy as np
from scipy.spatial import distance as scipy_distance

def points_pad(locs, tl=[30,30]):
    tp, lf = tl
    ipos = locs.copy()
    ipos[:,0] += lf
    ipos[:,1] += tp
    return ipos

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


def rbf_kernal(X, Y, sigma2=None):
    assert X.shape[1] == Y.shape[1]
    (N, D) = X.shape
    M = Y.shape[0]
    Dist2 = scipy_distance.cdist(X, Y, "sqeuclidean").T
    if sigma2 is None:
        sigma2 = np.sum(Dist2) / (D*N*M) 
    P = np.exp( -Dist2/ (2 * sigma2) )
    return P, sigma2
    

def ccf_deformable_transform_point(Yt, W=None, Y=None, Ym=0, Ys=1, Xm=0, Xs=1, beta=1, 
                                     G=None, Q=None, S=None):
    assert not W is None
    Y_d = (Yt -Ym)/Ys
    if (Y is None) or (np.array_equal(Y, Y_d.astype(Y.dtype))):
        assert Yt.shape == W.shape, 'the shape of Y and W must be same.'
        if (G is None):
            G = Q @ S @ Q.T
        assert not G is None
    else:
        G = rbf_kernal(Y_d, Y, sigma2=beta)[0]

    tmat = G @ W
    Y_n = (Y_d + tmat)* Xs + Xm
    return Y_n

def homotransform_point(locs, tmat, inverse=False, swap_xy=False):
    if locs is None:
        return locs
    tmat = np.array(tmat, dtype=np.float64)
    ndim = tmat.shape[0]-1
    locs = np.array(locs, dtype=np.float64).copy()
    new_locs = locs.copy()[:,:ndim]
    if swap_xy:
        new_locs[:,[1,0]] = new_locs[:,[0,1]]
    new_locs = np.c_[new_locs, np.ones(new_locs.shape[0], dtype=np.float64)]

    if inverse:
        new_locs =  new_locs @ np.linalg.inv(tmat).T
    else:
        new_locs =  new_locs @ tmat.T

    # locs[:,:2] = new_locs[:,:2]
    locs[:,:ndim] = new_locs[:,:ndim]/new_locs[:,[ndim]]
    if swap_xy:
        locs[:,[1,0]] = locs[:,[0,1]]

    return locs

def homotransform_points(locs, tmats, inverse=False, swap_xy=False):
    assert len(locs) == len(tmats), 'the length between locs and tmats must be same.'
    new_locs = [ homotransform_point(locs[i], tmats[i], inverse=inverse, swap_xy=swap_xy)
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
