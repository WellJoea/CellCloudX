import numpy as np
from scipy.spatial import distance_matrix
import torch as th
from tqdm import tqdm

from ...plotting._colors import colrows, color_palette
from .neighbors_ensemble import ko_Neighbors, Neighbors

def get_pos_by_bps( start, end, bps, zspace=0.1, xp=th):
    bps = xp.asarray(bps)

    assert bps.shape[0] >1, "bps should be a vector of length at least 2"
    if xp.asarray(zspace).ndim == 0:
        assert zspace < 1, "zspace should be between 0 and 1"
    else:
        assert  all(zspace < 1), "zspace should be smaller than 1"

    Xzmin = start
    Xzmax = end
    
    zdist = (bps[1:] - bps[:-1])/2.0
    zdist = zdist*(1 -zspace)
    # zdist = xp.hstack([ zdist[0]*0.01, zdist, zdist[-1]*0.01])
    zdist = xp.hstack([ zdist[0]*0.8, zdist, zdist[-1]*0.8])

    rpos = bps + zdist[1:]
    lpos = bps - zdist[:-1]

    
    Yzmin = lpos[0].clone()
    Yzmax = rpos[-1].clone()

    scale = (Xzmax - Xzmin) / (Yzmax - Yzmin)
    offset = Xzmin - Yzmin * scale
    lpos = lpos * scale + offset
    rpos = rpos * scale + offset

    # add floating point error
    rpos[-1] = Xzmax + 1e-3
    lpos[ 0] = Xzmin - 1e-3
    return lpos, rpos

def show_bins_by_bps(Xs, fscale = 5.0, size=0.5, werror=0, herror=0, nrow=None,
                      ncol=None, colors=None, titles=None):
    import matplotlib.pyplot as plt
    XL, L = len(Xs[0]), len(Xs)
    nrow, ncol = colrows(L, nrows=nrow, ncols=ncol)

    if colors is None or len(colors) ==0:
        colors = color_palette(XL)
    if isinstance(size, (int, float)):
        size = [ size for i in range(XL)] 
    assert len(size) >= XL

    if titles is None or len(titles) ==0:
        titles = range(L)

    fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey=True, 
                            figsize=((fscale+werror)*ncol,(fscale+herror)*nrow))

    for i in range(L):
        ax = axs[i//ncol, i%ncol]
        for xl in range(XL):
            iX = Xs[i][xl]
            if th.is_tensor(iX):
                iX = iX.cpu().numpy()
            ax.scatter( iX[:,0], iX[:,1], c=colors[xl], edgecolors='None', s=size[xl])

        ax.set_title( titles[i])
        ax.set_aspect('equal')
    
    if nrow*ncol - L >0:
        for j in range(nrow*ncol - L):
            fig.delaxes(axs.flatten()[-j-1])
    fig.tight_layout()
    plt.show()

def split_bins_by_bps(*Xs, bps=None, min_points=100, min_each=True, axis = -1, 
                      device=None, zspace=0.0, show=False,  verbose=0, **kargs):
    bps = th.asarray(bps, device=device)
    L, XL = bps.shape[0], len(Xs)

    zspace = th.asarray(zspace, device=device)
    if zspace.ndim==0:
        zspace = zspace.expand(L-1, XL)
    elif zspace.ndim==1:
        if zspace.shape[0] == XL:
            zspace = zspace.expand(L-1, XL)
        elif zspace.shape[0] == L-1:
            zspace = zspace.expand(XL, L-1).T
        else:
            raise ValueError(f'zspace must be of shape {L-1} or {XL}')
    elif zspace.ndim==2:
        if zspace.shape == (L-1, XL):
            pass
        elif zspace.shape == (XL, L-1):
            zspace = zspace.T
        else:
            raise ValueError(f'zspace must be of shape {L-1, XL} or {XL, L-1}')
    else:
        raise ValueError(f'zspace must be of a scale or a 1D or 2D array')

    LPs, RPs, minpts = [], [], th.zeros( (L, XL))
    start = min([ ixs[:,axis].min() for ixs in Xs])
    end   = max([ ixs[:,axis].max() for ixs in Xs])
    for xl in range(XL):
        X = th.asarray(Xs[xl]).to(device=device)
        lpos, rpos = get_pos_by_bps(start, end, bps, zspace=zspace[:, xl], xp=th)
        LPs.append(lpos)
        RPs.append(rpos)
        for iL in range(L):
            l,r = lpos[iL], rpos[iL]
            idx = (X[:,axis] >= l) & (X[:,axis] <= r)
            n_points = idx.sum()
            minpts[iL, xl] = float(n_points)

    Xa, Xins, = [], []
    for iL in range(L):
        if min_each:
            for xl in range(XL):
                aps = minpts[iL, xl]
                while aps < min_points:
                    X = th.asarray(Xs[xl]).to(device=device)
                    l,r = LPs[xl][iL], RPs[xl][iL]
                    zl = (r - l)*0.05
                    l -= zl
                    r += zl
                    idx = (X[:,axis] >= l) & (X[:,axis] <= r)
                    aps += float(idx.sum())
                    LPs[xl][iL], RPs[xl][iL] = l,r
        else:
            mps = minpts[iL].sum()
            if mps < min_points:
                aps = mps
                while aps < min_points:
                    aps = 0
                    for xl in range(XL):
                        X = th.asarray(Xs[xl]).to(device=device)
                        l,r = LPs[xl][iL], RPs[xl][iL]
                        zl = (r - l)*0.05
                        l -= zl
                        r += zl
                        idx = (X[:,axis] >= l) & (X[:,axis] <= r)
                        aps += float(idx.sum())
                        LPs[xl][iL], RPs[xl][iL] = l,r

        XsL, XsI = [], []
        for xl in range(XL):
            iX = th.asarray(Xs[xl]).to(device=device)
            l,r = LPs[xl][iL], RPs[xl][iL]
            idx = (iX[:,axis] >= l) & (iX[:,axis] <= r)

            XsI.append(idx)
            XsL.append(iX[idx].to(device))
        Xa.append(XsL)
        Xins.append(XsI)

    if show:
        show_bins_by_bps(Xa, **kargs)
    return Xa, Xins, LPs, RPs

def split_bins_by_bps0(X, bps, min_points=100, axis = -1, zspace=0.0, xp=th):
    bps = xp.asarray(bps)
    X = xp.asarray(X)

    lpos, rpos = get_pos_by_bps(X, bps, axis = axis, zspace=zspace, xp=xp)
    Xs, Xins = [], []
    for l,r in zip(lpos, rpos):
        idx = (X[:,axis] >= l) & (X[:,axis] <= r)
        n_points = idx.sum()
        while n_points < min_points:
            zl = (r - l)*0.1
            l -= zl
            r += zl
            idx = (X[:,axis] >= l) & (X[:,axis] <= r)
            n_points = idx.sum()
        ixz = X[idx]
        Xins.append(idx)
        Xs.append(ixz) 
    return (lpos, rpos, Xs, Xins)

def normal_X( X, Xm=None, Xs=None,  xp = th):
    X = X.clone()
    N,D = X.shape
    Xm = xp.mean(X, 0) if Xm is None else Xm

    X -= Xm 
    Xs = xp.sqrt(xp.sum(xp.square(X))/(N*D/2)) if Xs is None else Xs
    X /= Xs
    return [X, Xm, Xs]

def normal_F(X,  xp = th):
    l2x = xp.linalg.norm(X, ord=None, dim=1, keepdim=True)
    l2x[l2x == 0] = 1
    return X/l2x

def sigma_square_cos(X, Y, xp=th):
    [N, D] = X.shape
    [M, D] = Y.shape
    sigma2 = (M*xp.sum(X * X) + 
                N*xp.sum(Y * Y) - 
                2* xp.sum(xp.sum(X, 0) * xp.sum(Y, 0)))
    sigma2 /= (N*M) * 2 # TODO: D
    return sigma2

def sigma_square( X, Y, xp=th):
    [N, D] = X.shape
    [M, D] = Y.shape
    sigma2 = (M*xp.sum(X * X) + 
                N*xp.sum(Y * Y) - 
                2* xp.sum(xp.sum(X, 0) * xp.sum(Y, 0)))
    sigma2 /= (N*M*D)
    return sigma2

def windows_pair(L, k, k_term='both', sym =False):
    pairs = []
    for il in range(L):
        sl = list(range(max(il-k, 0), il))
        sr = list(range(min(L, il+1), min(L, il+k+1)))
        if k_term == 'both':
            idx = sl + sr
        elif k_term == 'left':
            idx = sl
        elif k_term == 'right':
            idx = sr
        else:
            raise ValueError('Invalid k_term value')
        pairs += [[il, iy] for iy in idx]
    pairs = np.array(pairs, dtype=np.int64)
    if sym:
        pairs = np.unique(np.sort(pairs, 1), axis=0)
    return pairs

def scale_array( X,
                zero_center = True,
                anis_var = False,
                axis = 0,
    ):
    if issparse(X):
        X = X.toarray()
    X = X.copy()
    N,D = X.shape

    mean = np.expand_dims(np.mean(X, axis=axis), axis=axis)

    if anis_var:
        std  = np.expand_dims(np.std(X, axis=axis, ddof=0), axis=axis)
        std[std == 0] = 1
    else:
        std = np.std(X)

    if zero_center:
        X -= mean
    X /=  std

    mean = np.squeeze(mean)
    std  = np.squeeze(std)
    Xf = np.eye(D+1, dtype=np.float64)
    Xf[:D,:D] *= std
    Xf[:D, D] = mean

    return X, mean, std, Xf

def is_positive_semi_definite(R):
    if not isinstance(R, (np.ndarray, np.generic)):
        raise ValueError('Encountered an error while checking if the matrix is positive semi definite. \
            Expected a numpy array, instead got : {}'.format(R))
    return np.all(np.linalg.eigvals(R) > 0)

def gaussian_kernel(X, beta, Y=None):
    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta**2))

def Gmm(X_emb, Y_emb, norm=False, sigma2=None, temp=1, shift=0, xp = None):
    if xp is None:
        xp = np
    assert X_emb.shape[1] == Y_emb.shape[1]
    (N, D) = X_emb.shape
    M = Y_emb.shape[0]

    if norm:
        # X_emb = (X_emb - np.mean(X_emb, axis=0)) / np.std(X_emb, axis=0)
        # Y_emb = (Y_emb - np.mean(Y_emb, axis=0)) / np.std(Y_emb, axis=0)
        X_l2 =  X_emb/np.linalg.norm(X_emb, ord=None, axis=1, keepdims=True)
        Y_l2 =  Y_emb/np.linalg.norm(Y_emb, ord=None, axis=1, keepdims=True)
    else:
        X_l2 = X_emb
        Y_l2 = Y_emb
    
    Dist = dist_matrix(X_l2, Y_l2)
    if sigma2 is None:
        sigma2 =np.sum(Dist) / (D*N*M)
    P = np.exp( (shift-Dist) / (2 * sigma2 * temp))
    return P, sigma2

def kGmm(X_emb, Y_emb, col, row, temp=1):
    assert X_emb.shape[1] == Y_emb.shape[1]
    (N, D) = X_emb.shape
    M = Y_emb.shape[0]

    Dist = (X_emb[row] - Y_emb[col])**2
    Dist = np.sum(Dist, axis=-1)

    sigma2 =np.mean(Dist) / D
    # sigma2 =np.sum(Dist) / (D*N*M)
    P = np.exp( -Dist / (2 * sigma2 * temp))
    return P, sigma2

def dist_matrix(X, Y, p=2, threshold=1000000):
    dist = distance_matrix(X, Y, p=p, threshold=threshold).T #D,M -> M,D
    return dist ** 2
    # (N, D) = X.shape
    # (M, _) = Y.shape
    # if chunck is None:
    #     diff = X[None, :, :] - Y[:, None, :] # (1, N, D) - (M ,1, D)
    #     dist = diff ** 2
    #     return np.sum(dist, axis=-1)
    # else:
    #     C = min(chunck, N)
    #     splits = np.arange(0, N+C, C).clip(0, N)
    #     Xb = X[None, :, :] 
    #     Yb = Y[:, None, :] 
    #     dist = []
    #     for idx in range(len(splits)-1):
    #         islice = slice(splits[idx], splits[idx+1])
    #         idist = np.sum( (Xb[:,islice,:] - Yb)** 2, axis=-1)
    #         dist.append(idist)
    #     return np.concatenate(dist, axis=0)

def normalAdj(A):
    d1 = np.sqrt(np.sum(A, axis = 1))
    d1[(np.isnan(d1) | (np.isinf(d1) ))] = 0
    d1 = np.diag(1/d1)

    d2 = np.sqrt(np.sum(A, axis = 0))
    d2[(np.isnan(d2) | (np.isinf(d2) ))] = 0
    d2 = np.diag(1/d2)

    return d1 @ A @ d2