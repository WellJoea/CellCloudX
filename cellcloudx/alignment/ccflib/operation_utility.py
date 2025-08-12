import numpy as np
from scipy.spatial import distance_matrix
import torch as th
from tqdm import tqdm


from .neighbors_ensemble import ko_Neighbors, Neighbors

def get_pos_by_bps( X, bps, axis = -1, zspace=0.1, xp=th):
    bps = xp.asarray(bps)

    assert bps.shape[0] >1, "bps should be a vector of length at least 2"
    if xp.asarray(zspace).ndim == 0:
        assert zspace < 1, "zspace should be between 0 and 1"
    else:
        assert  all(zspace < 1), "zspace should be smaller than 1"

    Xzmin = X[:,axis].min()
    Xzmax = X[:,axis].max()
    
    zdist = (bps[1:] - bps[:-1])/2
    zpass = zdist*zspace

    Yzmin = bps[ 0] - zdist[ 0]
    Yzmax = bps[-1] + zdist[-1]

    rpos = xp.hstack([ bps[:-1] + zdist-zpass, Yzmax])
    lpos = xp.hstack([Yzmin, bps[1: ] - zdist+zpass ])

    rpos = (rpos-Yzmin)/(Yzmax-Yzmin)*(Xzmax-Xzmin) + Xzmin
    lpos = (lpos-Yzmin)/(Yzmax-Yzmin)*(Xzmax-Xzmin) + Xzmin
    return lpos, rpos

def split_bins_by_bps(X, bps, min_points=100, axis = -1, zspace=0.0, xp=th):
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

def registration_score(Xs, Fs=None, k = 1, knn=30,
                        normal_x = True,  normal_f = True,
                       sigma2 = 1.0, tau2 = 1.0, device=None, dtype=None, temp=1.0,
                       k_term = 'both', wf=1, xp =th, use_keops = True):
    Xs = [ np.array(ix.detach().cpu() if hasattr(ix, 'detach') else ix).copy() for ix in Xs]
    L = len(Xs)
    if Fs is not None:
        Fs = [ np.array(ifx.detach().cpu() if hasattr(ifx, 'detach') else ifx).copy() for ifx in Fs]
        assert L == len(Fs)

    if xp.__name__ == 'torch':
        dtype = th.float32 if dtype is None else dtype
        Xs = [ xp.tensor(ix, device=device, dtype=dtype) for ix in Xs]
        if Fs is not None:
            Fs = [ xp.tensor(ifx, device=device, dtype=dtype) for ifx in Fs]

    if normal_x:
        Xm, Xn = normal_X( xp.cat(Xs,dim=0), xp=xp )[1:3]
        Xa = [ normal_X(ix, Xm=Xm, Xs=Xn, xp=xp)[0] for ix in Xs ]
    else:
        Xa = Xs
    
    if not Fs is None and normal_f:
        Fs = list(Fs)
        Fa = [ normal_F(ifx, xp=xp) for ifx in Fs ]
    else:
        Fa = Fs

    use_all = (knn is None) or (knn == 0)
    pairs = th.tensor(windows_pair(L, k, k_term=k_term, sym=use_all), dtype=th.int64)

    scores = xp.zeros((L,L))
    pbar = tqdm( range(len(pairs)), total=len(pairs), colour="#000000", desc=f'registration score')

    if use_all:
        for ipr in pbar:
            xid, yid = pairs[ipr]
            ix = Xa[xid]
            iy = Xa[yid]

            if sigma2 is None:
                isig2 = sigma_square(ix, iy, xp=xp)
            else:
                isig2 = sigma2

            if not Fs is None:
                ifx = Fa[xid]
                ify = Fa[yid]
        
                if (tau2 is None) and not Fs is None:
                    if normal_f:
                        itau2 = sigma_square_cos(ifx, ify, xp=xp)
                    else:
                        itau2 = sigma_square(ifx, ify, xp=xp)
                else:
                    itau2 = tau2

            N, M = ix.shape[0], iy.shape[0]
            if use_keops:
                d2 = kodist2(ix, iy) * float( -temp/isig2)
                if Fa is not None:
                    d2 = d2 + float(-temp * wf / itau2) * kodist2(ifx, ify)
                d2 = d2.exp()
                d2 = d2.sum(dim=0).sum()
            else:
                d2 = thdist2(ix, iy)
                d2.mul_(-1.0/float(isig2))

                if Fa is not None:
                    df2 = thdist2(ifx, ify)
                    d2.add_( df2, alpha=float(-wf/itau2))
                d2.exp_()
                d2 = d2.sum()
            iscore = d2 / (N * M)
            scores[xid, yid] = iscore
            mscore = scores[pairs[:ipr+1,0], pairs[:ipr+1, 1]].mean()    
            pbar.set_postfix({'iscore': f'{iscore:.4f}', 'mscore': f'{mscore:.4f}'})
        pbar.close()
    else:
        for ipr in pbar:
            xid, yid = pairs[ipr]
            ix = Xa[xid]
            iy = Xa[yid]

            kok = ko_Neighbors(iy, ix, K=knn)
            kok.knn()
            xidx = kok.dst
            yidx = kok.src

            if sigma2 is None:
                isig2 = sigma_square(ix, iy, xp=xp)
            else:
                isig2 = sigma2

            if not Fs is None:
                ifx = Fa[xid]
                ify = Fa[yid]
        
                if (tau2 is None) and not Fs is None:
                    itau2 = sigma_square_cos(ifx, ify, xp=xp)
                else:
                    itau2 = tau2
            
            d2  = (ix[xidx] - iy[yidx]).pow(2).sum(dim=1)
            d2.mul_(-1.0/float(isig2))
            # d2.exp_()
            if Fa is not None:
                df2 = (ifx[xidx] - ify[yidx]).pow(2).sum(dim=1)
                d2.add_( df2, alpha=float(-wf/itau2))
                # df2 = (1 + (ifx[xidx] * ify[yidx]).sum(dim=1)) /2.0
                # d2 = d2 + float(-wf/itau2) * df2
            d2.exp_()
            iscore = d2.mean()
            scores[xid, yid] = iscore
            mscore = scores[pairs[:ipr+1,0], pairs[:ipr+1, 1]].mean()    
            pbar.set_postfix({'iscore': f'{iscore:.4f}', 'mscore': f'{mscore:.4f}'})
    return scores, scores[pairs[:,0], pairs[:,1]].mean()  

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