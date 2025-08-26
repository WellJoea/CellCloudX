import torch as th
import numpy as np
try:
    import pykeops
    pykeops.set_verbose(False)
    from pykeops.torch import LazyTensor
except:
    pass
    # raise ImportError('pykeops is not installed, `pip install pykeops`')

import scipy.sparse as ssp
from ...utilis._clean_cache import clean_cache

def centerlize(X, Xm=None, Xs=None, device=None, xp = th):
    device = X.device if device is None else device
    if X.is_sparse: 
        X = X.to_dense()

    X = X.clone().to(device)
    N,D = X.shape
    Xm = xp.mean(X, 0) if Xm is None else Xm.to(device)

    X -= Xm
    Xs = xp.sqrt(xp.sum(xp.square(X))/(N*D/2)) if Xs is None else Xs.to(device) # N
    X /= Xs
    Xf = xp.eye(D+1, dtype=X.dtype, device=device)
    Xf[:D,:D] *= Xs
    Xf[:D, D] = Xm
    return [X, Xm, Xs, Xf]

def normalize(X, device=None, xp = th):
    device = X.device if device is None else device
    if X.is_sparse: 
        X = X.to_dense()

    X = X.clone().to(device)
    l2x = xp.linalg.norm(X, ord=None, dim=1, keepdim=True)
    l2x[l2x == 0] = 1
    return X/l2x #*((self.DF/2.0)**0.5)

def scaling(X, anis_var=False, zero_center = True, device=None):
    device = X.device if device is None else device
    if X.is_sparse: 
        X = X.to_dense()

    X = X.clone().to(device)
    mean = X.mean(dim=0, keepdim=True)

    if anis_var:
        std  = X.std(dim=0, keepdim=True)
        std[std == 0] = 1
    else:
        std =  X.std() or 1
    if zero_center:
        X -= mean
    X /=  std
    return X
    
def spsparse_to_thsparse(X):
    import torch as th
    XX = X.tocoo()
    values = XX.data
    indices = np.vstack((XX.row, XX.col))
    i = th.LongTensor(indices)
    v = th.tensor(values, dtype=th.float64)
    shape = th.Size(XX.shape)
    return th.sparse_coo_tensor(i, v, shape)

def thsparse_to_spsparse(X):
    XX = X.to_sparse_coo().coalesce()
    values = XX.values().detach().cpu().numpy()
    indices = XX.indices().detach().cpu().numpy()
    shape = XX.shape
    return ssp.csr_array((values, indices), shape=shape)

def kodist2(X, Y):
    x_i = LazyTensor(X[:, None, :])
    y_j = LazyTensor(Y[None, :, :])
    return ((x_i - y_j)**2).sum(dim=2)

def thdist2(X, Y):
    D = th.cdist(X, Y, p=2)
    D.pow_(2)
    return D

def sigma_square_cos(X, Y, xp =th):
    [N, D] = X.shape
    [M, D] = Y.shape
    # sigma2 = (M*np.trace(np.dot(np.transpose(X), X)) + 
    #           N*np.trace(np.dot(np.transpose(Y), Y)) - 
    #           2*np.dot(np.sum(X, axis=0), np.transpose(np.sum(Y, axis=0))))
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

def default_transparas(kargs={}):
    dp= {
        'E':dict(
            fix_s=True, s_clip=None,
        ),
        'S':dict(isoscale=False,
                fix_R=False, fix_t=False, fix_s=False,
                s_clip=None),
        'A':dict(delta=0.1, gamma1=0, kw= 15,  kd_method='sknn'),
        'D':dict( beta=3.0, alpha=5e2, delta=0,
                low_rank= 3000,
                low_rank_type = 'keops',
                fast_low_rank = 8000, num_eig=100, 
                gamma1=0, gamma2=0,  kw= 15, kl=15,
                alpha_decayto = 0.5,  use_p1=False, p1_thred = 0,
                gamma_growto = 1, kd_method='sknn'),
        'P':dict(gamma1=None, lr=0.005, lr_stepsize=None,
                 lr_gamma=0.5, opt='LBFGS', momentum=0.3,
                 opt_iter=70),
        **similarity_paras()
        }
    for k,v in kargs.items():
        dp[k].update(v)
    return dp
    
def similarity_paras():
    return {
        'T': dict(isoscale=True, 
                            fix_R=True,
                            fix_t=False,
                            fix_s=True,),
        'R':dict(isoscale=True, 
                            fix_R=False,
                            fix_t=True,
                            fix_s=True,),
        'I': dict(isoscale=True, 
                            fix_R=True,
                            fix_t=False,
                            fix_s=False,),
        'L': dict(isoscale=False,
                            fix_R=True,
                            fix_t=False,
                            fix_s=False,),
        'O': dict(isoscale=True,
                            fix_R=False,
                            fix_t=False,
                            fix_s=False,),
        # 'S': dict(isoscale=False,
        #                     fix_R=False,
        #                     fix_t=False,
        #                     fix_s=False,),
    }

def feature_pdist2(XF, YF, tau2=None, use_keops=False,
                    xids=None, yids=None,
                    tau2_prediv=True, device=None, dtype=None,
                    temp=1.0 ):
    # if xids is None:
    #     iXF = XF.to(device, dtype=dtype)
    # else:
    #     iXF = XF[xids].to(device, dtype=dtype)
    # if yids is None:
    #     iYF = YF.to(device, dtype=dtype)
    # else:
    #     iYF = YF[yids].to(device, dtype=dtype)

    iXF = XF.to(device, dtype=dtype)
    iYF = YF.to(device, dtype=dtype)
    if use_keops:
        d2f = kodist2(iYF, iXF)
        # d2f = d2f - d2f.min() #mul in cs TODO 
        if tau2_prediv:
            base = -1.0/float(tau2)/temp
            d2f = d2f*base
    else:
        try:
            d2f = thdist2(iYF, iXF)
            # d2f.sub_(d2f.min())
            if tau2_prediv:
                d2f.mul_(-1.0/float(tau2)/temp)
        except:
            clean_cache()
            raise('Memory Error in computing d2f')
    return d2f

def features_pdist2(XFs, YFs, tau2s=None, use_keops=False, tau2_prediv=True, **kargs):
    lenf = len(XFs)
    if tau2_prediv:
        d2fs = feature_pdist2(XFs[0],YFs[0], tau2s[0],
                            use_keops=use_keops, 
                            tau2_prediv=tau2_prediv, temp=1.0, **kargs)
        for il in range(1,lenf):
            if use_keops:
                d2fs = d2fs + feature_pdist2(XFs[il], YFs[il], tau2s[il], 
                                use_keops=use_keops, 
                                tau2_prediv=tau2_prediv, temp=1.0, **kargs)
            else:
                d2fs.add_(feature_pdist2(XFs[il], YFs[il], tau2s[il], 
                                use_keops=use_keops, 
                                tau2_prediv=tau2_prediv, temp=1.0, **kargs))
    else:
        d2fs = []
        try:
            for il in range(lenf):
                d2fs.append(feature_pdist2(XFs[il], YFs[il],
                                use_keops=use_keops, 
                                tau2_prediv=tau2_prediv, **kargs))
        except:
            clean_cache()
            raise('Memory Error in computing d2fs')
    return d2fs

def features_pdist2_df(XFs, YFs, tau2s, use_keops=False, **kargs):
    lenf = len(XFs)
    d2f = feature_pdist2(XFs[0],YFs[0], tau2s[0],
                            use_keops=use_keops, 
                            tau2_prediv=True, temp=1.0, **kargs)
    if use_keops:
        for il in range(1,lenf):
            d2f = d2f + feature_pdist2(XFs[il],YFs[il], tau2s[il],
                                use_keops=use_keops, 
                                tau2_prediv=True, temp=1.0, **kargs)
        cdff = d2f.exp()
        cdff = cdff.sum(dim=0).flatten()
        return [d2f, cdff]
    else:
        for il in range(1,lenf):
            d2f.add_(feature_pdist2(XFs[il], YFs[il], tau2s[il],
                                use_keops=use_keops, 
                                tau2_prediv=True, temp=1.0, **kargs))
        d2f.exp_()
        cdff = d2f.sum(dim=0).to_dense()
        d2f.div_(cdff)
        return d2f

def update_tau2(Pt1, P1, P, Np, XF, YF, use_keops=True, feat_normal=['cos'], DK=[1], 
                device=None, xp=th ):
    FL = len(XF)
    tau2s = xp.zeros(FL, device=device)
    for lf in range(FL):
        # tau2 = xp.einsum('ij,ij->', P, d2f)/Np/DF
        iXF = XF[lf] #.to(device)
        iYF = YF[lf] #.to(device)

        if use_keops:
            PF = P[0] @ (iXF*P[1].unsqueeze(1))
        else:
            PF = P @ iXF
        trPfg = xp.sum(iYF * PF)

        if  feat_normal[lf] in ['cos', 'cosine']:
            tau2 = (2* Np - 2 * trPfg) / (Np * DK[lf] )
        else:
            trfpf = xp.sum( Pt1 * xp.sum(iXF **2, axis=1))
            trgpg = xp.sum( P1  * xp.sum(iYF **2, axis=1))
            tau2 = (trfpf - 2 * trPfg + trgpg) / (Np * DK[lf] )
        tau2s[lf] = tau2
    return tau2s

def expectation_xp(X, TY, sigma2, gs, xp = th, d2f=None, tau2=None, 
                        tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1], 
                        feat_normal=['cos'], XF=None, YF=None, device=None,
                    ):
    P = update_P_xp(X, TY, sigma2, gs, xp = xp, d2f=d2f, tau2=tau2,
                    tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha, DF=DF)
    
    Pt1 = xp.sum(P, 0).to_dense()
    P1 = xp.sum(P, 1).to_dense()
    Np = xp.sum(P1)
    PX = P @ X

    if tau2_auto and (not XF is None) and (not YF is None):
        tau2s = update_tau2(Pt1, P1, P, Np, XF, YF, use_keops=False, 
                            feat_normal=feat_normal, DK=DF, device=device, xp=xp)
    else:
        tau2s = tau2
    return Pt1, P1, PX, Np, tau2s

def update_P_xp(X, Y, sigma2, gs, xp = th, d2f=None, tau2=None, 
             tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=1):
    D = X.shape[1]
    P = thdist2(Y, X)
    P.mul_(-1.0/ (2.0*sigma2) )
    cs = 0.5* D* xp.log(2.0* xp.pi* sigma2) 
    if not d2f is None:
        if tau2_auto:
            for il in range(len(tau2)):
                P.add_(d2f[il], alpha=tau2_alpha/(-2.0*tau2[il]))
                cs += 0.5* DF[il]* xp.log(2.0*xp.pi*tau2[il])
        else:
            P.add_(d2f, alpha=tau2_alpha)
            cs += 0
    if True:
        P.exp_()
        cs = xp.exp(cs)*gs
        cdfs = xp.sum(P, 0).to_dense() + cs
        cdfs.masked_fill_(cdfs == 0, 1.0)
        P.mul_(1.0/cdfs)
    else:
        cs = cs + xp.log(gs+eps)
        log_cdfs = xp.logsumexp(P, axis=0)
        log_cdfs = xp.logaddexp(log_cdfs, cs)
        P.sub_(log_cdfs)
        P.exp_()
    return P

def expectation_ko(X, TY, sigma2, gs, d2f=None, tau2=None, 
                    tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1],
                     feat_normal=['cos'], XF=None, YF=None, device=None,
                     xp = th,
                    ):
    P, c = update_P_ko(X, TY, sigma2, gs, xp = xp, d2f=d2f, tau2=tau2, 
                        tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha, DF=DF)
    ft1 = P.sum(dim=0).flatten()
    a = (ft1 + c)
    a.masked_fill_(a == 0, 1.0)
    a = 1.0/a

    Pt1 = ft1*a #1 - a*c
    P1 = P @ a
    PX = P @ (X*a.unsqueeze(1))
    Np = xp.sum(Pt1)

    if tau2_auto and (not XF is None) and (not YF is None):
        tau2s = update_tau2(Pt1, P1, (P,a), Np, XF, YF, use_keops=True, 
                            feat_normal=feat_normal, DK=DF, device=device, xp=xp)
    else:
        tau2s = tau2
    return Pt1, P1, PX, Np, tau2s

def update_P_ko(X, Y, sigma2, gs, xp = th, d2f=None, tau2=None, 
                tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=1):
    D = X.shape[1]
    P = kodist2(Y, X)
    P = P* (-1.0/ (2.0*sigma2) )
    cs = 0.5* D* xp.log(2.0* xp.pi* sigma2)

    if not d2f is None:
        if tau2_auto:
            for il in range(len(tau2)):
                P = P  + tau2_alpha/(-2.0*tau2)*d2f[il]
                cs += 0.5* DF[il]* xp.log(2.0*xp.pi*tau2[il])
        else: 
            P = P + d2f*tau2_alpha
            cs += 0

    P = P.exp()
    cs = xp.exp(cs)*gs
    return P, cs

def expectation_xp_df(X, TY, sigma2, gs, xp = th, d2f=None, cdff=None):
    P = update_P_xp_df(X, TY, sigma2, gs, xp = xp, d2f=d2f,)
    
    Pt1 = xp.sum(P, 0).to_dense()
    P1 = xp.sum(P, 1).to_dense()
    Np = xp.sum(P1)
    PX = P @ X
    return Pt1, P1, PX, Np

def update_P_xp_df(X, Y, sigma2, gs, xp = th, d2f=None, **kargs):
    D = X.shape[1]
    P = thdist2(Y, X)
    P.mul_(-1.0/ (2.0*sigma2) ).exp_()
    cs = 0.5* D* xp.log(2.0* xp.pi* sigma2) 
    cs = xp.exp(cs)*gs

    cdfs = xp.sum(P, 0).to_dense() + cs
    cdfs.masked_fill_(cdfs == 0, 1.0)

    if not d2f is None:
        P.multiply_(d2f)
    P.mul_(1.0/cdfs)
    return P

def expectation_ko_df(X, TY, sigma2, gs, xp = th, d2f=None, cdff=None,):
    P, a = update_P_ko_df(X, TY, sigma2, gs, xp = xp, d2f=d2f, cdff=cdff)
    ft1 = P.sum(dim=0).flatten()

    a.masked_fill_(a == 0, 1.0)
    a = 1.0/a

    Pt1 = ft1*a #1 - a*c
    P1 = P @ a
    PX = P @ (X*a.unsqueeze(1))
    Np = xp.sum(Pt1)
    return Pt1, P1, PX, Np

def update_P_ko_df( X, Y, sigma2, gs, xp = th, d2f=None,  cdff=None, **kargs):
    D = X.shape[1]
    P = kodist2(Y, X)
    P = P* (-1.0/ (2.0*sigma2) )
    cs = 0.5* D* xp.log(2.0* xp.pi* sigma2)
    cs = xp.exp(cs)*gs

    cdfs = P.exp()
    cdfs = cdfs.sum(dim=0).flatten() + cs

    if not d2f is None:
        P = P + d2f
        cdfs = cdfs * cdff

    P = P.exp()
    #cdfs.masked_fill_(cdfs == 0, 1.0)
    return P, cdfs

def neighbor_weight(w, w_term='both', L= None, xp =None, offset=0, root=None):
    assert w_term in ['both', 'left', 'right', 'R', 'L', 'B']
    assert offset>= 0
    if xp is None:
        try:
            xp = th
        except:
            xp = np
    if w is None:
        assert L is not None, 'w or L should be provided'
        P = xp.zeros((L, L))

    if type(w) in [float, int] or np.isscalar(w) :
        P = xp.ones((L, L)) * w
    elif type(w) in [list, np.ndarray, th.Tensor] :
        try:
            w = xp.asarray(w)
            if w.ndim == 1:
                typep = 'vector'
            elif w.ndim == 2:
                if w.shape == (L, L):
                    typep = 'matrix'
                else:
                    typep = 'list'
            else:
                raise ValueError(f'w term should be a float, list of float, list of length <= L or a tensor of shape ({L,L})')
        except:
            typep = 'list'

        if typep == 'matrix':
            P = w
        elif typep == 'vector':
            P = xp.zeros((L,L))
            for i in range(len(w)):
                il = xp.ones(L-i-offset) *w[i]
                P += xp.diag(il, -i-offset)
            P = P+ P.T- xp.diag(P.diagonal())

        elif typep == 'list': # offset TODO
            P = xp.zeros((L,L),dtype=xp.float64)
            for i in range(len(w)):
                igm = w[i]
                if igm is not None:
                    igm = xp.tensor(igm)
                    if igm.ndim== 0:
                        igm = igm.unsqueeze(0)
                    il = len(igm)
                    P[i, i: min(i+il, L)] = igm[: min(L-i, il)]
                    P[i, max(i-il+1, 0): (i+1)] = igm[: min(i+1, il)].flip(0)
        else:
            raise ValueError(f'the length of w should be less than {L-1} or equal to {L}')
    else:
        try:
            P= xp.asarray(w)
            assert (P.shape[0] == L) and (P.shape[1] == L)
        except:
            raise ValueError(f'w term should be a float, list of float, list of length L or a tensor of shape ({L,L})')

    if root is not None:
        assert type(root) in [int]
        P[root] = 0
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if (i <=root) and ((j <=i) or (j >root)):
                    P[i,j] = 0
                elif (i >root) and ((j >=i) or (j <root)):
                    P[i,j] = 0
    else:
        if w_term in ['L', 'left']:
            P = xp.tril(P)
        elif w_term in ['R', 'right']:
            P = xp.triu(P)
        # else:
        #     raise ValueError(f'gamma_term should be in ["both", "L", "R"]')
    return P

def rigid_outlier(X, Y):
    # chulls = [np.ptp(i, axis=0).prod() for i in [X, Y]]
    phulls = [ int(i.shape[0]) for i in [X, Y]]
    try:
        from scipy.spatial import ConvexHull
        chulls = [ ConvexHull(i.detach().cpu().numpy()).volume 
                for i in [X, Y]]
        fhulls = [chulls, phulls]
    except:
        fhulls = [phulls]
    w = 1 - min([ min([i[0]/i[1], i[1]/i[0]]) for i in fhulls])
    w = float(w)
    return w

def init_tmat(T, D, N=None, R = None, s = None, t=None, A=None, B = None,
              d=1.0, W=None,
              s_clip=None, xp =th, device=None, dtype=th.float32):
    itam = {}
    if T in 'ESRTILOAP':
        if t is None:
            itam['t'] = xp.zeros(D, device=device, dtype=dtype).clone()
        else:
            itam['t'] = xp.asarray(t, device=device, dtype=dtype).clone()
            assert (itam['t'].shape == (D,))

        if T in 'AP':
            if A is None:
                itam['A'] = xp.eye(D, device=device, dtype=dtype).clone()
            else:
                itam['A'] = xp.asarray(A, device=device, dtype=dtype).clone()
                assert (itam['A'].shape == (D,D))

        if T in 'P':
            if B is None:
                itam['B'] = xp.zeros(D, device=device, dtype=dtype).clone()
            else:
                itam['B'] = xp.asarray(B, device=device, dtype=dtype).clone()
                assert (itam['B'].shape == (D,))
            if d is None:
                itam['d'] = xp.tensor(1.0, device=device, dtype=dtype).clone()
            else:
                itam['d'] = xp.asarray(d, device=device, dtype=dtype).clone()
                assert (itam['d'].shape == ())

        if T in 'ESRTILO':
            if R is None:
                itam['R'] = xp.eye(D, device=device, dtype=dtype).clone()
            else:
                itam['R'] = xp.asarray(R, device=device, dtype=dtype).clone()
                assert (itam['R'].shape == (D,D))

            if T in 'E':
                if s is None:
                    itam['s'] = xp.tensor(1.0, device=device, dtype=dtype).clone()
                else:
                    itam['s'] = xp.asarray(s, device=device, dtype=dtype).clone()
                    assert (itam['s'].shape == ())
            else:
                if s is None:
                    itam['s'] = xp.ones(D, device=device, dtype=dtype).clone()
                else:
                    itam['s'] = xp.asarray(s, device=device, dtype=dtype).clone()
                    assert (itam['s'].shape == (D,))

            if s_clip is not None:
                itam['s'] = xp.clip(itam['s'], *s_clip)
    elif T in 'D':
        if W is None:
            itam['W'] = xp.zeros((N,D), device=device, dtype=dtype).clone()
        else:
            itam['W'] = xp.asarray(W, device=device, dtype=dtype).clone()
            assert (itam['W'].shape == (N,D))
    else:
        raise ValueError(f'Invalid transformation type {T}, should be in "ESRTILOAPD"')
    return itam
