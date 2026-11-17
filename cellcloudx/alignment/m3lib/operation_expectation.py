import torch as th
from torch import Tensor
import numpy as np
import math
try:
    import pykeops
    pykeops.set_verbose(False)
    from pykeops.torch import LazyTensor
except:
    pass
    # raise ImportError('pykeops is not installed, `pip install pykeops`')

import scipy.sparse as ssp
from ...utilis._clean_cache import clean_cache
from .refreshed_budgets import (imbids_otm_refresh_budget, uv_solver,
                                imbids_split_budget, exbids_split_budget,
                                imbids_qt_budget)

def centerlize(X, Xm=None, Xs=None, *, device=None, xp = th):
    device = X.device if device is None else device
    if X.is_sparse: 
        X = X.to_dense()

    X = X.clone().to(device)
    N,D = X.shape
    Xm = xp.mean(X, 0) if Xm is None else xp.as_tensor(Xm, dtype=X.dtype, device=device)

    X -= Xm
    Xs = xp.sqrt(xp.sum(xp.square(X))/(N*D/2)) if Xs is None else xp.as_tensor(Xs, dtype=X.dtype, device=device)
    Xs = xp.where(Xs == 0, xp.ones_like(Xs), Xs)
    X /= Xs
    Xf = xp.eye(D+1, dtype=X.dtype, device=device)
    Xf[:D,:D] *= Xs
    Xf[:D, D] = Xm
    return [X, [Xm, Xs, Xf]]

def normalize(X, *, Xs=None, device=None, xp = th):
    device = X.device if device is None else device
    if X.is_sparse: 
        X = X.to_dense()

    X = X.clone().to(device)
    if Xs is None:
        l2x = xp.linalg.norm(X, ord=None, dim=1, keepdim=True)
        # l2x[l2x == 0] = 1
    else:
        l2x = xp.as_tensor(Xs, dtype=X.dtype, device=device)
    l2x = xp.where(l2x == 0, xp.ones_like(l2x), l2x)
    return [X/l2x, [l2x]]


def center_normalize(X, Xm=None, *, Xs=None,  device=None, xp = th):
    device = X.device if device is None else device
    if X.is_sparse: 
        X = X.to_dense()

    X = X.clone().to(device)
    # X -= X.mean(dim=0, keepdim=True)
    Xm = xp.mean(X, 0) if Xm is None else xp.as_tensor(Xm, dtype=X.dtype, device=device)
    X -= Xm

    # For new samples reuse Xm, but recompute each sample's row norm.
    l2x = xp.linalg.norm(X, ord=None, dim=1, keepdim=True) if Xs is None else xp.as_tensor(Xs, dtype=X.dtype, device=device)
    l2x = xp.where(l2x == 0, xp.ones_like(l2x), l2x)
    return [X/l2x, [Xm, l2x]]


def scale_normalize(X, *, Xm=None, Xs=None, device=None, xp = th):
    device = X.device if device is None else device
    if X.is_sparse: 
        X = X.to_dense()

    X = X.clone().to(device)
    Xm = X.mean(dim=1, keepdim=True) if Xm is None else xp.as_tensor(Xm, dtype=X.dtype, device=device)
    X -= Xm
    l2x = xp.linalg.norm(X, ord=None, dim=1, keepdim=True) if Xs is None else xp.as_tensor(Xs, dtype=X.dtype, device=device)
    l2x = xp.where(l2x == 0, xp.ones_like(l2x), l2x)
    return [X/l2x, [Xm, l2x]]


def zscore_normalize(X, *, Xm=None, Xs=None, device=None, xp=th, eps=1e-12, sample_std=False, unit_l2=True):
    device = X.device if device is None else device
    if X.is_sparse:
        X = X.to_dense()
    X = X.clone().to(device)
    Xm = X.mean(dim=1, keepdim=True) if Xm is None else xp.as_tensor(Xm, dtype=X.dtype, device=device)
    Xc = X - Xm
    if Xs is None:
        if unit_l2:
            std = xp.linalg.norm(Xc, ord=None, dim=1, keepdim=True)
        else:
            correction = int(sample_std and X.shape[1] > 1)
            std = Xc.std(dim=1, keepdim=True, correction=correction).clamp_min(max(eps, xp.finfo(X.dtype).tiny))
    else:
        std = xp.as_tensor(Xs, dtype=X.dtype, device=device)
    std = xp.where(std == 0, xp.ones_like(std), std)
    Z = Xc / std
    # Return the actual divisor, including L2 normalization when requested.
    return [Z, [Xm, std]]

def scaling(X, Xm=None, Xs=None, *, anis_var=False, zero_center = True, device=None):
    device = X.device if device is None else device
    if X.is_sparse: 
        X = X.to_dense()

    X = X.clone().to(device)
    Xm = X.mean(0) if Xm is None else th.as_tensor(Xm, dtype=X.dtype, device=device)

    if Xs is None:
        if anis_var:
            std = X.std(dim=0, keepdim=True, correction=int(X.shape[0] > 1))
        else:
            std =  X.std() or 1
    else:
        std = th.as_tensor(Xs, dtype=X.dtype, device=device)
    std = th.where(std == 0, th.ones_like(std), std)

    if zero_center:
        X -= Xm
    X /=  std
    return [X, [Xm, std]]

def image_normalize(
    *Fs,
    Xm=None,
    Xs=None,
    device=None,
    xp=th,
    eps=1e-12,
    max_samples=100_000,
):
    if len(Fs) == 0:
        raise ValueError("At least one feature matrix is required")

    device = Fs[0].device if device is None else device
    dtype = Fs[0].dtype if Fs[0].is_floating_point() else th.get_default_dtype()
    Fs = [
        (X.to_dense() if X.is_sparse else X).clone().to(device=device, dtype=dtype)
        for X in Fs
    ]

    if any(X.ndim != 2 for X in Fs):
        raise ValueError("All feature matrices must be two-dimensional")
    D = Fs[0].shape[1]
    if any(X.shape[1] != D for X in Fs):
        raise ValueError("All feature matrices must have the same dimension")

    # n = min(min(X.shape[0] for X in Xs), max_samples)
    # if n == 0:
    #     raise ValueError("Feature matrices cannot be empty")

    Fm = None if Xm is None else xp.as_tensor(Xm, dtype=dtype, device=device)
    Fscale = None if Xs is None else xp.as_tensor(Xs, dtype=dtype, device=device)
    if Fm is None or Fscale is None:
        n = min(min(X.shape[0] for X in Fs), max_samples)
        if n < 1:
            raise ValueError("max_samples must be positive")

        F = xp.cat([
            X[xp.linspace(
                0, X.shape[0] - 1, n,
                device=device,
            ).long()]
            for X in Fs
        ], dim=0)

        Fm = xp.median(F, dim=0, keepdim=True).values if Fm is None else Fm
        if Fscale is None:
            Fscale = xp.sqrt(
                xp.median(
                    xp.sum(xp.square(F - Fm), dim=1)
                )
            )
    Fscale = xp.where(
        Fscale > eps,
        Fscale,
        xp.ones_like(Fscale),
    )

    return [*[(X - Fm) / Fscale for X in Fs], [Fm, Fscale]]

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
    # [N, D] = X.shape
    # [M, D] = Y.shape
    # # sigma2 = (M*np.trace(np.dot(np.transpose(X), X)) + 
    # #           N*np.trace(np.dot(np.transpose(Y), Y)) - 
    # #           2*np.dot(np.sum(X, axis=0), np.transpose(np.sum(Y, axis=0))))
    # sigma2 = (M*xp.sum(X * X) + 
    #             N*xp.sum(Y * Y) - 
    #             2* xp.sum(xp.sum(X, 0) * xp.sum(Y, 0)))
    # sigma2 /= (N*M) * 2 # TODO: D
    sigma2 = (xp.mean(X * X) + 
            xp.mean(Y * Y) - 
            2* xp.mean( xp.mean(X, 0) * xp.mean(Y, 0) ))
    sigma2 *= X.shape[1]/2.0
    return sigma2

def sigma_square( X, Y, xp=th):
    # [N, D] = X.shape
    # [M, D] = Y.shape
    # sigma2 = (M*xp.sum(X * X) + 
    #             N*xp.sum(Y * Y) - 
    #             2* xp.sum(xp.sum(X, 0) * xp.sum(Y, 0)))
    # sigma2 /= (N*M*D)

    sigma2 = (xp.mean(X * X) + 
             xp.mean(Y * Y) - 
             2* xp.mean( xp.mean(X, 0) * xp.mean(Y, 0) ))
    return sigma2

def default_transparas(s_clip = [1e-4, None], **kargs):
    dp= {
        'N':dict(),
        'E':dict(
            fix_s=True, s_clip=s_clip,
        ),
        'S':dict(isoscale=False,
                fix_R=False, fix_t=False, fix_s=False,
                s_clip=s_clip),
        'A':dict(delta=0.1, gamma1=0, kw= 15,  kd_method='sknn'),
        'D':dict( beta=2.5, alpha=5e2, delta=0,
                 beta_mrf = 0.5, alpha_mrf=0.8,
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
    dp['H'] = {**dp['D'], 'fix_s': False, 's_clip': s_clip}
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
            if len(P) == 3:
                PF = P[2][:, None] * PF
        else:
            PF = P @ iXF
        trPfg = xp.sum(iYF * PF)

        if  feat_normal[lf] in ['cos', 'cosine', 'ccosine', 'ccos', 'restc']:
            tau2 = (2* Np - 2 * trPfg) / (Np * DK[lf] )
        else:
            trfpf = xp.sum( Pt1 * xp.sum(iXF **2, axis=1))
            trgpg = xp.sum( P1  * xp.sum(iYF **2, axis=1))
            tau2 = (trfpf - 2 * trPfg + trgpg) / (Np * DK[lf] )
        tau2s[lf] = tau2
    return tau2s

def update_tau2_v1(Pt1, P1, trPfgs, Np, XF, YF, use_keops=True, feat_normal=['cos'], DK=[1], 
                device=None, xp=th ):
    FL = len(XF)
    tau2s = xp.zeros(FL, device=device)
    for lf in range(FL):
        # tau2 = xp.einsum('ij,ij->', P, d2f)/Np/DF
        iXF = XF[lf] #.to(device)
        iYF = YF[lf] #.to(device)
        trPfg = trPfgs[lf]
        if  feat_normal[lf] in ['cos', 'cosine', 'ccosine', 'ccos', 'restc']:
            tau2 = (2* Np - 2 * trPfg) / (Np * DK[lf] )
        else:
            trfpf = xp.sum( Pt1 * xp.sum(iXF **2, axis=1))
            trgpg = xp.sum( P1  * xp.sum(iYF **2, axis=1))
            tau2 = (trfpf - 2 * trPfg + trgpg) / (Np * DK[lf] )
        tau2s[lf] = tau2
    return tau2s

def update_tau2_otm(Pt1, P1, P, Np, XF, YF, feat_normal=['cos'], DK=[1], 
                device=None, xp=th ):
    """
    P = (u, K, v), where transport plan is implicitly:
        Pmn = u_m * K_mn * v_n
    """
    FL = len(XF)
    tau2s = xp.zeros(FL, device=device)
    iscomx = isinstance(P, (tuple, list))
    if iscomx:
        u, K, v = P

    for lf in range(FL):
        # tau2 = xp.einsum('ij,ij->', P, d2f)/Np/DF
        iXF = XF[lf] #.to(device)
        iYF = YF[lf] #.to(device)

        if iscomx:
            PF = u[:, None]* (K @ (v[:, None] * iXF))
        else:
            PF = P @ iXF
    
        trPfg = xp.sum(iYF * PF)

        if  feat_normal[lf] in ['cos', 'cosine', 'ccosine', 'ccos', 'restc']:
            tau2 = (2* Np - 2 * trPfg) / (Np * DK[lf] )
        else:
            trfpf = xp.sum( Pt1 * xp.sum(iXF **2, axis=1))
            trgpg = xp.sum( P1  * xp.sum(iYF **2, axis=1))
            tau2 = (trfpf - 2 * trPfg + trgpg) / (Np * DK[lf] )
        tau2s[lf] = tau2
    return tau2s

def expectation_xp(X, TY, sigma2, gs, xp = th, d2f=None, tau2=None, 
                        tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1], 
                        feat_normal=['cos'], XF=None, YF=None, 
                        use_stable=False, device=None,
                    ):
    P = update_P_xp(X, TY, sigma2, gs, xp = xp, d2f=d2f, tau2=tau2,
                    tau2_auto=tau2_auto, use_stable=False, eps=eps, tau2_alpha=tau2_alpha, DF=DF)
    
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

def expectation_xp_reg(X, TY, sigma2, gs, 
                        A=None, B=None, use_gating=True, delta_h = 0, use_lowm = True,
                        xp = th, d2f=None, tau2=None, 
                        tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1], 
                        feat_normal=['cos'], XF=None, YF=None, 
                        use_stable=False, device=None,
                    ):
    D = X.shape[1]
    P = thdist2(TY, X)
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
    P.exp_()
    C = xp.exp(cs)*gs

    up_tau2 = bool(tau2_auto and (not XF is None) and (not YF is None))
    tau2s = tau2
    dh = float(delta_h)
    lh = 1.0 - dh
    use_dh = bool(dh >0)
    if use_lowm:
        SK = xp.sum(P, 0).to_dense()
        SKH = (A * (P.T @ B)).sum(1)

        if use_dh:
            SKH = dh * SK + lh * SKH
        if use_gating:
            Cn = C + SK
            # Pt1 = SK / Cn
         
            # CnSKH =  Cn * SKH
            # CnSKH.masked_fill_(CnSKH < eps * 10, 1.0)
            # CnHnr = SK/ CnSKH
            # valid = SKH > (eps * 10)
            # CnHnr = xp.zeros_like(SKH)
            # CnHnr[valid] = SK[valid] / (Cn[valid] * SKH[valid])

            valid = SKH >  0 #eps * 10
            # Pt1 = SK/ Cn
            Pt1 = xp.zeros_like(SK)
            Pt1[valid] = SK[valid] / Cn[valid]

            CnHnr = xp.zeros_like(SKH)
            CnHnr[valid] = Pt1[valid] / SKH[valid]

            if use_dh:
                base = dh * CnHnr
                ACH = A * (lh * CnHnr)[:, None]
            else:
                ACH = A * CnHnr[:, None]
        else:
            CH = C + SKH
            Pt1 = SKH / CH
            if use_dh:
                base = dh / CH
                ACH = A * (lh / CH)[:, None]
            else:
                ACH = A / CH[:, None]

        Np  = xp.sum(Pt1)
        PX  = xp.zeros_like(TY)
        if use_dh:
            P1 = (B * (P @ ACH)).sum(1) + P @ base
            for iD in range(D):
                PX[:, iD] = (B * (P @ (ACH * X[:, [iD]]))).sum(1) + P @ (base * X[:, iD])
        else:
            P1 = (B * (P @ ACH) ).sum(1)
            for iD in range(D):
                PX[:, iD] = (B * (P @ (ACH * X[:, [iD]]))).sum(1)
        if up_tau2:
            trPfgs = []
            for i, xf in enumerate(XF):
                trPfg = 0
                iDF = xf.shape[1]
                for idf in range(iDF):
                    trPfg += ((YF[i][:,[idf]] *B) * (P @ (ACH * xf[:,[idf]])) ).sum()
                    if use_dh:
                        trPfg += (YF[i][:, idf] * (P @ (base * xf[:, idf]))).sum()
                trPfgs.append(trPfg)

            tau2s = update_tau2_v1(Pt1, P1, trPfgs, Np, XF, YF, use_keops=False, 
                                feat_normal=feat_normal, DK=DF, device=device, xp=xp)
    else:
        if use_gating:
            SK = P.sum(0)
            if use_dh:
                P.mul_( (lh *B) @ A.T + dh)
            else:
                P.mul_(B @ A.T)
            SKH = P.sum(0)
            Cn = C + SK

            valid = SKH > 0 #(eps * 10)
            # Pt1 = SK / Cn
            Pt1 = xp.zeros_like(SK)
            Pt1[valid] = SK[valid] / Cn[valid]
            CnHnr = xp.zeros_like(SKH)
            CnHnr[valid] = Pt1[valid] / SKH[valid]
            # valid = SKH > (eps * 10)
            # CnHnr = xp.zeros_like(SKH)
            # CnHnr[valid] = SK[valid] / (Cn[valid] * SKH[valid])
            ## vP = P[:, ~valid].clone() / Cn[~valid]
            P.mul_(CnHnr)
            ## P[:, ~valid] = vP
        else:
            if use_dh:
                P.mul_( (lh *B) @ A.T + dh)
            else:
                P.mul_(B @ A.T)
            CH = xp.sum(P, 0).to_dense() + C
            P.mul_(1.0/CH)
            Pt1 = xp.sum(P, 0).to_dense()

        P1 = xp.sum(P, 1).to_dense()
        Np = xp.sum(P1)
        PX = P @ X
        if up_tau2:
            trPfgs = []
            for i, xf in enumerate( XF):
                trPfg = (YF[i] * (P @ xf)).sum()
                trPfgs.append(trPfg)
            tau2s = update_tau2_v1(Pt1, P1, trPfgs, Np, XF, YF, use_keops=False, 
                                feat_normal=feat_normal, DK=DF, device=device, xp=xp)
    return Pt1, P1, PX, Np, tau2s

def expectation_ko_reg(X, TY, sigma2, gs,
                    A=None, B=None, use_gating=True, delta_h = 0, use_lowm = True,
                    d2f=None, tau2=None, 
                    tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1],
                    feat_normal=['cos'], XF=None, YF=None, device=None,
                    xp = th, use_stable=False,
                    ):
    D = X.shape[1]
    P = kodist2(TY, X)
    P = P* (-1.0/ (2.0*sigma2) )
    cs = 0.5* D* xp.log(2.0* xp.pi* sigma2)

    if not d2f is None:
        if tau2_auto:
            for il in range(len(tau2)):
                P = P  + tau2_alpha/(-2.0*tau2[il])*d2f[il]
                cs += 0.5* DF[il]* xp.log(2.0*xp.pi*tau2[il])
        else: 
            P = P + d2f*tau2_alpha
            cs += 0

    P = P.exp()
    C = xp.exp(cs)*gs

    up_tau2 = bool(tau2_auto and (not XF is None) and (not YF is None))
    tau2s = tau2
    dh = float(delta_h)
    lh = 1.0 - dh
    use_dh = bool(dh >0)

    if use_lowm:
        M, J = B.shape
        one = xp.ones((M, 1), dtype=B.dtype,device=B.device,)
        KTB = P.t() @ xp.cat([one, B], dim=1)
        SK = KTB[:, 0]
        SKH = (A * KTB[:, 1:]).sum(1)
        if use_dh:
            SKH = dh * SK + lh * SKH
        
        # if use_dh:
        #     SK = xp.sum(P, 0).to_dense()
        #     SKH = dh * SK + lh * (A * (P.T @ B)).sum(1)
        # else:
        #     SK  = xp.sum(P, 0).to_dense()
        #     SKH = (A * (P.T @ B)).sum(1)
        if use_gating:
            Cn = C + SK
            
            # Pt1 = SK/ Cn
            # CnSKH =  Cn * SKH
            # CnSKH.masked_fill_(CnSKH < eps * 10, 1.0)
            # CnHnr = SK/ CnSKH

            valid = SKH > 0
            Pt1 = xp.zeros_like(SK)
            Pt1[valid] = SK[valid] / Cn[valid]
            CnHnr = xp.zeros_like(SKH)
            CnHnr[valid] = Pt1[valid] / SKH[valid]

            if use_dh:
                base = dh * CnHnr
                ACH = A * (lh * CnHnr)[:, None]
            else:
                ACH = A * CnHnr[:, None]
        else:
            CH  = C + SKH
            Pt1 = SKH/ CH
            if use_dh:
                base = dh / CH
                ACH = A * (lh / CH)[:, None]
            else:
                ACH = A / CH[:, None]

        # Np  = xp.sum(Pt1)
        # PX  = xp.zeros_like(TY)
        
        B_i = LazyTensor(B.contiguous()[:, None, :])
        ACH_j = LazyTensor(ACH.contiguous()[None, :, :])
        Q = (B_i | ACH_j)
        if use_dh:
            base_j = LazyTensor(base.contiguous()[None, :, None])
            Q = Q + base_j
        PH = P * Q
    else:
        B_i = LazyTensor(B.contiguous()[:, None, :])
        A_j = LazyTensor(A.contiguous()[None, :, :])
        H = (B_i * A_j).sum(-1)
        if use_dh:
            H = lh * H + dh
            
        if use_gating:
            SK = P.sum(dim=0).reshape(-1)
            SKH = (P * H).sum(dim=0).reshape(-1)

            Cn = C + SK
            # Pt1 = SK / Cn
            # CnSKH =  Cn * SKH
            # CnSKH.masked_fill_(CnSKH < eps * 10, 1.0)
            # CnHnr = SK/ CnSKH
            
            valid = SKH > 0
            # Pt1 = SK / Cn
            Pt1 = xp.zeros_like(SK)
            Pt1[valid] = SK[valid] / Cn[valid]
            CnHnr = xp.zeros_like(SKH)
            CnHnr[valid] = Pt1[valid] / SKH[valid]

            CnHnr_j = LazyTensor(CnHnr.contiguous()[None, :, None])
            PH = P * H * CnHnr_j

        else:
            SKH = (P * H).sum(dim=0).reshape(-1)
            CH = C + SKH
            Pt1 = SKH / CH
            CH_j = LazyTensor((1.0 / CH).contiguous()[None, :, None])
            PH = P * H * CH_j

    Np  = xp.sum(Pt1)
    X1 = th.cat( [th.ones_like(X[:, :1]), X], dim=1, ).contiguous()
    X1_j = LazyTensor(X1[None, :, :])
    Pstat = (PH * X1_j).sum(dim=1)

    P1 = Pstat[:, 0]
    PX = Pstat[:, 1:]

    if up_tau2:
        trPfgs = []
        for i, xf in enumerate(XF):
            YF_i = LazyTensor(YF[i].contiguous()[:, None, :])
            XF_j = LazyTensor(xf.contiguous()[None, :, :])
            Fij = (YF_i * XF_j).sum(-1)
            trPfgs.append((PH * Fij).sum(dim=1).sum())

        tau2s = update_tau2_v1(
            Pt1, P1, trPfgs, Np, XF, YF,
            use_keops=False, feat_normal=feat_normal,
            DK=DF, device=device, xp=xp
        )
    return Pt1, P1, PX, Np, tau2s

def expectation_ko_reg_v0(X, TY, sigma2, gs, 
                        A, B, use_gating=True, delta_h = 0, use_lowm = True,
                        xp = th, d2f=None, tau2=None, 
                        tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1], 
                        feat_normal=['cos'], XF=None, YF=None, 
                        use_stable=False, device=None,
                    ):
    D = X.shape[1]
    P = kodist2(TY, X)
    P = P* (-1.0/ (2.0*sigma2) )
    cs = 0.5* D* xp.log(2.0* xp.pi* sigma2)

    if not d2f is None:
        if tau2_auto:
            for il in range(len(tau2)):
                P = P  + tau2_alpha/(-2.0*tau2[il])*d2f[il]
                cs += 0.5* DF[il]* xp.log(2.0*xp.pi*tau2[il])
        else: 
            P = P + d2f*tau2_alpha
            cs += 0

    P = P.exp()
    C = xp.exp(cs)*gs

    up_tau2 = bool(tau2_auto and (not XF is None) and (not YF is None))
    tau2s = tau2
    dh = float(delta_h)
    lh = 1.0 - dh
    use_dh = bool(dh >0)

    if use_lowm:
        if use_dh:
            SK = P.sum(0).flatten()
            SKH = dh * SK + lh * (A * (P.T @ B)).sum(1)
        else:
            SK  = P.sum(0).flatten()
            SKH = (A * (P.T @ B)).sum(1)
        if use_gating:
            Cn = C + SK
            Pt1 = SK/ Cn
         
            CnSKH =  Cn * SKH
            CnSKH.masked_fill_(CnSKH < eps * 10, 1.0)
            CnHnr = SK/ CnSKH
            # valid = SKH > (eps * 10)
            # CnHnr = xp.zeros_like(SKH)
            # CnHnr[valid] = SK[valid] / (Cn[valid] * SKH[valid])
            if use_dh:
                base = dh * CnHnr
                ACH = A * (lh * CnHnr)[:, None]
            else:
                ACH = A * CnHnr[:, None]
        else:
            CH  = C + SKH
            Pt1 = SKH/ CH
            if use_dh:
                base = dh / CH
                ACH = A * (lh / CH)[:, None]
            else:
                ACH = A / CH[:, None]

        Np  = xp.sum(Pt1)
        PX  = xp.zeros_like(TY)
        if use_dh:
            P1 = (B * (P @ ACH)).sum(1) + P @ base
            for iD in range(D):
                PX[:, iD] = (B * (P @ (ACH * X[:, [iD]]))).sum(1) + P @ (base * X[:, iD])
        else:
            P1 = (B * (P @ ACH) ).sum(1)
            for iD in range(D):
                PX[:, iD] = (B * (P @ (ACH * X[:, [iD]]))).sum(1)

        if up_tau2:
            trPfgs = []
            for i, xf in enumerate(XF):
                trPfg = 0
                iDF = xf.shape[1]
                for idf in range(iDF):
                    trPfg += ((YF[i][:,[idf]] *B) * (P @ (ACH * xf[:,[idf]])) ).sum()
                    if use_dh:
                        trPfg += (YF[i][:, idf] * (P @ (base * xf[:, idf]))).sum()
                trPfgs.append(trPfg)

            tau2s = update_tau2_v1(Pt1, P1, trPfgs, Np, XF, YF, use_keops=False, 
                                feat_normal=feat_normal, DK=DF, device=device, xp=xp)
    else:
        if use_gating:
            SK = P.sum(0)
            if use_dh:
                P.mul_( (lh *B) @ A.T + dh)
            else:
                P.mul_(B @ A.T)
            SKH = P.sum(0)
            # SKH.masked_fill_(SKH < 1e-5, 1.0)
            CnSKH =  (C + SK) * SKH
            CnSKH.masked_fill_(CnSKH < eps * 100, 1.0)
            CnHnr = SK/ CnSKH
            
            # valid = SKH > (eps * 10)
            # CnHnr = xp.zeros_like(SKH)
            # CnHnr[valid] = SK[valid] / (Cn[valid] * SKH[valid])
            ## vP = P[:, ~valid].clone() / Cn[~valid]
            P.mul_(CnHnr)
            ## P[:, ~valid] = vP

        else:
            if use_dh:
                P.mul_( (lh *B) @ A.T + dh)
            else:
                P.mul_(B @ A.T)
            CH = xp.sum(P, 0).to_dense() + C
            P.mul_(1.0/CH)
        Pt1 = xp.sum(P, 0).to_dense()
        P1 = xp.sum(P, 1).to_dense()
        Np = xp.sum(P1)
        PX = P @ X

        if up_tau2:
            trPfgs = []
            for i, xf in enumerate( XF):
                trPfg = (YF[i] * (P @ xf)).sum()
                trPfgs.append(trPfg)
            tau2s = update_tau2_v1(Pt1, P1, trPfgs, Np, XF, YF, use_keops=False, 
                                feat_normal=feat_normal, DK=DF, device=device, xp=xp)
    return Pt1, P1, PX, Np, tau2s

def expectation_xp_reg1(X, TY, sigma2, piu, pi0,
                    A=None, B=None, use_gating=True, delta_h = 0, use_lowm = True,
                    xp = th, d2f=None, tau2=None, 
                    tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1], 
                    feat_normal=['cos'], XF=None, YF=None, 
                    use_stable=False, device=None,
                    ):
    D = X.shape[1]
    P = thdist2(TY, X)
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
    if not use_stable:
        P.exp_()
        P.mul_(piu[:, None])
        C = xp.exp(cs) * pi0
    else:
        P.add_(xp.log(piu[:, None] + eps))
        cs = cs + xp.log(pi0 + eps)

        pmax = xp.maximum(xp.amax(P, dim=0), cs)
        P.sub_(pmax[None, :])
        P.exp_()
        C = xp.exp(cs - pmax)

    up_tau2 = bool(tau2_auto and (not XF is None) and (not YF is None))
    tau2s = tau2
    dh = float(delta_h)
    lh = 1.0 - dh
    use_dh = bool(dh > 0)

    if use_lowm:
        SK = xp.sum(P, 0).to_dense()
        SKH = (A * (P.T @ B)).sum(1)

        if use_dh:
            SKH = dh * SK + lh * SKH
        if use_gating:
            Cn = C + SK
            
            valid = SKH >  0
            # Pt1 = SK/ Cn
            Pt1 = xp.zeros_like(SK)
            Pt1[valid] = SK[valid] / Cn[valid]
            CnHnr = xp.zeros_like(SKH)
            CnHnr[valid] = Pt1[valid] / SKH[valid]

            if use_dh:
                base = dh * CnHnr
                ACH = A * (lh * CnHnr)[:, None]
            else:
                ACH = A * CnHnr[:, None]
        else:
            CH = C + SKH
            Pt1 = SKH / CH
            if use_dh:
                base = dh / CH
                ACH = A * (lh / CH)[:, None]
            else:
                ACH = A / CH[:, None]

        Np  = xp.sum(Pt1)
        PX  = xp.zeros_like(TY)
        if use_dh:
            P1 = (B * (P @ ACH)).sum(1) + P @ base
            for iD in range(D):
                PX[:, iD] = (B * (P @ (ACH * X[:, [iD]]))).sum(1) + P @ (base * X[:, iD])
        else:
            P1 = (B * (P @ ACH) ).sum(1)
            for iD in range(D):
                PX[:, iD] = (B * (P @ (ACH * X[:, [iD]]))).sum(1)
        if up_tau2:
            trPfgs = []
            for i, xf in enumerate(XF):
                trPfg = 0
                iDF = xf.shape[1]
                for idf in range(iDF):
                    trPfg += ((YF[i][:,[idf]] *B) * (P @ (ACH * xf[:,[idf]])) ).sum()
                    if use_dh:
                        trPfg += (YF[i][:, idf] * (P @ (base * xf[:, idf]))).sum()
                trPfgs.append(trPfg)

            tau2s = update_tau2_v1(
                Pt1, P1, trPfgs, Np, XF, YF,
                use_keops=False,
                feat_normal=feat_normal,
                DK=DF,
                device=device,
                xp=xp,
            )
    else:
        if use_gating:
            SK = P.sum(0)
            if use_dh:
                P.mul_( (lh *B) @ A.T + dh)
            else:
                P.mul_(B @ A.T)
            SKH = P.sum(0)
            Cn = C + SK

            valid = SKH > 0
            # Pt1 = SK / Cn
            Pt1 = xp.zeros_like(SK)
            Pt1[valid] = SK[valid] / Cn[valid]
            CnHnr = xp.zeros_like(SKH)
            CnHnr[valid] = Pt1[valid] / SKH[valid]
            
            P.mul_(CnHnr)
            ## P[:, ~valid] = vP

        else:
            if use_dh:
                P.mul_( (lh *B) @ A.T + dh)
            else:
                P.mul_(B @ A.T)
            CH = xp.sum(P, 0).to_dense() + C
            P.mul_(1.0/CH)
            Pt1 = xp.sum(P, 0).to_dense()

        P1 = xp.sum(P, 1).to_dense()
        Np = xp.sum(P1)
        PX = P @ X
        if up_tau2:
            trPfgs = []
            for i, xf in enumerate( XF):
                trPfg = (YF[i] * (P @ xf)).sum()
                trPfgs.append(trPfg)
            tau2s = update_tau2_v1(
                Pt1, P1, trPfgs, Np, XF, YF,
                use_keops=False,
                feat_normal=feat_normal,
                DK=DF,
                device=device,
                xp=xp,
            )
    return Pt1, P1, PX, Np, tau2s

def expectation_ko_reg1v1(
    X, TY, sigma2, piu, pi0,
    A=None, B=None, use_gating=True, delta_h = 0, use_lowm = True,
    d2f=None, tau2=None, 
    tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1],
    feat_normal=['cos'], XF=None, YF=None, device=None,
    xp = th, use_stable=False, backend='auto',
    ):
    D = X.shape[1]
    P = kodist2(TY, X)
    P = P* (-1.0/ (2.0*sigma2) )
    cs = 0.5* D* xp.log(2.0* xp.pi* sigma2)

    if not d2f is None:
        if tau2_auto:
            for il in range(len(tau2)):
                P = P  + tau2_alpha/(-2.0*tau2[il])*d2f[il]
                cs += 0.5* DF[il]* xp.log(2.0*xp.pi*tau2[il])
        else: 
            P = P + d2f*tau2_alpha
            cs += 0

    P = P.exp()
    piu_i = LazyTensor( piu.reshape(-1).contiguous()[:, None, None])
    P = P * piu_i

    C = xp.exp(cs) * pi0

    up_tau2 = bool(tau2_auto and (not XF is None) and (not YF is None))
    tau2s = tau2
    dh = float(delta_h)
    lh = 1.0 - dh
    use_dh = bool(dh >0)

    if use_lowm:
        M, J = B.shape
        if use_gating or use_dh:
            one = xp.ones((M, 1), dtype=B.dtype,device=B.device,)
            B1_i = LazyTensor(xp.cat([one, B], dim=1).contiguous()[:, None, :])
            KTB = (P * B1_i).sum(dim=0, backend=backend)
            SK = KTB[:, 0]
            SKH = (A * KTB[:, 1:]).sum(1)
            if use_dh:
                SKH = dh * SK + lh * SKH
        else:
            B_i = LazyTensor(B.contiguous()[:, None, :])
            KTB = (P * B_i).sum(dim=0, backend=backend)
            SKH = (A * KTB).sum(1)
        
        # if use_dh:
        #     SK = xp.sum(P, 0).to_dense()
        #     SKH = dh * SK + lh * (A * (P.T @ B)).sum(1)
        # else:
        #     SK  = xp.sum(P, 0).to_dense()
        #     SKH = (A * (P.T @ B)).sum(1)
        if use_gating:
            Cn = C + SK

            # CnSKH =  Cn * SKH
            # CnSKH.masked_fill_(CnSKH < eps * 10, 1.0)
            # CnHnr = SK/ CnSKH

            # Only columns with positive label-compatible mass are valid.
            # For delta_h=0, SKH=0 means strict label rejection.
            valid = SKH >  0 #0
            # Pt1 = SK/ Cn
            Pt1 = xp.zeros_like(SK)
            Pt1[valid] = SK[valid] / Cn[valid]
            CnHnr = xp.zeros_like(SKH)
            CnHnr[valid] = Pt1[valid] / SKH[valid]

            if use_dh:
                base = dh * CnHnr
                ACH = A * (lh * CnHnr)[:, None]
            else:
                ACH = A * CnHnr[:, None]
        else:
            CH  = C + SKH
            Pt1 = SKH/ CH
            if use_dh:
                base = dh / CH
                ACH = A * (lh / CH)[:, None]
            else:
                ACH = A / CH[:, None]

        # Np  = xp.sum(Pt1)
        # PX  = xp.zeros_like(TY)
        
        B_i = LazyTensor(B.contiguous()[:, None, :])
        ACH_j = LazyTensor(ACH.contiguous()[None, :, :])
        Q = (B_i | ACH_j)
        if use_dh:
            base_j = LazyTensor(base.contiguous()[None, :, None])
            Q = Q + base_j
        PH = P * Q
    else:
        B_i = LazyTensor(B.contiguous()[:, None, :])
        A_j = LazyTensor(A.contiguous()[None, :, :])
        H = (B_i * A_j).sum(-1)
        if use_dh:
            H = lh * H + dh
            
        if use_gating:
            one_j = LazyTensor(xp.ones_like(A[:, :1]).contiguous()[None, :, :])
            SKs = (P * one_j.concat(H)).sum(dim=0, backend=backend)
            SK = SKs[:, 0]
            SKH = SKs[:, 1]

            Cn = C + SK

            # valid = SKH > (eps * 10)
            # CnHnr = xp.zeros_like(SKH)
            # CnHnr[valid] = SK[valid] / (Cn[valid] * SKH[valid])
            # CnSKH =  Cn * SKH
            # CnSKH.masked_fill_(CnSKH < eps * 10, 1.0)
            # CnHnr = SK/ CnSKH
            
            valid = SKH > 0
            # Pt1 = SK / Cn
            Pt1 = xp.zeros_like(SK)
            Pt1[valid] = SK[valid] / Cn[valid]
            CnHnr = xp.zeros_like(SKH)
            CnHnr[valid] = Pt1[valid] / SKH[valid]

            CnHnr_j = LazyTensor(CnHnr.contiguous()[None, :, None])
            PH = P * H * CnHnr_j

        else:
            SKH = (P * H).sum(dim=0, backend=backend).reshape(-1)
            CH = C + SKH
            Pt1 = SKH / CH
            CH_j = LazyTensor((1.0 / CH).contiguous()[None, :, None])
            PH = P * H * CH_j

    Np  = xp.sum(Pt1)
    X1 = th.cat( [th.ones_like(X[:, :1]), X], dim=1, ).contiguous()
    if up_tau2:
        X1 = th.cat([X1] + list(XF), dim=1).contiguous()
    X1_j = LazyTensor(X1[None, :, :])
    Pstat = (PH * X1_j).sum(dim=1, backend=backend)
    P1 = Pstat[:, 0]
    PX = Pstat[:, 1:D + 1]

    if up_tau2:
        trPfgs = []
        off = D + 1
        for i, xf in enumerate(XF):
            d = xf.shape[1]
            trPfgs.append((YF[i] * Pstat[:, off:off + d]).sum())
            off += d

        tau2s = update_tau2_v1(
            Pt1, P1, trPfgs, Np, XF, YF,
            use_keops=False, feat_normal=feat_normal,
            DK=DF, device=device, xp=xp
        )
    return Pt1, P1, PX, Np, tau2s 

def expectation_ko_reg1(
    X, TY, sigma2, piu, pi0,
    A=None, B=None, use_gating=True, delta_h = 0, use_lowm = True,
    d2f=None, tau2=None, 
    tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1],
    feat_normal=['cos'], XF=None, YF=None, device=None,
    xp = th, use_stable=False,
    ):
    D = X.shape[1]
    P = kodist2(TY, X)
    P = P* (-1.0/ (2.0*sigma2) )
    cs = 0.5* D* xp.log(2.0* xp.pi* sigma2)

    if not d2f is None:
        if tau2_auto:
            for il in range(len(tau2)):
                P = P  + tau2_alpha/(-2.0*tau2[il])*d2f[il]
                cs += 0.5* DF[il]* xp.log(2.0*xp.pi*tau2[il])
        else: 
            P = P + d2f*tau2_alpha
            cs += 0

    P = P.exp()
    piu_i = LazyTensor( piu.reshape(-1).contiguous()[:, None, None])
    P = P * piu_i

    C = xp.exp(cs) * pi0

    up_tau2 = bool(tau2_auto and (not XF is None) and (not YF is None))
    tau2s = tau2
    dh = float(delta_h)
    lh = 1.0 - dh
    use_dh = bool(dh >0)

    if use_lowm:
        M, J = B.shape
        one = xp.ones((M, 1), dtype=B.dtype,device=B.device,)
        KTB = P.t() @ xp.cat([one, B], dim=1)
        SK = KTB[:, 0]
        SKH = (A * KTB[:, 1:]).sum(1)
        if use_dh:
            SKH = dh * SK + lh * SKH
        
        # if use_dh:
        #     SK = xp.sum(P, 0).to_dense()
        #     SKH = dh * SK + lh * (A * (P.T @ B)).sum(1)
        # else:
        #     SK  = xp.sum(P, 0).to_dense()
        #     SKH = (A * (P.T @ B)).sum(1)
        if use_gating:
            Cn = C + SK

            # CnSKH =  Cn * SKH
            # CnSKH.masked_fill_(CnSKH < eps * 10, 1.0)
            # CnHnr = SK/ CnSKH

            # Only columns with positive label-compatible mass are valid.
            # For delta_h=0, SKH=0 means strict label rejection.
            valid = SKH >  0 #0
            # Pt1 = SK/ Cn
            Pt1 = xp.zeros_like(SK)
            Pt1[valid] = SK[valid] / Cn[valid]
            CnHnr = xp.zeros_like(SKH)
            CnHnr[valid] = Pt1[valid] / SKH[valid]

            if use_dh:
                base = dh * CnHnr
                ACH = A * (lh * CnHnr)[:, None]
            else:
                ACH = A * CnHnr[:, None]
        else:
            CH  = C + SKH
            Pt1 = SKH/ CH
            if use_dh:
                base = dh / CH
                ACH = A * (lh / CH)[:, None]
            else:
                ACH = A / CH[:, None]

        # Np  = xp.sum(Pt1)
        # PX  = xp.zeros_like(TY)
        
        B_i = LazyTensor(B.contiguous()[:, None, :])
        ACH_j = LazyTensor(ACH.contiguous()[None, :, :])
        Q = (B_i | ACH_j)
        if use_dh:
            base_j = LazyTensor(base.contiguous()[None, :, None])
            Q = Q + base_j
        PH = P * Q
    else:
        B_i = LazyTensor(B.contiguous()[:, None, :])
        A_j = LazyTensor(A.contiguous()[None, :, :])
        H = (B_i * A_j).sum(-1)
        if use_dh:
            H = lh * H + dh
            
        if use_gating:
            SK = P.sum(dim=0).reshape(-1)
            SKH = (P * H).sum(dim=0).reshape(-1)

            Cn = C + SK

            # valid = SKH > (eps * 10)
            # CnHnr = xp.zeros_like(SKH)
            # CnHnr[valid] = SK[valid] / (Cn[valid] * SKH[valid])
            # CnSKH =  Cn * SKH
            # CnSKH.masked_fill_(CnSKH < eps * 10, 1.0)
            # CnHnr = SK/ CnSKH
            
            valid = SKH > 0
            # Pt1 = SK / Cn
            Pt1 = xp.zeros_like(SK)
            Pt1[valid] = SK[valid] / Cn[valid]
            CnHnr = xp.zeros_like(SKH)
            CnHnr[valid] = Pt1[valid] / SKH[valid]

            CnHnr_j = LazyTensor(CnHnr.contiguous()[None, :, None])
            PH = P * H * CnHnr_j

        else:
            SKH = (P * H).sum(dim=0).reshape(-1)
            CH = C + SKH
            Pt1 = SKH / CH
            CH_j = LazyTensor((1.0 / CH).contiguous()[None, :, None])
            PH = P * H * CH_j

    Np  = xp.sum(Pt1)
    X1 = th.cat( [th.ones_like(X[:, :1]), X], dim=1, ).contiguous()
    X1_j = LazyTensor(X1[None, :, :])
    Pstat = (PH * X1_j).sum(dim=1)
    P1 = Pstat[:, 0]
    PX = Pstat[:, 1:]

    if up_tau2:
        trPfgs = []
        for i, xf in enumerate(XF):
            YF_i = LazyTensor(YF[i].contiguous()[:, None, :])
            XF_j = LazyTensor(xf.contiguous()[None, :, :])
            Fij = (YF_i * XF_j).sum(-1)
            trPfgs.append((PH * Fij).sum(dim=1).sum())

        tau2s = update_tau2_v1(
            Pt1, P1, trPfgs, Np, XF, YF,
            use_keops=False, feat_normal=feat_normal,
            DK=DF, device=device, xp=xp
        )
    return Pt1, P1, PX, Np, tau2s

def expectation_xp_reg2(
    X, TY, sigma2, piu, pi0,
    A=None, B=None, use_gating=True, delta_h=0, use_lowm=True,
    xp=th, d2f=None, tau2=None,
    tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1],
    feat_normal=['cos'], XF=None, YF=None,
    use_stable=False, device=None, log_qt=True, qt=0.75,
):
    D = X.shape[1]
    P = thdist2(TY, X)
    P.mul_(-1.0 / (2.0 * sigma2))

    if d2f is not None:
        if tau2_auto:
            for il in range(len(tau2)):
                P.add_( d2f[il], alpha=tau2_alpha / (-2.0 * tau2[il]))
        else:
            P.add_(d2f, alpha=tau2_alpha)

    if not use_stable:
        P.exp_()
        P.mul_(piu[:, None])

        SK = xp.sum(P, 0).to_dense()
        C = xp.quantile(SK, qt) * pi0 / (1.0 - pi0)

    else:
        P.add_(xp.log(piu[:, None] + eps))

        log_SK = xp.logsumexp(P, dim=0)
        lmax = log_SK.max()

        if log_qt:
            c0u = xp.exp( xp.quantile( (log_SK - lmax), qt ))
            log_C = xp.log( c0u * pi0 / (1.0 - pi0)) + lmax
        else:
            c0u = xp.quantile( xp.exp(log_SK - lmax), qt ) 
            log_C = xp.log( c0u * pi0 / (1.0 - pi0)) + lmax
        pmax = xp.maximum( xp.amax(P, dim=0),log_C)

        P.sub_(pmax[None, :])
        P.exp_()

        SK = xp.sum(P, 0).to_dense()
        C = xp.exp(log_C - pmax)

    up_tau2 = bool(tau2_auto and (not XF is None) and (not YF is None))

    tau2s = tau2
    dh = float(delta_h)
    lh = 1.0 - dh
    use_dh = bool(dh > 0)

    if use_lowm:
        SKH = (A * (P.T @ B)).sum(1)
        if use_dh:
            SKH = dh * SK + lh * SKH

        if use_gating:
            Cn = C + SK

            valid = SKH > 0
            Pt1 = xp.zeros_like(SK)
            Pt1[valid] = SK[valid] / Cn[valid]
            CnHnr = xp.zeros_like(SKH)
            CnHnr[valid] = Pt1[valid] / SKH[valid]

            if use_dh:
                base = dh * CnHnr
                ACH = A * (lh * CnHnr)[:, None]
            else:
                ACH = A * CnHnr[:, None]
        else:
            CH = C + SKH
            # CH.masked_fill_(CH == 0, 1.0)
            Pt1 = SKH / CH
            if use_dh:
                base = dh / CH
                ACH = A * (lh / CH)[:, None]
            else:
                ACH = A / CH[:, None]

        Np = xp.sum(Pt1)
        PX = xp.zeros_like(TY)

        if use_dh:
            P1 = (B * (P @ ACH)).sum(1) + P @ base
            for iD in range(D):
                PX[:, iD] = (B * (P @ (ACH * X[:, [iD]]))).sum(1) + P @ (base * X[:, iD])
        else:
            P1 = (B * (P @ ACH)).sum(1)
            for iD in range(D):
                PX[:, iD] = (B * (P @ (ACH * X[:, [iD]]))).sum(1)

        if up_tau2:
            trPfgs = []
            for i, xf in enumerate(XF):
                trPfg = 0
                iDF = xf.shape[1]
                for idf in range(iDF):
                    trPfg += ( (YF[i][:, [idf]] * B) * (P @ (ACH * xf[:, [idf]]))).sum()
                    if use_dh:
                        trPfg += (YF[i][:, idf]* (P @ (base * xf[:, idf]))).sum()
                trPfgs.append(trPfg)

            tau2s = update_tau2_v1(
                Pt1, P1, trPfgs, Np, XF, YF,
                use_keops=False,
                feat_normal=feat_normal,
                DK=DF,
                device=device,
                xp=xp,
            )

    else:
        if use_gating:
            if use_dh:
                P.mul_((lh * B) @ A.T + dh)
            else:
                P.mul_(B @ A.T)
            SKH = xp.sum(P, 0).to_dense()
            Cn = C + SK

            valid = SKH > 0
            Pt1 = xp.zeros_like(SK)
            Pt1[valid] = SK[valid] / Cn[valid]
            CnHnr = xp.zeros_like(SKH)
            CnHnr[valid] = Pt1[valid] / SKH[valid]
    
            P.mul_(CnHnr)
        else:
            if use_dh:
                P.mul_((lh * B) @ A.T + dh)
            else:
                P.mul_(B @ A.T)
            CH = xp.sum(P, 0).to_dense() + C
            CH.masked_fill_(CH == 0, 1.0) #TOCHECK
            P.mul_(1.0 / CH)
            Pt1 = xp.sum(P, 0).to_dense()

        P1 = xp.sum(P, 1).to_dense()
        Np = xp.sum(P1)
        PX = P @ X

        if up_tau2:
            trPfgs = []
            for i, xf in enumerate(XF):
                trPfgs.append( (YF[i] * (P @ xf)).sum() )
            tau2s = update_tau2_v1(
                Pt1, P1, trPfgs, Np, XF, YF,
                use_keops=False,
                feat_normal=feat_normal,
                DK=DF,
                device=device,
                xp=xp,
            )
    return Pt1, P1, PX, Np, tau2s

def expectation_ko_reg2(
    X, TY, sigma2, piu, pi0,
    A=None, B=None, use_gating=True, delta_h=0, use_lowm=True,
    d2f=None, tau2=None,
    tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1],
    feat_normal=['cos'], XF=None, YF=None, device=None,
    xp=th, use_stable=False, log_qt =True, qt=0.75,
):
    P = kodist2(TY, X)
    P = P * (-1.0 / (2.0 * sigma2))
    if d2f is not None:
        if tau2_auto:
            for il in range(len(tau2)):
                P = P + tau2_alpha / (-2.0 * tau2[il]) * d2f[il]
        else:
            P = P + d2f * tau2_alpha
    P = P.exp()
    piu_i = LazyTensor( piu.reshape(-1).contiguous()[:, None, None])
    P = P * piu_i

    up_tau2 = bool(tau2_auto and (not XF is None) and (not YF is None))
    tau2s = tau2
    dh = float(delta_h)
    lh = 1.0 - dh
    use_dh = bool(dh > 0)

    if use_lowm:
        M, J = B.shape
        one = xp.ones((M, 1), dtype=B.dtype,device=B.device,)
        KTB = P.t() @ xp.cat([one, B], dim=1)
        SK = KTB[:, 0]
        SKH = (A * KTB[:, 1:]).sum(1)

        if log_qt:
            c0u = xp.quantile( xp.log1p(SK), qt)
            c0 = xp.expm1(c0u) / (1.0 - pi0)
        else:
            c0 = xp.quantile(SK, qt) / (1.0 - pi0)
        C = c0 * pi0

        if use_dh:
            SKH = dh * SK + lh * SKH

        if use_gating:
            Cn = C + SK
            Pt1 = SK / Cn

            valid = SKH > 0
            Pt1 = xp.zeros_like(SK)
            Pt1[valid] = SK[valid] / Cn[valid]
            CnHnr = xp.zeros_like(SKH)
            CnHnr[valid] = Pt1[valid] / SKH[valid]

            if use_dh:
                base = dh * CnHnr
                ACH = A * (lh * CnHnr)[:, None]
            else:
                ACH = A * CnHnr[:, None]

        else:
            CH = C + SKH
            # CH.masked_fill_(CH == 0, 1.0)
            Pt1 = SKH / CH

            if use_dh:
                base = dh / CH
                ACH = A * (lh / CH)[:, None]
            else:
                ACH = A / CH[:, None]

        B_i = LazyTensor(B.contiguous()[:, None, :])
        ACH_j = LazyTensor(ACH.contiguous()[None, :, :])
        Q = (B_i | ACH_j)
        if use_dh:
            base_j = LazyTensor(base.contiguous()[None, :, None])
            Q = Q + base_j
        PH = P * Q
    else:
        B_i = LazyTensor(B.contiguous()[:, None, :])
        A_j = LazyTensor(A.contiguous()[None, :, :])
        H = (B_i * A_j).sum(-1)

        if use_dh:
            H = lh * H + dh

        SK = P.sum(dim=0).reshape(-1)
        if log_qt:
            c0u = xp.quantile( xp.log1p(SK), qt)
            c0 = xp.expm1(c0u) / (1.0 - pi0)
        else:
            c0 = xp.quantile(SK, qt) / (1.0 - pi0)
        C = c0 * pi0

        if use_gating:
            SKH = (P * H).sum(dim=0).reshape(-1)
            Cn = C + SK

            valid = SKH > 0
            Pt1 = xp.zeros_like(SK)
            Pt1[valid] = SK[valid] / Cn[valid]
            CnHnr = xp.zeros_like(SKH)
            CnHnr[valid] = Pt1[valid] / SKH[valid]

            CnHnr_j = LazyTensor(CnHnr.contiguous()[None, :, None])
            PH = P * H * CnHnr_j
        else:
            SKH = (P * H).sum(dim=0).reshape(-1)
            CH = C + SKH
            # CH.masked_fill_(CH == 0, 1.0)
            Pt1 = SKH / CH
            CH_j = LazyTensor((1.0 / CH).contiguous()[None, :, None])
            PH = P * H * CH_j

    Np = xp.sum(Pt1)
    X1 = th.cat( [th.ones_like(X[:, :1]), X], dim=1, ).contiguous()
    X1_j = LazyTensor(X1[None, :, :])
    Pstat = (PH * X1_j).sum(dim=1)

    P1 = Pstat[:, 0]
    PX = Pstat[:, 1:]

    if up_tau2:
        trPfgs = []
        for i, xf in enumerate(XF):
            YF_i = LazyTensor(YF[i].contiguous()[:, None, :])
            XF_j = LazyTensor(xf.contiguous()[None, :, :])
            Fij = (YF_i * XF_j).sum(-1)
            trPfgs.append((PH * Fij).sum(dim=1).sum())

        tau2s = update_tau2_v1(
            Pt1, P1, trPfgs, Np, XF, YF,
            use_keops=False, feat_normal=feat_normal,
            DK=DF, device=device, xp=xp
        )
    return Pt1, P1, PX, Np, tau2s

def update_P_xp(X, Y, sigma2, gs, xp = th, d2f=None, tau2=None, 
             tau2_auto=False, use_stable=False, eps=1e-9, tau2_alpha=1.0, DF=1):
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
    if not use_stable:
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
                     xp = th, use_stable=False,
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
                P = P  + tau2_alpha/(-2.0*tau2[il])*d2f[il]
                cs += 0.5* DF[il]* xp.log(2.0*xp.pi*tau2[il])
        else: 
            P = P + d2f*tau2_alpha
            cs += 0

    P = P.exp()
    cs = xp.exp(cs)*gs
    return P, cs

def update_Cost_ko(X, Y, sigma2, xp = th, d2f=None, tau2=None, 
                tau2_auto=False, eps=1e-9, tau2_alpha=1.0):
    D = X.shape[1]
    P = kodist2(Y, X)
    P = P* (-1.0/ (2.0*sigma2) )

    if not d2f is None:
        if tau2_auto:
            for il in range(len(tau2)):
                P = P  + tau2_alpha/(-2.0*tau2[il])*d2f[il]
        else: 
            P = P + d2f*tau2_alpha
    return P

def update_Cost_xp(X, Y, sigma2, xp = th, d2f=None, tau2=None, 
             tau2_auto=False, eps=1e-12, tau2_alpha=1.0):
    D = X.shape[1]
    P = thdist2(Y, X)
    P.mul_(-1.0/ (2.0*sigma2) )
    if not d2f is None:
        if tau2_auto:
            for il in range(len(tau2)):
                P.add_(d2f[il], alpha=tau2_alpha/(-2.0*tau2[il]))
        else:
            P.add_(d2f, alpha=tau2_alpha)
    return P

def ot_relerr(x, y, base=1.0):
    return (x - y).abs().max() / th.maximum(
        th.maximum(x.abs().max(), y.abs().max()), x.new_tensor(base)
    )

def _kernel_power(K, logK=None, power=1.0, use_keops=True):
    power = float(power)
    if abs(power - 1.0) < 1e-8:
        return K
    if use_keops:
        return (power * logK).exp()
    return K.pow(power) 

def expectation_imbids_otm(
    X, TY, sigma2,
    C=None,
    d2f=None,
    tau2=None,
    tau2_auto=False,
    eps=1e-9,
    tau2_alpha=1.0,
    feat_normal=['cos'],
    XF=None,
    YF=None,
    device=None,
    xp=th,
    DF=[1],
    use_keops=True,
    ppas=None,
    iL=None,
    use_stable=False,
):
    """
    Entropic implicit Bi-DS-OTM expectation step.

    Split-budget optimized path:
        1. build pre-budget kernel K
        2. keep current split budget a,b,h,g fixed inside UV loop
        3. solve UV by the selected marginal solver
        4. update next split budget mainly from posterior P1/Pt1
        5. use Direct-KL support evidence only as a weak prior

    The key correction relative to the old split path is:
        Direct-KL evidence no longer overwrites a,b,h,g before UV.
        It only participates after UV as a weak prior for the next budget.

    solver_mode:
        'closed' : split residual KL with closed-form gate; exact only when tau = eps + rho
        'uot'    : ordinary UOT update without residual KL
        'bisect' : exact split residual KL by bisection
        'newton' : split residual KL by Newton
        'linear' : linearized residual KL
    """
    D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]
    uv_iter = ppas['otm_uv_iter']
    uv_tol = ppas['otm_tol']
    ab_iter = ppas['otm_ab_iter']
    ab_tol = 10

    epsv = max(float(ppas['otm_eps'][iL]), max(float(eps), 1e-9))
    tiny = max(float(eps), 1e-9)

    if 'otm_the_x' in ppas:
        thy = float(ppas['otm_the_y'][iL])
        thx = float(ppas['otm_the_x'][iL])
        rho_y = epsv * thy / max(1.0 - thy, tiny)
        rho_x = epsv * thx / max(1.0 - thx, tiny)
    else:
        rho_y = float(ppas['otm_rho_y'][iL])
        rho_x = float(ppas['otm_rho_x'][iL])
        thy = rho_y / (rho_y + epsv)
        thx = rho_x / (rho_x + epsv)

    solver = ppas['otm_solver'][iL]
    zqt = ppas['qt'][iL]
    rho_o = ppas['otm_rho_o'][iL]

    # closed solver is exact only when tau = eps + rho.
    # For the other split solvers tau is only the residual consistency weight.
    if solver == 'closed':
        tau_y = epsv + rho_y
        tau_x = epsv + rho_x
    elif solver == 'uot':
        tau_y = 0.0
        tau_x = 0.0
    else:
        tau_y = float(rho_o) * rho_y
        tau_x = float(rho_o) * rho_x

    a, b = ppas['as'][iL], ppas['bs'][iL]
    u, v = ppas['us'][iL], ppas['vs'][iL]
    iwo = ppas['wo'][iL]
    w_clip = ppas['w_clip']

    g_dr = ppas['g_dr']
    w_dr = ppas['w_dr']

    if solver == 'uot':
        budget_gamma = ppas['budget_gamma'][iL]
        smooth_mode = ppas['smooth_mode'][iL]
        budget_mode  = ppas['budget_mode'][iL]
    else:
        budget_eta = ppas['budget_eta'][iL]

    # gamma_k = ppas.get('split_gamma', 1.0)
    # eta_k = ppas.get('split_eta_k', 1.0)

    # # zeta: budget damping inside ab-loop.
    # zeta = ppas.get('split_zeta', 0.75)
    # budget_eta = ppas.get('split_budget_eta', 0.80)
    # loq = ppas.get('split_loq', 0.005)
    # hiq = ppas.get('split_hiq', 0.995)

    # # Posterior should dominate budget update.
    # # Direct-KL is only a weak prior.
    # post_w = ppas.get('split_post_w', 0.80)
    # shape_w = ppas.get('split_shape_w', 0.10)
    # prior_w = ppas.get('split_prior_w', 0.10)

    S_y = None
    S_x = None

    # ------------------------------------------------------------
    # Build logK
    # Case A: C is provided externally as a true OT cost
    #         logK = -C / eps
    # Case B: C is None, build score S from Gaussian/feature term
    #         S = -cost_base, so logK = S / eps
    # ------------------------------------------------------------
    use_stable = False  # TODO
    if C is None:
        if use_keops:
            S = update_Cost_ko(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            logK = S / epsv
            K = logK.exp()
        else:
            K = update_Cost_xp(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            K.div_(epsv).exp_()
            logK = None
    else:
        if use_keops:
            C = C.to(device=X.device, dtype=X.dtype)
            logK = C / (-epsv)
            K = logK.exp()
        else:
            # inplace: C will be overwritten by K to save memory
            K = C
            K.div_(-epsv).exp_()
            logK = None

    h = g = None
    P1 = Pt1 = Np = None

    Nref = X.new_tensor(float(Ni))
    sf = Nref
    mu = (Nref / Mi).expand(Mi)
    nu = (Nref / Ni).expand(Ni)

    for _ab in range(ab_iter):
        if a is None:
            a = (mu * iwo)
        if b is None:
            b = (nu * iwo)

        if u is None:
            u = th.ones_like(a)
        if v is None:
            v = th.ones_like(b)

        # --------------------------------------------------------
        # Current split budget is fixed inside this UV loop.
        # Direct-KL does NOT overwrite a,b,h,g here.
        # --------------------------------------------------------
        if solver != 'uot':
            h = mu - a
            g = nu - b
        else:
            h = None
            g = None

        if solver == 'linear':
            r0 = a
            c0 = b
        else:
            r0 = None
            c0 = None

        # --------------------------------------------------------
        # Sinkhorn under fixed budget.
        # --------------------------------------------------------
        for _uv in range(uv_iter):
            u_prev = u.clone()
            v_prev = v.clone()

            Kv = (K @ v).view(-1).clamp_min(tiny)
            r, u = uv_solver(
                Kv, a, h, mu, epsv, rho_y, tau_y,
                solver=solver, r0=r0, the=thy, eps=tiny
            )

            KTu = (K.T @ u).view(-1).clamp_min(tiny)
            c, v = uv_solver(
                KTu, b, g, nu, epsv, rho_x, tau_x,
                solver=solver, r0=c0, the=thx, eps=tiny
            )

            if solver == 'linear':
                r0 = r
                c0 = c

            err_u = ot_relerr(u, u_prev, base=1.0)
            err_v = ot_relerr(v, v_prev, base=1.0)
            err = 0.5 * (err_u + err_v)

            if err < uv_tol:
                break

        P1 = u * (K @ v).view(-1)
        Pt1 = v * (K.T @ u).view(-1)
        Np = P1.sum().clamp_min(tiny)
        wuot = (Np / Nref).clamp(0, 1)

        if solver == 'uot':
            w_pre= None
            iwo_n = wuot.clip(0,1)
            iwo = w_dr * iwo + (1.0 - w_dr) * iwo_n
            # iwo = w_dr * ppas['wo'][iL] + (1.0 - w_dr) * iwo_n
            iwo = xp.clip(iwo, *ppas['w_clip'])
            kappa = Nref * iwo

            a = imbids_otm_refresh_budget(
                P1,
                a=a,
                bm=budget_mode,
                sm=smooth_mode,
                gamma=budget_gamma,
                kappa=kappa,
                smoother=None,
                eps=eps,
            )
            b = imbids_otm_refresh_budget(
                Pt1,
                a=b,
                bm=budget_mode,
                sm=smooth_mode,
                gamma=budget_gamma,
                kappa=kappa,
                smoother=None,
                eps=eps,
            )
        else:
            zeta = g_dr
            gamma_k = 1.0
            eta_k = 1.0
            loq = 0.005
            hiq = 0.995
    
            prior_w =  1.0
            post_w = 1.0 - prior_w

            # if post_w > 0:
            #     alpha_post = P1 / mu
            #     beta_post = Pt1 / nu
            alpha_shape, beta_shape = 0, 0
            alpha_prior, beta_prior = 0, 0

            if post_w > 0:
                alpha_shape = imbids_qt_budget(
                    P1, iwo,
                    gamma=gamma_k,
                    zqt=zqt,
                    smoother=S_y,
                    loq=loq,
                    hiq=hiq,
                    eps=tiny,
                )

                beta_shape = imbids_qt_budget(
                    Pt1, iwo,
                    gamma=gamma_k,
                    zqt=zqt,
                    smoother=S_x,
                    loq=loq,
                    hiq=hiq,
                    eps=tiny,
                )

            if prior_w > 0:
                q0 = a * (sf / a.sum())
                p0 = b * (sf / b.sum())

                Ke = _kernel_power(
                    K, logK=logK, power=eta_k, use_keops=use_keops
                )

                E = (Ke @ p0).view(-1).clamp_min(tiny)
                F = (Ke.T @ q0).view(-1).clamp_min(tiny)

                alpha_prior = imbids_qt_budget(
                    E, iwo,
                    gamma=gamma_k,
                    zqt=zqt,
                    smoother=S_y,
                    loq=loq,
                    hiq=hiq,
                    eps=tiny,
                )

                beta_prior = imbids_qt_budget(
                    F, iwo,
                    gamma=gamma_k,
                    zqt=zqt,
                    smoother=S_x,
                    loq=loq,
                    hiq=hiq,
                    eps=tiny,
                )

            ws = post_w  + prior_w
            alpha = ( post_w * alpha_shape + prior_w * alpha_prior ) / ws
            beta = ( post_w * beta_shape + prior_w * beta_prior ) / ws

            rq = mu * alpha
            cq = nu * beta

            a = zeta * a + (1.0 - zeta) * rq
            b = zeta * b + (1.0 - zeta) * cq

            if solver != 'uot':
                h = mu - a
                g = nu - b

            w_pre = (a.sum() / mu.sum() * b.sum() / nu.sum()).sqrt()
            iwo_n = (w_pre ** budget_eta) * (wuot ** (1.0 - budget_eta))
            iwo = w_dr * iwo + (1.0 - w_dr) * iwo_n
            iwo = xp.clip(iwo, *w_clip)

    ppas['wo'][iL] = iwo
    ppas['as'][iL] = a
    ppas['bs'][iL] = b
    # ppas['us'][iL] = u
    # ppas['vs'][iL] = v

    for ik, iv in zip(['us', 'vs'], [u, v]):
        if ppas[ik][iL] is None:
            ppas[ik][iL] = iv
        else:
            ppas[ik][iL] = g_dr * ppas[ik][iL] + (1.0 - g_dr) * iv

    PX = u[:, None] * (K @ (v[:, None] * X))

    if tau2_auto and (XF is not None) and (YF is not None):
        tau2s = update_tau2_otm(
            Pt1, P1, (u, K, v), Np, XF, YF,
            feat_normal=feat_normal, DK=DF, device=device, xp=xp
        )
    else:
        tau2s = tau2

    return Pt1, P1, PX, Np, tau2s, ppas

def expectation_imbids_otm_m5(
    X, TY, sigma2,
    C=None,
    d2f=None,
    tau2=None,
    tau2_auto=False,
    eps=1e-9,
    tau2_alpha=1.0,
    feat_normal=['cos'],
    XF=None,
    YF=None,
    device=None,
    xp=th,
    DF=[1],
    use_keops=True,
    ppas=None,
    iL=None,
    use_stable=False,
):
    """
    Entropic implicit Bi-DS-OTM expectation step.

    Split-budget optimized path:
        1. build pre-budget kernel K
        2. keep current split budget a,b,h,g fixed inside UV loop
        3. solve UV by the selected marginal solver
        4. update next split budget mainly from posterior P1/Pt1
        5. use Direct-KL support evidence only as a weak prior

    The key correction relative to the old split path is:
        Direct-KL evidence no longer overwrites a,b,h,g before UV.
        It only participates after UV as a weak prior for the next budget.

    solver_mode:
        'closed' : split residual KL with closed-form gate; exact only when tau = eps + rho
        'uot'    : ordinary UOT update without residual KL
        'bisect' : exact split residual KL by bisection
        'newton' : split residual KL by Newton
        'linear' : linearized residual KL
    """
    D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]
    uv_iter = ppas['otm_uv_iter']
    uv_tol = ppas['otm_tol']
    ab_iter = ppas['otm_ab_iter']
    ab_tol = 10

    epsv = max(float(ppas['otm_eps'][iL]), max(float(eps), 1e-9))
    tiny = max(float(eps), 1e-9)

    if 'otm_the_x' in ppas:
        thy = float(ppas['otm_the_y'][iL])
        thx = float(ppas['otm_the_x'][iL])
        rho_y = epsv * thy / max(1.0 - thy, tiny)
        rho_x = epsv * thx / max(1.0 - thx, tiny)
    else:
        rho_y = float(ppas['otm_rho_y'][iL])
        rho_x = float(ppas['otm_rho_x'][iL])
        thy = rho_y / (rho_y + epsv)
        thx = rho_x / (rho_x + epsv)

    solver = ppas['otm_solver'][iL]
    zqt = ppas['qt'][iL]
    rho_o = ppas['otm_rho_o'][iL]

    # closed solver is exact only when tau = eps + rho.
    # For the other split solvers tau is only the residual consistency weight.
    if solver == 'closed':
        tau_y = epsv + rho_y
        tau_x = epsv + rho_x
    elif solver == 'uot':
        tau_y = 0.0
        tau_x = 0.0
    else:
        tau_y = float(rho_o) * rho_y
        tau_x = float(rho_o) * rho_x

    a, b = ppas['as'][iL], ppas['bs'][iL]
    u, v = ppas['us'][iL], ppas['vs'][iL]
    iwo = ppas['wo'][iL]
    w_clip = ppas['w_clip']

    g_dr = ppas['g_dr']
    w_dr = ppas['w_dr']

    if solver == 'uot':
        budget_gamma = ppas['budget_gamma'][iL]
        smooth_mode = ppas['smooth_mode'][iL]
        budget_mode  = ppas['budget_mode'][iL]
    else:
        budget_eta = ppas['budget_eta'][iL]

    # gamma_k = ppas.get('split_gamma', 1.0)
    # eta_k = ppas.get('split_eta_k', 1.0)

    # # zeta: budget damping inside ab-loop.
    # zeta = ppas.get('split_zeta', 0.75)
    # budget_eta = ppas.get('split_budget_eta', 0.80)
    # loq = ppas.get('split_loq', 0.005)
    # hiq = ppas.get('split_hiq', 0.995)

    # # Posterior should dominate budget update.
    # # Direct-KL is only a weak prior.
    # post_w = ppas.get('split_post_w', 0.80)
    # shape_w = ppas.get('split_shape_w', 0.10)
    # prior_w = ppas.get('split_prior_w', 0.10)

    S_y = None
    S_x = None

    # ------------------------------------------------------------
    # Build logK
    # Case A: C is provided externally as a true OT cost
    #         logK = -C / eps
    # Case B: C is None, build score S from Gaussian/feature term
    #         S = -cost_base, so logK = S / eps
    # ------------------------------------------------------------
    use_stable = False  # TODO
    if C is None:
        if use_keops:
            S = update_Cost_ko(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            logK = S / epsv
            K = logK.exp()
        else:
            K = update_Cost_xp(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            K.div_(epsv).exp_()
            logK = None
    else:
        if use_keops:
            C = C.to(device=X.device, dtype=X.dtype)
            logK = C / (-epsv)
            K = logK.exp()
        else:
            # inplace: C will be overwritten by K to save memory
            K = C
            K.div_(-epsv).exp_()
            logK = None

    h = g = None
    P1 = Pt1 = Np = None

    Nref = X.new_tensor(float(Ni))
    sf = Nref
    mu = (Nref / Mi).expand(Mi)
    nu = (Nref / Ni).expand(Ni)

    for _ab in range(ab_iter):
        if a is None:
            a = (mu * iwo)
        if b is None:
            b = (nu * iwo)

        if u is None:
            u = th.ones_like(a)
        if v is None:
            v = th.ones_like(b)

        # --------------------------------------------------------
        # Current split budget is fixed inside this UV loop.
        # Direct-KL does NOT overwrite a,b,h,g here.
        # --------------------------------------------------------
        if solver != 'uot':
            h = mu - a
            g = nu - b
        else:
            h = None
            g = None

        if solver == 'linear':
            r0 = a
            c0 = b
        else:
            r0 = None
            c0 = None

        # --------------------------------------------------------
        # Sinkhorn under fixed budget.
        # --------------------------------------------------------
        for _uv in range(uv_iter):
            u_prev = u.clone()
            v_prev = v.clone()

            Kv = (K @ v).view(-1).clamp_min(tiny)
            r, u = uv_solver(
                Kv, a, h, mu, epsv, rho_y, tau_y,
                solver=solver, r0=r0, the=thy, eps=tiny
            )

            KTu = (K.T @ u).view(-1).clamp_min(tiny)
            c, v = uv_solver(
                KTu, b, g, nu, epsv, rho_x, tau_x,
                solver=solver, r0=c0, the=thx, eps=tiny
            )

            if solver == 'linear':
                r0 = r
                c0 = c

            err_u = ot_relerr(u, u_prev, base=1.0)
            err_v = ot_relerr(v, v_prev, base=1.0)
            err = 0.5 * (err_u + err_v)

            if err < uv_tol:
                break

        P1 = u * (K @ v).view(-1)
        Pt1 = v * (K.T @ u).view(-1)
        Np = P1.sum().clamp_min(tiny)
        wuot = (Np / Nref).clamp(0, 1)

        if solver == 'uot':
            w_pre= None
            iwo_n = wuot.clip(0,1)
            iwo = w_dr * iwo + (1.0 - w_dr) * iwo_n
            # iwo = w_dr * ppas['wo'][iL] + (1.0 - w_dr) * iwo_n
            iwo = xp.clip(iwo, *ppas['w_clip'])
            kappa = Nref * iwo

            a = imbids_otm_refresh_budget(
                P1,
                a=a,
                bm=budget_mode,
                sm=smooth_mode,
                gamma=budget_gamma,
                kappa=kappa,
                smoother=None,
                eps=eps,
            )
            b = imbids_otm_refresh_budget(
                Pt1,
                a=b,
                bm=budget_mode,
                sm=smooth_mode,
                gamma=budget_gamma,
                kappa=kappa,
                smoother=None,
                eps=eps,
            )
        else:
            zeta = g_dr
            gamma_k = 1.0
            eta_k = 1.0
            loq = 0.005
            hiq = 0.995
    
            prior_w =  1.0
            post_w = 1.0 - prior_w

            # if post_w > 0:
            #     alpha_post = P1 / mu
            #     beta_post = Pt1 / nu
            alpha_shape, beta_shape = 0, 0
            alpha_prior, beta_prior = 0, 0

            if post_w > 0:
                alpha_shape = imbids_qt_budget(
                    P1, iwo,
                    gamma=gamma_k,
                    zqt=zqt,
                    smoother=S_y,
                    loq=loq,
                    hiq=hiq,
                    eps=tiny,
                )

                beta_shape = imbids_qt_budget(
                    Pt1, iwo,
                    gamma=gamma_k,
                    zqt=zqt,
                    smoother=S_x,
                    loq=loq,
                    hiq=hiq,
                    eps=tiny,
                )

            if prior_w > 0:
                q0 = a * (sf / a.sum())
                p0 = b * (sf / b.sum())

                Ke = _kernel_power(
                    K, logK=logK, power=eta_k, use_keops=use_keops
                )

                E = (Ke @ p0).view(-1).clamp_min(tiny)
                F = (Ke.T @ q0).view(-1).clamp_min(tiny)

                alpha_prior = imbids_qt_budget(
                    E, iwo,
                    gamma=gamma_k,
                    zqt=zqt,
                    smoother=S_y,
                    loq=loq,
                    hiq=hiq,
                    eps=tiny,
                )

                beta_prior = imbids_qt_budget(
                    F, iwo,
                    gamma=gamma_k,
                    zqt=zqt,
                    smoother=S_x,
                    loq=loq,
                    hiq=hiq,
                    eps=tiny,
                )

            ws = post_w  + prior_w
            alpha = ( post_w * alpha_shape + prior_w * alpha_prior ) / ws
            beta = ( post_w * beta_shape + prior_w * beta_prior ) / ws

            rq = mu * alpha
            cq = nu * beta

            a = zeta * a + (1.0 - zeta) * rq
            b = zeta * b + (1.0 - zeta) * cq

            if solver != 'uot':
                h = mu - a
                g = nu - b

            w_pre = (a.sum() / mu.sum() * b.sum() / nu.sum()).sqrt()
            iwo_n = (w_pre ** budget_eta) * (wuot ** (1.0 - budget_eta))
            iwo = w_dr * iwo + (1.0 - w_dr) * iwo_n
            iwo = xp.clip(iwo, *w_clip)

    ppas['wo'][iL] = iwo
    ppas['as'][iL] = a
    ppas['bs'][iL] = b
    # ppas['us'][iL] = u
    # ppas['vs'][iL] = v

    for ik, iv in zip(['us', 'vs'], [u, v]):
        if ppas[ik][iL] is None:
            ppas[ik][iL] = iv
        else:
            ppas[ik][iL] = g_dr * ppas[ik][iL] + (1.0 - g_dr) * iv

    PX = u[:, None] * (K @ (v[:, None] * X))

    if tau2_auto and (XF is not None) and (YF is not None):
        tau2s = update_tau2_otm(
            Pt1, P1, (u, K, v), Np, XF, YF,
            feat_normal=feat_normal, DK=DF, device=device, xp=xp
        )
    else:
        tau2s = tau2

    return Pt1, P1, PX, Np, tau2s, ppas

def expectation_imbids_otm_v2(
    X, TY, sigma2,
    C=None,
    d2f=None,
    tau2=None,
    tau2_auto=False,
    eps=1e-9,
    tau2_alpha=1.0,
    feat_normal=['cos'],
    XF=None,
    YF=None,
    device=None,
    xp=th,
    DF=[1],
    use_keops=True,
    ppas=None,
    iL=None,
    use_stable=False,
):
    """
    Entropic implicit Bi-DS-OTM expectation step.

    Core path:
        1. build pre-budget kernel K
        2. fix current budget a,b,h,g
        3. solve Sinkhorn under fixed budget
        4. update next budget by posterior P1/Pt1 + weak Direct-KL prior

    solver_mode:
        'closed' : split residual KL with closed-form gate, default
        'uot'    : budget-only split + ordinary UOT update
        'bisect' : exact split residual KL with bisection
        'linear' : linearized residual KL
    """
    D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]
    uv_iter = ppas['otm_uv_iter']
    uv_tol = ppas['otm_tol']
    ab_iter = ppas['otm_ab_iter']
    ab_tol = 10

    epsv = max(float(ppas['otm_eps'][iL]), max(float(eps), 1e-9))
    tiny = max(float(eps), 1e-9)

    if 'otm_the_x' in ppas:
        thy = float(ppas['otm_the_y'][iL])
        thx = float(ppas['otm_the_x'][iL])
        rho_y = epsv * thy / max(1.0 - thy, tiny)
        rho_x = epsv * thx / max(1.0 - thx, tiny)
    else:
        rho_y = float(ppas['otm_rho_y'][iL])
        rho_x = float(ppas['otm_rho_x'][iL])
        thy = rho_y / (rho_y + epsv)
        thx = rho_x / (rho_x + epsv)

    solver = ppas['otm_solver'][iL]
    zqt = ppas['qt'][iL]
    rho_o = ppas['otm_rho_o'][iL]

    # ------------------------------------------------------------
    # closed solver is exact only when tau = eps + rho.
    # If solver == 'closed', do not use arbitrary rho_o.
    # ------------------------------------------------------------
    if solver == 'closed':
        tau_y = epsv + rho_y
        tau_x = epsv + rho_x
    else:
        tau_y = rho_o * rho_y
        tau_x = rho_o * rho_x

    a, b = ppas['as'][iL], ppas['bs'][iL]
    u, v = ppas['us'][iL], ppas['vs'][iL]
    iwo = ppas['wo'][iL]
    w_clip = ppas['w_clip']

    g_dr = ppas['g_dr']
    w_dr = ppas['w_dr']

    if solver == 'uot':
        budget_gamma = ppas['budget_gamma'][iL]
        smooth_mode = ppas['smooth_mode'][iL]
        budget_mode  = ppas['budget_mode'][iL]

    gamma_k = 1.0
    eta_k = 1.0

    # zeta now means posterior-budget damping inside ab-loop.
    # It is no longer used to force Direct-KL shape to dominate.
    zeta = 0.75
    budget_eta = 0.80
    loq = 0.005
    hiq = 0.995

    S_y = None
    S_x = None

    # ------------------------------------------------------------
    # Build logK
    # Case A: C is provided externally as a true OT cost
    #         logK = -C / eps
    # Case B: C is None, build score S from Gaussian/feature term
    #         S = -cost_base, so logK = S / eps
    # ------------------------------------------------------------
    use_stable = False #TODO
    if C is None:
        if use_keops:
            S = update_Cost_ko(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            logK = S / epsv
            K = logK.exp()
        else:
            K = update_Cost_xp(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            K.div_(epsv).exp_()
            logK = None
    else:
        if use_keops:
            C = C.to(device=X.device, dtype=X.dtype)
            logK = C / (-epsv)
            K = logK.exp()
        else:
            # inplace: C will be overwritten by K to save memory
            K = C
            K.div_(-epsv).exp_()
            logK = None

    h = g = None
    P1 = Pt1 = Np = None

    Nref = X.new_tensor(float(Ni))
    sf = Nref
    mu = (Nref / Mi).expand(Mi)
    nu = (Nref / Ni).expand(Ni)
    # mu = (Nref / Mi)
    # nu = (Nref / Ni)

    for _ab in range(ab_iter):
        if a is None:
            a = (mu * iwo)
        if b is None:
            b = (nu * iwo)

        if u is None:
            u = th.ones(Mi, device=device, dtype=X.dtype)
        if v is None:
            v = th.ones(Ni, device=device, dtype=X.dtype)

        if solver != 'uot':
            h = (mu - a).clamp_min(0)
            g = (nu - b).clamp_min(0)

        if solver == 'linear':
            r0 = mu - h
            c0 = nu - g
        else:
            r0 = None
            c0 = None
        # --------------------------------------------------------
        # Sinkhorn.
        # Budget is fixed inside this UV loop.
        # --------------------------------------------------------
        for _uv in range(uv_iter):
            u_prev = u.clone()
            v_prev = v.clone()

            Kv = (K @ v).view(-1).clamp_min(tiny)
            r, u = uv_solver(
                Kv, a, h, mu, epsv, rho_y, tau_y,
                solver=solver, r0=r0, the=thy, eps=tiny
            )

            KTu = (K.T @ u).view(-1).clamp_min(tiny)
            c, v = uv_solver(
                KTu, b, g, nu, epsv, rho_x, tau_x,
                solver=solver, r0=c0, the=thx, eps=tiny
            )

            if solver == 'linear':
                r0 = r
                c0 = c

            err_u = ot_relerr(u, u_prev, base=1.0)
            err_v = ot_relerr(v, v_prev, base=1.0)
            err = 0.5 * (err_u + err_v)

            if err < uv_tol:
                break

        P1 = u * (K @ v).view(-1)
        Pt1 = v * (K.T @ u).view(-1)
        Np = P1.sum().clamp_min(tiny)

        prior_w = 0.0

        if prior_w > 0:
            # alpha_u = (P1 / mu).clamp(0.0, 1.0)
            # beta_u = (Pt1 / nu).clamp(0.0, 1.0)
        
            q0 = a * (sf / a.sum())
            p0 = b * (sf / b.sum())

            Ke = _kernel_power(
                K, logK=logK, power=eta_k, use_keops=use_keops
            )

            E = (Ke @ p0).view(-1).clamp_min(tiny)
            F = (Ke.T @ q0).view(-1).clamp_min(tiny)
        else:
            E = P1
            F = Pt1
            
        alpha_p = imbids_qt_budget(
            E, iwo,
            gamma=gamma_k,
            zqt=zqt,
            smoother=S_y,
            loq=loq,
            hiq=hiq,
            eps=tiny,
        )

        beta_p = imbids_qt_budget(
            F, iwo,
            gamma=gamma_k,
            zqt=zqt,
            smoother=S_x,
            loq=loq,
            hiq=hiq,
            eps=tiny,
        )
        # # rq ,cq = P1 * (mu/P1.sum()), Pt1 * (nu/Pt1.sum())
        # rq = (prior_w * mu) * alpha_p + (1.0 - prior_w) * rq
        # cq = (prior_w * nu) * beta_p  + (1.0 - prior_w) * cq
        rq = mu * alpha_p
        cq = nu * beta_p

        a = zeta * a + (1.0 - zeta) * rq
        b = zeta * b + (1.0 - zeta) * cq
        h = mu - a
        g = nu - b
        
        print(33333, mu.max(),  nu.max(), 
              h.max(), h.min(), g.max(), g.min())
        wuot = (Np / Nref).clamp(*w_clip)
        w_pre = (a.sum() / mu.sum() * b.sum() / nu.sum() ).sqrt()
        iwo_n = (w_pre ** budget_eta) * (wuot ** (1.0 - budget_eta))
        iwo = w_dr * iwo + (1.0 - w_dr) * iwo_n
        # iwo = w_dr * ppas['wo'][iL] + (1.0 - w_dr) * iwo_n
        iwo = xp.clip(iwo, *ppas['w_clip'])

        print(11111111111111, eps, iwo, w_pre, wuot, iwo_n, Np, Np / Ni, _uv ,a,b,u,v, 22222222222222)

    ppas['wo'][iL] = iwo
    ppas['as'][iL] = a
    ppas['bs'][iL] = b

    # for ik, iv in zip(['as', 'bs', 'us', 'vs'], [a, b, u, v]):
    for ik, iv in zip([ 'us', 'vs'], [ u, v]):
        if ppas[ik][iL] is None:
            ppas[ik][iL] = iv
        else:
            ppas[ik][iL] = g_dr * ppas[ik][iL] + (1.0 - g_dr) * iv

    PX = u[:, None] * (K @ (v[:, None] * X))

    if tau2_auto and (XF is not None) and (YF is not None):
        tau2s = update_tau2_otm(
            Pt1, P1, (u, K, v), Np, XF, YF,
            feat_normal=feat_normal, DK=DF, device=device, xp=xp
        )
    else:
        tau2s = tau2

    return Pt1, P1, PX, Np, tau2s, ppas

def expectation_imbids_otm_v3(
    X, TY, sigma2,
    C=None,
    d2f=None,
    tau2=None,
    tau2_auto=False,
    eps=1e-9,
    tau2_alpha=1.0,
    feat_normal=['cos'],
    XF=None,
    YF=None,
    device=None,
    xp=th,
    DF=[1],
    use_keops=True,
    ppas=None,
    iL=None,
    use_stable=False,
):
    """
    Entropic implicit Bi-DS-OTM expectation step.

    Core path:
        1. build pre-budget kernel K
        2. fix current budget a,b,h,g
        3. solve Sinkhorn under fixed budget
        4. update next budget by posterior P1/Pt1 + weak Direct-KL prior

    solver_mode:
        'closed' : split residual KL with closed-form gate, default
        'uot'    : budget-only split + ordinary UOT update
        'bisect' : exact split residual KL with bisection
        'linear' : linearized residual KL
    """
    D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]
    uv_iter = ppas['otm_uv_iter']
    uv_tol = ppas['otm_tol']
    ab_iter = ppas['otm_ab_iter']
    ab_tol = 10

    epsv = max(float(ppas['otm_eps'][iL]), max(float(eps), 1e-9))
    tiny = max(float(eps), 1e-9)

    if 'otm_the_x' in ppas:
        thy = float(ppas['otm_the_y'][iL])
        thx = float(ppas['otm_the_x'][iL])
        rho_y = epsv * thy / max(1.0 - thy, tiny)
        rho_x = epsv * thx / max(1.0 - thx, tiny)
    else:
        rho_y = float(ppas['otm_rho_y'][iL])
        rho_x = float(ppas['otm_rho_x'][iL])
        thy = rho_y / (rho_y + epsv)
        thx = rho_x / (rho_x + epsv)

    solver = ppas['otm_solver'][iL]
    zqt = ppas['qt'][iL]
    rho_o = ppas['otm_rho_o'][iL]

    if solver == 'closed':
        tau_y = epsv + rho_y
        tau_x = epsv + rho_x
    else:
        tau_y = rho_o * rho_y
        tau_x = rho_o * rho_x

    a, b = ppas['as'][iL], ppas['bs'][iL]
    u, v = ppas['us'][iL], ppas['vs'][iL]
    iwo = ppas['wo'][iL]
    w_clip = ppas['w_clip']

    g_dr = ppas['g_dr']
    w_dr = ppas['w_dr']

    if solver == 'uot':
        budget_gamma = ppas['budget_gamma'][iL]
        smooth_mode = ppas['smooth_mode'][iL]
        budget_mode  = ppas['budget_mode'][iL]

    gamma_k = 1.0
    eta_k = 1.0

    # zeta now means posterior-budget damping inside ab-loop.
    # It is no longer used to force Direct-KL shape to dominate.
    zeta = 0.75
    budget_eta = 0.80
    loq = 0.005
    hiq = 0.995

    S_y = None
    S_x = None

    # ------------------------------------------------------------
    # Build logK
    # Case A: C is provided externally as a true OT cost
    #         logK = -C / eps
    # Case B: C is None, build score S from Gaussian/feature term
    #         S = -cost_base, so logK = S / eps
    # ------------------------------------------------------------
    use_stable = False #TODO
    if C is None:
        if use_keops:
            S = update_Cost_ko(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            logK = S / epsv
            K = logK.exp()
        else:
            K = update_Cost_xp(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            K.div_(epsv).exp_()
            logK = None
    else:
        if use_keops:
            C = C.to(device=X.device, dtype=X.dtype)
            logK = C / (-epsv)
            K = logK.exp()
        else:
            # inplace: C will be overwritten by K to save memory
            K = C
            K.div_(-epsv).exp_()
            logK = None

    h = g = None
    P1 = Pt1 = Np = None

    Nref = X.new_tensor(float(Ni))
    sf = Nref
    mu = (Nref / Mi).expand(Mi)
    nu = (Nref / Ni).expand(Ni)

    for _ab in range(ab_iter):
        if a is None:
            a = (mu * iwo)
        if b is None:
            b = (nu * iwo)

        if u is None:
            u = th.ones(Mi, device=device, dtype=X.dtype)
        if v is None:
            v = th.ones(Ni, device=device, dtype=X.dtype)

        if solver != 'uot':
            h = (mu - a).clamp_min(0)
            g = (nu - b).clamp_min(0)

        if solver == 'linear':
            r0 = mu - h
            c0 = nu - g
        else:
            r0 = None
            c0 = None
        # --------------------------------------------------------
        # Sinkhorn.
        # Budget is fixed inside this UV loop.
        # --------------------------------------------------------
        for _uv in range(uv_iter):
            u_prev = u.clone()
            v_prev = v.clone()

            Kv = (K @ v).view(-1).clamp_min(tiny)
            r, u = uv_solver(
                Kv, a, h, mu, epsv, rho_y, tau_y,
                solver=solver, r0=r0, the=thy, eps=tiny
            )

            KTu = (K.T @ u).view(-1).clamp_min(tiny)
            c, v = uv_solver(
                KTu, b, g, nu, epsv, rho_x, tau_x,
                solver=solver, r0=c0, the=thx, eps=tiny
            )

            if solver == 'linear':
                r0 = r
                c0 = c

            err_u = ot_relerr(u, u_prev, base=1.0)
            err_v = ot_relerr(v, v_prev, base=1.0)
            err = 0.5 * (err_u + err_v)

            if err < uv_tol:
                break

        P1 = (u * (K @ v).view(-1))
        Pt1 = (v * (K.T @ u).view(-1))
        Np = P1.sum().clamp_min(tiny)
        wuot = (Np / Nref).clamp(0, 1)

        # --------------------------------------------------------
        # Budget update after Sinkhorn.
        # This update is for the next ab-loop / next E-step.
        # --------------------------------------------------------
        if solver != 'uot':
            # ----------------------------------------------------
            # Direct-KL support evidence as weak prior.
            # It is computed after Sinkhorn and mixed with posterior.
            # ----------------------------------------------------
            # rt = P1/mu
            
            # Direct-KL evidence is only a weak prior.
            prior_w = 0.0
            rq ,cq = P1 * (mu/P1.sum()), Pt1 * (nu/Pt1.sum())
            if prior_w > 0:
                # alpha_u = (P1 / mu).clamp(0.0, 1.0)
                # beta_u = (Pt1 / nu).clamp(0.0, 1.0)
            
                q0 = a * (sf / a.sum())
                p0 = b * (sf / b.sum())

                Ke = _kernel_power(
                    K, logK=logK, power=eta_k, use_keops=use_keops
                )

                E = (Ke @ p0).view(-1).clamp_min(tiny)
                F = (Ke.T @ q0).view(-1).clamp_min(tiny)

                alpha_p = imbids_qt_budget(
                    E, iwo,
                    gamma=gamma_k,
                    zqt=zqt,
                    smoother=S_y,
                    loq=loq,
                    hiq=hiq,
                    eps=tiny,
                )

                beta_p = imbids_split_budget(
                    F, iwo,
                    gamma=gamma_k,
                    zqt=zqt,
                    smoother=S_x,
                    loq=loq,
                    hiq=hiq,
                    eps=tiny,
                )

                rq = (prior_w * mu) * alpha_p + (1.0 - prior_w) * rq
                cq = (prior_w * nu) * beta_p  + (1.0 - prior_w) * cq

            a = zeta * a + (1.0 - zeta) * rq
            b = zeta * b + (1.0 - zeta) * cq
            h = mu - a
            g = nu - b

            w_pre = (a.sum() / mu.sum() * b.sum() / nu.sum() ).sqrt()
            iwo_n = (w_pre ** budget_eta) * (wuot ** (1.0 - budget_eta))
            # iwo = w_dr * iwo + (1.0 - w_dr) * iwo_n
            iwo = w_dr * ppas['wo'][iL] + (1.0 - w_dr) * iwo_n
            iwo = xp.clip(iwo, *ppas['w_clip'])
        else:
            # ic_qt = 0.5 * (xp.quantile(P1, float(zqt)) + xp.quantile(P1, float(zqt)))
            # w_pre0 =  Np / (ic_qt*N +Np)
            # w_p1 = wuot * Np / ( wuot * Np  + (1-wuot) *  xp.quantile(P1, float(zqt)) * float(Mi))
            # w_p2 = wuot * Np / ( wuot * Np  + (1-wuot) *  xp.quantile(Pt1, float(zqt)) * float(Ni))
            # w_pre = (w_p1 * w_p2).sqrt()
            # iwo_n = (w_pre ** budget_eta) * (wuot ** (1.0 - budget_eta))

            w_pre = None
            iwo_n = wuot
            iwo = w_dr * iwo + (1.0 - w_dr) * iwo_n
            # iwo = (iwo ** w_dr) * (iwo_n **(1.0 - w_dr))
            # iwo = w_dr * ppas['wo'][iL] + (1.0 - w_dr) * iwo_n
            iwo = xp.clip(iwo, *ppas['w_clip'])
            kappa = Nref * iwo

            a = imbids_otm_refresh_budget(
                P1,
                a=a,
                bm=budget_mode,
                sm=smooth_mode,
                gamma=budget_gamma,
                kappa=kappa,
                smoother=None,
                eps=eps,
            )

            b = imbids_otm_refresh_budget(
                Pt1,
                a=b,
                bm=budget_mode,
                sm=smooth_mode,
                gamma=budget_gamma,
                kappa=kappa,
                smoother=None,
                eps=eps,
            )

    ppas['wo'][iL] = iwo
    for ik, iv in zip(['as', 'bs', 'us', 'vs'], [a, b, u, v]):
        if ppas[ik][iL] is None:
            ppas[ik][iL] = iv
        else:
            ppas[ik][iL] = g_dr * ppas[ik][iL] + (1.0 - g_dr) * iv
    print(11111111222211111, _uv, u, v, a,b, P1, Pt1, iwo, iwo_n)

    PX = u[:, None] * (K @ (v[:, None] * X))

    if tau2_auto and (XF is not None) and (YF is not None):
        tau2s = update_tau2_otm(
            Pt1, P1, (u, K, v), Np, XF, YF,
            feat_normal=feat_normal, DK=DF, device=device, xp=xp
        )
    else:
        tau2s = tau2

    return Pt1, P1, PX, Np, tau2s, ppas

def expectation_imbids_otm_v4(
    X, TY, sigma2,
    C=None,
    d2f=None,
    tau2=None,
    tau2_auto=False,
    eps=1e-9,
    tau2_alpha=1.0,
    feat_normal=['cos'],
    XF=None,
    YF=None,
    device=None,
    xp=th,
    DF=[1],
    use_keops=True,
    ppas=None,
    iL=None,
    use_stable=False,
):
    """
    Entropic implicit Bi-DS-OTM expectation step.

    Core path:
        1. build pre-budget kernel K
        2. estimate overlap shape by Direct-KL evidence
        3. split total mass into inlier budget a,b and residual budget h,g
        4. solve Sinkhorn by closed-form gate update by default

    solver_mode:
        'closed' : split residual KL with closed-form gate, default
        'uot'    : budget-only split + ordinary UOT update
        'bisect' : exact split residual KL with bisection
        'linear' : linearized residual KL
    """
    D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]
    uv_iter = ppas['otm_uv_iter']
    uv_tol = ppas['otm_tol']
    ab_iter = ppas['otm_ab_iter']
    ab_tol = 10

    epsv = max(float(ppas['otm_eps'][iL]), max(float(eps), 1e-9))
    tiny = max(float(eps), 1e-9)

    if 'otm_the_x' in ppas:
        thy = ppas['otm_the_y'][iL]
        thx = ppas['otm_the_x'][iL]
        rho_y = epsv * thy / (1.0 - thy)
        rho_x = epsv * thx / (1.0 - thx)
    else:
        rho_y = float(ppas['otm_rho_y'][iL])
        rho_x = float(ppas['otm_rho_x'][iL])
        thy = rho_y / (rho_y + epsv)
        thx = rho_x / (rho_x + epsv)

    solver = ppas['otm_solver'][iL]
    zqt = ppas['qt'][iL]
    rho_o = ppas['otm_rho_o'][iL]

    if solver == 'closed':
        tau_y = epsv + rho_y
        tau_x = epsv + rho_x
    else:
        tau_y = rho_o * rho_y
        tau_x = rho_o * rho_x

    a, b = ppas['as'][iL], ppas['bs'][iL]
    u, v = ppas['us'][iL], ppas['vs'][iL]
    iwo = ppas['wo'][iL]

    g_dr = ppas['g_dr']
    w_dr = ppas['w_dr']

    # rx, ry, otm_reg = None, None, 'entropy' #  TODO otm_reg='kl'

    if solver == 'uot':
        budget_gamma = ppas['budget_gamma'][iL]
        smooth_mode = ppas['smooth_mode'][iL]
        budget_mode  = ppas['budget_mode'][iL]

    gamma_k = 1.0
    eta_k = 1.0
    zeta = 0.9
    budget_eta = 0.8
    loq = 0.005
    hiq = 0.995

    S_y = None
    S_x = None

    # ------------------------------------------------------------
    # Build logK
    # Case A: C is provided externally as a true OT cost
    #         logK = -C / eps
    # Case B: C is None, build score S from Gaussian/feature term
    #         S = -cost_base, so logK = S / eps
    # ------------------------------------------------------------
    use_stable = False #TODO
    if C is None:
        if use_keops:
            S = update_Cost_ko(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            logK = S / epsv
            K = logK.exp()
        else:
            K = update_Cost_xp(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            K.div_(epsv).exp_()
            logK = None
    else:
        if use_keops:
            C = C.to(device=X.device, dtype=X.dtype)
            logK = C / (-epsv)
            K = logK.exp()
        else:
            # inplace: C will be overwritten by K to save memory
            K = C
            K.div_(-epsv).exp_()
            logK = None

    h = g = None
    P1 = Pt1 = Np = None
    Nref = X.new_tensor(float(Ni))
    sf = Nref
    mu = (Nref / Mi).expand(Mi)
    nu = (Nref / Ni).expand(Ni)

    for _ab in range(ab_iter):
        if a is None:
            a = (mu * iwo)
        if b is None:
            b = (nu * iwo)

        if u is None:
            u = th.ones_like(a)
        if v is None:
            v = th.ones_like(b)

        # --------------------------------------------------------
        # Direct-KL support evidence and split budget.
        # --------------------------------------------------------
        if solver != 'uot':
            q0 = a * (sf / a.sum())
            p0 = b * (sf / b.sum())
            Ke = _kernel_power(K, logK=logK, power=eta_k, use_keops=use_keops)

            E = (Ke @ p0).view(-1).clamp_min(tiny)  # E *sf
            F = (Ke.T @ q0).view(-1).clamp_min(tiny)  # F *sf
 
            a, alpha, h = imbids_split_budget(
                E, mu, iwo, gamma=gamma_k, zqt=zqt, smoother=S_y,
                loq=loq, hiq=hiq, eps=tiny,
            )
            b, beta, g  = imbids_split_budget(
                F, nu, iwo, gamma=gamma_k, zqt=zqt, smoother=S_x,
                loq=loq, hiq=hiq, eps=tiny
            )

        if (solver == 'linear'):
            r0 = mu - h
            c0 = nu - g
        else:
            r0 = None
            c0 = None

        # --------------------------------------------------------
        # Sinkhorn.
        # --------------------------------------------------------
        for _uv in range(uv_iter):
            u_prev = u.clone()
            v_prev = v.clone()

            Kv = (K @ v).view(-1).clamp_min(tiny)
            r, u = uv_solver(Kv, a, h, mu, epsv, rho_y, tau_y, solver=solver, r0=r0, the=thy, eps=tiny)

            KTu = (K.T @ u).view(-1).clamp_min(tiny)
            c, v = uv_solver(KTu, b, g, nu, epsv, rho_x, tau_x, solver=solver, r0=c0, the=thx, eps=tiny)

            if solver == 'linear':
                r0 = r
                c0 = c

            err_u = ot_relerr(u, u_prev, base=1.0)
            err_v = ot_relerr(v, v_prev, base=1.0)
            # err = xp.maximum(err_u, err_v)
            err = 0.5*(err_u + err_v)
            if err < uv_tol:
                break

        P1 = (u * (K @ v).view(-1)).clamp_min(tiny)
        Pt1 = (v * (K.T @ u).view(-1)).clamp_min(tiny)
        Np = P1.sum().clamp_min(tiny)

        wuot = Np / Nref
        # --------------------------------------------------------
        # Optional post-UOT shape correction.
        # Direct-KL q,p remain the main shape; P1/Pt1 only refine them.
        # --------------------------------------------------------
        if solver != 'uot':
            # # alpha_u = P1 / (P1 + h).clamp_min(tiny)
            # # beta_u  = Pt1 / (Pt1 + g).clamp_min(tiny)
            # alpha_u = (P1 / mu).clamp(0.0, 1.0) ##
            # beta_u = (Pt1 / nu).clamp(0.0, 1.0) ##
            
            # alpha = (zeta * alpha + (1.0 - zeta) * alpha_u).clamp(0.0, 1.0)
            # beta = (zeta * beta + (1.0 - zeta) * beta_u).clamp(0.0, 1.0)

            # a = mu * alpha
            # h = mu * (1.0 - alpha)
            # b = nu * beta
            # g = nu * (1.0 - beta)

            # # q = zeta * q + (1.0 - zeta) * sf /P1.sum() * P1
            # # p = zeta * p + (1.0 - zeta) * sf /Pt1.sum() * Pt1

            # # a = ((mu.sum() * iwo/sf) * q).clamp_min(tiny)
            # # b = ((nu.sum() * iwo/sf) * p).clamp_min(tiny)
            # # h = (mu - a).clamp_min(tiny)
            # # g = (nu - b).clamp_min(tiny)

            w_pre = (a.sum() / mu.sum() * b.sum() / nu.sum() ).sqrt()
            iwo_n = (w_pre ** budget_eta) * (wuot ** (1.0 - budget_eta))
            iwo = w_dr * iwo + (1.0 - w_dr) * iwo_n
            # iwo = w_dr * ppas['wo'][iL] + (1.0 - w_dr) * iwo_n
            iwo = xp.clip(iwo, *ppas['w_clip'])
        else:
            w_pre= None
            iwo_n = wuot.clip(0,1)
            iwo = w_dr * iwo + (1.0 - w_dr) * iwo_n
            # iwo = w_dr * ppas['wo'][iL] + (1.0 - w_dr) * iwo_n
            # iwo = xp.clip(iwo, *ppas['w_clip'])
            kappa = Nref * iwo

            a = imbids_otm_refresh_budget(P1, a =a, 
                    bm = budget_mode,
                    sm=smooth_mode, 
                    gamma=budget_gamma,
                    kappa = kappa,
                    # rho = rho_y, 
                    smoother=None,
                    eps=eps) #TODO
            b = imbids_otm_refresh_budget(Pt1, a =b, 
                    bm = budget_mode,
                    sm=smooth_mode, 
                    gamma=budget_gamma,
                    kappa = kappa,
                    # rho = rho_x, 
                    smoother=None,
                    eps=eps) #TODO

    ppas['wo'][iL] = iwo
    for ik, iv in zip(['as', 'bs', 'us', 'vs'], [a, b, u, v]):
        if ppas[ik][iL] is None:
            ppas[ik][iL] = iv
        else:
            ppas[ik][iL] = g_dr * ppas[ik][iL] + (1.0 - g_dr) * iv
    print(11111111222211111, _uv, u, v, a,b, P1, Pt1, iwo, iwo_n)

    PX = u[:, None] * (K @ (v[:, None] * X))
    if tau2_auto and (XF is not None) and (YF is not None):
        tau2s = update_tau2_otm(
            Pt1, P1, (u, K, v), Np, XF, YF,
            feat_normal=feat_normal, DK=DF, device=device, xp=xp
        )
    else:
        tau2s = tau2

    return Pt1, P1, PX, Np, tau2s, ppas

def expectation_imbids_otm_v1(
    X, TY, sigma2,
    C=None,
    d2f=None, tau2=None,
    tau2_auto=False, eps=1e-9, tau2_alpha=1.0,
    feat_normal=['cos'], XF=None, YF=None, device=None,
    xp=th, DF=[1], use_keops=True, ppas = None, iL = None, use_stable=False,
):
    """
    Entropic UOT expectation step (normal domain).
    Supports:
      - external OT cost C (true cost matrix), or
      - internally built Gaussian/feature score from X, TY, sigma2, d2f, tau2.

    Returns
    -------
    Pt1, P1, PX, Np, tau2s, u, v
    """
    D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]
    uv_iter, uv_tol, ab_iter, ab_tol = ppas['otm_uv_iter'], ppas['otm_tol'], ppas['otm_ab_iter'], 10
    eps_c, rho_y, rho_x = ppas['otm_eps'][iL], ppas['otm_rho_y'][iL],  ppas['otm_rho_x'][iL]

    u,v = ppas['us'][iL], ppas['vs'][iL]
    q0, p0 = ppas['as'][iL], ppas['bs'][iL]
    iwo = ppas['wo'][iL]
    budget_gamma , smooth_mode, budget_mode = ppas['budget_gamma'], ppas['smooth_mode'][iL], ppas['budget_mode'][iL]

    g_dr, w_dr = ppas['g_dr'], ppas['w_dr']
    tiny = max(float(eps), 1e-9)
    epsv = max(float(eps_c), tiny)

    thy =  rho_y / (rho_y + epsv)
    thx =  rho_x / (rho_x + epsv)

    rx, ry, otm_reg = None, None, 'entropy' #  TODO otm_reg='kl'

    if C is None:
        if use_keops:
            S = update_Cost_ko(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            logK = S / epsv
            K = logK.exp()
        else:
            K = update_Cost_xp(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            K.div_(epsv).exp_()
    else:
        C = C.to(device=X.device, dtype=X.dtype)
        logK = -C / epsv
        K = logK.exp()

    ## Direct-KL Refresh Budgets
    # scale to Nref, stable data p -> p* Nref; q -> q * Nref
    Nref = float(Ni)
    sf = X.new_tensor(Nref)
    if q0 is None:
        q0 = X.new_tensor(sf / Mi).expand(Mi)
    else:
        q0 = q0 * (sf / q0.sum())
    if p0 is None:
        p0 = X.new_tensor(sf / Ni).expand(Ni)
    else:
        p0 = p0 * (sf / p0.sum())

    Ey = K @ p0
    Ex = K.T @ q0
    budget_alpha = 0.05
    q = q0 * Ey.pow(float(budget_alpha))
    p = p0 * Ex.pow(float(budget_alpha))
    # q = q / q.sum()
    # p = p / p.sum()
    q = q * (sf / q.sum())
    p = p * (sf / p.sum())
    
    budget_eta = 0.0
    budget_zeta = 0.1
    if budget_eta > 0:
        Hq = -(q * q.clamp_min(tiny).log()).sum()/sf + sf.log()
        Hp = -(p * p.clamp_min(tiny).log()).sum()/sf + sf.log()

        iwo_yc = (Hq - xp.log(q.new_tensor(float(Mi)))).exp()
        iwo_xc = (Hp - xp.log(p.new_tensor(float(Ni)))).exp()
        iwo_c = (iwo_yc * iwo_xc).sqrt()

        if False:
            K_sim  = K
            iwo_s = (q * (K_sim @ p)).sum()/(sf*sf)
        else:
            iwo_s = 1.0
        iwo_pre = (iwo_c * iwo_s)
    else:
        iwo_pre = 1.0


    q_all = q
    p_all = p
    for _ab in range(ab_iter):
        iwo_all = (iwo_pre**budget_eta) * (iwo**(1-budget_eta))
        kappa = Nref * iwo_all / sf
        if u is None:
            u = th.ones_like(q)
        if v is None:
            v = th.ones_like(p)

        u = u.clamp_min(eps)
        v = v.clamp_min(eps)
        a = kappa * q_all
        b = kappa * p_all

        for _uv in range(uv_iter): #ra, rb
            u_prev = u.clone()
            v_prev = v.clone()

            Kv = K @ v
            Kv = Kv.clamp_min(tiny)
            u = (a / Kv).pow(thy)

            KTu = K.T @ u
            KTu = KTu.clamp_min(tiny)
            v = (b / KTu).pow(thx)

            err_u = ot_relerr(u, u_prev, base=1.0)
            err_v = ot_relerr(v, v_prev, base=1.0)
            # err = xp.maximum(err_u, err_v)
            err = 0.5*(err_u + err_v)
            if err < uv_tol:
                break

        P1  = u * (K @ v)               # row sums, shape (M,)
        Pt1 = v * (K.T @ u)             # col sums, shape (N,)
        Np  = xp.sum(P1)

        iwo = w_dr * ppas['wo'][iL] + (1.0 - w_dr) * (Np / Nref)
        iwo = xp.clip(iwo, min=ppas['w_clip'][0])

        q_all = (budget_zeta * q + (1-budget_zeta)*sf/Np * P1)  #/  (budget_eta * q.sum() + (1-budget_eta)* Np)
        p_all = (budget_zeta * p + (1-budget_zeta)*sf/Np * Pt1) #/  (budget_eta * p.sum() + (1-budget_eta)* Np)

    ppas['wo'][iL] = iwo
    ppas['bs'][iL] = p_all
    ppas['as'][iL] = q_all

    mapk =  zip(['bs','as', 'us', 'vs'], [p_all, q_all,u,v])
    # mapk =  zip(['us', 'vs'], [u,v])
    for ik, iv in mapk:
        if ppas[ik][iL] is None:
            ppas[ik][iL] = iv
        else:
            ppas[ik][iL] = g_dr * ppas[ik][iL] + (1.0 - g_dr) * iv

    PX  = u[:, None] * (K @ (v[:, None] * X))   # shape (M, D)
    if tau2_auto and (XF is not None) and (YF is not None):
        tau2s = update_tau2_otm(
            Pt1, P1, (u, K, v), Np, XF, YF,
            feat_normal=feat_normal, DK=DF, device=device, xp=xp
        )
    else:
        tau2s = tau2
    return Pt1, P1, PX, Np, tau2s, ppas

def expectation_imbids_otm_v0(
    X, TY, sigma2,
    C=None,
    d2f=None, tau2=None,
    tau2_auto=False, eps=1e-9, tau2_alpha=1.0,
    feat_normal=['cos'], XF=None, YF=None, device=None,
    xp=th, DF=[1], use_keops=True, ppas = None, iL = None, use_stable = False
):
    """
    Entropic UOT expectation step (normal domain).
    Supports:
      - external OT cost C (true cost matrix), or
      - internally built Gaussian/feature score from X, TY, sigma2, d2f, tau2.

    Returns
    -------
    Pt1, P1, PX, Np, tau2s, u, v
    """
    D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]
    uv_iter, uv_tol, ab_iter, ab_tol = ppas['otm_uv_iter'], ppas['otm_tol'], ppas['otm_ab_iter'], 10
    eps_c, thy, thx = ppas['otm_eps'][iL], ppas['otm_the_y'][iL],  ppas['otm_the_x'][iL]

    a, b, u,v = ppas['as'][iL], ppas['bs'][iL], ppas['us'][iL], ppas['vs'][iL]
    iwo = ppas['wo'][iL]
    budget_gamma , smooth_mode, budget_mode = ppas['budget_gamma'][iL], ppas['smooth_mode'][iL], ppas['budget_mode'][iL]    
    budget_eta = ppas['budget_eta'][iL]
    qt = ppas['qt'][iL]

    g_dr, w_dr = ppas['g_dr'], ppas['w_dr']
    tiny = max(float(eps), 1e-9)
    epsv = max(float(eps_c), tiny)

    # thy =  rho_y / (rho_y + epsv)
    # thx =  rho_x / (rho_x + epsv)

    rx, ry, otm_reg = None, None, 'entropy' #  TODO otm_reg='kl'

    # ------------------------------------------------------------
    # Build logK
    # Case A: C is provided externally as a true OT cost
    #         logK = -C / eps
    # Case B: C is None, build score S from Gaussian/feature term
    #         S = -cost_base, so logK = S / eps
    # ------------------------------------------------------------
    if C is None:
        if use_keops:
            S = update_Cost_ko(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            logK = S / epsv
            K = logK.exp()
        else:
            K = update_Cost_xp(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            K.div_(epsv).exp_()
    else:
        if use_keops:
            C = C.to(device=X.device, dtype=X.dtype)
            logK = C / (-epsv)
            K = logK.exp()
        else:
            K = C
            K.div_(-epsv).exp_()

    for _ab in range(ab_iter):
        kappa = float(Ni) * iwo
        if a is None:
            a = (kappa / Mi).expand(Mi)
        if b is None:
            b = (kappa / Ni).expand(Ni)
        if u is None:
            u = th.ones_like(a)
        if v is None:
            v = th.ones_like(b)
        
        if False: #TODO in dense 
            log_u = th.log(u.clamp_min(tiny))
            log_v = th.log(v.clamp_min(tiny))
            log_a = th.log(a.clamp_min(tiny))
            log_b = th.log(b.clamp_min(tiny))

            # --------------------------------------------------------
            # Log-domain UOT Sinkhorn.
            # --------------------------------------------------------
            for _uv in range(uv_iter):
                u_prev = log_u.exp()
                v_prev = log_v.exp()

                # logKv = _logsumexp_Kv(logK, log_v, use_keops=use_keops)
                if use_keops:
                    # log_v_j = LazyTensor(log_v[None, :, None])
                    logKv = (logK + log_v[None, :, None]).logsumexp(dim=1).view(-1)
                else:
                    logKv = (logK + log_v[None, :]).logsumexp(dim=1).view(-1)
                log_u = thy * (log_a - logKv)

                if use_keops:
                    # log_u_i = LazyTensor(log_u[:, None, None])
                    logKTu =  (logK + log_u[:, None, None]).logsumexp(dim=0).view(-1)
                else:
                    logKTu = (logK + log_u[:, None]).logsumexp(dim=0).view(-1) 
                log_v = thx * (log_b - logKTu)

                u = log_u.exp()
                v = log_v.exp()

                err_u = ot_relerr(u, u_prev, base=1.0)
                err_v = ot_relerr(v, v_prev, base=1.0)
                # err = xp.maximum(err_u, err_v)
                err = 0.5*(err_u + err_v)
                if err < uv_tol:
                    break
            if use_keops:
                logP = logK + log_u[:, None, None] + log_v[None, :, None]
            else:
                # logP = logK + log_u[:, None] + log_v[None, :]
                logK.add_(log_u[:, None])
                logK.add_(log_v[None, :])
                logP = logK
            P1 = logP.logsumexp(dim=1).view(-1).exp()
            Pt1 = logP.logsumexp(dim=0).view(-1).exp()
            Np = P1.sum()
        else:
            a = a.clamp_min(eps)
            b = b.clamp_min(eps)
            u = u.clamp_min(eps)
            v = v.clamp_min(eps)

            for _uv in range(uv_iter): #ra, rb
                u_prev = u.clone()
                v_prev = v.clone()

                Kv = K @ v
                Kv = Kv.clamp_min(tiny)
                u = (a / Kv).pow(thy)

                KTu = K.T @ u
                KTu = KTu.clamp_min(tiny)
                v = (b / KTu).pow(thx)

                err_u = ot_relerr(u, u_prev, base=1.0)
                err_v = ot_relerr(v, v_prev, base=1.0)
                # err = xp.maximum(err_u, err_v)
                err = 0.70*(err_u + err_v)
                if err < uv_tol:
                    break
        
            P1  = u * (K @ v)               # row sums, shape (M,)
            Pt1 = v * (K.T @ u)             # col sums, shape (N,)
            Np  = xp.sum(P1)

        iwo = w_dr * ppas['wo'][iL] + (1.0 - w_dr) * (Np / Ni)
        iwo = xp.clip(iwo, min=ppas['w_clip'][0])
        kappa = float(Ni) * iwo

        budget_zeta =1.0
        if budget_eta > 0 :
            # wowi = (1.0- iwo)/iwo
            wowi = 1.0
            if use_keops:
                Kb = (budget_eta * logK).exp()
                S = Kb.sum(1).view(-1)
                T = Kb.sum(0).view(-1)

                C0y = xp.quantile(S, qt)
                C0x = xp.quantile(T, qt)

                wy = (S + wowi* C0y).rsqrt()
                wx = (T + wowi *C0x).rsqrt()

                if budget_zeta > 0:
                    Keff = ((budget_eta + budget_zeta) * logK).exp()
                else:
                    Keff = Kb
                r_cur = (u * wy) * (Keff @ (v * wx))
                c_cur = (v * wx) * (Keff.T @ (u * wy))
            else:
                K.pow_(budget_eta)
                S = K.sum(1)
                T = K.sum(0)

                C0y = xp.quantile(S, qt)
                C0x = xp.quantile(T, qt)

                wy = (S + wowi* C0y).rsqrt()
                wx = (T + wowi* C0x).rsqrt()
                
                if budget_zeta > 0:
                    K.pow_(budget_zeta /budget_eta + 1)
                r_cur = (u * wy) * (K @ (v * wx))
                c_cur = (v * wx) * (K.T @ (u * wy))
                
                K.pow_(1.0 / (budget_zeta+budget_eta))
        else:
            r_cur = P1
            c_cur = Pt1
        a = imbids_otm_refresh_budget(r_cur, a =a, 
                bm = budget_mode,
                sm=smooth_mode, 
                gamma=budget_gamma,
                kappa = kappa,
                # rho = rho_y, 
                smoother=None,
                eps=eps) #TODO
        b = imbids_otm_refresh_budget(c_cur, a =b, 
                bm = budget_mode,
                sm=smooth_mode, 
                gamma=budget_gamma,
                kappa = kappa,
                # rho = rho_x, 
                smoother=None,
                eps=eps) #TODO
        print(11111111111111, eps, iwo, Np, Np / Ni, _uv ,a,b,u,v, 22222222222222)

    # mapk =  zip(['as','bs', 'us', 'vs'], [a,b,u,v])
    # a,b,u,v = [ iv if ppas[ik][iL] is None
    #                 else g_dr * ppas[ik][iL] + (1.0 - g_dr) * iv
    #             for ik, iv in mapk ]
    # ppas['as'][iL], ppas['bs'][iL], ppas['us'][iL], ppas['vs'][iL] = a,b,u,v
    ppas['wo'][iL] = iwo
    # mapk =  zip(['us', 'vs'], [u,v])
    mapk =  zip(['as','bs', 'us', 'vs'], [a,b,u,v])
    for ik, iv in mapk:
        if ppas[ik][iL] is None:
            ppas[ik][iL] = iv
        else:
            ppas[ik][iL] = g_dr * ppas[ik][iL] + (1.0 - g_dr) * iv

    PX  = u[:, None] * (K @ (v[:, None] * X))   # shape (M, D)
    if tau2_auto and (XF is not None) and (YF is not None):
        tau2s = update_tau2_otm(
            Pt1, P1, (u, K, v), Np, XF, YF,
            feat_normal=feat_normal, DK=DF, device=device, xp=xp
        )
    else:
        tau2s = tau2
    return Pt1, P1, PX, Np, tau2s, ppas

def expectation_imbids_otm_log(
    X, TY, sigma2,
    C=None,
    d2f=None, tau2=None,
    tau2_auto=False, eps=1e-9, tau2_alpha=1.0,
    feat_normal=['cos'], XF=None, YF=None, device=None,
    xp=th, DF=[1], use_keops=True, ppas=None, iL=None,
):
    """
    Log-domain entropic UOT expectation step for implicit OTM.

    This version keeps the original input/output interface.
    Main change:
        normal-domain u, v, K updates are replaced by log_u, log_v, logK.
    """
    D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]

    uv_iter = ppas['otm_uv_iter']
    uv_tol = ppas['otm_tol']
    ab_iter = ppas['otm_ab_iter']

    eps_c = ppas['otm_eps'][iL]
    rho_y = ppas['otm_rho_y'][iL]
    rho_x = ppas['otm_rho_x'][iL]

    a = ppas['as'][iL]
    b = ppas['bs'][iL]
    u = ppas['us'][iL]
    v = ppas['vs'][iL]

    iwo = ppas['wo'][iL]
    budget_gamma = ppas['budget_gamma']
    smooth_mode = ppas['smooth_mode'][iL]
    budget_mode = ppas['budget_mode'][iL]

    g_dr = ppas['g_dr']
    w_dr = ppas['w_dr']

    tiny = max(float(eps), 1e-10)
    epsv = max(float(eps_c), tiny)

    theta_y = rho_y / (rho_y + epsv)
    theta_x = rho_x / (rho_x + epsv)

    # ------------------------------------------------------------
    # Build logK directly.
    # Case A: C is true cost, logK = -C / eps.
    # Case B: C is None, S is Gaussian/feature score, logK = S / eps.
    # ------------------------------------------------------------
    if C is None:
        if use_keops:
            S = update_Cost_ko(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            logK = S / epsv
        else:
            logK = update_Cost_xp(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            logK.div_(epsv)
    else:
        C = C.to(device=X.device, dtype=X.dtype)
        logK = -C / epsv

    # ------------------------------------------------------------
    # Initialize log_u, log_v from previous normal-domain u, v.
    # This keeps compatibility with the original ppas['us'], ppas['vs'].
    # ------------------------------------------------------------
    log_u = None if u is None else th.log(u.clamp_min(tiny))
    log_v = None if v is None else th.log(v.clamp_min(tiny))

    for _ab in range(ab_iter):
        kappa = float(Ni) * iwo
        if a is None:
            a = (kappa / Mi).expand(Mi)
        if b is None:
            b = (kappa / Ni).expand(Ni)
        if u is None:
            u = th.ones_like(a)
        if v is None:
            v = th.ones_like(b)

        log_u = th.log(u.clamp_min(tiny))
        log_v = th.log(v.clamp_min(tiny))
        log_a = th.log(a.clamp_min(tiny))
        log_b = th.log(b.clamp_min(tiny))

        # --------------------------------------------------------
        # Log-domain UOT Sinkhorn.
        # --------------------------------------------------------
        for _uv in range(uv_iter):
            u_prev = log_u.exp()
            v_prev = log_v.exp()

            # logKv = _logsumexp_Kv(logK, log_v, use_keops=use_keops)
            if use_keops:
                # log_v_j = LazyTensor(log_v[None, :, None])
                logKv = (logK + log_v[None, :, None]).logsumexp(dim=1).view(-1)
            else:
                logKv = (logK + log_v[None, :]).logsumexp(dim=1).view(-1)
            log_u = theta_y * (log_a - logKv)

            if use_keops:
                # log_u_i = LazyTensor(log_u[:, None, None])
                logKTu =  (logK + log_u[:, None, None]).logsumexp(dim=0).view(-1)
            else:
                logKTu = (logK + log_u[:, None]).logsumexp(dim=0).view(-1) 
            log_v = theta_x * (log_b - logKTu)

            u = log_u.exp()
            v = log_v.exp()

            err_u = ot_relerr(u, u_prev, base=1.0)
            err_v = ot_relerr(v, v_prev, base=1.0)
            # err = xp.maximum(err_u, err_v)
            err = 0.5*(err_u + err_v)
            if err < uv_tol:
                break

        # --------------------------------------------------------
        # Sufficient statistics from logP.
        # --------------------------------------------------------
        # Pt1, P1, Np = _log_otm_stats(
        #     logK, log_u, log_v, X,
        #     use_keops=use_keops
        # )
        if use_keops:
            logP = logK + log_u[:, None, None] + log_v[None, :, None]
        else:
            # logP = logK + log_u[:, None] + log_v[None, :]
            logK.add_(log_u[:, None])
            logK.add_(log_v[None, :])
            logP = logK
        P1 = logP.logsumexp(dim=1).view(-1).exp()
        Pt1 = logP.logsumexp(dim=0).view(-1).exp()
        Np = P1.sum()

        # iwo = w_dr * iwo + (1.0 - w_dr) * (Np / Ni) #each update
        iwo = w_dr * ppas['wo'][iL] + (1.0 - w_dr) * (Np / Ni)
        iwo = xp.clip(iwo, min=ppas['w_clip'][0])
        kappa = float(Ni) * iwo
        a = imbids_otm_refresh_budget(
            P1,
            a=a,
            bm=budget_mode,
            sm=smooth_mode,
            gamma=budget_gamma,
            kappa=kappa,
            smoother=None,
            eps=eps
        )
        b = imbids_otm_refresh_budget(
            Pt1,
            a=b,
            bm=budget_mode,
            sm=smooth_mode,
            gamma=budget_gamma,
            kappa=kappa,
            smoother=None,
            eps=eps
        )

    ppas['wo'][iL] = iwo
    mapk =  zip(['as','bs', 'us', 'vs'], [a,b,u,v])
    for ik, iv in mapk:
        if ppas[ik][iL] is None:
            ppas[ik][iL] = iv
        else:
            ppas[ik][iL] = g_dr * ppas[ik][iL] + (1.0 - g_dr) * iv

    if use_keops:
        P = logP.exp()
    else:
        logP.exp_()
        P = logP

    # PX  = u[:, None] * (K @ (v[:, None] * X))   # shape (M, D)
    PX = P @ X
    if tau2_auto and (XF is not None) and (YF is not None):
        tau2s = update_tau2_otm(
            Pt1, P1, P, Np, XF, YF,
            feat_normal=feat_normal,
            DK=DF,
            device=device,
            xp=xp
        )
    else:
        tau2s = tau2

    return Pt1, P1, PX, Np, tau2s, ppas

def expectation_exbids_otm_11(
    X, TY, sigma2,
    C=None,
    d2f=None, tau2=None,
    tau2_auto=False, eps=1e-9, tau2_alpha=1.0,
    feat_normal=['cos'], XF=None, YF=None, device=None,
    xp=th, DF=[1], use_keops=True, ppas = None, iL = None,
    # rx=None, ry=None,  otm_reg='entropy',
):
    """
    Entropic UOT expectation step (normal domain).
    Supports:
      - external OT cost C (true cost matrix), or
      - internally built Gaussian/feature score from X, TY, sigma2, d2f, tau2.

    Returns
    -------
    Pt1, P1, PX, Np, tau2s, u, v

    Notes
    -----
    Corner-slack explicit dustbin patch.
    Keep the original augmented dustbin structure

        [ K    ky ]
        [ kx   kss]

    but separate true unmatched masses s_x,s_y from dummy-dummy corner mass z.
    The dummy row / column budgets are dx0=s_x+z and dy0=s_y+z, while the
    real-row / real-column totals stay A=kappa+s_y and B=kappa+s_x.
    """
    D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]
    uv_iter, uv_tol = ppas['otm_uv_iter'], ppas['otm_tol']
    ab_iter, ab_tol = ppas['otm_ab_iter'], ppas.get('otm_ab_tol', ppas['otm_tol'])

    a, b, u, v, u_d, v_d = (
        ppas['as'][iL], ppas['bs'][iL], ppas['us'][iL], ppas['vs'][iL],
        ppas['uds'][iL], ppas['vds'][iL]
    )
    dustbin_mode, budget_mode = ppas['dustbin_mode'][iL], ppas['budget_mode'][iL]

    k_ss, kx, ky = ppas['otm_kss'][iL], ppas['kxs'][iL], ppas['kys'][iL]
    iwo, iwx, iwy = ppas['wo'][iL], ppas['wx'][iL], ppas['wy'][iL]
    g_dr, w_dr, qt = ppas['g_dr'], ppas['w_dr'], ppas['qt'][iL]

    eps_c = ppas['otm_eps'][iL]
    tiny = max(float(eps), 1e-10)
    epsv = max(float(eps_c), tiny)

    # ------------------------------------------------------------
    # Evidence-regularized explicit dustbin / budget refresh.
    # Direct support evidence is evaluated after the UV solve, and
    # only refreshes next-step dustbin / budget variables.
    # ------------------------------------------------------------
    prebudget = ppas['prebudget'][iL]
    eta_k = 1.0
    gamma_k = 1.0
    loq, hiq = 0.005, 0.995
    zeta = g_dr

    # corner slack parameters for k_ss / p_ss
    lam_ss = float(ppas.get('otm_lam_ss', 0.25))
    zmax_ratio = float(ppas.get('otm_zmax_ratio', 0.50))
    ss_ratio_clip = ppas.get('otm_ss_ratio_clip', (0.25, 4.0))

    if (ppas.get('otm_the_x') is not None) and (ppas['otm_the_x'][iL] is not None):
        thy = ppas['otm_the_y'][iL]
        thx = ppas['otm_the_x'][iL]
        rho_y = epsv * thy / (1.0 - thy)
        rho_x = epsv * thx / (1.0 - thx)
    else:
        rho_y = float(ppas['otm_rho_y'][iL])
        rho_x = float(ppas['otm_rho_x'][iL])
        thy = rho_y / (rho_y + epsv)
        thx = rho_x / (rho_x + epsv)

    rx, ry, otm_reg = None, None, 'entropy'
    if otm_reg == 'kl':
        if (rx is None) or (ry is None):
            rx = b
            ry = a
    else:
        rx, ry = 1, 1

    # ------------------------------------------------------------
    # Build logK
    # Case A: C is provided externally as a true OT cost
    #         logK = -C / eps
    # Case B: C is None, build score S from Gaussian/feature term
    #         S = -cost_base, so logK = S / eps
    # ------------------------------------------------------------
    use_stable = False #TODO
    if C is None:
        if use_keops:
            S = update_Cost_ko(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            logK = S / epsv
            K = logK.exp()
        else:
            K = update_Cost_xp(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            K.div_(epsv).exp_()
            logK = None
    else:
        if use_keops:
            C = C.to(device=X.device, dtype=X.dtype)
            logK = C / (-epsv)
            K = logK.exp()
        else:
            # inplace: C will be overwritten by K to save memory
            K = C
            K.div_(-epsv).exp_()
            logK = None

    Nref = X.new_tensor(float(min(Ni, Mi)))
    sf = Nref
    Kv, KTu = None, None
    for _ab in range(ab_iter):
        iwo_t = iwo
        iwx_t = iwx
        iwy_t = iwy

        kappa = Nref * iwo_t
        eta_x = iwx_t / (1.0 - iwx_t).clamp_min(tiny)
        eta_y = iwy_t / (1.0 - iwy_t).clamp_min(tiny)

        # true unmatched masses, excluding dummy-dummy corner
        s_x = kappa * eta_x
        s_y = kappa * eta_y

        # independent corner slack target for p_ss
        z = lam_ss * (s_x * s_y).clamp_min(tiny).sqrt()
        z = z.clamp_max(zmax_ratio * kappa.clamp_min(tiny))

        # real-row / real-column target totals
        A0 = (kappa + s_y).clamp_min(tiny)
        B0 = (kappa + s_x).clamp_min(tiny)

        # dummy-row / dummy-column target totals
        dx0 = (s_x + z).clamp_min(tiny)
        dy0 = (s_y + z).clamp_min(tiny)

        if a is None:
            a = (A0 / Mi).expand(Mi)
        else:
            a = a.clamp_min(tiny)
            a = a * (A0 / a.sum().clamp_min(tiny))
        if b is None:
            b = (B0 / Ni).expand(Ni)
        else:
            b = b.clamp_min(tiny)
            b = b * (B0 / b.sum().clamp_min(tiny))
        if u is None:
            u = th.ones_like(a)
        if v is None:
            v = th.ones_like(b)
        if u_d is None:
            u_d = u.mean()
        if v_d is None:
            v_d = v.mean()

        a_safe = a.clamp_min(tiny)
        b_safe = b.clamp_min(tiny)

        # --------------------------------------------------------
        # Current augmented kernel / budget is fixed inside UV.
        # Prebudget evidence is not injected before this solve.
        # --------------------------------------------------------
        if ky is None:
            ky = eta_y * K.sum(1).squeeze()
        if kx is None:
            kx = eta_x * K.sum(0).squeeze()
        if k_ss is None:
            k_ss = (kx.clamp_min(tiny).mean() * ky.clamp_min(tiny).mean()).sqrt()
        elif not th.is_tensor(k_ss):
            k_ss = X.new_tensor(float(k_ss))
        else:
            k_ss = k_ss.to(device=X.device, dtype=X.dtype)

        kx_safe = kx.clamp_min(tiny)
        ky_safe = ky.clamp_min(tiny)
        k_ss_safe = k_ss.clamp_min(tiny)

        for _uv in range(uv_iter): #ra, rb
            u_prev = u.clone()
            v_prev = v.clone()
            # u_d_prev = u_d.clone()
            # v_d_prev = v_d.clone()

            Kv = (K @ v)
            Kv_aug = Kv + ky_safe * v_d
            u = (a_safe / Kv_aug.clamp_min(tiny)).pow(thy)

            ud_den = (kx_safe * v).sum() + k_ss_safe * v_d
            u_d = (dx0 / ud_den.clamp_min(tiny)).pow(thy)

            KTu = (K.T @ u)
            KTu_aug = KTu + kx_safe * u_d
            v = (b_safe / KTu_aug.clamp_min(tiny)).pow(thx)

            vd_den = (ky_safe * u).sum() + k_ss_safe * u_d
            v_d = (dy0 / vd_den.clamp_min(tiny)).pow(thx)

            err_u = ot_relerr(u, u_prev, base=1.0)
            err_v = ot_relerr(v, v_prev, base=1.0)
            # err = xp.maximum(err_u, err_v)
            err = 0.5 * (err_u + err_v)
            if err < uv_tol:
                break

        Kv = K @ v
        KTu = K.T @ u

        P1 = u * Kv
        Pt1 = v * KTu
        p_y = u * (ky_safe * v_d)
        p_x = v * (kx_safe * u_d)
        p_ss = u_d * k_ss_safe * v_d

        r_t = P1 + p_y
        r_d = p_x.sum() + p_ss
        c_t = Pt1 + p_x
        c_d = p_y.sum() + p_ss
        Np = xp.sum(P1)

        # true dustbin masses, excluding p_ss
        Sx = p_x.sum().clamp_min(0.0)
        Sy = p_y.sum().clamp_min(0.0)
        Sx_aug = r_d
        Sy_aug = c_d

        # --------------------------------------------------------
        # Prebudget support evidence for next-step refresh.
        # qK,pK adjust true-budget shape; chi_y,chi_x adjust dustbin channel.
        # --------------------------------------------------------
        if prebudget:
            # q0 = a * (sf / (a.sum() + tiny))
            # p0 = b * (sf / (b.sum() + tiny))

            q0 = r_t * (sf / r_t.sum().clamp_min(tiny))
            p0 = c_t * (sf / c_t.sum().clamp_min(tiny))

            Ke = _kernel_power(K, logK=logK, power=eta_k, use_keops=use_keops)
            Ey = (Ke @ p0).view(-1)  # E * sf
            Ex = (Ke.T @ q0).view(-1)  # F * sf

            wy_in = 1.0 - iwy_t
            wx_in = 1.0 - iwx_t

            chi_y, qK = exbids_split_budget(Ey, q0, sf, w=wy_in,
                                             gamma=gamma_k, zqt=qt,
                                             loq=loq, hiq=hiq, eps=tiny)
            chi_x, pK = exbids_split_budget(Ex, p0, sf, w=wx_in,
                                             gamma=gamma_k, zqt=qt,
                                             loq=loq, hiq=hiq, eps=tiny)
        else:
            chi_y = X.new_ones(Mi)
            chi_x = X.new_ones(Ni)
            qK, pK = None, None

        if dustbin_mode == 'adaptive':
            # target true dustbin masses, excluding p_ss
            px_shape = (Pt1.detach() * chi_x).clamp_min(tiny)
            py_shape = (P1.detach() * chi_y).clamp_min(tiny)
            px_shape = px_shape / px_shape.sum().clamp_min(tiny)
            py_shape = py_shape / py_shape.sum().clamp_min(tiny)

            px_tgt = s_x * px_shape
            py_tgt = s_y * py_shape

            kx_hat = px_tgt / (u_d.detach().clamp_min(tiny) * v.detach().clamp_min(tiny))
            ky_hat = py_tgt / (u.detach().clamp_min(tiny) * v_d.detach().clamp_min(tiny))

            kx = kx_safe.pow(g_dr) * kx_hat.clamp_min(tiny).pow(1.0 - g_dr)
            ky = ky_safe.pow(g_dr) * ky_hat.clamp_min(tiny).pow(1.0 - g_dr)

            # calibrate k_ss so p_ss follows the independent corner slack z
            z_tgt = z.detach().clamp_min(tiny)
            ratio_ss = (z_tgt / (p_ss.detach() + tiny)).clamp(*ss_ratio_clip)
            k_ss_hat = k_ss_safe * ratio_ss
            k_ss = k_ss_safe.pow(g_dr) * k_ss_hat.clamp_min(tiny).pow(1.0 - g_dr)

            # kx = g_dr * kx + (1.0 - g_dr) * kx_hat
            # ky = g_dr * ky + (1.0 - g_dr) * ky_hat
            # log-domain damping for kernel-scale quantities
            # kx = (beta_k * kx_safe.log() + (1.0 - beta_k) * kx_hat.clamp_min(tiny).log()).exp()
            # ky = (beta_k * ky_safe.log() + (1.0 - beta_k) * ky_hat.clamp_min(tiny).log()).exp()
        elif dustbin_mode == 'static':
            kx = kx
            ky = ky
            k_ss = k_ss_safe
        else:
            raise ValueError("Unknown dustbin_mode: %s" % dustbin_mode)

        if budget_mode in ['slack_fixed', 'slack_free']:

            # A/B are real-side totals. Do not subtract p_ss here.
            # p_ss is represented separately by z through dx0=s_x+z, dy0=s_y+z.
            A = (kappa + s_y).clamp_min(tiny)
            B = (kappa + s_x).clamp_min(tiny)

            if prebudget:
                r_t = zeta * r_t + ((1.0 - zeta) * r_t.sum().clamp_min(tiny)) * qK
                c_t = zeta * c_t + ((1.0 - zeta) * c_t.sum().clamp_min(tiny)) * pK

            a = r_t * (A / r_t.sum().clamp_min(tiny))
            b = c_t * (B / c_t.sum().clamp_min(tiny))

            if budget_mode == 'slack_fixed':
                iwx_n = s_x / (kappa + s_x).clamp_min(tiny)
                iwy_n = s_y / (kappa + s_y).clamp_min(tiny)
            elif budget_mode == 'slack_free':
                # iwx_n = Sx_aug / (Np + Sx_aug)
                # iwy_n = Sy_aug / (Np + Sy_aug)
                iwx_n = Sx / (Np + Sx).clamp_min(tiny)
                iwy_n = Sy / (Np + Sy).clamp_min(tiny)

        elif budget_mode == 'free':
            # use true dustbin masses, excluding p_ss
            s_x = Sx
            s_y = Sy

            A = (Np + Sy).clamp_min(tiny)
            B = (Np + Sx).clamp_min(tiny)
            a = r_t * (A / r_t.sum().clamp_min(tiny))
            b = c_t * (B / c_t.sum().clamp_min(tiny))

            iwx_n = Sx / (Np + Sx).clamp_min(tiny)
            iwy_n = Sy / (Np + Sy).clamp_min(tiny)

            # iwx_n = Sx_aug / (Np + Sx_aug)
            # iwy_n = Sy_aug / (Np + Sy_aug)
        elif budget_mode == 'fixed':
            a = a
            b = b
            s_x = s_x
            s_y = s_y
            iwx_n = s_x / (kappa + s_x).clamp_min(tiny)
            iwy_n = s_y / (kappa + s_y).clamp_min(tiny)
            # iwy_n = s_y/ (a.sum())
            # iwx_n = s_x/ (b.sum())
        else:
            raise ValueError("Unknown budget_mode: %s" % budget_mode)

        iwo_n = (Np / Nref).clamp_min(tiny).clamp_max(1.0 - tiny)
        mapk = zip([iwo_n, iwx_n, iwy_n], [iwo_t, iwx_t, iwy_t])
        iwo, iwx, iwy = [xp.clip(w_dr * i_ol + (1.0 - w_dr) * i_ne, *ppas['w_clip'])
                            for i_ne, i_ol in mapk]
        # if err_ab < ab_tol:
        #     break

        if ppas.get('otm_debug', False):
            print(33333, eps, iwo, iwo_n, iwx_n,  iwy_n,  Np,  _uv,
                  s_x, s_y, z, p_ss, a, b, u, v, 33333)

    ppas['wo'][iL], ppas['wx'][iL], ppas['wy'][iL] = iwo, iwx, iwy
    ppas['kxs'][iL], ppas['kys'][iL] = kx, ky
    ppas['otm_kss'][iL] = k_ss

    mapk = zip(['as','bs', 'us', 'vs', 'uds', 'vds',],
               [a,b,u,v,u_d,v_d])
    for ik, iv in mapk:
        if ppas[ik][iL] is None:
            ppas[ik][iL] = iv
        else:
            ppas[ik][iL] = g_dr * ppas[ik][iL] + (1.0 - g_dr) * iv

    # Kv = K @ v
    # KTu = K.T @ u
    # P1 = u * Kv
    # Pt1 = v * KTu
    # Np  = xp.sum(P1)
    PX = u[:, None] * (K @ (v[:, None] * X))   # shape (M, D)

    if tau2_auto and (XF is not None) and (YF is not None):
        tau2s = update_tau2_otm(
            Pt1, P1, (u, K, v), Np, XF, YF,
            feat_normal=feat_normal, DK=DF, device=device, xp=xp
        )
    else:
        tau2s = tau2

    return Pt1, P1, PX, Np, tau2s, ppas

def expectation_exbids_otm_00(
    X, TY, sigma2,
    C=None,
    d2f=None, tau2=None,
    tau2_auto=False, eps=1e-9, tau2_alpha=1.0,
    feat_normal=['cos'], XF=None, YF=None, device=None,
    xp=th, DF=[1], use_keops=True, ppas=None, iL=None,
    # rx=None, ry=None,  otm_reg='entropy',
):
    """
    Explicit dustbin UOT expectation step with capacity-constrained overlap.

    Main change from the previous explicit OTM:
      - wo controls matched mass kappa.
      - wx/wy are no longer free self-updated odds.
      - side dustbin masses are derived from physical capacities:
            s_x = N_x - kappa,
            s_y = N_y - kappa.
      - overlap refresh uses posterior matched mass plus pre-budget support evidence.

    Returns
    -------
    Pt1, P1, PX, Np, tau2s, ppas
    """
    D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]
    uv_iter, uv_tol = ppas['otm_uv_iter'], ppas['otm_tol']
    ab_iter = ppas['otm_ab_iter']

    a, b, u, v, u_d, v_d = (
        ppas['as'][iL], ppas['bs'][iL], ppas['us'][iL], ppas['vs'][iL],
        ppas['uds'][iL], ppas['vds'][iL]
    )
    dustbin_mode = ppas['dustbin_mode'][iL]
    budget_mode = ppas['budget_mode'][iL]

    k_ss, kx, ky = ppas['otm_kss'][iL], ppas['kxs'][iL], ppas['kys'][iL]
    iwo = ppas['wo'][iL]
    g_dr, w_dr, qt = ppas['g_dr'], ppas['w_dr'], ppas['qt'][iL]

    eps_c = ppas['otm_eps'][iL]
    tiny = max(float(eps), 1e-10)
    epsv = max(float(eps_c), tiny)

    prebudget = ppas['prebudget'][iL]
    eta_k = float(ppas.get('otm_prebudget_power', 1.0))
    zeta = g_dr

    if (ppas.get('otm_the_x') is not None) and (ppas['otm_the_x'][iL] is not None):
        thy = ppas['otm_the_y'][iL]
        thx = ppas['otm_the_x'][iL]
    else:
        rho_y = float(ppas['otm_rho_y'][iL])
        rho_x = float(ppas['otm_rho_x'][iL])
        thy = rho_y / (rho_y + epsv)
        thx = rho_x / (rho_x + epsv)

    # ------------------------------------------------------------
    # Build kernel.
    # C provided: C is a cost, K = exp(-C / eps).
    # C is None: update_Cost_* returns score S = -cost, K = exp(S / eps).
    # ------------------------------------------------------------
    if C is None:
        if use_keops:
            S = update_Cost_ko(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            logK = S / epsv
            K = logK.exp()
        else:
            K = update_Cost_xp(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            K.div_(epsv).exp_()
            logK = None
    else:
        if use_keops:
            C = C.to(device=X.device, dtype=X.dtype)
            logK = C / (-epsv)
            K = logK.exp()
        else:
            K = C
            K.div_(-epsv).exp_()
            logK = None

    # overlap is measured relative to X side, as in the original code
    Nref = X.new_tensor(float(Ni))
    cap_x = X.new_tensor(float(Ni))  # column / X side capacity
    cap_y = X.new_tensor(float(Mi))  # row / TY side capacity
    cap_o = th.minimum(cap_x, cap_y)

    for _ab in range(ab_iter):
        # --------------------------------------------------------
        # Capacity-constrained explicit dustbin mass.
        # No free wx/wy odds are optimized here.
        # --------------------------------------------------------
        kappa = th.minimum(Nref * iwo, cap_o)
        s_x = cap_x - kappa
        s_y = cap_y - kappa

        eta_x = s_x / kappa.clamp_min(tiny)
        eta_y = s_y / kappa.clamp_min(tiny)

        A0 = kappa + s_y
        B0 = kappa + s_x

        if a is None:
            a = (A0 / Mi).expand(Mi)
        else:
            a = a * (A0 / a.sum().clamp_min(tiny))

        if b is None:
            b = (B0 / Ni).expand(Ni)
        else:
            b = b * (B0 / b.sum().clamp_min(tiny))

        if u is None:
            u = th.ones_like(a)
        if v is None:
            v = th.ones_like(b)
        if u_d is None:
            u_d = X.new_tensor(1.0)
        if v_d is None:
            v_d = X.new_tensor(1.0)

        a_safe = a.clamp_min(tiny)
        b_safe = b.clamp_min(tiny)

        if ky is None:
            ky = eta_y * K.sum(1).squeeze()
        if kx is None:
            kx = eta_x * K.sum(0).squeeze()

        kx_safe = kx.clamp_min(tiny)
        ky_safe = ky.clamp_min(tiny)

        # --------------------------------------------------------
        # Generalized Sinkhorn on the explicit augmented system.
        # --------------------------------------------------------
        for _uv in range(uv_iter):
            u_prev = u.clone()
            v_prev = v.clone()

            Kv = K @ v
            u = (a_safe / (Kv + ky_safe * v_d).clamp_min(tiny)).pow(thy)
            u_d = (s_x / ((kx_safe * v).sum() + k_ss * v_d).clamp_min(tiny)).pow(thy)

            KTu = K.T @ u
            v = (b_safe / (KTu + kx_safe * u_d).clamp_min(tiny)).pow(thx)
            v_d = (s_y / ((ky_safe * u).sum() + k_ss * u_d).clamp_min(tiny)).pow(thx)

            err_u = ot_relerr(u, u_prev, base=1.0)
            err_v = ot_relerr(v, v_prev, base=1.0)
            err = 0.6 * (err_u + err_v)
            if err < uv_tol:
                break

        Kv = K @ v
        KTu = K.T @ u

        P1 = u * Kv
        Pt1 = v * KTu
        p_y = u * (ky_safe * v_d)
        p_x = v * (kx_safe * u_d)
        p_ss = u_d * k_ss * v_d

        r_t = P1 + p_y
        c_t = Pt1 + p_x
        Np = xp.sum(P1)

        # --------------------------------------------------------
        # Pre-budget support evidence.
        # Use budget prior a/b, not posterior r_t/c_t, to avoid
        # feeding current dustbin errors back into next budgets.
        # --------------------------------------------------------
        if prebudget:
            Ke = _kernel_power(K, logK=logK, power=eta_k, use_keops=use_keops)

            Ey = (Ke @ b_safe).view(-1)
            Ex = (Ke.T @ a_safe).view(-1)

            zy = th.quantile(Ey.detach(), qt)
            zx = th.quantile(Ex.detach(), qt)

            gy = Ey / (Ey + zy + tiny)
            gx = Ex / (Ex + zx + tiny)

            qE = a_safe * gy
            pE = b_safe * gx

            qK = qE / qE.sum().clamp_min(tiny)
            pK = pE / pE.sum().clamp_min(tiny)

            kappa_e = th.minimum(qE.sum(), pE.sum())
            kappa_e = th.minimum(kappa_e, cap_o)

            chi_y = 1.0 - gy
            chi_x = 1.0 - gx
        else:
            qK, pK = None, None
            kappa_e = kappa
            chi_y = X.new_ones(Mi)
            chi_x = X.new_ones(Ni)

        # --------------------------------------------------------
        # Adaptive dustbin kernels.
        # eta_x/eta_y are now capacity-derived, so they cannot blow up
        # through wx/(1-wx) self-feedback.
        # --------------------------------------------------------
        if dustbin_mode == 'adaptive':
            kx_hat = eta_x * chi_x * KTu / u_d.clamp_min(tiny)
            ky_hat = eta_y * chi_y * Kv  / v_d.clamp_min(tiny)
            kx = g_dr * kx + (1.0 - g_dr) * kx_hat
            ky = g_dr * ky + (1.0 - g_dr) * ky_hat
        elif dustbin_mode == 'static':
            pass
        else:
            raise ValueError("Unknown dustbin_mode: %s" % dustbin_mode)

        # --------------------------------------------------------
        # Refresh true-side budget shape only.  The total dustbin mass
        # remains capacity-constrained through kappa, s_x, s_y.
        # --------------------------------------------------------
        if budget_mode in ['slack_fixed', 'slack_free', 'free']:
            A = kappa + s_y - p_ss
            B = kappa + s_x - p_ss

            if prebudget:
                r_t = zeta * r_t + ((1.0 - zeta) * r_t.sum()) * qK
                c_t = zeta * c_t + ((1.0 - zeta) * c_t.sum()) * pK

            a = r_t * (A / r_t.sum().clamp_min(tiny))
            b = c_t * (B / c_t.sum().clamp_min(tiny))
        elif budget_mode == 'fixed':
            pass
        else:
            raise ValueError("Unknown budget_mode: %s" % budget_mode)

        # --------------------------------------------------------
        # Overlap update.
        # Posterior Np alone is not trusted; it is mixed with
        # pre-budget support evidence and then damped in mass space.
        # --------------------------------------------------------
        alpha_e = float(ppas.get('otm_wo_evi_alpha', 0.5))
        kappa_n = (1.0 - alpha_e) * Np + alpha_e * kappa_e
        kappa_n = th.minimum(kappa_n, cap_o)

        kappa_m = w_dr * kappa + (1.0 - w_dr) * kappa_n
        kappa_m = th.minimum(kappa_m, cap_o)

        iwo = kappa_m / Nref
        iwx = (cap_x - kappa_m) / cap_x
        iwy = (cap_y - kappa_m) / cap_y

    ppas['wo'][iL], ppas['wx'][iL], ppas['wy'][iL] = iwo, iwx, iwy
    ppas['kxs'][iL], ppas['kys'][iL] = kx, ky

    for ik, iv in zip(['as', 'bs', 'us', 'vs', 'uds', 'vds'], [a, b, u, v, u_d, v_d]):
        if ppas[ik][iL] is None:
            ppas[ik][iL] = iv
        else:
            ppas[ik][iL] = g_dr * ppas[ik][iL] + (1.0 - g_dr) * iv

    PX = u[:, None] * (K @ (v[:, None] * X))

    if tau2_auto and (XF is not None) and (YF is not None):
        tau2s = update_tau2_otm(
            Pt1, P1, (u, K, v), Np, XF, YF,
            feat_normal=feat_normal, DK=DF, device=device, xp=xp
        )
    else:
        tau2s = tau2

    return Pt1, P1, PX, Np, tau2s, ppas

def expectation_exbids_otm_m3(
    X, TY, sigma2,
    C=None,
    d2f=None, tau2=None,
    tau2_auto=False, eps=1e-9, tau2_alpha=1.0,
    feat_normal=['cos'], XF=None, YF=None, device=None,
    xp=th, DF=[1], use_keops=True, ppas=None, iL=None,
    # rx=None, ry=None,  otm_reg='entropy',
):
    """
    Entropic UOT expectation step with explicit dustbins.

    Key update in this version
    --------------------------
    wo, wx, wy are not damped directly.  They are interpreted as

        kappa = Nref * wo
        s_x   = kappa * wx / (1 - wx)
        s_y   = kappa * wy / (1 - wy)

    and the refresh is performed in mass space: kappa, s_x, s_y.
    Then wo, wx, wy are mapped back from the refreshed masses.

    Returns
    -------
    Pt1, P1, PX, Np, tau2s, ppas
    """
    D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]
    uv_iter, uv_tol = ppas['otm_uv_iter'], ppas['otm_tol']
    ab_iter, ab_tol = ppas['otm_ab_iter'], ppas.get('otm_ab_tol', ppas['otm_tol'])

    a, b, u, v, u_d, v_d = (
        ppas['as'][iL], ppas['bs'][iL], ppas['us'][iL], ppas['vs'][iL],
        ppas['uds'][iL], ppas['vds'][iL]
    )
    dustbin_mode, budget_mode = ppas['dustbin_mode'][iL], ppas['budget_mode'][iL]

    k_ss, kx, ky = ppas['otm_kss'][iL], ppas['kxs'][iL], ppas['kys'][iL]
    iwo, iwx, iwy = ppas['wo'][iL], ppas['wx'][iL], ppas['wy'][iL]
    g_dr, w_dr, qt = ppas['g_dr'], ppas['w_dr'], ppas['qt'][iL]

    eps_c = ppas['otm_eps'][iL]
    tiny = max(float(eps), 1e-10)
    epsv = max(float(eps_c), tiny)

    # ------------------------------------------------------------
    # Evidence-regularized explicit dustbin / budget refresh.
    # Direct support evidence is evaluated after the UV solve, and
    # only refreshes next-step dustbin / budget variables.
    # ------------------------------------------------------------
    prebudget = ppas['prebudget'][iL]
    eta_k = 1.0
    gamma_k = 1.0
    loq, hiq = 0.005, 0.995
    zeta = g_dr

    if (ppas.get('otm_the_x') is not None) and (ppas['otm_the_x'][iL] is not None):
        thy = ppas['otm_the_y'][iL]
        thx = ppas['otm_the_x'][iL]
        rho_y = epsv * thy / (1.0 - thy)
        rho_x = epsv * thx / (1.0 - thx)
    else:
        rho_y = float(ppas['otm_rho_y'][iL])
        rho_x = float(ppas['otm_rho_x'][iL])
        thy = rho_y / (rho_y + epsv)
        thx = rho_x / (rho_x + epsv)

    rx, ry, otm_reg = None, None, 'entropy'
    if otm_reg == 'kl':
        if (rx is None) or (ry is None):
            rx = b
            ry = a
    else:
        rx, ry = 1, 1

    # ------------------------------------------------------------
    # Build logK
    # Case A: C is provided externally as a true OT cost
    #         logK = -C / eps
    # Case B: C is None, build score S from Gaussian/feature term
    #         S = -cost_base, so logK = S / eps
    # ------------------------------------------------------------
    use_stable = False  # TODO
    if C is None:
        if use_keops:
            S = update_Cost_ko(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            logK = S / epsv
            K = logK.exp()
        else:
            K = update_Cost_xp(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            K.div_(epsv).exp_()
            logK = None
    else:
        if use_keops:
            C = C.to(device=X.device, dtype=X.dtype)
            logK = C / (-epsv)
            K = logK.exp()
        else:
            # inplace: C will be overwritten by K to save memory
            K = C
            K.div_(-epsv).exp_()
            logK = None

    # Nref = X.new_tensor(float(min(Ni, Mi)))
    Nref = X.new_tensor(float(Ni))
    sf = Nref
    Kv, KTu = None, None

    for _ab in range(ab_iter):
        iwo_t = iwo
        iwx_t = iwx
        iwy_t = iwy

        # --------------------------------------------------------
        # Interpret wo/wx/wy as masses.
        # kappa: true XY mass; s_x/s_y: side unmatched masses.
        # --------------------------------------------------------
        kappa = Nref * iwo_t
        eta_x = iwx_t / (1.0 - iwx_t).clamp_min(tiny)
        eta_y = iwy_t / (1.0 - iwy_t).clamp_min(tiny)
        s_x = kappa * eta_x
        s_y = kappa * eta_y

        A0 = kappa + s_y
        B0 = kappa + s_x
        if a is None:
            a = ((kappa + s_y) / Mi).expand(Mi)
        else:
            a = a * (A0 / a.sum().clamp_min(tiny))
        if b is None:
            b = ((kappa + s_x) / Ni).expand(Ni)
        else:
            b = b * (B0 / b.sum().clamp_min(tiny))
        if u is None:
            u = th.ones_like(a)
        if v is None:
            v = th.ones_like(b)
        if u_d is None:
            u_d = 1.0
        if v_d is None:
            v_d = 1.0

        a_safe = a.clamp_min(tiny)
        b_safe = b.clamp_min(tiny)

        # --------------------------------------------------------
        # Current augmented kernel / budget is fixed inside UV.
        # Prebudget evidence is not injected before this solve.
        # --------------------------------------------------------
        if ky is None:
            ky = eta_y * K.sum(1).squeeze()
        if kx is None:
            kx = eta_x * K.sum(0).squeeze()

        kx_safe = kx.clamp_min(tiny)
        ky_safe = ky.clamp_min(tiny)

        for _uv in range(uv_iter):
            u_prev = u.clone()
            v_prev = v.clone()

            Kv = K @ v
            Kv_aug = Kv + ky_safe * v_d
            u = (a_safe / Kv_aug.clamp_min(tiny)).pow(thy)
            u_d = (s_x / ((kx_safe * v).sum() + k_ss * v_d).clamp_min(tiny)).pow(thy)

            KTu = K.T @ u
            KTu_aug = KTu + kx_safe * u_d
            v = (b_safe / KTu_aug.clamp_min(tiny)).pow(thx)
            v_d = (s_y / ((ky_safe * u).sum() + k_ss * u_d).clamp_min(tiny)).pow(thx)

            err_u = ot_relerr(u, u_prev, base=1.0)
            err_v = ot_relerr(v, v_prev, base=1.0)
            err = 0.6 * (err_u + err_v)
            if err < uv_tol:
                break

        Kv = K @ v
        KTu = K.T @ u

        P1 = u * Kv
        Pt1 = v * KTu
        p_y = u * (ky_safe * v_d)
        p_x = v * (kx_safe * u_d)
        p_ss = u_d * k_ss * v_d

        r_t = P1 + p_y
        r_d = p_x.sum() + p_ss
        c_t = Pt1 + p_x
        c_d = p_y.sum() + p_ss

        Np = xp.sum(P1)

        # Physical unmatched masses.  Exclude dustbin-dustbin filler p_ss.
        Sx = (r_d - p_ss).clamp_min(0.0)
        Sy = (c_d - p_ss).clamp_min(0.0)

        # --------------------------------------------------------
        # Prebudget support evidence for next-step refresh.
        # qK,pK adjust true-budget shape; chi_y,chi_x adjust dustbin channel.
        # --------------------------------------------------------
        if prebudget:
            q0 = r_t * (sf / r_t.sum().clamp_min(tiny))
            p0 = c_t * (sf / c_t.sum().clamp_min(tiny))

            Ke = _kernel_power(K, logK=logK, power=eta_k, use_keops=use_keops)
            Ey = (Ke @ p0).view(-1)       # E * sf
            Ex = (Ke.T @ q0).view(-1)     # F * sf

            wy_in = 1.0 - iwy_t
            wx_in = 1.0 - iwx_t

            chi_y, qK = exbids_split_budget(
                Ey, q0, sf, w=wy_in,
                gamma=gamma_k, zqt=qt,
                loq=loq, hiq=hiq, eps=tiny
            )
            chi_x, pK = exbids_split_budget(
                Ex, p0, sf, w=wx_in,
                gamma=gamma_k, zqt=qt,
                loq=loq, hiq=hiq, eps=tiny
            )
        else:
            chi_y = X.new_ones(Mi)
            chi_x = X.new_ones(Ni)
            qK, pK = None, None

        if dustbin_mode == 'adaptive':
            kx_hat = eta_x * chi_x * KTu / u_d.clamp_min(tiny)
            ky_hat = eta_y * chi_y * Kv / v_d.clamp_min(tiny)
            kx = g_dr * kx + (1.0 - g_dr) * kx_hat
            ky = g_dr * ky + (1.0 - g_dr) * ky_hat
        elif dustbin_mode == 'static':
            kx = kx
            ky = ky
        else:
            raise ValueError("Unknown dustbin_mode: %s" % budget_mode)

        # --------------------------------------------------------
        # Mass-space budget refresh.
        # Do not damp wo/wx/wy directly.  Refresh kappa/s_x/s_y,
        # then map them back to wo/wx/wy.
        # --------------------------------------------------------
        if budget_mode in ['slack_fixed', 'slack_free']:

            if prebudget:
                r_t = zeta * r_t + ((1.0 - zeta) * r_t.sum()) * qK
                c_t = zeta * c_t + ((1.0 - zeta) * c_t.sum()) * pK

            if budget_mode == 'slack_fixed':
                # update true mass only; keep dustbin odds fixed
                kappa_n = Np
                s_x_n = kappa_n * eta_x
                s_y_n = kappa_n * eta_y

            elif budget_mode == 'slack_free':
                # update true mass and side unmatched masses
                kappa_n = Np
                s_x_n = Sx
                s_y_n = Sy

        elif budget_mode == 'free':
            # fully free mass refresh; use Sx/Sy, not r_d/c_d.
            kappa_n = Np
            s_x_n = Sx
            s_y_n = Sy

        elif budget_mode == 'fixed':
            # keep all masses fixed
            kappa_n = kappa
            s_x_n = s_x
            s_y_n = s_y

        else:
            raise ValueError("Unknown budget_mode: %s" % budget_mode)

        # Damping in physical mass space.
        kappa = w_dr * kappa + (1.0 - w_dr) * kappa_n
        s_x = w_dr * s_x + (1.0 - w_dr) * s_x_n
        s_y = w_dr * s_y + (1.0 - w_dr) * s_y_n

        # Refresh a,b using damped masses and current posterior shape.
        if budget_mode != 'fixed':
            A = kappa + s_y
            B = kappa + s_x
            a = r_t * (A / r_t.sum().clamp_min(tiny))
            b = c_t * (B / c_t.sum().clamp_min(tiny))

        # Map physical masses back to wo/wx/wy.
        iwo = kappa / Nref.clamp_min(tiny)
        iwx = s_x / (kappa + s_x).clamp_min(tiny)
        iwy = s_y / (kappa + s_y).clamp_min(tiny)

        # Debug-only new ratios before damping.
        iwo_n = kappa_n / Nref.clamp_min(tiny)
        iwx_n = s_x_n / (kappa_n + s_x_n).clamp_min(tiny)
        iwy_n = s_y_n / (kappa_n + s_y_n).clamp_min(tiny)

        # if err_ab < ab_tol:
        #     break

        print(11111, eps, iwo, iwo_n, iwx_n, iwy_n, Np, _uv, s_x, s_y, a, b, u, v, 33333)

    ppas['wo'][iL], ppas['wx'][iL], ppas['wy'][iL] = iwo, iwx, iwy
    ppas['kxs'][iL], ppas['kys'][iL] = kx, ky

    mapk = zip(['as', 'bs', 'us', 'vs', 'uds', 'vds'], [a, b, u, v, u_d, v_d])
    for ik, iv in mapk:
        if ppas[ik][iL] is None:
            ppas[ik][iL] = iv
        else:
            ppas[ik][iL] = g_dr * ppas[ik][iL] + (1.0 - g_dr) * iv

    PX = u[:, None] * (K @ (v[:, None] * X))

    if tau2_auto and (XF is not None) and (YF is not None):
        tau2s = update_tau2_otm(
            Pt1, P1, (u, K, v), Np, XF, YF,
            feat_normal=feat_normal, DK=DF, device=device, xp=xp
        )
    else:
        tau2s = tau2

    return Pt1, P1, PX, Np, tau2s, ppas

def expectation_exbids_otm_m4(
    X, TY, sigma2,
    C=None,
    d2f=None, tau2=None,
    tau2_auto=False, eps=1e-9, tau2_alpha=1.0,
    feat_normal=['cos'], XF=None, YF=None, device=None,
    xp=th, DF=[1], use_keops=True, ppas = None, iL = None,
    # rx=None, ry=None,  otm_reg='entropy',
):
    """
    Entropic UOT expectation step (normal domain).
    Supports:
      - external OT cost C (true cost matrix), or
      - internally built Gaussian/feature score from X, TY, sigma2, d2f, tau2.

    Returns
    -------
    Pt1, P1, PX, Np, tau2s, u, v
    """
    D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]
    uv_iter, uv_tol = ppas['otm_uv_iter'], ppas['otm_tol']
    ab_iter, ab_tol = ppas['otm_ab_iter'], ppas.get('otm_ab_tol', ppas['otm_tol'])

    a, b, u, v, u_d, v_d = (
        ppas['as'][iL], ppas['bs'][iL], ppas['us'][iL], ppas['vs'][iL],
        ppas['uds'][iL], ppas['vds'][iL]
    )
    dustbin_mode, budget_mode = ppas['dustbin_mode'][iL], ppas['budget_mode'][iL]

    k_ss, kx, ky = ppas['otm_kss'][iL], ppas['kxs'][iL], ppas['kys'][iL]
    iwo, iwx, iwy = ppas['wo'][iL], ppas['wx'][iL], ppas['wy'][iL]
    g_dr, w_dr, qt = ppas['g_dr'], ppas['w_dr'], ppas['qt'][iL]

    eps_c = ppas['otm_eps'][iL]
    tiny = max(float(eps), 1e-10)
    epsv = max(float(eps_c), tiny)

    # ------------------------------------------------------------
    # Evidence-regularized explicit dustbin / budget refresh.
    # Direct support evidence is evaluated after the UV solve, and
    # only refreshes next-step dustbin / budget variables.
    # ------------------------------------------------------------
    prebudget = ppas['prebudget'][iL]
    eta_k = 1.0
    gamma_k = 1.0
    loq, hiq = 0.005, 0.995
    zeta = g_dr

    if (ppas.get('otm_the_x') is not None) and (ppas['otm_the_x'][iL] is not None):
        thy = ppas['otm_the_y'][iL]
        thx = ppas['otm_the_x'][iL]
        rho_y = epsv * thy / (1.0 - thy)
        rho_x = epsv * thx / (1.0 - thx)
    else:
        rho_y = float(ppas['otm_rho_y'][iL])
        rho_x = float(ppas['otm_rho_x'][iL])
        thy = rho_y / (rho_y + epsv)
        thx = rho_x / (rho_x + epsv)

    rx, ry, otm_reg = None, None, 'entropy'
    if otm_reg == 'kl':
        if (rx is None) or (ry is None):
            rx = b
            ry = a
    else:
        rx, ry = 1, 1

    # ------------------------------------------------------------
    # Build logK
    # Case A: C is provided externally as a true OT cost
    #         logK = -C / eps
    # Case B: C is None, build score S from Gaussian/feature term
    #         S = -cost_base, so logK = S / eps
    # ------------------------------------------------------------
    use_stable = False #TODO
    if C is None:
        if use_keops:
            S = update_Cost_ko(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            logK = S / epsv
            K = logK.exp()
        else:
            K = update_Cost_xp(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            K.div_(epsv).exp_()
            logK = None
    else:
        if use_keops:
            C = C.to(device=X.device, dtype=X.dtype)
            logK = C / (-epsv)
            K = logK.exp()
        else:
            # inplace: C will be overwritten by K to save memory
            K = C
            K.div_(-epsv).exp_()
            logK = None

    # Nref = X.new_tensor(float(min(Ni, Mi)))
    Nref = X.new_tensor(float(Ni))
    sf = Nref
    Kv, KTu = None, None
    for _ab in range(ab_iter):
        iwo_t = iwo
        iwx_t = iwx
        iwy_t = iwy

        kappa = Nref * iwo_t
        eta_x = iwx_t / (1.0 - iwx_t).clamp_min(tiny)
        eta_y = iwy_t / (1.0 - iwy_t).clamp_min(tiny)
        s_x = kappa * eta_x
        s_y = kappa * eta_y

        # p_ss = u_d * k_ss * v_d
        A0 = kappa + s_y #- p_ss
        B0 = kappa + s_x #- p_ss
        if a is None:
            a = ((kappa + s_y) / Mi).expand(Mi)
        else:
            a = a * (A0 / a.sum())
        if b is None:
            b = ((kappa + s_x) / Ni).expand(Ni)
        else:
            b = b * (B0 / b.sum())
        if u is None:
            u = th.ones_like(a)
        if v is None:
            v = th.ones_like(b)
        if u_d is None:
            u_d = 1.0
        if v_d is None:
            v_d = 1.0

        a_safe = a.clamp_min(tiny)
        b_safe = b.clamp_min(tiny)

        # --------------------------------------------------------
        # Current augmented kernel / budget is fixed inside UV.
        # Prebudget evidence is not injected before this solve.
        # --------------------------------------------------------
        if ky is None:
            ky = eta_y * K.sum(1).squeeze()
        if kx is None:
            kx = eta_x * K.sum(0).squeeze()

        kx_safe = kx.clamp_min(tiny)
        ky_safe = ky.clamp_min(tiny)

        for _uv in range(uv_iter): #ra, rb
            u_prev = u.clone()
            v_prev = v.clone()
            # u_d_prev = u_d.clone()
            # v_d_prev = v_d.clone()

            Kv = (K @ v)
            Kv_aug = Kv + ky_safe * v_d
            u = (a_safe / Kv_aug.clamp_min(tiny)).pow(thy)
            u_d = (s_x / ((kx_safe * v).sum() + k_ss * v_d).clamp_min(tiny)).pow(thy)

            KTu = (K.T @ u)
            KTu_aug = KTu + kx_safe * u_d
            v = (b_safe / KTu_aug.clamp_min(tiny)).pow(thx)
            v_d = (s_y / ((ky_safe * u).sum() + k_ss * u_d).clamp_min(tiny)).pow(thx)

            err_u = ot_relerr(u, u_prev, base=1.0)
            err_v = ot_relerr(v, v_prev, base=1.0)
            # err = xp.maximum(err_u, err_v)
            err = 0.6 * (err_u + err_v)
            if err < uv_tol:
                break

        Kv = K @ v
        KTu = K.T @ u

        P1 = u * Kv
        Pt1 = v * KTu
        p_y = u * (ky_safe * v_d)
        p_x = v * (kx_safe * u_d)
        p_ss = u_d * k_ss * v_d

        r_t = P1 + p_y
        r_d = p_x.sum() + p_ss
        c_t = Pt1 + p_x
        c_d = p_y.sum() + p_ss
        Np = xp.sum(P1)
        Sx = (r_d - p_ss).clamp_min(0.0)
        Sy = (c_d - p_ss).clamp_min(0.0)
        Sx_aug = r_d
        Sy_aug = c_d

        # --------------------------------------------------------
        # Prebudget support evidence for next-step refresh.
        # qK,pK adjust true-budget shape; chi_y,chi_x adjust dustbin channel.
        # --------------------------------------------------------
        if prebudget:
            q0 = a 
            p0 = b 

            # q0 = r_t 
            # p0 = c_t 

            Ke = _kernel_power(K, logK=logK, power=eta_k, use_keops=use_keops)
            Ey = (Ke @ p0).view(-1)  # E * sf
            Ex = (Ke.T @ q0).view(-1)  # F * sf

            wy_in = 1.0 - iwy_t
            wx_in = 1.0 - iwx_t

            chi_y, qK = exbids_split_budget(Ey, q0, w=wy_in,
                                             gamma=gamma_k, zqt=qt,
                                             loq=loq, hiq=hiq, eps=tiny)
            chi_x, pK = exbids_split_budget(Ex, p0,  w=wx_in,
                                             gamma=gamma_k, zqt=qt,
                                             loq=loq, hiq=hiq, eps=tiny)
        else:
            chi_y = X.new_ones(Mi)
            chi_x = X.new_ones(Ni)
            qK, pK = None, None

        kx_hat = eta_x * chi_x * KTu / u_d.clamp_min(tiny)
        ky_hat = eta_y * chi_y * Kv / v_d.clamp_min(tiny)

        kx = g_dr * kx_safe + (1.0 - g_dr) * kx_hat
        ky = g_dr * ky_safe + (1.0 - g_dr) * ky_hat
        # kx = kx_safe.pow(g_dr) * kx_hat.pow(1.0 - g_dr)
        # ky = ky_safe.pow(g_dr) * ky_hat.pow(1.0 - g_dr)
        # log-domain damping for kernel-scale quantities
        # kx = (beta_k * kx_safe.log() + (1.0 - beta_k) * kx_hat.clamp_min(tiny).log()).exp()
        # ky = (beta_k * ky_safe.log() + (1.0 - beta_k) * ky_hat.clamp_min(tiny).log()).exp()

        if budget_mode in ['slack_fixed', 'slack_free']:

            A = kappa + s_y - p_ss
            B = kappa + s_x - p_ss

            if prebudget:
                r_t = zeta * r_t + ((1.0 - zeta) * r_t.sum()) * qK
                c_t = zeta * c_t + ((1.0 - zeta) * c_t.sum()) * pK

            a = r_t * (A / r_t.sum())
            b = c_t * (B / c_t.sum())

            if budget_mode == 'slack_fixed':
                iwx_n = s_x / (kappa + s_x)
                iwy_n = s_y / (kappa + s_y)
            elif budget_mode == 'slack_free':
                # iwx_n = Sx_aug / (Np + Sx_aug)
                # iwy_n = Sy_aug / (Np + Sy_aug)
                iwx_n = Sx / (Np + Sx)
                iwy_n = Sy / (Np + Sy)

        elif budget_mode == 'free':
            s_x = r_d
            s_y = c_d
            A = kappa + s_y - p_ss
            B = kappa + s_x - p_ss

            a = r_t * (A / r_t.sum())
            b = c_t * (B / c_t.sum())
            # a = r_t
            # b = c_t

            iwx_n = s_x / (Np + s_x)
            iwy_n = s_y / (Np + s_y)

            # iwx_n = Sx / (Np + Sx)
            # iwy_n = Sy / (Np + Sy)
        elif budget_mode == 'fixed':
            a = None
            b = None
            s_x = s_x
            s_y = s_y
            iwx_n = s_x / (kappa + s_x)
            iwy_n = s_y / (kappa + s_y)
            # iwy_n = s_y/ (a.sum())
            # iwx_n = s_x/ (b.sum())
        else:
            raise ValueError("Unknown budget_mode: %s" % budget_mode)

        # wo_a = (a.sum() / Nref) * (1.0 - iwy_n)
        # wo_b = (b.sum() / Nref) * (1.0 - iwx_n)
        # wo_budget = (wo_a * wo_b).sqrt()
        # iwo_uot = (Np / Nref) #.clip(0,1)
        # iwo_n = 0.5* (wo_budget + iwo_uot)
        iwo_n = (Np / Nref) #.clip(0,1)
        mapk = zip([iwo_n, iwx_n, iwy_n], [iwo_t, iwx_t, iwy_t])
        iwo, iwx, iwy = [xp.clip(w_dr * i_ol + (1.0 - w_dr) * i_ne, *ppas['w_clip'])
                            for i_ne, i_ol in mapk]
        # if err_ab < ab_tol:
        #     break

        print(666666, eps, iwo, iwo_n, iwx_n,  iwy_n,  Np,  _uv , s_x, s_y, a,b,u,v, 33333)

    ppas['wo'][iL], ppas['wx'][iL], ppas['wy'][iL] = iwo, iwx, iwy
    ppas['kxs'][iL], ppas['kys'][iL] = kx, ky

    mapk = zip(['as','bs', 'us', 'vs', 'uds', 'vds',],
               [a,b,u,v,u_d,v_d])
    for ik, iv in mapk:
        if ppas[ik][iL] is None:
            ppas[ik][iL] = iv
        else:
            ppas[ik][iL] = g_dr * ppas[ik][iL] + (1.0 - g_dr) * iv

    # Kv = K @ v
    # KTu = K.T @ u
    # P1 = u * Kv
    # Pt1 = v * KTu
    # Np  = xp.sum(P1)
    PX = u[:, None] * (K @ (v[:, None] * X))   # shape (M, D)

    if tau2_auto and (XF is not None) and (YF is not None):
        tau2s = update_tau2_otm(
            Pt1, P1, (u, K, v), Np, XF, YF,
            feat_normal=feat_normal, DK=DF, device=device, xp=xp
        )
    else:
        tau2s = tau2

    return Pt1, P1, PX, Np, tau2s, ppas

def expectation_exbids_otm(
    X, TY, sigma2,
    C=None,
    d2f=None, tau2=None,
    tau2_auto=False, eps=1e-9, tau2_alpha=1.0,
    feat_normal=['cos'], XF=None, YF=None, device=None,
    xp=th, DF=[1], use_keops=True, ppas=None, iL=None,
    # rx=None, ry=None,  otm_reg='entropy',
):
    """
    Entropic UOT expectation step (normal domain).
    Supports:
      - external OT cost C (true cost matrix), or
      - internally built Gaussian/feature score from X, TY, sigma2, d2f, tau2.

    Returns
    -------
    Pt1, P1, PX, Np, tau2s, u, v
    """
    D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]
    uv_iter, uv_tol = ppas['otm_uv_iter'], ppas['otm_tol']
    ab_iter, ab_tol = ppas['otm_ab_iter'], ppas.get('otm_ab_tol', ppas['otm_tol'])

    a, b, u, v, u_d, v_d = (
        ppas['as'][iL], ppas['bs'][iL], ppas['us'][iL], ppas['vs'][iL],
        ppas['uds'][iL], ppas['vds'][iL]
    )
    dustbin_mode, budget_mode = ppas['dustbin_mode'][iL], ppas['budget_mode'][iL]

    k_ss, kx, ky = ppas['otm_kss'][iL], ppas['kxs'][iL], ppas['kys'][iL]
    iwo, iwx, iwy = ppas['wo'][iL], ppas['wx'][iL], ppas['wy'][iL]
    g_dr, w_dr, qt = ppas['g_dr'], ppas['w_dr'], ppas['qt'][iL]

    eps_c = ppas['otm_eps'][iL]
    tiny = max(float(eps), 1e-10)
    epsv = max(float(eps_c), tiny)

    # ------------------------------------------------------------
    # Evidence-regularized explicit dustbin / budget refresh.
    # Direct support evidence is evaluated after the UV solve, and
    # only refreshes next-step dustbin / budget variables.
    # ------------------------------------------------------------
    prebudget = ppas['prebudget'][iL]
    eta_k = 1.0
    gamma_k = 1.0
    loq, hiq = 0.005, 0.995
    zeta = g_dr

    if (ppas.get('otm_the_x') is not None) and (ppas['otm_the_x'][iL] is not None):
        thy = ppas['otm_the_y'][iL]
        thx = ppas['otm_the_x'][iL]
        rho_y = epsv * thy / (1.0 - thy)
        rho_x = epsv * thx / (1.0 - thx)
    else:
        rho_y = float(ppas['otm_rho_y'][iL])
        rho_x = float(ppas['otm_rho_x'][iL])
        thy = rho_y / (rho_y + epsv)
        thx = rho_x / (rho_x + epsv)

    rx, ry, otm_reg = None, None, 'entropy'
    if otm_reg == 'kl':
        if (rx is None) or (ry is None):
            rx = b
            ry = a
    else:
        rx, ry = 1, 1

    # ------------------------------------------------------------
    # Build logK
    # Case A: C is provided externally as a true OT cost
    #         logK = -C / eps
    # Case B: C is None, build score S from Gaussian/feature term
    #         S = -cost_base, so logK = S / eps
    # ------------------------------------------------------------
    use_stable = False
    if C is None:
        if use_keops:
            S = update_Cost_ko(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            logK = S / epsv
            K = logK.exp()
        else:
            K = update_Cost_xp(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            K.div_(epsv).exp_()
            logK = None
    else:
        if use_keops:
            C = C.to(device=X.device, dtype=X.dtype)
            logK = C / (-epsv)
            K = logK.exp()
        else:
            K = C
            K.div_(-epsv).exp_()
            logK = None

    # Nref = X.new_tensor(float(min(Ni, Mi)))
    Nref = X.new_tensor(float(Ni))
    sf = Nref
    Kv, KTu = None, None

    for _ab in range(ab_iter):
        iwo_t = iwo
        iwx_t = iwx
        iwy_t = iwy

        kappa = Nref * iwo_t
        eta_x = iwx_t / (1.0 - iwx_t).clamp_min(tiny)
        eta_y = iwy_t / (1.0 - iwy_t).clamp_min(tiny)
        s_x = kappa * eta_x
        s_y = kappa * eta_y

        # p_ss = u_d * k_ss * v_d
        A0 = kappa + s_y - p_ss
        B0 = kappa + s_x - p_ss
        if a is None:
            a = (A0 / Mi).expand(Mi)
        else:
            a = a * (A0 / a.sum())
        if b is None:
            b = (B0 / Ni).expand(Ni)
        else:
            b = b * (B0 / b.sum())

        if u is None:
            u = th.ones_like(a)
        if v is None:
            v = th.ones_like(b)
        if u_d is None:
            u_d = 1.0
        if v_d is None:
            v_d = 1.0

        a_safe = a.clamp_min(0)
        b_safe = b.clamp_min(0)

        # --------------------------------------------------------
        # Current augmented kernel / budget is fixed inside UV.
        # --------------------------------------------------------
        Ky0 = K.sum(1).squeeze()
        Kx0 = K.sum(0).squeeze()

        if ky is None:
            ky = eta_y * Ky0
        if kx is None:
            kx = eta_x * Kx0

        kx_safe = kx.clamp_min(tiny)
        ky_safe = ky.clamp_min(tiny)

        for _uv in range(uv_iter):
            u_prev = u.clone()
            v_prev = v.clone()

            Kv = K @ v
            Kv_aug = Kv + ky_safe * v_d
            u = (a_safe / Kv_aug.clamp_min(tiny)).pow(thy)
            u_d = (s_x / ((kx_safe * v).sum() + k_ss * v_d).clamp_min(tiny)).pow(thy)

            KTu = K.T @ u
            KTu_aug = KTu + kx_safe * u_d
            v = (b_safe / KTu_aug.clamp_min(tiny)).pow(thx)
            v_d = (s_y / ((ky_safe * u).sum() + k_ss * u_d).clamp_min(tiny)).pow(thx)

            err_u = ot_relerr(u, u_prev, base=1.0)
            err_v = ot_relerr(v, v_prev, base=1.0)
            err = 0.6 * (err_u + err_v)
            if err < uv_tol:
                break

        Kv = K @ v
        KTu = K.T @ u

        P1 = u * Kv
        Pt1 = v * KTu
        p_y = u * (ky_safe * v_d)
        p_x = v * (kx_safe * u_d)
        p_ss = u_d * k_ss * v_d

        r_t = P1 + p_y
        r_d = p_x.sum() + p_ss
        c_t = Pt1 + p_x
        c_d = p_y.sum() + p_ss

        Np = xp.sum(P1)
        Sx = (r_d - p_ss).clamp_min(0.0)
        Sy = (c_d - p_ss).clamp_min(0.0)
        Sx_aug = r_d
        Sy_aug = c_d
        A = kappa + s_y - p_ss
        B = kappa + s_x - p_ss

        if prebudget:
            # q0 = P1
            # p0 = Pt1
            q0 = th.ones_like(P1)
            p0 = th.ones_like(Pt1)
            if abs(eta_k -1.0) < 1e-4:
                Ey = Ky0
                Ex = Kx0
            else:
                # Ke = _kernel_power(K, logK=logK, power=eta_k, use_keops=use_keops)
                # Ey = (Ke @ p0).view(-1)
                # Ex = (Ke.T @ q0).view(-1)
                Ke = _kernel_power(K, logK=logK, power=eta_k, use_keops=use_keops)
                Ey = Ke.sum(1).squeeze()
                Ex = Ke.sum(0).squeeze()
                
            wy_in = 1.0 - iwy_t
            wx_in = 1.0 - iwx_t

            chi_y, qK = exbids_split_budget(
                Ey, q0,  w=wy_in,
                gamma=gamma_k, zqt=qt,
                loq=loq, hiq=hiq, eps=tiny
            )
            chi_x, pK = exbids_split_budget(
                Ex, p0, w=wx_in,
                gamma=gamma_k, zqt=qt,
                loq=loq, hiq=hiq, eps=tiny
            )

            qO = q0 * chi_y
            pO = p0 * chi_x
            qO = qO / qO.sum().clamp_min(tiny)
            pO = pO / pO.sum().clamp_min(tiny)
        else:
            chi_y = X.new_ones(Mi)
            chi_x = X.new_ones(Ni)

        # --------------------------------------------------------
        # Correct a,b refresh.
        # a,b are augmented true row/column budgets, but their shape is
        # split into matched-only and dustbin-only components.
        # --------------------------------------------------------
        if budget_mode in ['slack_fixed', 'slack_free']:
            Ay = (s_y - p_ss).clamp_min(0.0)
            Bx = (s_x - p_ss).clamp_min(0.0)

            if prebudget:
                if 'sab' in dustbin_mode:
                    # qI = P1 / Np
                    # pI = Pt1 / Np

                    # qD = (p_y ) / (Sy + tiny )
                    # pD = (p_x ) / (Sx + tiny)
                    # qI_n = zeta * qI + (1.0 - zeta) * qK
                    # pI_n = zeta * pI + (1.0 - zeta) * pK
                    # qO_n = zeta * qD + (1.0 - zeta) * qO
                    # pO_n = zeta * pD + (1.0 - zeta) * pO

                    # a = kappa * qI_n + Ay * qO_n
                    # b = kappa * pI_n + Bx * pO_n
                    qI_n = zeta * kappa / Np * P1 + (1.0 - zeta) * kappa / Np * qK
                    pI_n = zeta * kappa / Np * Pt1 + (1.0 - zeta) * kappa / Np * pK
                    qO_n = zeta * Ay / (Sy + tiny) * p_y + (1.0 - zeta) * Ay / (Sy + tiny ) * qO
                    pO_n = zeta * Bx / (Sx + tiny) * p_x + (1.0 - zeta) * Bx / (Sx + tiny ) * pO

                    #TOCHECK-COMPARE
                    # qI_n = kappa * (zeta * P1 / Np + (1.0 - zeta) * qK)
                    # pI_n = kappa * (zeta * Pt1 / Np + (1.0 - zeta) * pK)
                    # qO_n = Ay * (zeta * p_y / (Sy + tiny) + (1.0 - zeta) * qO)
                    # pO_n = Bx * (zeta * p_x / (Sx + tiny) + (1.0 - zeta) * pO)

                    a = qI_n + qO_n
                    b = pI_n + pO_n
                else:
                    a = zeta * r_t + ((1.0 - zeta) * r_t.sum()) * qK
                    b = zeta * c_t + ((1.0 - zeta) * c_t.sum()) * pK
                a = a * (A / a.sum())
                b = b * (B / b.sum())
            else:
                if 'sab' in dustbin_mode:
                    a = P1 * (kappa/Np)  + p_y * (Ay /(Sy + tiny ))
                    b = Pt1 * (kappa/Np) + p_x * (Bx /(Sx + tiny ))
                else:
                    a = r_t 
                    b = c_t 
                a = a * (A / a.sum())
                b = b * (B / b.sum())

            if budget_mode == 'slack_fixed':
                iwx_n = s_x / (kappa + s_x)
                iwy_n = s_y / (kappa + s_y)
            elif budget_mode == 'slack_free':
                # iwx_n = Sx_aug / (Np + Sx_aug)
                # iwy_n = Sy_aug / (Np + Sy_aug)
                iwx_n = Sx / (Np + Sx)
                iwy_n = Sy / (Np + Sy)

        elif budget_mode == 'free':
            s_x = r_d
            s_y = c_d

            # s_x = Sx
            # s_y = Sy
            # A = Np + s_y
            # B = Np + s_x
            a = r_t * (A / r_t.sum())
            b = c_t * (B / c_t.sum())
            iwx_n = s_x / (Np + s_x)
            iwy_n = s_y / (Np + s_y)

            # iwx_n = Sx / (Np + Sx)
            # iwy_n = Sy / (Np + Sy)

        elif budget_mode == 'fixed':
            a = None
            b = None
            s_x = s_x
            s_y = s_y
            iwx_n = s_x / (kappa + s_x)
            iwy_n = s_y / (kappa + s_y)
            # iwy_n = s_y/ (a.sum())
            # iwx_n = s_x/ (b.sum())
        else:
            raise ValueError("Unknown budget_mode: %s" % budget_mode)

        # wo_a = (a.sum() / Nref) * (1.0 - iwy_n)
        # wo_b = (b.sum() / Nref) * (1.0 - iwx_n)
        # wo_budget = (wo_a * wo_b).sqrt()
        # iwo_uot = (Np / Nref) #.clip(0,1)
        # iwo_n = 0.5* (wo_budget + iwo_uot)
        iwo_n = (Np / Nref) #.clip(0,1)
        mapk = zip([iwo_n, iwx_n, iwy_n], [iwo_t, iwx_t, iwy_t])
        iwo, iwx, iwy = [xp.clip(w_dr * i_ol + (1.0 - w_dr) * i_ne, *ppas['w_clip'])
                            for i_ne, i_ol in mapk]

        # --------------------------------------------------------
        # Adaptive dustbin channel for next outer iteration.
        # --------------------------------------------------------
        if 'kxy' in dustbin_mode:
            eta_x = iwx / (1.0 - iwx)
            eta_y = iwy / (1.0 - iwy)
            kx = eta_x * chi_x * KTu / u_d.clamp_min(tiny)
            ky = eta_y * chi_y * Kv / v_d.clamp_min(tiny)
            # kx_hat = eta_x * chi_x * KTu / u_d.clamp_min(tiny)
            # ky_hat = eta_y * chi_y * Kv / v_d.clamp_min(tiny)
            # kx = g_dr * kx_safe + (1.0 - g_dr) * kx_hat
            # ky = g_dr * ky_safe + (1.0 - g_dr) * ky_hat
        elif 'zxy' in dustbin_mode:
            eta_x = iwx / (1.0 - iwx)
            eta_y = iwy / (1.0 - iwy)
            ky = eta_y * Ky0
            kx = eta_x * Kx0
        else:
            kx, ky = None, None

        # if err_ab < ab_tol:
        #     break

    mapk = zip([ 'wo', 'wx', 'wy'], [ iwo,iwx,iwy])
    for ik, iv in mapk:
        ppas[ik][iL] = iv

    mapk = [ [ 'as', a, A], ['bs', b, B] ]
    for ik, iv, ivt in mapk:
        if ppas[ik][iL] is None:
            ppas[ik][iL] = iv
        else:
            ppas[ik][iL] = (g_dr * ivt /  ppas[ik][iL].sum() * ppas[ik][iL]) + (1.0 - g_dr) * iv

    mapk = zip([ 'us', 'vs', 'uds', 'vds'],
               [ u,v,u_d,v_d])
    mapk = zip(['us', 'vs', 'uds', 'vds', 'kxs', 'kys'],
               [u,v,u_d,v_d, kx, ky])
    for ik, iv in mapk:
        if ppas[ik][iL] is None:
            ppas[ik][iL] = iv
        else:
            ppas[ik][iL] = g_dr * ppas[ik][iL] + (1.0 - g_dr) * iv

    # Kv = K @ v
    # KTu = K.T @ u
    # P1 = u * Kv
    # Pt1 = v * KTu
    # Np  = xp.sum(P1)
    PX = u[:, None] * (K @ (v[:, None] * X))   # shape (M, D)

    if tau2_auto and (XF is not None) and (YF is not None):
        tau2s = update_tau2_otm(
            Pt1, P1, (u, K, v), Np, XF, YF,
            feat_normal=feat_normal, DK=DF, device=device, xp=xp
        )
    else:
        tau2s = tau2

    return Pt1, P1, PX, Np, tau2s, ppas

def expectation_exbids_otm_m0(
    X, TY, sigma2,
    C=None, 
    d2f=None, tau2=None,
    tau2_auto=False, eps=1e-9, tau2_alpha=1.0,
    feat_normal=['cos'], XF=None, YF=None, device=None,
    xp=th, DF=[1], use_keops=True, ppas = None, iL = None,
    # rx=None, ry=None,  otm_reg='entropy',
):
    """
    Entropic UOT expectation step (normal domain).
    Supports:
      - external OT cost C (true cost matrix), or
      - internally built Gaussian/feature score from X, TY, sigma2, d2f, tau2.

    Returns
    -------
    Pt1, P1, PX, Np, tau2s, u, v
    """
    D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]
    uv_iter, uv_tol = ppas['otm_uv_iter'], ppas['otm_tol']
    ab_iter, ab_tol = ppas['otm_ab_iter'], ppas.get('otm_ab_tol', ppas['otm_tol'])

    a,b,u,v,u_d,v_d = ppas['as'][iL], ppas['bs'][iL], ppas['us'][iL], ppas['vs'][iL], ppas['uds'][iL], ppas['vds'][iL]
    dustbin_mode, budget_mode = ppas['dustbin_mode'][iL], ppas['budget_mode'][iL]

    k_ss, kx, ky = ppas['otm_kss'][iL] or 0, ppas['kxs'][iL], ppas['kys'][iL]
    iwo, iwx, iwy  = ppas['wo'][iL], ppas['wx'][iL], ppas['wy'][iL]
    # budget_gamma = ppas['budget_gamma']
    g_dr, w_dr, qt = ppas['g_dr'], ppas['w_dr'], ppas['qt'][iL]

    eps_c = ppas['otm_eps'][iL]
    tiny = max(float(eps), 1e-10)
    epsv = max(float(eps_c), tiny)

    if 'otm_the_x' in ppas:
        thy = ppas['otm_the_y'][iL]
        thx = ppas['otm_the_x'][iL]
        rho_y = epsv * thy / (1.0 - thy)
        rho_x = epsv * thx / (1.0 - thx)
    else:
        rho_y = float(ppas['otm_rho_y'][iL])
        rho_x = float(ppas['otm_rho_x'][iL])
        thy = rho_y / (rho_y + epsv)
        thx = rho_x / (rho_x + epsv)


    rx, ry, otm_reg =None, None, 'entropy'
    if otm_reg=='kl':
        if (rx is None) or (ry is None):
            rx = b
            ry = a
    else:
        rx, ry = 1,1

    # ------------------------------------------------------------
    # Build logK
    # Case A: C is provided externally as a true OT cost
    #         logK = -C / eps
    # Case B: C is None, build score S from Gaussian/feature term
    #         S = -cost_base, so logK = S / eps
    # ------------------------------------------------------------

    if C is None:
        if use_keops:
            S = update_Cost_ko(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            logK = S / epsv
            K = logK.exp()
        else:
            K = update_Cost_xp(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            K.div_(epsv).exp_()
    else:
        if use_keops:
            C = C.to(device=X.device, dtype=X.dtype)
            logK = C / (-epsv)
            K = logK.exp()
        else:
            K = C
            K.div_(-epsv).exp_()

    Kv, KTu = None, None
    for _ab in range(ab_iter):
        kappa = X.new_tensor(float(Ni)) * iwo
        eta_x = iwx / (1.0 - iwx).clamp_min(tiny)
        eta_y = iwy / (1.0 - iwy).clamp_min(tiny)
        s_x = kappa * eta_x
        s_y = kappa * eta_y

        if a is None:
            a = ((kappa + s_y) / Mi).expand(Mi)
        if b is None:
            b = ((kappa + s_x) / Ni).expand(Ni)
        if u is None:
            u = th.ones_like(a)
        if v is None:
            v = th.ones_like(b)
        if u_d is None:
            u_d = u.mean()
        if v_d is None:
            v_d = v.mean()
        if ky is None:
            ky = eta_y * K.sum(1).squeeze()
        if kx is None:
            kx = eta_x * K.sum(0).squeeze()

        a_safe = a.clamp_min(tiny)
        b_safe = b.clamp_min(tiny)
        kx_safe =  kx.clamp_min(tiny)
        ky_safe =  ky.clamp_min(tiny)

        for _uv in range(uv_iter): #ra, rb
            u_prev = u.clone()
            v_prev = v.clone()

            Kv = (K @ v)
            Kv_aug = Kv + ky_safe * v_d
            u = (a_safe / Kv_aug.clamp_min(tiny)).pow(thy)
            u_d = (s_x / ((kx_safe * v).sum() + k_ss * v_d).clamp_min(tiny)).pow(thy)

            KTu = (K.T @ u)
            KTu_aug = KTu + kx_safe * u_d
            v = (b_safe / KTu_aug.clamp_min(tiny)).pow(thx)
            v_d = (s_y / ((ky_safe * u).sum() + k_ss * u_d).clamp_min(tiny)).pow(thx)

            # err = max(
            #     (u - u_prev).abs().max().item(),
            #     (v - v_prev).abs().max().item(),
            # )
            # err_u = (u - u_prev).abs().max() / xp.maximum(
            #     xp.maximum(u.abs().max(), u_prev.abs().max()), u.new_tensor(1.0)
            # )
            # err_v = (v - v_prev).abs().max() / xp.maximum(
            #     xp.maximum(v.abs().max(), v_prev.abs().max()), v.new_tensor(1.0)
            # )
            err_u = ot_relerr(u, u_prev, base=1.0)
            err_v = ot_relerr(v, v_prev, base=1.0)
            # err = xp.maximum(err_u, err_v)
            err = 0.70*(err_u + err_v)
            if err < uv_tol:
                break
    
        Kv = (K @ v)
        KTu = (K.T @ u)

        P1  = u * Kv
        Pt1 = v * KTu
        p_y = u * (ky_safe * v_d)
        p_x = v * (kx_safe * u_d)
        p_ss = u_d * k_ss * v_d

        r_t = P1 + p_y
        r_d = p_x.sum() + p_ss
        c_t = Pt1 + p_x
        c_d = p_y.sum() + p_ss
        Np  = xp.sum(P1)
        Sx  = (r_d - p_ss).clamp_min(0.0)
        Sy  = (c_d - p_ss).clamp_min(0.0)

        if dustbin_mode == 'adaptive':
            kx = eta_x * KTu / u_d
            ky = eta_y * Kv / v_d
            # kx_hat = eta_x * KTu / u_d
            # ky_hat = eta_y * Kv / v_d
            # kx = g_dr * kx + (1- g_dr) * kx_hat
            # ky = g_dr * ky + (1- g_dr) * ky_hat
        elif dustbin_mode == 'static':
            kx = kx
            ky = ky
        else:
            raise ValueError("Unknown dustbin_mode: %s" % dustbin_mode)
        
        if budget_mode in ['slack_fixed', 'slack_free']:
            A = (kappa + s_y - p_ss)
            B = (kappa + s_x - p_ss)
            a = A * r_t/r_t.sum()
            b = B * c_t/c_t.sum()

            if budget_mode  == 'slack_fixed':
                iwx_n = s_x / (kappa + s_x)
                iwy_n = s_y / (kappa + s_y)
            elif budget_mode == 'slack_free':
                iwx_n = Sx / (Np + Sx)
                iwy_n = Sy / (Np + Sy)

        elif budget_mode == 'free':
            a = r_t
            b = c_t
            s_x = r_d
            s_y = c_d
            iwx_n = Sx / (Np + Sx)
            iwy_n = Sy / (Np + Sy)
        elif budget_mode == 'fixed':
            a = a
            b = b
            s_x = s_x
            s_y = s_y 
            iwx_n = s_x / (kappa + s_x) 
            iwy_n = s_y / (kappa + s_y)
            # iwy_n = s_y/ (a.sum())
            # iwx_n = s_x/ (b.sum())
        else:
            raise ValueError("Unknown budget_mode: %s" % budget_mode)

        iwo_n = (Np / Ni)
        mapk =  zip([iwo_n, iwx_n, iwy_n], [iwo, iwx, iwy])
        iwo, iwx, iwy =[ xp.clip(w_dr * i_ol + (1.0 - w_dr) * i_ne, *ppas['w_clip']) 
                            for i_ne, i_ol in mapk ]
        # if err_ab < ab_tol:
        #     break
        print(232323232323, eps, iwo, Np, Np / Ni, _uv ,a,b,u,v, 232323232323)
        print(111111111111, np.histogram(P1.detach().cpu(), bins=100))
        print(22222222222222, np.histogram(Pt1.detach().cpu(), bins=100))

    ppas['wo'][iL], ppas['wx'][iL], ppas['wy'][iL] = iwo, iwx, iwy
    mapk =  zip(['as','bs', 'us', 'vs', 'uds', 'vds', 'kxs', 'kys'], 
                [a,b,u,v,u_d,v_d,kx,ky])
    for ik, iv in mapk:
        if ppas[ik][iL] is None:
            ppas[ik][iL] = iv
        else:
            ppas[ik][iL] = g_dr * ppas[ik][iL] + (1.0 - g_dr) * iv 

    # Kv = K @ v
    # KTu = K.T @ u
    # P1 = u * Kv          
    # Pt1 = v * KTu
    # Np  = xp.sum(P1)
    PX  = u[:, None] * (K @ (v[:, None] * X))   # shape (M, D)

    if tau2_auto and (XF is not None) and (YF is not None):
        tau2s = update_tau2_otm(
            Pt1, P1, (u, K, v), Np, XF, YF,
            feat_normal=feat_normal, DK=DF, device=device, xp=xp
        )
    else:
        tau2s = tau2

    return Pt1, P1, PX, Np, tau2s, ppas

def expectation_exbids_otm_m1(
    X, TY, sigma2,
    C=None, 
    d2f=None, tau2=None,
    tau2_auto=False, eps=1e-9, tau2_alpha=1.0,
    feat_normal=['cos'], XF=None, YF=None, device=None,
    xp=th, DF=[1], use_keops=True, ppas = None, iL = None,
    # rx=None, ry=None,  otm_reg='entropy',
):
    """
    Entropic UOT expectation step (normal domain).
    Supports:
      - external OT cost C (true cost matrix), or
      - internally built Gaussian/feature score from X, TY, sigma2, d2f, tau2.

    Returns
    -------
    Pt1, P1, PX, Np, tau2s, u, v
    """
    D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]
    uv_iter, uv_tol = ppas['otm_uv_iter'], ppas['otm_tol']
    ab_iter, ab_tol = ppas['otm_ab_iter'], ppas.get('otm_ab_tol', ppas['otm_tol'])

    a, b, u, v, u_d, v_d = (
        ppas['as'][iL], ppas['bs'][iL], ppas['us'][iL], ppas['vs'][iL],
        ppas['uds'][iL], ppas['vds'][iL]
    )
    dustbin_mode, budget_mode = ppas['dustbin_mode'][iL], ppas['budget_mode'][iL]

    k_ss, kx, ky = ppas['otm_kss'][iL], ppas['kxs'][iL], ppas['kys'][iL]
    iwo, iwx, iwy = ppas['wo'][iL], ppas['wx'][iL], ppas['wy'][iL]
    # budget_gamma = ppas['budget_gamma']
    g_dr, w_dr, qt = ppas['g_dr'], ppas['w_dr'], ppas['qt'][iL]

    eps_c = ppas['otm_eps'][iL]
    tiny = max(float(eps), 1e-10)
    epsv = max(float(eps_c), tiny)

    # ------------------------------------------------------------
    # Evidence-regularized explicit dustbin / budget refresh.
    # Fixed local parameters are used here to avoid introducing new ppas keys.
    # Suggested ranges are listed in the accompanying notes.
    # ------------------------------------------------------------
    prebudget = ppas['prebudget'][iL]
    # prebudget = True # TODO
    eta_k = 1.0
    gamma_k = 0.95
    loq, hiq = 0.005, 0.995

    if 'otm_the_x' in ppas:
        thy = ppas['otm_the_y'][iL]
        thx = ppas['otm_the_x'][iL]
        rho_y = epsv * thy / (1.0 - thy)
        rho_x = epsv * thx / (1.0 - thx)
    else:
        rho_y = float(ppas['otm_rho_y'][iL])
        rho_x = float(ppas['otm_rho_x'][iL])
        thy = rho_y / (rho_y + epsv)
        thx = rho_x / (rho_x + epsv)


    rx, ry, otm_reg =None, None, 'entropy'
    if otm_reg == 'kl':
        if (rx is None) or (ry is None):
            rx = b
            ry = a
    else:
        rx, ry = 1,1

    # ------------------------------------------------------------
    # Build logK
    # Case A: C is provided externally as a true OT cost
    #         logK = -C / eps
    # Case B: C is None, build score S from Gaussian/feature term
    #         S = -cost_base, so logK = S / eps
    # ------------------------------------------------------------
    use_stable = False #TODO
    if C is None:
        if use_keops:
            S = update_Cost_ko(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            logK = S / epsv
            K = logK.exp()
        else:
            K = update_Cost_xp(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            K.div_(epsv).exp_()
            logK = None
    else:
        if use_keops:
            C = C.to(device=X.device, dtype=X.dtype)
            logK = C / (-epsv)
            K = logK.exp()
        else:
            # inplace: C will be overwritten by K to save memory
            K = C
            K.div_(-epsv).exp_()
            logK = None

    Nref = X.new_tensor(float(Ni))
    sf = Nref
    Kv, KTu = None, None
    for _ab in range(ab_iter):
        iwo_t = iwo
        iwx_t = iwx
        iwy_t = iwy

        kappa = Nref * iwo_t
        eta_x = iwx_t / (1.0 - iwx_t).clamp_min(tiny)
        eta_y = iwy_t / (1.0 - iwy_t).clamp_min(tiny)
        s_x = kappa * eta_x
        s_y = kappa * eta_y

        if a is None:
            a = ((kappa + s_y) / Mi).expand(Mi)
        if b is None:
            b = ((kappa + s_x) / Ni).expand(Ni)
        if u is None:
            u = th.ones_like(a)
        if v is None:
            v = th.ones_like(b)
        if u_d is None:
            u_d = u.mean()
        if v_d is None:
            v_d = v.mean()

        a_safe = a.clamp_min(tiny)
        b_safe = b.clamp_min(tiny)

        # --------------------------------------------------------
        # Pre-budget support evidence.
        # q0,p0 are true-point budget shapes, not augmented budgets.
        # --------------------------------------------------------
        if prebudget:
            q0 = a * (sf / a.sum())
            p0 = b * (sf / b.sum())

            Ke = _kernel_power(K, logK=logK, power=eta_k, use_keops=use_keops)
            Ey = (Ke @ p0).view(-1)  # E *sf
            Ex = (Ke.T @ q0).view(-1)  # F *sf

            chi_y, qK = exbids_split_budget(Ey, q0, sf, gamma=gamma_k, zqt=qt, loq=loq, hiq=hiq, eps=tiny)
            chi_x, pK = exbids_split_budget(Ex, p0, sf, gamma=gamma_k, zqt=qt, loq=loq, hiq=hiq, eps=tiny)
        else:
            chi_y = X.new_ones(Mi)
            chi_x = X.new_ones(Ni)
            qK, pK = None, None

        if ky is None:
            ky = eta_y * chi_y * K.sum(1).squeeze()
        if kx is None:
            kx = eta_x * chi_x * K.sum(0).squeeze()

        kx_safe =  kx.clamp_min(tiny)
        ky_safe =  ky.clamp_min(tiny)

        for _uv in range(uv_iter): #ra, rb
            u_prev = u.clone()
            v_prev = v.clone()
            # u_d_prev = u_d.clone()
            # v_d_prev = v_d.clone()

            Kv = (K @ v)
            Kv_aug = Kv + ky_safe * v_d
            u = (a_safe / Kv_aug.clamp_min(tiny)).pow(thy)
            u_d = (s_x / ((kx_safe * v).sum() + k_ss * v_d).clamp_min(tiny)).pow(thy)

            KTu = (K.T @ u)
            KTu_aug = KTu + kx_safe * u_d
            v = (b_safe / KTu_aug.clamp_min(tiny)).pow(thx)
            v_d = (s_y / ((ky_safe * u).sum() + k_ss * u_d).clamp_min(tiny)).pow(thx)

            err_u = ot_relerr(u, u_prev, base=1.0)
            err_v = ot_relerr(v, v_prev, base=1.0)
            # err = xp.maximum(err_u, err_v)
            err = 0.60*(err_u + err_v)
            if err < uv_tol:
                break
    
        Kv = K @ v
        KTu = K.T @ u

        P1 = u * Kv
        Pt1 = v * KTu
        p_y = u * (ky_safe * v_d)
        p_x = v * (kx_safe * u_d)
        p_ss = u_d * k_ss * v_d

        r_t = P1 + p_y
        r_d = p_x.sum() + p_ss
        c_t = Pt1 + p_x
        c_d = p_y.sum() + p_ss
        Np = xp.sum(P1)
        Sx = (r_d - p_ss).clamp_min(0.0)
        Sy = (c_d - p_ss).clamp_min(0.0)

        if dustbin_mode == 'adaptive':
            kx_hat = eta_x * chi_x * KTu / u_d.clamp_min(tiny)
            ky_hat = eta_y * chi_y * Kv / v_d.clamp_min(tiny)
            kx = g_dr * kx_safe + (1- g_dr) * kx_hat
            ky = g_dr * ky_safe + (1- g_dr) * ky_hat
            # kx = kx_safe.pow(g_dr) * kx_hat.pow(1.0 - g_dr)
            # ky = ky_safe.pow(g_dr) * ky_hat.pow(1.0 - g_dr)
            # log-domain damping for kernel-scale quantities
            # kx = (beta_k * kx_safe.log() + (1.0 - beta_k) * kx_hat.clamp_min(tiny).log()).exp()
            # ky = (beta_k * ky_safe.log() + (1.0 - beta_k) * ky_hat.clamp_min(tiny).log()).exp()
        elif dustbin_mode == 'static':
            kx = kx
            ky = ky
        else:
            raise ValueError("Unknown dustbin_mode: %s" % dustbin_mode)
        
        if budget_mode in ['slack_fixed', 'slack_free']:

            # pss = th.minimum(pss, 0.95 * (kappa + s_y), 0.95 * (kappa + s_x))
            A = (kappa + s_y - p_ss)
            B = (kappa + s_x - p_ss)

            if prebudget:
                r_t = g_dr * r_t + ( (1.0 - g_dr) * r_t.sum()) * qK
                c_t = g_dr * c_t + ( (1.0 - g_dr) * c_t.sum()) * pK

            a = r_t * (A / r_t.sum())
            b = c_t * (B / c_t.sum())

            if budget_mode  == 'slack_fixed':
                iwx_n = s_x / (kappa + s_x)
                iwy_n = s_y / (kappa + s_y)
            elif budget_mode == 'slack_free':
                iwx_n = Sx / (Np + Sx)
                iwy_n = Sy / (Np + Sy)

        elif budget_mode == 'free':
            a = r_t
            b = c_t
            s_x = r_d
            s_y = c_d
            iwx_n = Sx / (Np + Sx)
            iwy_n = Sy / (Np + Sy)
        elif budget_mode == 'fixed':
            a = a
            b = b
            s_x = s_x
            s_y = s_y 
            iwx_n = s_x / (kappa + s_x)
            iwy_n = s_y / (kappa + s_y)
            # iwy_n = s_y/ (a.sum())
            # iwx_n = s_x/ (b.sum())
        else:
            raise ValueError("Unknown budget_mode: %s" % budget_mode)

        iwo_n = (Np / Ni) #.clip(0,1)
        mapk =  zip([iwo_n, iwx_n, iwy_n], [iwo_t, iwx_t, iwy_t])
        iwo, iwx, iwy =[ xp.clip(w_dr * i_ol + (1.0 - w_dr) * i_ne, *ppas['w_clip']) 
                            for i_ne, i_ol in mapk ]
        # if err_ab < ab_tol:
        #     break

        print(11111111111111, eps, iwo, iwo_n, iwx_n,  iwy_n,  Np,  _uv , s_x, s_y, a,b,u,v, 22222222222222)
        
    ppas['wo'][iL], ppas['wx'][iL], ppas['wy'][iL] = iwo, iwx, iwy
    ppas['kxs'][iL], ppas['kys'][iL] = kx, ky

    mapk =  zip(['as','bs', 'us', 'vs', 'uds', 'vds',], 
                [a,b,u,v,u_d,v_d])
    for ik, iv in mapk:
        if ppas[ik][iL] is None:
            ppas[ik][iL] = iv
        else:
            ppas[ik][iL] = g_dr * ppas[ik][iL] + (1.0 - g_dr) * iv 

    # Kv = K @ v
    # KTu = K.T @ u
    # P1 = u * Kv          
    # Pt1 = v * KTu
    # Np  = xp.sum(P1)
    PX  = u[:, None] * (K @ (v[:, None] * X))   # shape (M, D)

    if tau2_auto and (XF is not None) and (YF is not None):
        tau2s = update_tau2_otm(
            Pt1, P1, (u, K, v), Np, XF, YF,
            feat_normal=feat_normal, DK=DF, device=device, xp=xp
        )
    else:
        tau2s = tau2

    return Pt1, P1, PX, Np, tau2s, ppas

def expectation_exbids_otm_m2(
    X, TY, sigma2,
    C=None,
    d2f=None, tau2=None,
    tau2_auto=False, eps=1e-9, tau2_alpha=1.0,
    feat_normal=['cos'], XF=None, YF=None, device=None,
    xp=th, DF=[1], use_keops=True, ppas = None, iL = None,
    # rx=None, ry=None,  otm_reg='entropy',
):
    """
    Entropic UOT expectation step (normal domain).
    Supports:
      - external OT cost C (true cost matrix), or
      - internally built Gaussian/feature score from X, TY, sigma2, d2f, tau2.

    Returns
    -------
    Pt1, P1, PX, Np, tau2s, u, v
    """
    D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]
    uv_iter, uv_tol = ppas['otm_uv_iter'], ppas['otm_tol']
    ab_iter, ab_tol = ppas['otm_ab_iter'], ppas.get('otm_ab_tol', ppas['otm_tol'])

    a, b, u, v, u_d, v_d = (
        ppas['as'][iL], ppas['bs'][iL], ppas['us'][iL], ppas['vs'][iL],
        ppas['uds'][iL], ppas['vds'][iL]
    )
    dustbin_mode, budget_mode = ppas['dustbin_mode'][iL], ppas['budget_mode'][iL]

    k_ss, kx, ky = ppas['otm_kss'][iL], ppas['kxs'][iL], ppas['kys'][iL]
    iwo, iwx, iwy = ppas['wo'][iL], ppas['wx'][iL], ppas['wy'][iL]
    g_dr, w_dr, qt = ppas['g_dr'], ppas['w_dr'], ppas['qt'][iL]

    eps_c = ppas['otm_eps'][iL]
    tiny = max(float(eps), 1e-10)
    epsv = max(float(eps_c), tiny)

    # ------------------------------------------------------------
    # Evidence-regularized explicit dustbin / budget refresh.
    # Direct support evidence is evaluated after the UV solve, and
    # only refreshes next-step dustbin / budget variables.
    # ------------------------------------------------------------
    prebudget = ppas['prebudget'][iL]
    eta_k = 1.0
    gamma_k = 1.0
    loq, hiq = 0.005, 0.995
    zeta = g_dr

    if (ppas.get('otm_the_x') is not None) and (ppas['otm_the_x'][iL] is not None):
        thy = ppas['otm_the_y'][iL]
        thx = ppas['otm_the_x'][iL]
        rho_y = epsv * thy / (1.0 - thy)
        rho_x = epsv * thx / (1.0 - thx)
    else:
        rho_y = float(ppas['otm_rho_y'][iL])
        rho_x = float(ppas['otm_rho_x'][iL])
        thy = rho_y / (rho_y + epsv)
        thx = rho_x / (rho_x + epsv)

    rx, ry, otm_reg = None, None, 'entropy'
    if otm_reg == 'kl':
        if (rx is None) or (ry is None):
            rx = b
            ry = a
    else:
        rx, ry = 1, 1

    # ------------------------------------------------------------
    # Build logK
    # Case A: C is provided externally as a true OT cost
    #         logK = -C / eps
    # Case B: C is None, build score S from Gaussian/feature term
    #         S = -cost_base, so logK = S / eps
    # ------------------------------------------------------------
    use_stable = False #TODO
    if C is None:
        if use_keops:
            S = update_Cost_ko(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            logK = S / epsv
            K = logK.exp()
        else:
            K = update_Cost_xp(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            )
            K.div_(epsv).exp_()
            logK = None
    else:
        if use_keops:
            C = C.to(device=X.device, dtype=X.dtype)
            logK = C / (-epsv)
            K = logK.exp()
        else:
            # inplace: C will be overwritten by K to save memory
            K = C
            K.div_(-epsv).exp_()
            logK = None

    Nref = X.new_tensor(float(min(Ni, Mi)))
    sf = Nref
    Kv, KTu = None, None
    for _ab in range(ab_iter):
        iwo_t = iwo
        iwx_t = iwx
        iwy_t = iwy

        kappa = Nref * iwo_t
        eta_x = iwx_t / (1.0 - iwx_t).clamp_min(tiny)
        eta_y = iwy_t / (1.0 - iwy_t).clamp_min(tiny)
        s_x = kappa * eta_x
        s_y = kappa * eta_y

        if a is None:
            a = ((kappa + s_y) / Mi).expand(Mi)
        if b is None:
            b = ((kappa + s_x) / Ni).expand(Ni)
        if u is None:
            u = th.ones_like(a)
        if v is None:
            v = th.ones_like(b)
        if u_d is None:
            u_d = u.mean()
        if v_d is None:
            v_d = v.mean()

        a_safe = a.clamp_min(tiny)
        b_safe = b.clamp_min(tiny)

        # --------------------------------------------------------
        # Current augmented kernel / budget is fixed inside UV.
        # Prebudget evidence is not injected before this solve.
        # --------------------------------------------------------
        if ky is None:
            ky = eta_y * K.sum(1).squeeze()
        if kx is None:
            kx = eta_x * K.sum(0).squeeze()

        kx_safe = kx.clamp_min(tiny)
        ky_safe = ky.clamp_min(tiny)

        for _uv in range(uv_iter): #ra, rb
            u_prev = u.clone()
            v_prev = v.clone()
            # u_d_prev = u_d.clone()
            # v_d_prev = v_d.clone()

            Kv = (K @ v)
            Kv_aug = Kv + ky_safe * v_d
            u = (a_safe / Kv_aug.clamp_min(tiny)).pow(thy)
            u_d = (s_x / ((kx_safe * v).sum() + k_ss * v_d).clamp_min(tiny)).pow(thy)

            KTu = (K.T @ u)
            KTu_aug = KTu + kx_safe * u_d
            v = (b_safe / KTu_aug.clamp_min(tiny)).pow(thx)
            v_d = (s_y / ((ky_safe * u).sum() + k_ss * u_d).clamp_min(tiny)).pow(thx)

            err_u = ot_relerr(u, u_prev, base=1.0)
            err_v = ot_relerr(v, v_prev, base=1.0)
            # err = xp.maximum(err_u, err_v)
            err = 0.5 * (err_u + err_v)
            if err < uv_tol:
                break

        Kv = K @ v
        KTu = K.T @ u

        P1 = u * Kv
        Pt1 = v * KTu
        p_y = u * (ky_safe * v_d)
        p_x = v * (kx_safe * u_d)
        p_ss = u_d * k_ss * v_d

        r_t = P1 + p_y
        r_d = p_x.sum() + p_ss
        c_t = Pt1 + p_x
        c_d = p_y.sum() + p_ss
        Np = xp.sum(P1)
        Sx = (r_d - p_ss).clamp_min(0.0)
        Sy = (c_d - p_ss).clamp_min(0.0)
        Sx_aug = r_d
        Sy_aug = c_d

        # --------------------------------------------------------
        # Prebudget support evidence for next-step refresh.
        # qK,pK adjust true-budget shape; chi_y,chi_x adjust dustbin channel.
        # --------------------------------------------------------
        if prebudget:
            # q0 = a * (sf / (a.sum() + tiny))
            # p0 = b * (sf / (b.sum() + tiny))

            q0 = r_t * (sf / r_t.sum())
            p0 = c_t * (sf / c_t.sum())

            Ke = _kernel_power(K, logK=logK, power=eta_k, use_keops=use_keops)
            Ey = (Ke @ p0).view(-1)  # E * sf
            Ex = (Ke.T @ q0).view(-1)  # F * sf

            wy_in = 1.0 - iwy_t
            wx_in = 1.0 - iwx_t

            chi_y, qK = exbids_split_budget(Ey, q0, sf, w=wy_in,
                                             gamma=gamma_k, zqt=qt,
                                             loq=loq, hiq=hiq, eps=tiny)
            chi_x, pK = exbids_split_budget(Ex, p0, sf, w=wx_in,
                                             gamma=gamma_k, zqt=qt,
                                             loq=loq, hiq=hiq, eps=tiny)
        else:
            chi_y = X.new_ones(Mi)
            chi_x = X.new_ones(Ni)
            qK, pK = None, None

        if dustbin_mode == 'adaptive':
            kx_hat = eta_x * chi_x * KTu / u_d.clamp_min(tiny)
            ky_hat = eta_y * chi_y * Kv / v_d.clamp_min(tiny)
            kx = g_dr * kx + (1.0 - g_dr) * kx_hat
            ky = g_dr * ky + (1.0 - g_dr) * ky_hat
            # kx = kx_safe.pow(g_dr) * kx_hat.pow(1.0 - g_dr)
            # ky = ky_safe.pow(g_dr) * ky_hat.pow(1.0 - g_dr)
            # log-domain damping for kernel-scale quantities
            # kx = (beta_k * kx_safe.log() + (1.0 - beta_k) * kx_hat.clamp_min(tiny).log()).exp()
            # ky = (beta_k * ky_safe.log() + (1.0 - beta_k) * ky_hat.clamp_min(tiny).log()).exp()
        elif dustbin_mode == 'static':
            kx = kx
            ky = ky
        else:
            raise ValueError("Unknown dustbin_mode: %s" % dustbin_mode)

        if budget_mode in ['slack_fixed', 'slack_free']:

            A = kappa + s_y - p_ss
            B = kappa + s_x - p_ss

            if prebudget:
                r_t = zeta * r_t + ((1.0 - zeta) * r_t.sum()) * qK
                c_t = zeta * c_t + ((1.0 - zeta) * c_t.sum()) * pK

            a = r_t * (A / r_t.sum())
            b = c_t * (B / c_t.sum())

            if budget_mode == 'slack_fixed':
                iwx_n = s_x / (kappa + s_x)
                iwy_n = s_y / (kappa + s_y)
            elif budget_mode == 'slack_free':
                # iwx_n = Sx_aug / (Np + Sx_aug)
                # iwy_n = Sy_aug / (Np + Sy_aug)
                iwx_n = Sx / (Np + Sx)
                iwy_n = Sy / (Np + Sy)

        elif budget_mode == 'free':
            a = r_t
            b = c_t
            s_x = r_d
            s_y = c_d
            iwx_n = s_x / (Np + s_x)
            iwy_n = s_y / (Np + s_y)

            # iwx_n = Sx / (Np + Sx)
            # iwy_n = Sy / (Np + Sy)
        elif budget_mode == 'fixed':
            a = a
            b = b
            s_x = s_x
            s_y = s_y
            iwx_n = s_x / (kappa + s_x)
            iwy_n = s_y / (kappa + s_y)
            # iwy_n = s_y/ (a.sum())
            # iwx_n = s_x/ (b.sum())
        else:
            raise ValueError("Unknown budget_mode: %s" % budget_mode)

        iwo_n = (Np / Nref) #.clip(0,1)
        mapk = zip([iwo_n, iwx_n, iwy_n], [iwo_t, iwx_t, iwy_t])
        iwo, iwx, iwy = [xp.clip(w_dr * i_ol + (1.0 - w_dr) * i_ne, *ppas['w_clip'])
                            for i_ne, i_ol in mapk]
        # if err_ab < ab_tol:
        #     break

        print(33333, eps, iwo, iwo_n, iwx_n,  iwy_n,  Np,  _uv , s_x, s_y, a,b,u,v, 33333)

    ppas['wo'][iL], ppas['wx'][iL], ppas['wy'][iL] = iwo, iwx, iwy
    ppas['kxs'][iL], ppas['kys'][iL] = kx, ky

    mapk = zip(['as','bs', 'us', 'vs', 'uds', 'vds',],
               [a,b,u,v,u_d,v_d])
    for ik, iv in mapk:
        if ppas[ik][iL] is None:
            ppas[ik][iL] = iv
        else:
            ppas[ik][iL] = g_dr * ppas[ik][iL] + (1.0 - g_dr) * iv

    # Kv = K @ v
    # KTu = K.T @ u
    # P1 = u * Kv
    # Pt1 = v * KTu
    # Np  = xp.sum(P1)
    PX = u[:, None] * (K @ (v[:, None] * X))   # shape (M, D)

    if tau2_auto and (XF is not None) and (YF is not None):
        tau2s = update_tau2_otm(
            Pt1, P1, (u, K, v), Np, XF, YF,
            feat_normal=feat_normal, DK=DF, device=device, xp=xp
        )
    else:
        tau2s = tau2

    return Pt1, P1, PX, Np, tau2s, ppas

def uot_sinkhorn_dense_general(
    C, a, b, eps_c, rho_y, rho_x,
    max_iter=200, tol=1e-4, mask=None,
    reg_mode='entropy', ref=None,
    tiny=1e-8, neg_inf=-1e30,
    log_u=None, log_v=None,
    return_log=True,
    check_mask=False,
):
    """
    Dense log-domain generalized UOT Sinkhorn (PyTorch version).

    Solve:
        min_P <C, P>
              + eps * Omega(P)
              + rho_y * KL(P1 || a)
              + rho_x * KL(P^T1 || b)

    Parameters
    ----------
    C : (M, N) th.Tensor
        Cost matrix.
    a : (M,) th.Tensor
        Reference marginal for row sums.
    b : (N,) th.Tensor
        Reference marginal for column sums.
    eps : float
        Entropic regularization strength.
    rho_y, rho_x : float
        KL marginal relaxation strengths for row / column marginals.
    reg_mode : str
        One of:
            - 'entropy'      : K = exp(-C/eps)
            - 'kl_separable' : K = diag(a) * exp(-C/eps) * diag(b)
            - 'kl_full'      : K = ref * exp(-C/eps)
    ref : (M, N) th.Tensor or None
        Full reference matrix when reg_mode='kl_full'.
    mask : (M, N) bool th.Tensor or None
        Feasible support mask. False entries are forbidden.
    return_log : bool
        If True, return (log_u, log_v, logK). Otherwise return dense P.

    Returns
    -------
    If return_log:
        log_u, log_v, logK
    else:
        P
    """
    M, N = C.shape
    epsv = max(float(eps_c), tiny)

    rho_y_f = float(rho_y)
    rho_x_f = float(rho_x)
    ry = 1.0 if math.isinf(rho_y_f) else rho_y_f / (rho_y_f + epsv)
    rx = 1.0 if math.isinf(rho_x_f) else rho_x_f / (rho_x_f + epsv)

    log_a = th.log(a.clamp_min(tiny))
    log_b = th.log(b.clamp_min(tiny))

    if reg_mode == 'entropy':
        logK = -C / epsv

    elif reg_mode in ['kl','kl_separable']:
        # K = diag(a) * exp(-C/eps) * diag(b)
        logK = -C / epsv + log_a[:, None] + log_b[None, :]

    elif reg_mode == 'kl_full':
        if ref is None:
            raise ValueError("ref must be provided when reg_mode='kl_full'.")
        logK = th.log(ref.clamp_min(tiny)) - C / epsv

    else:
        raise ValueError(
            "reg_mode must be one of {'entropy', 'kl_separable', 'kl_full'}"
        )

    if mask is not None:
        logK = th.where(mask, logK, th.full_like(logK, neg_inf))

        if check_mask:
            # positive-mass rows/cols should have at least one feasible entry
            bad_rows = (a > tiny) & (~mask.any(dim=1))
            bad_cols = (b > tiny) & (~mask.any(dim=0))
            if bad_rows.any():
                idx = th.nonzero(bad_rows).flatten()[:10].tolist()
                raise ValueError(
                    f"Mask leaves some positive-mass rows infeasible, e.g. rows {idx}"
                )
            if bad_cols.any():
                idx = th.nonzero(bad_cols).flatten()[:10].tolist()
                raise ValueError(
                    f"Mask leaves some positive-mass cols infeasible, e.g. cols {idx}"
                )

    if log_u is None:
        log_u = th.zeros_like(a)
    # else:
    #     log_u = log_u.clone()

    if log_v is None:
        log_v = th.zeros_like(b)
    # else:
    #     log_v = log_v.clone()

    for _ in range(max_iter):
        log_u_prev = log_u.clone()
        log_v_prev = log_v.clone()

        # Row-side update: P1 ~ a
        logKv = th.logsumexp(logK + log_v[None, :], dim=1)
        log_u = ry * (log_a - logKv)

        # Col-side update: Pt1 ~ b
        logKTu = th.logsumexp(logK.transpose(0, 1) + log_u[None, :], dim=1)
        log_v = rx * (log_b - logKTu)

        err = max(
            (log_u - log_u_prev).abs().max().item(),
            (log_v - log_v_prev).abs().max().item(),
        )
        if err < tol:
            break

    if return_log:
        return log_u, log_v, logK

    logP = log_u[:, None] + logK + log_v[None, :]
    if mask is not None:
        P = th.where(mask, th.exp(logP), th.zeros_like(logP))
    else:
        P = th.exp(logP)
    return P

def expectation_ko1(X, TY, sigma2, piu, pi0, d2f=None, tau2=None, 
                    tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1],
                     feat_normal=['cos'], XF=None, YF=None, device=None,
                     xp = th, use_stable=False,
                    ):
    P, c = update_P_ko1(X, TY, sigma2, xp = xp, d2f=d2f, tau2=tau2, 
                        tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha, DF=DF) # m*n
    # ft1 = P.T @ pim # n*1
    # a = (ft1 + c*pi0*U) # n*1
    ft1 = P.T @ piu
    a = (ft1 + c*pi0)

    a.masked_fill_(a == 0, 1.0)
    a = 1.0/a

    Pt1 = ft1*a #1 - a*c*pi0*U, n*1 #check
    P1 = piu * (P @ a)
    PX = piu[:, None] * (P @ (X*a.unsqueeze(1)))
    Np = xp.sum(Pt1)

    if tau2_auto and (not XF is None) and (not YF is None):
        tau2s = update_tau2(Pt1, P1, (P, a, piu), Np, XF, YF, use_keops=True, 
                            feat_normal=feat_normal, DK=DF, device=device, xp=xp)
    else:
        tau2s = tau2
    return Pt1, P1, PX, Np, tau2s

# def expectation_ko2(X, TY, sigma2, piu, pi0, d2f=None, tau2=None, 
#                     tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1],
#                      feat_normal=['cos'], XF=None, YF=None, device=None,
#                      xp = th, use_stable=False, qt = 0.75
#                     ):
#     P = kodist2(TY, X)
#     P = P* (-1.0/ (2.0*sigma2) )
#     if not d2f is None:
#         if tau2_auto:
#             for il in range(len(tau2)):
#                 P = P  + tau2_alpha/(-2.0*tau2[il])*d2f[il]
#         else: 
#             P = P + d2f*tau2_alpha
#     P = P.exp()

#     ft1 = P.T @ piu
#     c0 = xp.quantile(ft1, qt)/(1-pi0)

#     a = (ft1 + c0*pi0)
#     a.masked_fill_(a == 0, 1.0)
#     a = 1.0/a

#     Pt1 = ft1*a #1 - a*c*pi0*U, n*1 #check
#     P1 = piu * (P @ a)
#     PX = piu[:, None] * (P @ (X*a.unsqueeze(1)))
#     Np = xp.sum(Pt1)

#     if tau2_auto and (not XF is None) and (not YF is None):
#         tau2s = update_tau2(Pt1, P1, (P, a, piu), Np, XF, YF, use_keops=True, 
#                             feat_normal=feat_normal, DK=DF, device=device, xp=xp)
#     else:
#         tau2s = tau2
#     return Pt1, P1, PX, Np, tau2s

def expectation_ko2(X, TY, sigma2, piu, pi0, d2f=None, tau2=None, 
                    tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1],
                     feat_normal=['cos'], XF=None, YF=None, device=None,
                     xp = th, use_stable=False,  log_qt =True, qt = 0.75
                    ):
    P = kodist2(TY, X)
    P = P* (-1.0/ (2.0*sigma2) )
    if not d2f is None:
        if tau2_auto:
            for il in range(len(tau2)):
                P = P  + tau2_alpha/(-2.0*tau2[il])*d2f[il]
        else: 
            P = P + d2f*tau2_alpha

    if use_stable:
        log_piu = xp.log(piu)
        log_ft1 = ( P + log_piu[:, None]).logsumexp(dim=0).reshape(-1)


        log_q = xp.quantile(log_ft1, qt)
        log_cpi0 = (log_q + xp.log(pi0)- xp.log1p(-pi0))

        # log denominator:
        # log(ft1 + c0*pi0)
        log_den = xp.logaddexp( log_ft1, log_cpi0, )

        # Pt1 = ft1 / denominator
        Pt1 = xp.exp(log_ft1 - log_den)

        # Pa_mn = P_mn * a_n
        #       = exp(logP_mn - log_den_n)
        #
        Pa = ( P - log_den[None, :] ).exp()

        P1 = piu * Pa.sum(dim=1).reshape(-1)
        PX = piu[:, None] * (Pa @ X)
        Np = xp.sum(Pt1)

        if tau2_auto and (not XF is None) and (not YF is None):
            tau2s = update_tau2(
                Pt1, P1, (Pa, xp.ones_like(log_den), piu,), Np, XF, YF, 
                use_keops=True, feat_normal=feat_normal, DK=DF, device=device, xp=xp,
            )
        else:
            tau2s = tau2

    else:
        P = P.exp()
        ft1 = P.T @ piu
        if log_qt:
            c0u = xp.quantile( xp.log1p(ft1), qt)
            c0 = xp.expm1(c0u) / (1.0 - pi0)
        else:
            c0 = xp.quantile(ft1, qt) / (1.0 - pi0)

        a = ft1 + c0 * pi0
        a.masked_fill_(a == 0, 1.0)
        a = 1.0 / a

        Pt1 = ft1 * a
        P1 = piu * (P @ a)
        PX = piu[:, None] * ( P @ (X * a.unsqueeze(1)))
        Np = xp.sum(Pt1)

        if tau2_auto and (not XF is None) and (not YF is None):
            tau2s = update_tau2(Pt1, P1, (P, a, piu), Np, XF, YF, use_keops=True, 
                                feat_normal=feat_normal, DK=DF, device=device, xp=xp)
        else:
            tau2s = tau2
    return Pt1, P1, PX, Np, tau2s


def expectation_xp2(X, TY, sigma2, piu, pi0, d2f=None, tau2=None, 
                        tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1], 
                        feat_normal=['cos'], XF=None, YF=None, 
                        use_stable=False, device=None, xp = th, 
                        log_qt =True,  qt = 0.75
                    ):
    P = thdist2(TY, X)
    P.mul_(-1.0/ (2.0*sigma2) )

    if not d2f is None:
        if tau2_auto:
            for il in range(len(tau2)):
                P.add_(d2f[il], alpha=tau2_alpha/(-2.0*tau2[il]))
        else:
            P.add_(d2f, alpha=tau2_alpha)

    if not use_stable:
        P.exp_()
        P.mul_(piu[:, None]) # large deviation
        ft1 = P.sum(0)
        
        if log_qt:
            c0u = xp.quantile( xp.log1p(ft1), qt)
            c0 = xp.expm1(c0u) / (1.0 - pi0)
        else:
            c0 = xp.quantile(ft1, qt) / (1.0 - pi0)

        a = (ft1 + c0*pi0)
        a.masked_fill_(a == 0, 1.0)
        a = 1.0/a 

        P.mul_(a[None,:])
        # Pt1 = ft1*a 
        # P1 = xp.sum(P, 1).to_dense()
        # Np = xp.sum(P1)
        # PX = P @ X

    else: 
        # Better but slower
        P.add_(xp.log(piu[:, None]+eps))
        log_ft1 = xp.logsumexp(P, axis=0)
        if log_qt:
            log_cp = xp.quantile(log_ft1, qt)/(1-pi0)*pi0
        else:
            log_cp = xp.log(xp.quantile(log_ft1.exp(), qt)/(1-pi0)*pi0)
        log_a = xp.logaddexp(log_ft1, log_cp)
        P.sub_(log_a)
        P.exp_()

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

def update_P_xp1(X, Y, sigma2, piu, pi0, xp = th, d2f=None, tau2=None, 
             tau2_auto=False, use_stable=False, eps=1e-12, tau2_alpha=1.0, DF=1):
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
    if not use_stable:
        P.exp_()
        # P.mul_(pim[:, None]) # large deviation
        # cs = (xp.exp(cs)* U) *pi0
        P.mul_(piu[:, None]) # large deviation
        cs = xp.exp(cs) *pi0
        cdfs = xp.sum(P, 0).to_dense() + cs
        cdfs.masked_fill_(cdfs == 0, 1.0)
        P.mul_(1.0/cdfs)
    else: 
        # Better but slower
        cs = cs + xp.log(pi0+eps)
        P.add_(xp.log(piu[:, None]+eps))
        log_cdfs = xp.logsumexp(P, axis=0)
        log_cdfs = xp.logaddexp(log_cdfs, cs)
        P.sub_(log_cdfs)
        P.exp_()
    return P

def expectation_ko11(X, TY, sigma2, piu, pi0, d2f=None, tau2=None, 
                    tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1],
                     feat_normal=['cos'], XF=None, YF=None, device=None,
                     xp = th, use_stable=False,
                    ): #TODO: check + log stable
    P, c = update_P_ko1(X, TY, sigma2, xp = xp, d2f=d2f, tau2=tau2, 
                        tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha, DF=DF) # m*n
    # ft1 = P.T @ pim # n*1
    # a = (ft1 + c*pi0*U) # n*1
    piu_l = LazyTensor(piu[None,:,None])
    ft1 = (P.T * piu_l.squeeze(-1)).sum(dim=1).flatten()
    a = (ft1 + c*pi0)

    a.masked_fill_(a == 0, 1.0)
    a = 1.0/a

    Pt1 = ft1*a #1 - a*c*pi0*U, n*1 #check

    a_l= LazyTensor(a[None, :, None])
    X_l= LazyTensor(X[None, :, :])
    P1 = (piu_l* P * a_l.squeeze(-1)).sum(dim=1).flatten()
    PX = (piu_l* P * X_l * a_l.squeeze(-1)).sum(dim=1).flatten()

    Np = xp.sum(Pt1)

    if tau2_auto and (not XF is None) and (not YF is None):
        tau2s = update_tau2(Pt1, P1, (P, a, piu), Np, XF, YF, use_keops=True, 
                            feat_normal=feat_normal, DK=DF, device=device, xp=xp)
    else:
        tau2s = tau2
    return Pt1, P1, PX, Np, tau2s

def update_P_ko1(X, Y, sigma2,  xp = th, d2f=None, tau2=None, 
                tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=1):
    D = X.shape[1]
    P = kodist2(Y, X)
    P = P* (-1.0/ (2.0*sigma2) )
    cs = 0.5* D* xp.log(2.0* xp.pi* sigma2)
    if not d2f is None:
        if tau2_auto:
            for il in range(len(tau2)):
                P = P  + tau2_alpha/(-2.0*tau2[il])*d2f[il]
                cs += 0.5* DF[il]* xp.log(2.0*xp.pi*tau2[il])
        else: 
            P = P + d2f*tau2_alpha
            cs += 0 # xp.log(2.0*xp.pi*tau2[il])

    P = P.exp()
    cs = xp.exp(cs)
    return P, cs

def expectation_xp1(X, TY, sigma2, piu, pi0, d2f=None, tau2=None, 
                        tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1], 
                        feat_normal=['cos'], XF=None, YF=None, 
                        use_stable=False, device=None, xp = th,
                    ):
    P = update_P_xp1(X, TY, sigma2,  piu, pi0, xp = xp, d2f=d2f, tau2=tau2,
                    tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha, DF=DF,
                    use_stable=use_stable)

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

def update_P_xp1(X, Y, sigma2, piu, pi0, xp = th, d2f=None, tau2=None, 
             tau2_auto=False, use_stable=False, eps=1e-12, tau2_alpha=1.0, DF=1):
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
    if not use_stable:
        P.exp_()
        # P.mul_(pim[:, None]) # large deviation
        # cs = (xp.exp(cs)* U) *pi0
        P.mul_(piu[:, None]) # large deviation
        cs = xp.exp(cs) *pi0
        cdfs = xp.sum(P, 0).to_dense() + cs
        cdfs.masked_fill_(cdfs == 0, 1.0)
        P.mul_(1.0/cdfs)
    else: 
        # Better but slower
        cs = cs + xp.log(pi0+eps)
        P.add_(xp.log(piu[:, None]+eps))
        log_cdfs = xp.logsumexp(P, axis=0)
        log_cdfs = xp.logaddexp(log_cdfs, cs)
        P.sub_(log_cdfs)
        P.exp_()
    return P

def update_pi2(P1, Np, N, M, pi0, di=0.01, dtmp=1.0, piu_e = None,
               w_clip=[1e-4,1-1e-4], mrf_solver=None, eps =1e-10, xp=th):
    N, M = float(N), float(M)
    piu_e = float(piu_e or M)

    s_in = di*N/M
    s_0 = 0
    s_0 = pi0/(1- pi0)*di*N
    
    denom_pi = (N + s_0 + di*N)
    pi0_new = ((N - Np +s_0) /denom_pi).clip(*w_clip)
    piu_new = (P1 + s_in) 

    if  (mrf_solver is not None):
        piu_new = mrf_solver.smooth(piu_new)
    piu_new = xp.clip(piu_new, eps, None)
    
    s = xp.sum(piu_new)/denom_pi
    if s + pi0_new <= 0:
        piu_new = ((1.0 - pi0)*piu_e/M).expand(M)
    else:
        # normalize so that sum pi + pi0 = 1
        piu_new = piu_new ** dtmp
        piu_new = ((1.0 - pi0_new) * piu_e / xp.sum(piu_new)) * piu_new
        # piu_new = ((1.0 - pi0_new)* piu_e / (xp.sum(piu_new)**dtmp)) * (piu_new**dtmp)
    return pi0_new, piu_new

def update_pi(P1, Np, N, M, pi0, alpha=1.0, alpha0=1.0, U = None,
               w_clip=[1e-4,1-1e-4], mrf_solver=None,  eps =1e-10, xp=th):
    N, M = float(N), float(M)
    U = float(U or N)

    denom_pi = (N + alpha0 -1 + M*(alpha-1) + eps)
    piu_new = (P1 + alpha -1) #/denom_pi
    pi0_new = ((N - Np + alpha0 -1) /denom_pi).clip(*w_clip)

    if  (mrf_solver is not None):
        piu_new = mrf_solver.smooth(piu_new)

    piu_new = xp.clip(piu_new, eps, None)
    s = xp.sum(piu_new)/denom_pi

    if s + pi0_new <= 0:
        piu_new = ((1.0 - pi0)*U/M).expand(M)
    else:
        # normalize so that sum pi + pi0 = 1
        piu_new = (1.0 - pi0_new)* U/xp.sum(piu_new) * piu_new
    return pi0_new, piu_new

def update_pi0(P1, Np, N, M,  pi0, alpha=1.0, alpha0=1.0,  mrf_solver=None, wa=0.998, xp=th):
    denom_pi = N + alpha0 -1 + M*(alpha-1) + 1e-12
    pi_new = (P1 + alpha -1) / denom_pi
    pi0_new = ( N - Np + alpha0 -1) / denom_pi #xp.sum(r0)

    if  (mrf_solver is not None):
        pi_new = mrf_solver.smooth(pi_new)
        # After smoothing, re-normalize to keep mass sum <= 1 - pi0_new (we will renorm below)

    # ensure numerical stability and normalization
    pi_new = xp.clip(pi_new, 1e-13, None)
    s = xp.sum(pi_new)
    if s + pi0_new <= 0:
        pi_new = ((1.0 - pi0)/M).expand(M)
    else:
        # normalize so that sum pi + pi0 = 1
        pi_new = (1.0 - pi0_new)* pi_new/xp.sum(pi_new)
  
    pi0_new = wa *pi0 + (1-wa)*pi0_new
    pi_new  = wa *pi_new + (1-wa)*pi_new
    return pi0_new, pi_new

def expectation_xp_ref(X, TY, sigma2, gs, xp = th, d2f=None, tau2=None, 
                        tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1], 
                        feat_normal=['cos'], XF=None, YF=None, 
                        use_stable=False, device=None,
                    ):
    P = update_P_xp(X[:,:-1], TY[:,:-1], sigma2, gs, xp = xp, d2f=d2f, tau2=tau2,
                    tau2_auto=tau2_auto, use_stable=False, eps=eps, tau2_alpha=tau2_alpha, DF=DF)
    
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

def expectation_ko_ref(X, TY, sigma2, gs, d2f=None, tau2=None, 
                    tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1],
                     feat_normal=['cos'], XF=None, YF=None, device=None,
                     xp = th, use_stable=False,
                    ):
    P, c = update_P_ko(X[:,:-1], TY[:,:-1], sigma2, gs, xp = xp, d2f=d2f, tau2=tau2,
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

def expectation_xp1_ref(X, TY, sigma2, piu, pi0, d2f=None, tau2=None, 
                        tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1], 
                        feat_normal=['cos'], XF=None, YF=None, 
                        use_stable=False, device=None, xp = th,
                    ):
    P = update_P_xp1(X[:,:-1], TY[:,:-1], sigma2,  piu, pi0, xp = xp, d2f=d2f, tau2=tau2,
                    tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha, DF=DF,
                    use_stable=use_stable)
    
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

def expectation_ko1_ref(X, TY, sigma2, piu, pi0, d2f=None, tau2=None, 
                    tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1],
                     feat_normal=['cos'], XF=None, YF=None, device=None,
                     xp = th, use_stable=False,
                    ):
    P, c = update_P_ko1(X[:,:-1], TY[:,:-1], sigma2, xp = xp, d2f=d2f, tau2=tau2,
                        tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha, DF=DF) # m*n
    # ft1 = P.T @ pim # n*1
    # a = (ft1 + c*pi0*U) # n*1
    ft1 = P.T @ piu
    a = (ft1 + c*pi0)

    a.masked_fill_(a == 0, 1.0)
    a = 1.0/a

    Pt1 = ft1*a #1 - a*c*pi0*U, n*1 #check
    P1 = piu * (P @ a)
    PX = piu[:, None] * (P @ (X*a.unsqueeze(1)))
    Np = xp.sum(Pt1)

    if tau2_auto and (not XF is None) and (not YF is None):
        tau2s = update_tau2(Pt1, P1, (P, a, piu), Np, XF, YF, use_keops=True, 
                            feat_normal=feat_normal, DK=DF, device=device, xp=xp)
    else:
        tau2s = tau2
    return Pt1, P1, PX, Np, tau2s

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
        # elif w_term in ['B', 'both']:
        # else:
        #     raise ValueError(f'gamma_term should be in ["both", "L", "R"]')
    return P

def rigid_outlier(X, Y):
    # chulls = [np.ptp(i, axis=0).prod() for i in [X, Y]]
    phulls = [ float(i.shape[0]) for i in [X, Y]]
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

def rigid_overlap(X, Y):
    # chulls = [np.ptp(i, axis=0).prod() for i in [X, Y]]
    phulls = [ float(i.shape[0]) for i in [X, Y]] #+ \
            # [np.ptp(i, axis=0).prod() for i in [X, Y]]
    try:
        from scipy.spatial import ConvexHull
        chulls = [ ConvexHull(i.detach().cpu().numpy()).volume 
                for i in [X, Y]]
        fhulls = [chulls, phulls]
    except:
        fhulls = [phulls]
    w = max([ min([i[0]/i[1], i[1]/i[0]]) for i in fhulls])
    return float(w)

def init_tmat(T, D, N=None, R = None, s = None, t=None, A=None, B = None,
              d=1.0, W=None,
              s_clip=None, xp =th, device=None, dtype=th.float32):
    if not T in 'ESRTILOAPDNH':
        raise ValueError(f'Invalid transformation type {T}, should be in "ESRTILOAPDNH"')
    itam = {}
    if T in 'ESRTILOAPNH':
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

        if T in 'ESRTILONH':
            if R is None:
                itam['R'] = xp.eye(D, device=device, dtype=dtype).clone()
            else:
                itam['R'] = xp.asarray(R, device=device, dtype=dtype).clone()
                assert (itam['R'].shape == (D,D))

            if T in 'EH':
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
    if T in 'DH':
        if W is None:
            itam['W'] = xp.zeros((N,D), device=device, dtype=dtype).clone()
        else:
            itam['W'] = xp.asarray(W, device=device, dtype=dtype).clone()
            assert (itam['W'].shape == (N,D))

    return itam
