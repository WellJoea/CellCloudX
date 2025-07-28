import torch as th
import numpy as np
try:
    import pykeops
    pykeops.set_verbose(False)
    from pykeops.torch import LazyTensor
except:
    pass
    # raise ImportError('pykeops is not installed, `pip install pykeops`')

from tqdm import tqdm
import itertools

from .neighbors_ensemble import Neighbors
from ...utilis._clean_cache import clean_cache
from ...io._logger import logger

class pwExpection():
    def __init__(self):
        super().__init__()
        self.rigid_outlier = rigid_outlier
        self.kodist2 = kodist2
        self.thdist2 = thdist2
        self.update_P_xp = update_P_xp
        self.update_P_ko = update_P_ko
        self.expectation_ko = expectation_ko
        self.expectation_xp = expectation_xp
        self.expectation_ko_df = expectation_ko_df
        self.expectation_xp_df = expectation_xp_df

    def pwcompute_feauture(self):
        if self.fexist:
            self.d2f = features_pdist2(self.XF, self.YF, tau2s=self.tau2, 
                                       use_keops=self.use_keops, 
                                       tau2_prediv= self.tau2_prediv,
                                       device=self.device, dtype=self.floatx)
    
            if not self.pairs is None: #TODO
                iXF = self.XF.to(self.device)[self.pairs[0]]
                iYF = self.YF.to(self.device)[self.pairs[1]]
                self.dpf = (iXF - iYF).pow(2).sum(1)
                self.dpf = self.dpf.to(self.device)
                if not self.tau2_auto:
                    self.dpf.div_(-2*self.tau2)
        else:
            self.d2f = None

    def pwcompute_feauture_df(self):
        if self.fexist:
            dd =  features_pdist2_df(self.XF, self.YF, self.tau2, 
                                     use_keops=self.use_keops, 
                                     device=self.device, dtype=self.floatx)
            if self.use_keops:
                self.d2f, self.cdff = dd
            else:
                self.d2f = dd
                self.cdff = None
        else:
            self.d2f = None
            self.cdff = None

    def pwexpectation(self):
        self.Pt1, self.P1, self.PX, self.Np, tau2 = \
            self.expectation_func(self.X, self.TY, self.sigma2, self.gs, xp=self.xp,
                                  d2f=self.d2f, tau2=self.tau2, 
                                  tau2_auto=self.tau2_auto, eps=self.eps,
                                  tau2_alpha=self.tau2_grow[self.iteration], DF=self.DK, 
                                  feat_normal=self.feat_normal, XF=self.XF, YF=self.YF,
                                    device=self.device,
                                  )

        # self.w = (1- self.Pt1).clip(*self.w_clip)
        w = (1-self.Np/self.N).clip(*self.w_clip)
        self.w = self.wa*self.w + (1-self.wa)*w
        self.gs = (self.M/self.N)*self.w/(1-self.w)

        if self.fexist and self.tau2_auto:
            self.tau2 = tau2
        #     for lf in range(self.FL):
        #         # tau2 = self.xp.einsum('ij,ij->', P, self.d2f)/self.Np/self.DF
        #         if self.use_keops:
        #             PF = P[0] @ (self.XF[lf].to(self.device)*P[1].unsqueeze(1))
        #         else:
        #             PF = P @ self.XF[lf].to(self.device)
        #         trPfg = self.xp.sum(self.YF[lf].to(self.device) * PF)
        
        #         if  self.feat_normal[lf] in ['cos', 'cosine']:
        #             tau2 = (2* self.Np - 2 * trPfg) / (self.Np * self.DK[lf] )
        #         else:
        #             trfpf = self.xp.sum( self.Pt1 * self.xp.sum(self.XF[lf] **2, axis=1))
        #             trgpg = self.xp.sum( self.P1  * self.xp.sum(self.YF[lf] **2, axis=1))
        #             tau2 = (trfpf - 2 * trPfg + trgpg) / (self.Np * self.DK[lf] )
        #         if self.tau2_clip[lf] is not None:
        #             self.tau2[lf] = tau2.clip(*self.tau2_clip[lf])
        #         else:
        #             self.tau2[lf] = tau2
        self.update_P_ko_pairs()

    def pwexpectation_df(self):
        self.Pt1, self.P1, self.PX, self.Np = \
            self.expectation_df_func(self.X, self.TY, self.sigma2, self.gs, xp=self.xp,
                                  d2f=self.d2f, cdff=self.cdff)

        # self.w = (1- self.Pt1).clip(*self.w_clip)
        # w = (1.0-self.Np/self.N).clip(*self.w_clip)
        # self.w = self.wa*self.w + (1-self.wa)*w
        self.gs = (self.M/self.N)*self.w/(1-self.w)
        self.update_P_ko_pairs()

    def update_P_ko_pairs(self, a=None):
        if not self.pairs is None:
            xids, yidx = self.pairs[0], self.pairs[1]
            iX = self.X[xids]
            iY = self.TY[yidx]

            dists = (iX- iY).pow(2).sum(1)
            dists.mul_(-1.0/ (2.0*self.sigma2) )
            if self.fexist:
                if self.tau2_auto:
                    dists.add_(self.dpf, alpha=-self.ts_ratio/(2.0*self.tau2))
                else:
                    dists.add_(self.dpf)
            # dists.exp_().mul_(a[xids])
            dists.exp_()
            # dists.mul_(a[xids])
            # dists.mul(1./(2*self.xp.pi*self.sigma2)**(self.D/2))
            self.pairs_idx = dists >= self.c_threshold
            pairs = self.pairs[:, self.pairs_idx]
            if pairs.shape[1] > 0:
                c_Pt1, c_P1, c_PX = self.constrained_expectation(pairs)
                self.Pt1 = self.Pt1 + self.c_alpha*self.sigma2*c_Pt1
                self.P1 = self.P1 + self.c_alpha*self.sigma2*c_P1
                self.PX = self.PX + self.c_alpha*self.sigma2*c_PX
                self.Np = self.xp.sum(self.Pt1)

    def constrained_expectation(self, pairs, pp=None):
        if pp is None:
            pp = self.xp.ones(pairs.shape[1]).to(self.device, dtype=self.floatx)
        V = self.xp.sparse_coo_tensor( 
                    self.xp.vstack([pairs[1], pairs[0]]), pp, 
                    size=(self.M, self.N))
        c_Pt1 = self.xp.sum(V, 0).to_dense()
        c_P1 = self.xp.sum(V, 1).to_dense()
        c_PX = V @ self.X
        return c_Pt1, c_P1, c_PX

    def update_PK(self, X, Y, K):
        D, N, M = X.shape[1], X.shape[0], Y.shape[0]
        src, dst, P, sigma2 = self.cdist_k(X, Y, 
                                            knn=K,
                                            method=self.kd_method,
                                            sigma2=sigma2 )
        P.mul_(-0.5/sigma2)
        P.exp_()
        if self.fexist:
            P = P + self.d2f[dst, src]
            P = P.exp()
            cs = 0.5*(
                self.D*self.xp.log(2*self.xp.pi*self.sigma2) 
                # + self.DF*self.xp.log(2*self.xp.pi*self.tau2)
            )
        else:
            P = P.exp()
            cs = 0.5*(
                self.D*self.xp.log(2*self.xp.pi*self.sigma2)
            )

        P = self.xp.sparse_coo_tensor( self.xp.vstack([dst, src]), P, 
                                        size=(M, N), 
                                        dtype=self.floatx)
        cdfs = self.xp.sum(P, 0, keepdim=True).to_dense()

        Nx = self.xp.sum(cdfs>0)
        gs = M/Nx*self.w/(1. - self.w)
        cs = self.xp.exp(cs+gs)
        cdfs.add_(cs) 
        cdfs.masked_fill_(cdfs == 0, 1.0) 
        P.mul_(1.0/cdfs)
        return P

    def cdist_k(self, X, Y, sigma2=None, method='cunn',
                    metric='euclidean', 
                    knn=200, n_jobs=-1, **kargs):
        (N, D) = X.shape
        M = Y.shape[0]
        snn = Neighbors(method=method, metric=metric, 
                        device_index=self.device_index,
                        n_jobs=n_jobs)
        snn.fit(X, **kargs)
        ckdout = snn.transform(Y, knn=knn)
        nnidx = ckdout[1]

        src = self.xp.LongTensor(nnidx.flatten('C')).to(X.device)
        dst = self.xp.repeat_interleave(self.xp.arange(M, dtype=self.xp.int64), knn, dim=0).to(X.device)
        dist2 = self.xp.tensor(ckdout[0].flatten('C'), dtype=X.dtype, device=X.device)
        dist2.pow_(2)

        if sigma2 is None:
            sigma2 = self.xp.mean(dist2)/D
        return src, dst, dist2, sigma2

    def adjustable_paras(self, **kargs):
        for key, value in kargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)

    def echo_paras(self, paras=None, maxrows=10, ncols = 2, dvinfo=None,):
        if paras is None:
            paras = ['N', 'M', 'D', 'FDs', 'sigma2', 'sigma2_min', 'tau2', 'tau2_decayto', 'tau2_decaystop', 'tau2_auto',   
                     'alpha', 'beta', 'alpha_decayto', 'gamma_growto',  'wa',  'scale_factor',
                     'device', 'device_pre', 'feat_model', 'use_keops', 'floatx', 'floatxx', 
                     'gamma1', 'feat_normal', 'maxiter', 'reg_core', 'tol', 'df_version', 'num_eig',
                      'gamma2', 'kw', 'kl', 'beta_fg', 'use_p1', 'p1_thred', 'normal', 'low_rank', 
                      'fast_low_rank',  'low_rank_type', 'w', 'c', 'lr', 'lr_gamma', 'lr_stepsize','opt']
        logpara = [] 
        for ipara in paras:
            if hasattr(self, ipara):
                ivalue = getattr(self, ipara)
                try:
                    if self.xp.is_tensor(ivalue):
                        if ivalue.ndim == 0:
                            logpara.append([ipara, f'{ivalue.item():.4e}'])
                        else:
                            if self.xp.is_floating_point(ivalue) and ivalue.ndim == 1:
                                logpara.append([ipara, f'{[ round(float(x),4) for x in ivalue]}'])
                            else:
                                logpara.append([ipara, f'{ivalue.cpu().tolist()}'])
                    elif (type(ivalue) in [float]):
                        logpara.append([ipara, f'{ivalue:.4e}'])   
                    else:
                        logpara.append([ipara, f'{ivalue}'])
                except:
                    logpara.append([ipara, f'{ivalue}'])
        logpara = sorted(logpara, key=lambda x: x[0])
        lpadsize = max([len(ipara) for ipara, ivalue in logpara])
        rpadsize = max([len(str(ivalue)) for ipara, ivalue in logpara])

        logpara = [f'{ipara.ljust(lpadsize)} = {ivalue.ljust(rpadsize)}' for ipara, ivalue in logpara]

        if len(logpara)>maxrows:
            nrows = int(np.ceil(len(logpara)/ncols))
        else:
            nrows = len(logpara)
        logpara1 = []
        for i in range(nrows):
            ilog = []
            for j in range(ncols):
                idx = i + j*nrows
                if idx < len(logpara):
                    ilog.append(logpara[idx])
            ilog = '+ ' + ' + '.join(ilog) + ' +'
            logpara1.append(ilog)

        headsize= len(logpara1[0])
        headinfo = 'init parameters:'.center(headsize, '-')
        if dvinfo is None:
            dvinfo = ' '.join(sorted(set(self.dvinfo)))
            dvinfo = dvinfo.center(headsize, '-')
        elif dvinfo is False:
            dvinfo = ''
        else:
            dvinfo = ' '.join(sorted(set(dvinfo)))
            dvinfo = dvinfo.center(headsize, '-')
        logpara1 = '\n' + '\n'.join([headinfo] + logpara1 + [dvinfo])
        
        if self.verbose > 1:
            logger.info(logpara1)

class gwcompute_feauture(object):
    def __init__(self, keops_thr=None, xp=th, verbose=0):
        self.keops_thr = keops_thr
        self.verbose = verbose
        self.xp = xp
        self.features_pdist2 = features_pdist2

    def compute_pairs(self, Fa, tau2, tau2_prediv=None, mask=None, fexist=None, device=None, dtype=None):
        L = len(Fa)
        if tau2_prediv is None:
            tau2_prediv = [True] * L
        with tqdm(total=L*L, 
                    desc="feature fusion",
                    colour='#AAAAAA', 
                    disable=(self.verbose==0)) as pbar:
            for i, j in itertools.product(range(L), range(L)):
                pbar.set_postfix(dict(i=int(i), j=int(j)))
                pbar.update()
                if not fexist is None:
                    assert fexist[i] == (Fa[i] is not None), f'Fa[{i}] is None'
                    assert fexist[j] == (Fa[j] is not None), f'Fa[{j}] is None'

                if (Fa[i] is not None) and (Fa[j] is not None):
                    if (mask is not None) and (mask[i,j]==0):
                        continue
                    try:
                        use_keops = bool(self.keops_thr[i,j])
                    except:
                        use_keops = False

                    if use_keops:
                        fd = features_pdist2(Fa[j], Fa[i],
                                            tau2[i,j], 
                                            use_keops=use_keops, 
                                            tau2_prediv=tau2_prediv[i],
                                            device=device, dtype=dtype)
                        setattr(self,  f'f{i}_{j}', fd)
                    else:
                        if (hasattr(self, f'f{j}_{i}') 
                            and self.xp.all(tau2[i,j]== tau2[j,i]) 
                            and (tau2_prediv[i]== tau2_prediv[j])):
                            setattr(self, f'f{i}_{j}', getattr(self, f'f{j}_{i}').T)
                        else:
                            fd = features_pdist2(Fa[j], Fa[i], tau2[i,j], 
                                                use_keops=use_keops, 
                                                tau2_prediv=tau2_prediv[i], 
                                                device=device, dtype=dtype)
                            setattr(self,  f'f{i}_{j}', fd)
                # else:
                #     setattr(self,  f'f{i}_{j}', None)

def kodist2(X, Y):
    x_i = LazyTensor(X[:, None, :])
    y_j = LazyTensor(Y[None, :, :])
    return ((x_i - y_j)**2).sum(dim=2)

def thdist2(X, Y):
    D = th.cdist(X, Y, p=2)
    D.pow_(2)
    return D

def default_transparas(kargs={}):
    dp= {
        'E':dict(
            fix_s=True, s_clip=None,
        ),
        'S':dict(isoscale=False,
                fix_R=False, fix_t=False, fix_s=False,
                s_clip=None),
        'A':dict(delta=0.1),
        'D':dict( beta=4.0, alpha=5e2, 
                low_rank= 3000,
                low_rank_type = 'keops',
                fast_low_rank = 8000, num_eig=100, 
                gamma1=0, gamma2=0,  kw= 15, kl=15,
                alpha_decayto = 0.5,  use_p1=False, p1_thred = 0,
                gamma_growto = 1, kd_method='sknn'),
        'P':dict(gamma1=None, lr=0.005, lr_stepsize=None,
                 lr_gamma=0.5, opt='LBFGS', d=1.0,
                 opt_iter=70),
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

def expectation_ko(X, TY, sigma2, gs, xp = th, d2f=None, tau2=None, 
                    tau2_auto=False, eps=1e-9, tau2_alpha=1.0, DF=[1],
                     feat_normal=['cos'], XF=None, YF=None, device=None,
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