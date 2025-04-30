from builtins import super
import numpy as np
from tqdm import tqdm


from ._neighbors import Neighbors
from .emregistration import EMRegistration

from .th_operation import thopt
from ...io._logger import logger
from ...transform import homotransform_points, ccf_deformable_transform_point

class zlinear_recorrection(thopt):
    def __init__(self, X, Y, X_feat=None, Y_feat=None, 
                 normal='each', 
                 feat_normal='l2', 
                 feat_model = 'gmm',
                 transformer = 'affine-zc',
                 floatx = None,
                 device = None,
                 use_projection = False,
                 use_YZ = False,
                 device_pre = 'cpu',
                 knn = 50,
                 kiter = 0,
                 kd_method='cunn',
                 w=0, 
                 wa=0.995,
                 gamma = 1,
                 delta = 1,
                 phi = 0,
                 penal_term = 'both',
                 zspace= 0.1,
                 maxiter = 200,
                 tol=1e-9, 
                 seed = 200504,
                 verbose = 1, 
                 **kargs
                ):
        super().__init__(device=device, device_pre=device_pre,
                          floatx=floatx, seed=seed)
        self.verbose = verbose
        self.reg_core = None
        self.normal = normal

        self.feat_normal = feat_normal
        self.feat_model = feat_model

        self.maxiter = maxiter
        self.iteration = 0
        self.tol = tol
        self.w = w
        self.wa = wa
        self.zspace = zspace
        self.penal_term = penal_term
        self.transformer = transformer
        self.use_projection = use_projection
        self.use_YZ = use_YZ
        self.knn = knn
        self.kiter = kiter
        self.kd_method = kd_method

        self.diff = self.xp.inf
        self.q = self.xp.inf

        self.init_XY(X, Y, X_feat, Y_feat)
        self.gamma = self.init_penalty(gamma)
        self.delta = self.init_penalty(delta)
        self.phi = self.init_penalty(phi)
        self.init_zbins()
        self.init_transformer()
        self.homotransform_points = homotransform_points
        self.ccf_deformable_transform_point = ccf_deformable_transform_point
        self.dvinfo = [self.get_memory(idv) for idv in [self.device, self.device_pre]]

    def init_XY(self, X, Y, X_feat, Y_feat):
        self.D = X.shape[1]
        assert X.shape[1] == self.D, 'X should be of shape (N, 3)'
        assert Y.shape[1] == self.D, 'Y should be of shape (M, 3)'
        self.Xr = self.to_tensor(X, dtype=self.floatx, device='cpu')
        self.Yr = self.to_tensor(Y, dtype=self.floatx, device='cpu')

        if self.normal in ['each']:
            self.X, self.Xm, self.Xv, self.Xf = self.centerlize(self.Xr, Xm=None, Xs=None, device='cpu')
            self.Y, self.Ym, self.Yv, self.Yf = self.centerlize(self.Yr, Xm=None, Xs=None, device='cpu')
        else:
            self.X = self.Xr
            self.Y = self.Yr
            self.Xm, self.Xv, self.Xf = (self.xp.zeros(self.D), 
                                        self.xp.asarray(1), 
                                        self.xp.eye(self.D+1))
            self.Ym, self.Yv, self.Yf = (self.xp.zeros(self.D), 
                                        self.xp.asarray(1), 
                                        self.xp.eye(self.D+1))
        self.Zid = self.xp.unique(self.Yr[:,self.D-1])
        self.Lid, self.Ms = self.xp.unique(self.Y[:,self.D-1], return_counts=True)
        self.N, self.M = X.shape[0], Y.shape[0]
        self.L = self.Lid.shape[0]

    def init_penalty(self, penal):
        if penal is None:
            P = self.xp.zeros((self.L,self.L))
        if type(penal) in [float, int]:
            P = self.xp.diag( self.xp.ones(self.L) * penal)
        elif type(penal) in [list, np.ndarray, self.xp.Tensor]:
            try:
                penal = self.xp.asarray(penal)
                typep = 'vector' if penal.ndim == 1 else 'list'
            except:
                typep = 'list'

            if typep == 'vector':
                P = self.xp.zeros((self.L,self.L))
                for i in range(len(penal)):
                    il = self.xp.ones(self.L-i) *penal[i]
                    P += self.xp.diag(il, -i)
                P = P+P.T-np.diag(P.diagonal())

            elif typep == 'list':
                P = self.xp.zeros((self.L,self.L),dtype=self.floatx)
                for i in range(len(penal)):
                    igm = penal[i]
                    if igm is not None:
                        igm = self.xp.tensor(igm)
                        if igm.ndim== 0:
                            igm = igm.unsqueeze(0)
                        il = len(igm)
                        P[i, i: min(i+il, self.L)] = igm[: min(self.L-i, il)]
                        P[i, max(i-il+1, 0): (i+1)] = igm[: min(i+1, il)].flip(0)
            else:
                raise ValueError(f'the length of gamma should be less than {self.L-1} or equal to {self.L}')
        else:
            try:
                P= self.to_tensor(penal)
                assert (P.shape[0] == self.L) and (P.shape[1] == self.L)
            except:
                raise ValueError(f'penal term should be a float, list of float, list of length L or a tensor of shape ({self.L,self.L})')
        P = self.to_tensor(P, dtype=self.floatx, device=self.device)

        if self.penal_term in ['L', 'left']:
            P = self.xp.tril(P)
        elif self.penal_term in ['R', 'right']:
            P = self.xp.triu(P)
        # else:
        #     raise ValueError(f'gamma_term should be in ["both", "L", "R"]')
        return P

    def init_zbins(self):
        self.zspace = self.xp.tensor(self.zspace)
        if self.zspace.ndim == 0:
            self.zspace = self.xp.ones(self.L-1) * self.zspace
        elif self.zspace.ndim == 1:
            self.zspace = self.xp.asarray(self.zspace)
        else:
            raise ValueError(f'zspace should be a scalar or a vector of length {self.L-1}')

        K = self.D-1
        Xzmin = self.X[:,K].min()
        Xzmax = self.X[:,K].max()
        
        zdist = (self.Lid[1:] - self.Lid[:-1])/2
        zpass = zdist*self.zspace

        Yzmin = self.Lid[ 0] - zdist[ 0]
        Yzmax = self.Lid[-1] + zdist[-1]
        rpos = self.xp.hstack([ self.Lid[:-1] + zdist-zpass, Yzmax])
        lpos = self.xp.hstack([Yzmin, self.Lid[1: ] - zdist+zpass ])

        rpos = (rpos-Yzmin)/(Yzmax-Yzmin)*(Xzmax-Xzmin) + Xzmin
        lpos = (lpos-Yzmin)/(Yzmax-Yzmin)*(Xzmax-Xzmin) + Xzmin

        self.Xs = []
        for l,r in zip(lpos, rpos):
            idx = (self.X[:,K] >= l) & (self.X[:,K] <= r)
            ixz = self.X[idx]
            self.Xs.append(ixz.to(self.device)) 
            if ixz.shape[0] < 100:
                print(f'Few points in bin {l} to {r}, please lower the zspace')

        self.Ys = []
        self.Yins = []
        for i in self.Lid:
            idx = self.xp.where(self.Y[:,K] == i)[0]
            self.Yins.append(idx)
            if self.use_YZ:
                self.Ys.append(self.Y[idx][:, :self.D].to(self.device))
            else:
                self.Ys.append(self.Y[idx][:, :self.D-1].to(self.device))

    def init_transformer(self): #TODO
        self.reg_core = self.transformer
        if self.transformer == 'affine-zc':
            self.B = self.xp.zeros((self.L, self.D-1, self.Ys[0].shape[1]), device=self.device)
            self.tc =  self.xp.hstack([i[:, self.D-1].min() for i in self.Xs])
            self.tab = self.xp.zeros((self.L, self.D-1), device=self.device)

            self.B_tmp = self.xp.zeros((self.L, self.D-1, self.Ys[0].shape[1]), device=self.device)
            self.tc_tmp =  self.xp.hstack([i[:, self.D-1].min() for i in self.Xs])
            self.tab_tmp = self.xp.zeros((self.L, self.D-1), device=self.device)

    def init_params(self):
        self.w = self.xp.tensor(self.w)
        if self.w.ndim == 0:
            self.ws = self.xp.ones(self.L, device=self.device)*self.w
        elif self.zspace.ndim == 1:
            self.ws = self.xp.asarray(self.w, device=self.device)
        else:
            raise ValueError(f'w should be a scalar or a vector of length {self.L}')
    
        self.diff = 1.0
        self.Ns = self.xp.asarray([i.shape[0] for i in self.Xs])
        self.Ms = self.xp.asarray([i.shape[0] for i in self.Ys])
        self.YOs = [self.xp.ones((i,1), dtype=self.floatx,
                                 device=self.device) for i in self.Ms]
        if self.use_YZ:
            self.TYs = [ self.Ys[i].clone() for i in range(self.L)]
        else:
            self.TYs = [self.xp.hstack([self.Ys[i], self.YOs[i]]) for i in range(self.L)]
        self.sigma2 = self.sigma_square(self.X, self.Y)
        self.sigma2s = self.xp.ones(self.L, device=self.device)*self.sigma2
        self.Qs = self.xp.ones(self.L, device=self.device)*self.xp.inf

        self.Q = self.xp.inf
        self.Gam = self.gamma.sum(1)
        self.Del = self.delta.sum(1)
        self.Phi = self.phi.sum(1)

    def echo_paras(self, paras=None, maxrows=10, ncols = 2):
        if paras is None:
            paras = ['N', 'M', 'D', 'DF', 'sigma2', 'tau2', 'K', 'KF', 'alpha', 'beta', 
                     'device', 'device_pre', 'feat_model', 'data_level', 'L',
                     'gamma1', 'feat_normal', 'maxiter', 'reg_core', 'tol',
                      'gamma2', 'kw', 'kl', 'beta_fg', 'use_fg', 'normal', 'low_rank', 
                      'low_rank_g',  'w', 'c']
        logpara = []
        for ipara in paras:
            if hasattr(self, ipara):
                ivalue = getattr(self, ipara)
                try:
                    if ((type(ivalue) in [float]) or
                         (self.xp.is_floating_point(ivalue))
                        ):
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
        dvinfo = ' '.join(sorted(set(self.dvinfo)))
        dvinfo = dvinfo.center(headsize, '-')
        logpara1 = '\n' + '\n'.join([headinfo] + logpara1 + [dvinfo])
        logger.info(logpara1)

    def postfix(self, **kargs):
        iargs = {'tol': f'{self.diff :.3e}', 
                'Q': f'{self.Q :.3e}', }
        iargs.update(kargs)
        return iargs

    def register(self, callback= None, **kwargs):
        self.init_params()
        self.echo_paras()
        pbar = tqdm(range(self.maxiter), total=self.maxiter,
                     colour='red', desc=f'{self.reg_core}', 
                     disable=(self.verbose==0))
        for i in pbar:
            self.optimization()
            pbar.set_postfix(self.postfix())
        pbar.close()
        self.update_transform()
        self.del_cache_attributes()
        self.detach_to_cpu(to_numpy=True)

    def optimization(self):
        qpre = self.Q
        for iL in range(self.L):
            if self.iteration >= self.maxiter - self.kiter:
                Pt1, P1, Np, PX = self.expectation(iL, K=self.knn)
            else:
                Pt1, P1, Np, PX = self.expectation(iL)
            self.maximization(iL, Pt1, P1, Np, PX)

        # if self.iteration >preheat:
        self.B = self.B_tmp.clone()
        self.tab = self.tab_tmp.clone()
        self.tc = self.tc_tmp.clone()
 
        self.Q = self.xp.sum(self.Qs)
        self.diff = self.xp.abs(qpre - self.Q)
        self.iteration += 1
    
    def expectation(self, iL,  K=None):
        if K:
            P = self.update_PK(iL, K)
        else:
            P = self.update_P(iL)
        Pt1 = self.xp.sum(P, 0).to_dense()
        P1 = self.xp.sum(P, 1).to_dense()
        Np = self.xp.sum(P1)
        PX = P @ self.Xs[iL]

        Nx = self.xp.sum(Pt1>0)
        ww = self.xp.clip(1- Np/Nx, 0, 1-1e-8)
        self.ws[iL] = self.wa*self.ws[iL] + (1-self.wa)*ww

        # if self.fexist:
        #     self.tau2 = self.xp.einsum('ij,ij->', P, self.d2f)/self.Np/self.DF
        return Pt1, P1, Np, PX

    def update_P(self, iL):
        X, Y= self.Xs[iL], self.TYs[iL]
        D, Ni, Mi = X.shape[1], X.shape[0], Y.shape[0]

        P = self.xp.cdist(Y, X, p=2)
        P.pow_(2)
    
        if (not self.sigma2s[iL]) or (self.sigma2s[iL] <= 10*self.eps):
            self.sigma2s[iL] = max(self.xp.mean(P)/D, 10*self.eps)

        P.div_(-2* self.sigma2s[iL])
        P.exp_()
        cdfs = self.xp.sum(P, 0).to_dense()
        gs = Mi/Ni*self.ws[iL]/(1. - self.ws[iL])
        cs =(2 * self.xp.pi * self.sigma2s[iL]) ** (-0.5 * D)
        cs = gs*cs
    
        cdfs.add_(cs)
        cdfs.masked_fill_(cdfs == 0, 1.0)
        P.div_(cdfs)
        return P

    def update_PK(self,iL, K):
        X, Y= self.Xs[iL], self.TYs[iL]
        D, Ni, Mi = X.shape[1], X.shape[0], Y.shape[0]

        sigma2 = self.sigma2s[iL]
        if (not sigma2) or (sigma2 <= 10*self.eps):
            sigma2 = 10*self.eps

        src, dst, P, sigma2 = self.cdist_k(X, Y, 
                                          knn=K,
                                          method=self.kd_method,
                                          sigma2=sigma2 )
 
        P.mul_(-0.5/sigma2)
        P.exp_()
        P = self.xp.sparse_coo_tensor( self.xp.vstack([dst, src]), P, 
                                        size=(Mi, Ni), 
                                        dtype=self.floatx)
        cdfs = self.xp.sum(P, 0, keepdim=True).to_dense()
        Nx = self.xp.sum(cdfs>0)

        gs = Mi/Nx*self.w/(1. - self.w)
        cs =(2 * self.xp.pi * sigma2) ** (-0.5 * D)
        cs = gs*cs

        if cs < self.eps:
            cdfs.masked_fill_(cdfs == 0, 1)
        P.mul_(1/(cdfs+cs))
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
    
    def maximization(self, iL, Pt1, P1, Np, PX):
        iX, iY = self.Xs[iL], self.Ys[iL]
        sigma2 = self.sigma2s[iL]

        NpPab = Np + sigma2 * self.Del[iL]
        TabP = sigma2 * (self.delta[iL] @ self.tab)
        NpcPc = Np + sigma2 * self.Phi[iL]
        TcP = sigma2 * (self.phi[iL] @ self.tc)

        muXab = self.xp.divide(self.xp.sum(PX[:,:self.D-1], axis=0)+TabP, NpPab)
        muXc = self.xp.divide(self.xp.sum(PX[:,self.D-1])+TcP, NpcPc)
        muY = self.xp.divide(iY.T @ P1, NpPab)

        X_hat_ab = iX[:,:self.D-1] - muXab
        X_hat_c = iX[:,self.D-1] - muXc
        Y_hat = iY - muY

        XabPY = PX[:,:self.D-1].T @ Y_hat -self.xp.outer(muXab, P1 @ Y_hat)
        XabPY += sigma2*self.xp.einsum('l,lij->ij', self.gamma[iL], self.B)

        YPY = (Y_hat.T * P1) @ Y_hat
        YPY.diagonal().add_(sigma2 * self.Gam[iL])

        try:
            A_ab = XabPY @ self.xp.linalg.inv(YPY)
        except:
            B_pre = self.B_tmp[iL]
            YPYh = YPY.clone()
            YPYh.diagonal().add_(0.001*self.sigma2s[iL])
            A_ab = (XabPY+ 0.001*self.sigma2s[iL]*B_pre) @ self.xp.linalg.inv(YPYh)

        # A_ab = XabPY @ self.xp.linalg.inv(YPY)
        t_ab = muXab - A_ab @ muY
        t_c = muXc

        TY = iY @ A_ab.T + t_ab
        TY = self.xp.hstack([TY, self.YOs[iL] *t_c])

        trYPY = self.xp.trace(A_ab.T @ A_ab @ YPY)
        trXPY = self.xp.trace(A_ab.T @ XabPY)
        if self.use_projection:
            trXPX = self.xp.sum(Pt1 * (self.xp.sum(X_hat_ab ** 2, 1) + X_hat_c ** 2) )
            sigma2 = (trXPX - 2*trXPY + trYPY) / (Np * (self.D))
            # trxPx = self.xp.sum( Pt1 * self.xp.sum(iX **2, axis=1) )
            # tryPy = self.xp.sum( P1 * self.xp.sum( TY **2, axis=1))
            # trPxy = self.xp.sum( TY * PX)
            # sigma2 = (trxPx - 2 * trPxy + tryPy) / (Np * self.D)
        else:
            trXPX = self.xp.sum(Pt1 * (self.xp.sum(X_hat_ab ** 2, 1)) )
            sigma2 = (trXPX - 2*trXPY + trYPY) / (Np * (self.D-1))
        if sigma2 < 0:
            sigma2 = self.xp.abs(sigma2) * 5
        sigma2 = self.xp.clip(sigma2, 0, 1e10)

        iQ = self.D * Np/2 *(1+ self.xp.log(sigma2))  \
                + self.xp.einsum('l,lij->', self.gamma[iL], (A_ab - self.B)**2)/2 \
                + self.xp.einsum('l,li->',  self.delta[iL], (t_ab - self.tab)**2)/2 \
                + self.xp.einsum('l,l->',  self.phi[iL], (t_c - self.tc)**2)/2

        self.B_tmp[iL] = A_ab
        self.tab_tmp[iL] = t_ab
        self.tc_tmp[iL] = t_c
        self.TYs[iL] = TY
        self.Qs[iL] = iQ
        if sigma2 > self.eps:
            self.sigma2s[iL] = sigma2

    def update_transform(self):
        '''
        tmats: not scaled transform matrix
        tforms: scaled transform matrix
        '''
        self.TY = self.xp.zeros([self.M, self.D])
        self.tforms = self.xp.zeros([self.L, self.D+1, self.D+1])
        self.tmats  = self.xp.zeros([self.L, self.D+1, self.D+1])
        for iL in range(self.L):
            idx = self.Yins[iL]
            ity = self.TYs[iL]

            Bab = self.B[iL].cpu() * (self.Xv/self.Yv)
            K = Bab.shape[1]
            tab = (self.tab[iL].cpu() * self.Xv + self.Xm[:self.D-1]) - (self.Ym[:K] @ Bab.T)
            tc  = (self.tc[iL].cpu() * self.Xv + self.Xm[self.D-1])

            itform = self.tforms[iL]
            itform[:self.D-1, :K] = Bab
            itform[:self.D-1, self.D] = tab
            itform[self.D-1, self.D] = tc
            itform[self.D, self.D] = 1.0
    
            itmats = self.tmats[iL]
            itmats[:self.D-1, :K] = self.B[iL].cpu()
            itmats[:self.D-1, self.D] = self.tab[iL].cpu()
            itmats[self.D-1, self.D] = self.tc[iL].cpu()
            itmats[self.D, self.D] = 1.0

            self.TY[idx] = ity.cpu() * self.Xv + self.Xm

    def transform_point(self, Yr, tforms = None):
        tforms = self.tforms if tforms is None else tforms
        Yins = [ Yr[:,self.D-1] == i for i in self.Zid ]
        Ys =   [ Yr[i] for i in Yins ]
        return self.transform_points(Ys, tforms, Yins=Yins)

    def transform_points(self, Ys, tforms, Yins=None, inverse=False, xp=None):
        if xp is None:
            if 'torch.Tensor' in str(type(Ys[0])):
                import torch as th
                xp = th
                if self.use_YZ:
                    locs = [ i[:,:self.D].cpu() for i in Ys]
                else:
                    locs = [ xp.hstack([i[:,:self.D-1].cpu(), xp.ones([i.shape[0], 1])]) for i in Ys]   
            else:
                xp = np
                if self.use_YZ: 
                    locs = [ i[:,:self.D] for i in Ys]
                else:
                    locs = [ xp.hstack([i[:,:self.D-1], xp.ones([i.shape[0], 1])]) for i in Ys]
        
        TYs = self.homotransform_points(locs, tforms, inverse=inverse, xp=xp)

        if Yins is not None:
            M = sum([ i.shape[0] for i in Ys])
            T = self.xp.zeros([M, self.D])
            for iL in range(len(Ys)):
                T[Yins[iL]] = TYs[iL]
            TYs = T
        return TYs

    def detach_to_cpu(self, attributes=None, to_numpy=True):
        if attributes is None:
            attributes = ['R', 'A', 'B', 'tab', 'tc', 'd', 's',
                          'Xm', 'Xs', 'Xf', 'X', 'Ym', 'Ys', 'Y', 'Yf', 
                          'TYs', 'TY', 'B_tmp', 'tab_tmp', 'tc_tmp',
                          'beta', 'G', 'Q', 'S', 'W', 'inv_S',
                          'tmat', 'tmatinv', 'tform', 'tforminv',
                           'P', 'C']    
        for a in attributes:
            if hasattr(self, a):
                value = getattr(self, a)
                if to_numpy:
                    issp, spty = self.is_sparse(value)
                    if (spty=='torch'):
                        if issp:
                            value = self.thsparse_to_spsparse(value)
                        else:
                            value = value.detach().cpu().numpy()
                        setattr(self, a, value)
                else:
                    value = value.detach().cpu()
                    setattr(self, a, value)

    def del_cache_attributes(self, attributes=None):
        if attributes is None:
            attributes = ['Pf', 'P1', 'Pt1', 'cdff', 'PX', 'MY', 'Av', 'AvUCv', 
                          #Xr, Yr, 
                          'B_tmp', 'tab_tmp', 'tc_tmp', 
                          'XF', 'YF', 'VAv', 'Fv', 'F', 'MPG' ]                
        for attr in attributes:
            if hasattr(self, attr):
                delattr(self, attr)
        self.clean_cache()
    