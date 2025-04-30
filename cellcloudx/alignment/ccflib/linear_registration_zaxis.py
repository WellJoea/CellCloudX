from builtins import super
import numpy as np
from tqdm import tqdm
from ._neighbors import Neighbors
from .emregistration import EMRegistration
from .xp_utility import lle_w, gl_w
from ...io._logger import logger

class zlinear_registration(EMRegistration):
    def __init__(self, *args, transformer=None, delta=0.01, z_res = [3,4,5], kw=15, **kwargs):
        super().__init__(*args, **kwargs)

        self.transformer = transformer or 'affine-z'
        self.tol =  1e-9 if self.tol is None else self.tol
        self.kw = kw
        self.delta=delta
        self.z_res = z_res
        self.normal_ = False if self.normal_ is None else self.normal_ #or global
        self.maxiter = self.maxiter or 200
        self.eachiter = 50

        self.normalXY()
        self.normalXYfeatures()

        self.YZ = self.xp.ones((self.M, 1),
                               dtype=self.floatx , 
                               device=self.device)
        self.Y = self.xp.hstack([self.Y, self.YZ])
        self.TY = self.Y.clone()
        self.init_transformer()
        self.update_transformer()

    def init_transformer(self): #TODO
        self.D = 3
        self.tc =  self.X[:,2].min()
        self.tab = self.xp.zeros(self.D-1, device=self.device)
        self.reg_core = self.transformer
        if self.transformer == 'affine-z':
            self.B = self.xp.zeros((self.D-1, self.D), device=self.device)
        elif self.transformer == 'rigid-z':
            self.R = self.xp.eye(self.D-1, device=self.device)
        elif self.transformer in ['isosimilarity-z', 'similarity-z' ]:
            self.R = self.xp.eye(self.D-1, device=self.device)
            self.s = self.xp.eye(self.D-1, device=self.device)
        else:
            raise ValueError('Unknown transformer: %s' % self.transformer)

    def init_paras0(self):
        self.sigma2 = self.sigma_square(self.X[:,:2], self.TY[:,:2])
        self.sigma2c = self.sigma_square(self.X[:,[2]], self.TY[:,[2]])
        self.sigma = self.xp.sqrt(self.xp.stack([self.sigma2, self.sigma2, self.sigma2c]))
        self.tc = self.X[:, self.D-1].min()
        self.q = self.N/2 *(self.D-1)* (1 + self.xp.log(self.sigma2)) + \
                 self.N/2 *(1 + self.xp.log(self.sigma2c))
        self.diff = self.q

    def init_paras(self):
        self.sigma2 = self.sigma_square(self.X, self.TY)
        self.sigma = self.xp.sqrt(self.sigma2)
        self.tc = self.X[:, self.D-1].min()
        self.q = self.N/2 *(self.D)* (1 + self.xp.log(self.sigma2))
        self.Np = self.N
        self.diff = self.q

    def get_z_window(self, X, z_n = 5):
        ax = self.D-1
        xz_uni = self.xp.unique(X[:,ax])
        if z_n < xz_uni.shape[0]:
            z_window = self.xp.linspace(X[:,ax].min(), 
                            X[:,ax].max(), 
                            z_n+1)
            Xs = [ X[((X[:,ax]>=z_window[i]) &  (X[:,ax]<=z_window[i+1]))]  for i in range(z_n)]
            return [Xs, False]
        else:
            Xs = [ X[(X[:,ax] ==i)]  for i in xz_uni]
            return  [Xs, True]

    def get_z_bins(self, Xz, tc, z_n = 0.1):
        zran = Xz.max() - Xz.min()
        zbin = max(1/self.xp.unique(Xz).shape[0], z_n)/2
        error = True
        while error:
            zlen = zran*zbin
            idx = (Xz >= tc- zlen) & (Xz <= tc+zlen)
            zbin += 0.001
            error = idx.sum()< 100
        return idx

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

    def postfix(self, **kargs):
        iargs = {'tol': f'{self.diff :.3e}', 
                'Q': f'{self.q :.3e}', 
                'tc': f'{self.tc :.3e}',
                'np': f'{self.Np :.3e}',
                'sigma2': f'{self.sigma2 :.3e}',}
        iargs.update(kargs)
        return iargs
    
    def register(self, callback= None, **kwargs):
        self.init_paras()
        self.echo_paras()

        Xs = self.X
        for d, z_n in enumerate(self.z_res):
            Xs, is_zslice = self.get_z_window(Xs, z_n=z_n)
            iters = len(Xs)*self.maxiter
            pbar = tqdm(range(iters), total=iters, colour='red',
                         desc=f'{self.reg_core}-{d}', 
                         disable=(self.verbose==0))
            with pbar as pbar:
                min_i = 0
                tc = self.tc
                Q = self.xp.inf
                sigma2 = self.xp.inf
                ity = self.TY
                for i in range(len(Xs)):
                    iX = Xs[i]
                    self.sigma2 = None
                    self.tc = iX[:, self.D-1].min()
                    self.TY = self.Y
                    for j in range(self.maxiter):
                        self.optimization(iX, self.Y)
                        pbar.set_postfix(self.postfix(**{'min_i': min_i}))
                        pbar.update()
                        if  (self.xp.abs(self.sigma2) < self.eps):
                            break
                    if self.sigma2 < sigma2: #Q > self.q: #TODO
                        min_i = i
                        tc = self.tc
                        ity = self.TY
                        Q = self.q
                        sigma2 = self.sigma2
                        pbar.set_postfix(self.postfix(**{'min_i':min_i}))
                        pbar.update()
            pbar.close()
            Xs = Xs[min_i]
            self.TY = ity
            if is_zslice:
                break

        pbar = tqdm(range(self.maxiter), total=self.maxiter, colour='blue',
                        desc=f'{self.reg_core}-f', 
                        disable=(self.verbose==0))
        with pbar as pbar:
            iX = Xs
            self.sigma2 = None
            self.tc = iX[:, self.D-1].min()
            self.TX = iX
            self.TY = ity
            for j in range(self.maxiter):
                self.optimization(iX, self.Y)
                pbar.set_postfix(self.postfix())
                pbar.update()
                if  (self.xp.abs(self.sigma2) < self.eps):
                    break

        # pbar = tqdm(range(self.maxiter), total=self.maxiter, colour='blue',
        #                 desc=f'{self.reg_core}-f', 
        #                 disable=(self.verbose==0))
        # iX = self.X
        # iY = self.Y
        # self.TX = self.X

        # for j in pbar:
        #     self.optimization(self.X, self.Y)
        #     pbar.set_postfix(self.postfix())
        #     if  (self.xp.abs(self.sigma2) < self.eps):
        #         break
        pbar.close()
        self.update_normalize()
        self.del_cache_attributes()
        self.detach_to_cpu(to_numpy=True)

    def expectation(self, iX, iY, K=None):
        if K:
            P = self.update_PK(iX, self.TY, K)
        else:
            P = self.update_P(iX, self.TY)
        self.Pt1 = self.xp.sum(P, 0).to_dense()
        self.P1 = self.xp.sum(P, 1).to_dense()
        self.Np = self.xp.sum(self.P1)
        self.PX = P @ iX

        Nx = self.xp.sum(self.Pt1>0)
        ww = self.xp.clip(1- self.Np/Nx, 0, 1-1e-8)
        self.w = self.wa*self.w + (1-self.wa)*ww

    def update_P(self, X, Y):
        D, Nx, My = X.shape[1], X.shape[0], Y.shape[0]

        P = self.xp.cdist(Y, X, p=2)
        P.pow_(2)

        if not self.sigma2 or (self.sigma2 <= 5*self.eps):
            self.sigma2 = max(self.xp.mean(P)/D, 5*self.eps)
        P.div_(-2*self.sigma2)
        P.exp_()
        cdfs = self.xp.sum(P, 0).to_dense()

        gs = My/Nx*self.w/(1. - self.w)
        cs =(2 * self.xp.pi* self.sigma2) ** (-0.5 * D) #self.sigma.prod()
        cs = gs*cs

        if cs < self.eps:
            cdfs.masked_fill_(cdfs == 0, 1)
        P.div_(cdfs+cs)
        return P

    def update_PK(self, X, Y, K):
        src, dst, P, self.sigma2 = self.cdist_k(X, Y, 
                                          knn=K,
                                          method=self.kd_method,
                                          sigma2=self.sigma2 )
 
        P.mul_(-0.5/self.sigma2)
        P.exp_()
        P = self.xp.sparse_coo_tensor( self.xp.vstack([dst, src]), P, 
                                        size=(self.M, self.N), 
                                        dtype=self.floatx)
        cdfs = self.xp.sum(P, 0, keepdim=True).to_dense()
        Nx = self.xp.sum(cdfs>0)
        My = Y.shape[0]
        gs = My/Nx*self.w/(1. - self.w)
        cs =(2 * self.xp.pi * self.sigma2) ** (-0.5 * (self.D))
        cs = gs*cs

        if cs < self.eps:
            cdfs.masked_fill_(cdfs == 0, 1)
        P.mul_(1/(cdfs+cs))
        return P

    def optimization(self, iX, iY):
        self.expectation(iX, iY)
        self.maximization(iX, iY)
        # self.maximization(iX, iY[:, :self.D-1])
        self.iteration += 1

    def maximization0(self, iX, iY):
        muX = self.xp.divide(self.xp.sum(self.PX, axis=0), self.Np)
        muY = self.xp.divide(iY.T @ self.P1, self.Np)
        X_hat = iX - muX
        Y_hat = iY - muY

        XPX = self.Pt1 @ (X_hat**2)
        self.tc = muX[self.D-1]
        self.sigma2c = XPX[self.D-1] / self.Np

        A_ab = self.PX[:,:self.D-1].T @ Y_hat - self.xp.outer(muX[:self.D-1],  self.P1 @ Y_hat)

        YPYh = (Y_hat.T * self.P1) @ Y_hat
        if self.xp.det(YPYh) == 0:
            B_pre = self.B
            YPY = YPYh.clone()
            YPY.diagonal().add_(self.delta*self.sigma2)
            self.B = (A_ab+self.delta*self.sigma2*B_pre) @ self.xp.linalg.inv(YPY)
        else:
            self.B = A_ab @ self.xp.linalg.inv(YPYh)

        self.tab = muX[:self.D-1] - self.B @ muY

        # YPY = (iY.T * self.P1) @ iY
        # XPY = self.PX[:,:self.D-1].T @ iY
        # for i in range(100):
        #     YPT = self.xp.outer(self.tab, self.xp.sum(iY.T * self.P1, 1))
        #     self.B = (XPY - YPT) @ self.xp.linalg.inv(YPY)
        #     self.tab = muX[:self.D-1] - self.B @ muY

        self.TY = iY @ self.B.T +  self.tab        
        self.TY = self.xp.hstack([self.TY, self.YZ*self.tc])

        trAB = self.xp.trace(A_ab @ self.B.T)
        trXPX = self.xp.sum(XPX[:self.D-1])
        trAAYPY = self.xp.trace(self.B.T @ self.B @ YPYh)
        self.sigma21 = (trXPX - trAB) / (self.Np * (self.D-1))
        self.sigma2 = (trXPX - 2*trAB + trAAYPY) / (self.Np * (self.D-1))

        if self.sigma2 < 0:
            self.sigma2 = self.xp.abs(self.sigma2) * 1

        self.sigma = self.xp.sqrt(self.xp.stack([self.sigma2, self.sigma2, self.sigma2c]))
        self.sigma = self.xp.sqrt(self.sigma2)
        qprev = self.q
        self.q = (self.D-1) * self.Np/2 *(1 + self.xp.log(self.sigma2)) + self.Np/2 *(1 + self.xp.log(self.sigma2c))
        self.diff = self.xp.abs(self.q - qprev)

    def maximization(self, iX, iY, tc=None):
        muX = self.xp.divide(self.xp.sum(self.PX[:,:self.D-1], axis=0), self.Np)
        muY = self.xp.divide(iY.transpose(1,0) @ self.P1, self.Np)

        X_hat = iX[:,:self.D-1] - muX
        Y_hat = iY - muY

        A_ab = self.PX[:,:self.D-1].transpose(1,0) @ Y_hat - \
                 self.xp.outer(muX,  self.P1 @ Y_hat)
        YPYh = (Y_hat.transpose(1,0) * self.P1) @ Y_hat
        try:
            self.B = A_ab @ self.xp.linalg.inv(YPYh)
        except:
            B_pre = self.B
            YPY = YPYh.clone()
            YPY.diagonal().add_(self.delta*self.sigma)
            self.B = (A_ab+self.delta*self.sigma*B_pre) @ self.xp.linalg.inv(YPY)

        self.t = muX - self.B @ muY
        if tc is None:
            self.tc = self.xp.sum(self.PX[:,self.D-1])/self.Np
        else:
            self.tc = tc

        self.TY = iY @ self.B.transpose(1,0) +  self.t
        self.TY = self.xp.hstack([self.TY, self.YZ*self.tc])

        trAB = self.xp.trace(A_ab @ self.B.transpose(1,0))
        trXPX = self.xp.sum(self.Pt1 * self.xp.sum(X_hat * X_hat, 1))
        trAAYPY = self.xp.trace(self.B.T @ self.B @ YPYh)
        self.sigma2 = (trXPX - 2*trAB + trAAYPY) / (self.Np * (self.D-1))
        # self.sigma2 = (trXPX - trAB) / (self.Np * (self.D-1))

        if self.sigma2 < 0:
            self.sigma2 = self.xp.abs(self.sigma2) * 1
        self.sigma2  = self.xp.clip(self.sigma2, 0, 1e10)
        self.sigma = self.xp.sqrt(self.sigma2)
        qprev = self.q
        self.q = self.D * self.Np/2 *(1 + self.xp.log(self.sigma2))
        self.diff = self.xp.abs(self.q - qprev)

    def update_transformer(self, ):
        self.tmat = self.xp.eye(self.D+1, dtype=self.floatx, device=self.device)
        self.tmat[:self.D-1, :self.D] = self.B
        self.tmat[:self.D-1, self.D] = self.tab
        self.tmat[self.D-1, self.D] = self.tc
        self.tform = self.tmat

    def transform_point(self, Y=None):
        if Y is None:
            self.TY = self.homotransform_point(self.Y, self.tmat, inverse=False, xp=self.xp)
            return
        else:
            Yz = self.xp.hstack([Y, self.xp.ones([Y.shape[0], 1], device=Y.device)])
            return self.homotransform_point(Yz, self.tform, inverse=False, xp=self.xp)

    def update_normalize(self):
        '''
        tmat: not scaled transform matrix
        tform: scaled transform matrix
        '''
        # self.update_transformer()

        # self.B *= (self.Xs/self.Ys)
        # self.tab = (self.tab * self.Xs + self.Xm[:self.D-1]) - (self.Ym[:self.D-1] @ self.B.transpose(1,0))
        # self.tc = (self.c * self.Xs + self.Xm[self.D-1])
        # self.update_transformer()

        self.TY = self.TY * self.Xs + self.Xm
        # TY = self.transform_point(self.Yr)

    def get_transformer(self):
        pars = ['tform', 'B', 'R', 's', 'tab', 'tc']
        return { i: getattr(self, i) for i in pars if hasattr(self, i) }
