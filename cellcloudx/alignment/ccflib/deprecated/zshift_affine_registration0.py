from builtins import super
import numpy as np
from tqdm import tqdm

from .pairwise_emregistration import pwEMRegistration
from ._neighbors import Neighbors

from .xp_utility import lle_w, gl_w
from ...io._logger import logger

class ZAffineRegistration(pwEMRegistration):
    def __init__(self, *args, gamma1=None, theta=0.1, kw=15, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'zs-affine'

        self.tol =  1e-9 if self.tol is None else self.tol
        self.gamma1 = 0 if gamma1 is None else gamma1
        self.kw = kw
        self.normal_ = 'each' if self.normal_ is None else self.normal_ #or global
        self.maxiter = self.maxiter or 150

        self.normalXY()
        self.normalXYfeatures()

        self.D = 3
        self.B = self.xp.eye(self.D-1, device=self.device)
        self.tab = self.xp.zeros(self.D-1, device=self.device)
        self.tc = 3450 #self.X[:,2].min()
        self.YZ = self.xp.ones((self.M, 1), 
                               dtype=self.floatx , 
                               device=self.device)
        self.TY = self.xp.hstack([self.Y, self.YZ])
        self.TX = self.xp.hstack([self.X, self.YZ])
        self.HY = self.xp.hstack([self.Y, self.YZ])

        self.update_transformer()
        # self.init_sigma2()

    def init_L(self, kw=15, use_unique=False, method='sknn'):
        if self.gamma1 > 0:
            logger.info(f'compute Y lle...')
            W = lle_w(self.Y, use_unique = use_unique, kw=kw, method=method)
            # W = gl_w(self.Y, kw=kw, method=method)
            self.WY = (W @ self.Y).to(self.device)

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

    def cube_TX(self, tc, z_range=0.05):
        xz_range = self.X[:,2].max() - self.X[:,2].min()
        zspace = max(1/self.xp.unique(self.X[:,2]).shape[0], z_range)*xz_range/2
        zindex = (self.X[:,2] > tc - zspace) & (self.X[:,2] <= tc + zspace)
        self.TX = self.X[zindex,:]

    def register(self, callback= None, **kwargs):
        self.adjustable_paras(**kwargs)
        self.echo_paras()
        self.pre_cumpute_paras()
        if self.verbose:
            pbar = tqdm(range(self.maxiter), total=self.maxiter, colour='red', desc=f'{self.reg_core}')
            for i in pbar:
                if ((not self.sigma2_step is None) and 
                    (i % self.sigma2_step == 0) and 
                    (i // self.sigma2_step <= self.sigma2_iters)):
                    self.sigma2 = self.sigma_square(self.X, self.TY)
                    self.q = 1.0 + self.N * self.D * 0.5 * self.xp.log(self.sigma2)

                self.optimization()
                # if not self.w_clip is None:
                #     self.w = self.xp.clip(self.w, self.w_clip[0], self.w_clip[1])
                log_prt = {
                        'tol': f'{self.diff :.4e}', 
                        'Q': f'{self.q :.5e}', 
                        'w': f'{self.w :.5e}',
                        'np': f'{self.Np :.3e}',
                        'tau2': f'{self.tau2 :.4e}',
                        'tc': f'{self.tc :.4e}',
                        'sigma2': f'{self.sigma2 :.4e}'}

                pbar.set_postfix(log_prt)
                # if (self.p_epoch):
                #     pbar.update(self.p_epoch)
                if callable(callback):
                    kwargs = {'iteration': self.iteration,
                             'error': self.q, 'X': self.X, 'Y': self.TY}
                    callback(**kwargs)

                if  (self.xp.abs(self.sigma2) < self.eps): #(self.diff <= self.tol) or
                    pbar.close()
                    logger.info(f'Tolerance is lower than {self.tol}. Program ends early.')
                    break
            pbar.close()
        else:
            while self.diff > self.tol and (self.iteration < self.maxiter):
                self.optimization()

        if self.get_final_P:
           self.P = self.update_P(self.X, self.TY)
           self.P = self.P.detach().cpu()
        self.update_normalize()
        self.del_cache_attributes()
        self.detach_to_cpu(to_numpy=True)

    def expectation(self):
        # self.cube_TX(self.tc, z_range=0.05)
        # self.TX = self.X
        if self.K:
            P = self.update_PK(self.TX, self.TY)
        else:
            P = self.update_P(self.TX, self.TY)
        self.Pt1= self.xp.sum(P, 0).to_dense()
        self.P1 = self.xp.sum(P, 1).to_dense()
        self.Np = self.xp.sum(self.P1)
        # self.PX = (P.to_dense()[:,self.src] @ self.X[self.src,:])
        self.PX =(P @ self.TX)
        self.P = P
        ww = 1- self.Np/self.Nx
        self.w = self.wa*self.w + (1-self.wa)*ww

    def update_P(self, X, Y):
        P = self.xp.cdist(Y, X, p=2)
        P.pow_(2)
        if not self.sigma2:
            self.sigma2 = self.xp.mean(P)/(self.D-1)
        P.div_(-2*self.sigma2)

        self.Nx = X.shape[0]
        gs = self.M/self.Nx*self.w/(1. - self.w)

        if self.fexist:
            P.add_(self.d2f)
            P.exp_()
            cs = 0.5*(
                self.D*self.xp.log(2*self.xp.pi*self.sigma2) 
                # + self.DF*self.xp.log(2*self.xp.pi*self.tau2)
            )
            cs = self.xp.exp(cs+gs)
        else:
            P.exp_()
            cs = 0.5*(
                self.D*self.xp.log(2*self.xp.pi*self.sigma2)
            )
            cs = self.xp.exp(cs+gs)
        cs =0
        cdfs = self.xp.sum(P, 0).to_dense()
        cdfs.masked_fill_(cdfs == 0, self.eps)
        P.div_(cdfs+cs)
        return P
    
    def update_PK(self, X, Y):
        src, dst, P, self.sigma2 = self.cdist_k(X, Y, 
                                          knn=self.K,
                                          method=self.kd_method,
                                          sigma2=self.sigma2 )
 
        P.mul_(-0.5/self.sigma2)
        P.exp_()
        P = self.xp.sparse_coo_tensor( self.xp.vstack([dst, src]), P, 
                                        size=(self.M, self.N), 
                                        dtype=self.floatx)
        
        # src, dst, P, self.sigma2 = self.cdist_k(Y, X, 
        #                                   knn=self.K,
        #                                   method=self.kd_method,
        #                                   sigma2=self.sigma2 )
        # P.mul_(-0.5/self.sigma2)
        # P.exp_()
        # P = self.xp.sparse_coo_tensor( self.xp.vstack([src, dst]), P, 
        #                                 size=(self.M, self.N), 
        #                                 dtype=self.floatx)
        # self.src = self.xp.unique(src)
        # self.Nx = self.src.shape[0]

        cdfs = self.xp.sum(P, 0, keepdim=True).to_dense()
        self.Nx = self.xp.sum(cdfs>0)

        gs = self.M/self.Nx*self.w/(1. - self.w)
        cs = (2*self.xp.pi*self.sigma2)**(0.5*self.D)
        cs *= gs
        cs = 0

        # print(cdfs.shape, self.Nx, P.shape)
        if cs < self.eps:
            cdfs.masked_fill_(cdfs == 0, 1)
        P.mul_(1/(cdfs+cs))
        return P

    def optimization(self):
        self.expectation()
        self.maximization()
        self.iteration += 1


    def maximization(self):
        Px = self.PX[:,:2]
        TX = self.TX[:,:2]
        muX = self.xp.divide(self.xp.sum(Px, axis=0), self.Np)
        muY = self.xp.divide(self.Y.transpose(1,0) @ self.P1, self.Np)

        X_hat = TX - muX
        Y_hat = self.Y - muY

        A = Px.transpose(1,0) @ Y_hat - \
                 self.xp.outer(muX,  self.P1 @ Y_hat)
        YPY = (Y_hat.transpose(1,0) * self.P1) @ Y_hat

        self.B = A @ self.xp.linalg.inv(YPY)
        self.tab = muX - self.B @ muY
        # self.tc = self.xp.sum(self.Pt1 * self.X[:,2])/self.Np
        # self.tc1 = self.xp.sum(PX[:,-1])/self.Np

        # Y_aug = self.xp.hstack([Y_hat, self.YZ]).to(self.device)
        # W = self.xp.linalg.inv(Y_aug.T @ (self.P1 * Y_aug.T).T) @ Y_aug.T @ self.PX[:, :2]
        # B = W[:2, :2].T
        # t_ab =  muX[:2] - B @ muY
        # self.B = B
        # self.tab= t_ab

        self.TY = self.Y @ self.B.transpose(1,0) +  self.tab
        self.TY = self.xp.hstack([self.TY, self.YZ*self.tc])

        trAB = self.xp.trace(A @ self.B.transpose(1,0))
        trXPX = self.xp.sum(self.Pt1 * self.xp.sum(X_hat * X_hat, 1))
        trXTc = self.xp.sum(self.Pt1 * (self.TX[:,2] - self.tc)**2 )

        self.sigma2 = (trXPX - trAB + trXTc) / (self.Np * self.D)
        if self.sigma2 < 0:
            self.sigma2 = self.xp.abs(self.sigma2) * 5

        qprev = self.q
        self.q = self.D * self.Np/2 * (1+self.xp.log(self.sigma2))
        self.diff = self.xp.abs(self.q - qprev)

    def update_transformer(self, ):
        self.tmat = self.xp.eye(self.D+1, dtype=self.floatx, device=self.device)
        self.tmat[:self.D-1, :self.D-1] = self.B
        self.tmat[:self.D-1, self.D] = self.tab
        self.tmat[self.D-1, self.D] = self.tc

        self.tmatinv = self.xp.linalg.inv(self.tmat)
        self.tform = self.tmat

    def transform_point(self, Y=None):
        if Y is None:
            self.TY = self.homotransform_point(self.Y, self.tmat, inverse=False, xp=self.xp)
            return
        else:
            return self.homotransform_point(Y, self.tform, inverse=False, xp=self.xp)

    def update_normalize(self):
        '''
        tmat: not scaled transform matrix
        tform: scaled transform matrix
        '''
        self.update_transformer()
        # self.transform_point()

        # self.B *= (self.Xs/self.Ys)
        # self.t = (self.t * self.Xs + self.Xm) - (self.Ym @ self.B.transpose(1,0)) #*(self.Xs/self.Ys) scaled in B
        # self.s = self.Xs/self.Ys
        self.update_transformer()
        # self.tform = self.Xf @ self.tmat @ self.xp.linalg.inv(self.Yf)
        # self.tforminv = self.xp.linalg.inv(self.tform)
        # self.TY = self.TY * self.Xs + self.Xm
        # TY = self.transform_point(self.Yr)

    def get_transformer(self):
        return { 'tform': self.tform, 'B': self.B, 't': self.t, 's': self.s }