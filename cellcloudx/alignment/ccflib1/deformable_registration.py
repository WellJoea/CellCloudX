from builtins import super
import numpy as np

from .emregistration import EMRegistration
from .xp_utility import lle_w, gl_w, low_rank_eigen_grbf, low_rank_eigen, WoodburyB, WoodburyC
from ...io._logger import logger

class DeformableRegistration(EMRegistration):
    def __init__(self, *args, alpha=None, beta=None, gamma1=None, 
                gamma2=None, kw=15, kl=15,
                #  gamma_decay = 0.99, 
                #  gamma_decay_start = 150, 
                use_fg=False, beta_fg=None,
                low_rank_g =False, low_rank=True, num_eig=100,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'deformable'
        self.alpha = 0.5 if alpha is None else alpha
        self.beta_ = 1 if beta is None else beta
        self.gamma1 = 0 if gamma1 is None else gamma1
        self.gamma2 = 0 if gamma2 is None else gamma2
        self.tol =  1e-11 if self.tol is None else self.tol

        self.low_rank_g= low_rank_g
        if self.M > 5e5:
            #self.low_rank_g = True
            logger.warning('Both low_rank and low_rank_g should be set to True for large datasets.')
    
        self.low_rank_ = ((low_rank or low_rank_g) or (self.gamma1>0 or self.gamma2>0))
        self.num_eig_ = num_eig
        self.use_fg = use_fg
        self.beta_fg = beta_fg
        self.use_local =  (self.gamma1>0) or (self.gamma2>0)
        self.kw =kw
        self.kl =kl
        self.normal_ = 'each' if self.normal_ is None else self.normal_ # TODO global

        self.normalXY()
        self.normalXYfeatures()
        self.init_L( kw=self.kw, kl=self.kl, method=self.kd_method)
        self.update_transformer()

    @property
    def beta(self):
        return self.beta_
    @property
    def num_eig(self):
        return self.num_eig_
    @property
    def low_rank(self):
        return self.low_rank_

    def init_L(self, Y=None, kw=15, kl=20, use_unique=False, method='sknn'): #TODO float
        '''
        compute penalty matrix...
        '''

        if Y is None:
            Y = self.Y.detach().clone().to(self.device_pre)

        if (self.gamma1>0):
            logger.info('compute Y lle...')
            M = lle_w(Y, use_unique = use_unique, kw=kw, method=method)
            M = M.transpose(1,0) @ M
            self.MY = (self.gamma1 * (M @ Y))

        if (self.gamma2>0):
            logger.info('compute Y gl...')
            P = gl_w(Y, kw=kl, method=method)
            P = P.transpose(1,0) @ P

        if self.low_rank is True:
            logger.info(f'compute G eigendecomposition: low_rank_g = {self.low_rank_g}...')
            if self.low_rank_g:
                self.Q, self.S  = low_rank_eigen_grbf(Y, self.beta_**0.5, self.num_eig, sw_h=0)
            else:
                G = self.kernel_xmm(Y, Y, sigma2=self.beta_, temp=1/2)[0]
                self.Q, self.S = low_rank_eigen(G, self.num_eig, xp=self.xp)
                self.clean_cache(G)

            self.inv_S = self.xp.diag(1./self.S.clone())
            self.S = self.xp.diag(self.S)
    
            if self.use_local:
                U = 0
                if (self.gamma1>0):
                    U =  self.gamma1 * (M @ self.Q) + U
                    self.clean_cache(M)
                if (self.gamma2>0):
                    U =  self.gamma2 * (P @ self.Q) + U
                    self.clean_cache(P)
                # F = (self.alpha * self.xp.eye(self.M) + self.gamma1 * MG + self.gamma2 *PG)
                # self.Fv = np.linalg.inv(F)

                Cv = self.inv_S
                V = self.Q.T
                if self.data_level<=1:
                    Av = 1.0 / self.alpha * self.xp.eye(self.M, dtype=self.floatx, device=self.device_pre)
                    self.Fv = WoodburyB(Av, U, Cv, V, xp=self.xp)

                elif self.data_level>1:
                    Av = self.xp.asarray(1.0 / self.alpha, dtype=self.floatx)
                    UCv = self.xp.linalg.inv(Cv  + Av * (V @ U))
                    self.Av = Av
                    self.AvUCv = Av * (U @ UCv)
                    self.VAv = V * Av
                self.clean_cache(U)
            else:
                #self.Fv = 1 / self.alpha * self.xp.eye(self.M, dtype=self.floatx, device=self.device)
                self.Fv = self.xp.asarray(1.0 / self.alpha, dtype=self.floatx) #only for multiply data
        else:
            logger.info('compute G eigendecomposition...')
            self.G = self.kernel_xmm(Y, Y, sigma2=self.beta_, temp=1/2)[0]

            if self.use_fg: # TODO
                logger.info('compute feature Y gmm...')
                G1 = self.kernel_xmm(self.Y_feat, self.Y_feat, sigma2=self.beta_fg/2)[0]
                self.G *= G1.to(self.device_pre)

            if self.use_local:
                MPG = 0
                if (self.gamma1>0):
                    MPG = self.gamma1 * (M @ self.G) + MPG
                if (self.gamma2>0):
                    MPG =  self.gamma2 * (P @ self.G) + MPG
                self.MPG = MPG

                self.F = (self.alpha * self.xp.eye(self.M, dtype=self.floatx) + self.MPG)
            else:
                self.F = self.alpha * self.xp.eye(self.M, dtype=self.floatx)

        self.W = self.xp.zeros((self.M, self.D), dtype=self.floatx)
        device_attr = ['MY', 'Q', 'S', 'inv_S', 'Fv', 'Av', 'AvUCv', 
                       'VAv', 'G', 'W', 'MPG', 'F']
        for iattr in device_attr:
            if hasattr(self, iattr):
                setattr(self, iattr, getattr(self, iattr).to(self.device))
        self.clean_cache()

    def optimization(self):
        self.expectation()
        self.maximization()
        self.iteration += 1

    def maximization(self):
        self.update_transform()
        self.update_transformer()
        self.transform_point()
        self.update_variance()
    
    def update_transform(self):
        if self.low_rank is False:
            if (self.use_local):
                A = self.xp.multiply(self.P1, self.G).T + self.sigma2 * self.F
                B = self.PX - self.xp.multiply(self.Y.T, self.P1).T - self.sigma2*self.MY
                self.W = self.xp.linalg.solve(A, B)
            else: # same
                # A = self.xp.dot(self.xp.diag(self.P1), self.G) + \
                #     self.alpha * self.sigma2 * self.xp.eye(self.M)
                # B = self.PX - self.xp.dot(self.xp.diag(self.P1), self.Y)
                A = self.xp.multiply(self.P1, self.G).T + self.sigma2 * self.F
                B = self.PX - self.xp.multiply(self.Y.T, self.P1).T
                self.W = self.xp.linalg.solve(A, B)

        elif self.low_rank is True:
            if (self.use_local):
                B = self.PX - self.xp.multiply(self.Y.T, self.P1).T - self.sigma2*self.MY
                dPQ = self.xp.multiply(self.Q.T, self.P1).T

                if self.data_level<=1:
                    Fvs2v = self.Fv / self.sigma2
                    FB = Fvs2v @ B
                    FQ = Fvs2v @ dPQ
                    # Uv = np.linalg.inv(self.inv_S + self.Q.T @ FQ)
                    # self.W = FB - (FQ @ Uv) @ (self.Q.T @ FB)
                    del B, Fvs2v, dPQ
                    self.W = FB - FQ @ self.xp.linalg.solve(self.inv_S + self.Q.T @ FQ, self.Q.T @ FB)
                else:
                    FB = self.Av * B - self.AvUCv @ (self.VAv @ B)
                    FB /= self.sigma2
                    FQ = self.Av * dPQ - self.AvUCv @ (self.VAv @ dPQ)
                    FQ /= self.sigma2
                    del B, dPQ
                    self.W = FB - FQ @ self.xp.linalg.solve(self.inv_S + self.Q.T @ FQ, self.Q.T @ FB)
            else:  
                # B = self.PX - self.xp.multiply(self.Y.T, self.P1).T #- self.sigma2*self.MY
                # dPQ = self.xp.multiply(self.Q.T, self.P1).T
                # Fv = 1 / self.alpha * self.xp.eye(self.M, dtype=self.floatx, device=self.device)
                # Fvs2v = Fv / self.sigma2
                # FB = Fvs2v @ B
                # FQ = Fvs2v @ dPQ
                # del B, Fvs2v, dPQ
                # self.W = FB - FQ @ self.xp.linalg.solve(self.inv_S + self.Q.T @ FQ, self.Q.T @ FB)

                Fvs2v = self.Fv / self.sigma2
                FB = Fvs2v * (self.PX - self.xp.multiply(self.Y.T, self.P1).T)
                FQ = Fvs2v * (self.xp.multiply(self.Q.T, self.P1).T)
                self.W = FB - FQ @ self.xp.linalg.solve(self.inv_S + self.Q.T @ FQ, self.Q.T @ FB)
    
    def update_transformer(self):
        if self.low_rank is False:
            self.tmat = self.G @ self.W

        elif self.low_rank is True:
            self.tmat = self.xp.matmul(self.Q, self.xp.matmul(self.S, self.xp.matmul(self.Q.T, self.W)))

    def transform_point(self, Y=None ): #TODO
        if Y is None:
            self.TY = self.Y + self.tmat
        else:
            return self.ccf_deformable_transform_point(
                        Y, Y=self.Y, Ym=self.Ym, Ys=self.Ys, 
                        Xm=self.Xm, Xs=self.Xs, beta=self.beta, 
                        G=self.G, W=self.W, Q=self.Q, S=self.S)
            # Y_t = (Y -self.Ym)/self.Ys
            # if reset_G or (not np.array_equal(self.Y, Y.astype(self.Y.dtype))):
            #     G = self.kernal_gmm(Y_t, self.Y, sigma2=self.beta)[0]
            #     tmat = np.dot(G, self.W)
            # else:
            #     tmat = self.tmat
            # Y_n = (Y_t + tmat)* self.Xs + self.Xm
            # return Y_n

    def update_variance(self):
        qprev = self.sigma2        
        trxPx = self.xp.sum( self.Pt1 * self.xp.sum(self.X  * self.X, axis=1) )
        tryPy = self.xp.sum( self.P1 * self.xp.sum( self.TY * self.TY, axis=1))
        trPXY = self.xp.sum(self.TY * self.PX)

        self.sigma2 = (trxPx - 2 * trPXY + tryPy) / (self.Np * self.D)
        if self.sigma2 < 0:
            # self.sigma2 = self.xp.asarray(1/self.iteration)
            self.sigma2 = self.xp.abs(self.sigma2) * 10 #TODO

        self.diff = self.xp.abs(self.sigma2 - qprev)

    def update_normalize(self):
        self.s = self.Xs/self.Ys
        self.TY = self.TY * self.Xs + self.Xm
        self.tform = self.TY - (self.Y * self.Ys + self.Ym)

    def get_transformer(self, attributes=None): # TODO
        if attributes is None:
            attributes = ['W', 'Xm', 'Xs', 'Xf', 'Ym', 'Ys', 'Yf', 'Y',
                           'beta', 'Q', 'S', 'G', 'P', 'C']    
        paras = {}            
        for attr in attributes:
            if hasattr(self, attr):
                paras[attr] = getattr(self, attr)
        return paras
