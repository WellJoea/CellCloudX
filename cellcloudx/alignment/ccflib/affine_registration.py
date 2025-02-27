from builtins import super
import numpy as np
from .emregistration import EMRegistration

from .xp_utility import lle_w, gl_w
from ...io._logger import logger

class AffineRegistration(EMRegistration):
    def __init__(self, *args, gamma1=None, theta=0.1, kw=15, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'affine'
        self.B = self.xp.eye(self.D)
        self.t = self.xp.atleast_2d(self.xp.zeros((1, self.D)))

        self.tol =  1e-9 if self.tol is None else self.tol
        self.gamma1 = 0 if gamma1 is None else gamma1
        self.kw = kw
        self.normal_ = 'each' if self.normal_ is None else self.normal_ #or global

        self.normalXY()
        self.normalXYfeatures()
        self.update_transformer()
        self.init_L( kw=self.kw, method=self.kd_method)

    def init_L(self, kw=15, use_unique=False, method='sknn'):
        if self.gamma1 > 0:
            logger.info(f'compute Y lle...')
            W = lle_w(self.Y, use_unique = use_unique, kw=kw, method=method)
            # W = gl_w(self.Y, kw=kw, method=method)


            self.WY = (W @ self.Y).to(self.device)

    def optimization(self):
        self.expectation()
        self.maximization()
        self.iteration += 1

    def maximization(self):
        muX = self.xp.divide(self.xp.sum(self.PX, axis=0), self.Np)
        muY = self.xp.divide(self.Y.transpose(1,0) @ self.P1, self.Np)

        X_hat = self.X - muX
        Y_hat = self.Y - muY

        A = self.PX.transpose(1,0) @ Y_hat - \
                 self.xp.outer(muX,  self.P1 @ Y_hat)
        YPY = (Y_hat.transpose(1,0) * self.P1) @ Y_hat

        if self.gamma1 > 0: # TODO (P1WY)

            YtZY = (self.WY.transpose(1,0) * self.P1) @ self.WY
            # YtZY = (self.WY.transpose(1,0)  @ self.WY)
            YPY.add_( (2 * self.gamma1 * self.sigma2) * YtZY )
    
        self.B = A @ self.xp.linalg.inv(YPY)
        self.t = muX - self.B @ muY
        self.TY = self.Y @ self.B.transpose(1,0) +  self.t

        trAB = self.xp.trace(A @ self.B.transpose(1,0))
        # trXPX = np.sum( self.Pt1.T * np.sum(np.multiply(X_hat, X_hat), axis=1))
        trXPX = self.xp.sum(self.Pt1 * self.xp.sum(X_hat * X_hat, 1))

        self.sigma2 = (trXPX - trAB) / (self.Np * self.D)
        if self.sigma2 < 0:
            # self.sigma2 = self.xp.asarray(1/self.iteration)
            self.sigma2 = self.xp.abs(self.sigma2) * 10 #TODO

        qprev = self.q
        # self.q = (trXPX -  trAB ) / (2 * self.sigma2) + \
        #             self.D * self.Np/2 * self.xp.log(self.sigma2)
        self.q = self.D * self.Np/2 * (1+self.xp.log(self.sigma2))
        self.diff = self.xp.abs(self.q - qprev)

    def update_transformer(self, tmat=None, tmatinv=None):
        if not tmatinv is None:
            tmat = self.xp.linalg.inv( self.xp.asarray(tmatinv, dtype=self.floatx) )
        if not tmat is None:
            self.tmat = self.xp.asarray(tmat, dtype=self.floatx)
            self.tmatinv = self.xp.linalg.inv(self.tmat)
    
            #TODO
            self.B = self.tmat[:-1, :-1]
            self.t = self.tmat[:-1, [-1]].T
        else:
            self.tmat = self.xp.eye(self.D+1, dtype=self.floatx, device=self.device)
            self.tmat[:self.D, :self.D] = self.B
            self.tmat[:self.D, self.D] = self.t
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
        self.transform_point()

        self.B *= (self.Xs/self.Ys)
        self.t = (self.t * self.Xs + self.Xm) - (self.Ym @ self.B.transpose(1,0)) #*(self.Xs/self.Ys) scaled in B
        self.s = self.Xs/self.Ys
    
        self.tform = self.Xf @ self.tmat @ self.xp.linalg.inv(self.Yf)
        self.tforminv = self.xp.linalg.inv(self.tform)
        self.TY = self.TY * self.Xs + self.Xm
        # TY = self.transform_point(self.Yr)

    def get_transformer(self):
        return { 'tform': self.tform, 'B': self.B, 't': self.t, 's': self.s }