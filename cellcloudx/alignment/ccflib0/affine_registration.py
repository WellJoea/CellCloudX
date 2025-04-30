from builtins import super
import numpy as np
from .emregistration import EMRegistration
from .utility import is_positive_semi_definite
from .xmm import kernel_xmm, kernel_xmm_k, kernel_xmm_p, low_rank_eigen, lle_W

class AffineRegistration(EMRegistration):
    """
    Affine registration.

    Attributes
    ----------
    B: numpy array (semi-positive definite)
        DxD affine transformation matrix.

    t: numpy array
        1xD initial translation vector.
    """
    def __init__(self, *args, gamma=None, kw=25, use_lle=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'affine'
        self.B = np.eye(self.D)
        self.t = np.atleast_2d(np.zeros((1, self.D))) 
        self.normal_ = False if self.normal_ is None else self.normal_
        self.tol =  1e-9 if self.tol is None else self.tol

        self.gamma = 1 if gamma is None else gamma
        self.use_lle = False
        self.kw = kw

        self.init_normalize()
        self.init_L( kw=self.kw, rl_w=None, method='sknn')
        self.update_transformer()

    def init_L(self, kw=15, rl_w=None, method='sknn'):
        if self.use_lle:
            print(f'compute Y lle...')
            W = self.lle_M(self.Y, kw=kw, rl_w=rl_w, method=method, eps=self.eps)
            M = W.transpose().dot(W)
            self.GY = 2* self.gamma * M.dot(self.Y)

    def optimization(self):
        self.expectation()
        self.update_transform()
        self.update_transformer()
        self.transform_point()
        self.update_variance()
        self.iteration += 1

    def update_transform(self):
        muX = self.xp.divide(self.xp.sum(self.PX, axis=0), self.Np)
        muY = self.xp.divide(self.xp.dot(self.Y.T, self.P1), self.Np)

        X_hat = self.tX - muX
        Y_hat = self.Y - muY

        self.A = self.xp.dot(self.PX.T, Y_hat) - self.xp.outer(muX, self.xp.dot(self.P1.T, Y_hat))
        self.YPY  = self.xp.dot(self.xp.multiply(Y_hat.T, self.P1), Y_hat)

        if self.use_lle: # TODO
            self.YPY += self.sigma2* self.xp.dot(self.xp.multiply(self.Y.T, self.P1), self.GY)
        self.B = self.xp.dot(self.A, self.xp.linalg.inv(self.YPY))
        # self.B = np.linalg.solve(np.transpose(self.YPY), np.transpose(self.A))
        self.t = muX - self.xp.dot(self.B, muY)

        self.trAB = self.xp.trace(self.xp.dot(self.A, self.B.T))
        self.trXPX = np.sum( self.Pt1.T * np.sum(np.multiply(X_hat, X_hat), axis=1))

    def update_transformer(self, tmat=None, tmatinv=None):
        if not tmatinv is None:
            tmat = self.xp.linalg.inv(self.float(tmatinv))
        if not tmat is None:
            self.tmat = self.float(tmat)
            self.tmatinv = self.xp.linalg.inv(self.tmat)
    
            #TODO
            self.B = self.tmat[:-1, :-1]
            self.t = self.tmat[:-1, [-1]].T
        else:
            self.tmat = self.xp.eye(self.D+1, dtype=self.float)
            self.tmat[:self.D, :self.D] = self.B
            self.tmat[:self.D, self.D] = self.t
            self.tmatinv = self.xp.linalg.inv(self.tmat)
            self.tform = self.tmat

    def transform_point(self, Y=None):
        if Y is None:
            self.TY = self.homotransform_point(self.Y, self.tmat, inverse=False)
        else:
            return self.homotransform_point(Y, self.tmat, inverse=False)

    def update_variance(self):
        self.sigma2 = (self.trXPX - self.trAB) / (self.Np * self.D)
        if self.sigma2 <= 0:
            self.sigma2 = 1/self.iteration
        qprev = self.q
        self.q = (self.trXPX -  self.trAB ) / (2 * self.sigma2) + \
                    self.D * self.Np/2 * self.xp.log(self.sigma2)
        self.diff = self.xp.abs(self.q - qprev)

    def update_normalize(self):
        self.B *= (self.Xs/self.Ys)
        self.t = (self.t * self.Xs + self.Xm) - self.B @ self.Ym.T
        self.tform = self.Xf @ self.tmat @ np.linalg.inv(self.Yf)
        self.tforminv = np.linalg.inv(self.tform)
        self.update_transformer()
        self.TY = self.TY * self.Xs + self.Xm

    def get_transformer(self):
        return { 'tform': self.tform, 'B': self.B, 't': self.t }
