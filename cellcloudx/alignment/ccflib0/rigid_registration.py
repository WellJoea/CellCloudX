from builtins import super
import numpy as np
import numbers
from scipy.sparse import issparse, csr_array, csc_array, diags
from .emregistration import EMRegistration

class RigidRegistration(EMRegistration):
    def __init__(self, *args, scale=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'rigid'
        self.R = np.eye(self.D) 
        self.t = np.atleast_2d(np.zeros((1, self.D)))
        self.s = 1 
        self.scale = scale
        self.normal_ = False if self.normal_ is None else self.normal_
        self.tol =  1e-9 if self.tol is None else self.tol

        self.init_normalize()
        self.update_transformer()

    def optimization(self):
        self.expectation()
        self.update_transform()
        self.update_transformer()
        self.transform_point()
        self.update_variance()
        self.iteration += 1
    
    def update_transform(self):
        muX = self.xp.divide(self.xp.sum(self.PX, axis=0), self.Np)
        muY = self.xp.divide(self.xp.dot(self.xp.transpose(self.Y), self.P1), self.Np)

        X_hat = self.tX - muX # TODO
        Y_hat = self.Y - muY

        self.A = self.xp.dot(self.PX.T, Y_hat) - self.xp.outer(muX, self.xp.dot(self.P1.T, Y_hat))
        U, S, V = self.xp.linalg.svd(self.A, full_matrices=True)
        S = self.xp.diag(S)
        C = self.xp.eye(self.D)
        C[-1, -1] = self.xp.linalg.det(self.xp.dot(U, V))
        self.R = self.xp.dot(self.xp.dot(U, C), V)
        
        if np.linalg.det(self.R) < 0: #TODO
            U[:, -1] = -U[:, -1]
            self.R = self.xp.dot(self.xp.dot(U, C), V)

        # self.trAR = np.trace(S @ C)
        # self.trYPY = self.xp.dot(self.P1.T, 
        #                     self.xp.sum(self.xp.multiply(Y_hat, Y_hat), axis=1))
        # self.trXPX = self.xp.dot(self.Pt1.T, 
        #                     self.xp.sum(self.xp.multiply(X_hat, X_hat), axis=1))
        self.trAR = self.xp.trace(self.xp.dot(self.A.T, self.R))
        self.trXPX = np.sum( self.Pt1.T * np.sum(np.multiply(X_hat, X_hat), axis=1))
        self.trYPY = np.sum( self.P1.T * np.sum(np.multiply(Y_hat, Y_hat), axis=1))

        if self.scale is True:
            self.s = self.trAR/self.trYPY
        self.t = muX - self.s * self.xp.dot(self.R, muY)

    def update_transformer(self, tmat=None, tmatinv=None):
        if not tmatinv is None:
            tmat = np.linalg.inv(self.float(tmatinv))
        if not tmat is None:
            self.tmat = self.float(tmat)
            self.tmatinv = np.linalg.inv(self.tmat)
    
            #TODO
            B = self.tmat[:-1, :-1]
            self.s = np.linalg.det(B)**(1/(B.shape[0])) 
            self.R = B/self.s
            self.t = self.tmat[:-1, [-1]].T

        else:
            self.tmat = np.eye(self.D+1, dtype=self.float)
            self.tmat[:self.D, :self.D] = self.R * self.s
            self.tmat[:self.D,  self.D] = self.t
            self.tmatinv = np.linalg.inv(self.tmat)
            self.tform = self.tmat

    def transform_point(self, Y=None):
        if Y is None:
            self.TY = self.homotransform_point(self.Y, self.tmat, inverse=False)
            return
        else:
            return self.homotransform_point(Y, self.tmat, inverse=False)

    def update_variance(self):
        if self.scale is True:
            self.sigma2 = (self.trXPX - self.s * self.trAR) / (self.Np * self.D)
        else:
            self.sigma2 = (self.trXPX + self.s* self.s* self.trYPY - 2* self.s* self.trAR) / (self.Np * self.D)
            #scale * self.trAR
            #self.sigma2 = self.sigma_square(self.X, self.TY)

        if self.sigma2 <= 0:
            self.sigma2 = 1/self.iteration
    
        qprev = self.q
        self.q = (self.trXPX - 2 * self.s * self.trAR + self.s * self.s * self.trYPY) / (2 * self.sigma2) 
        self.q += self.D * self.Np/2 * np.log(self.sigma2)
        self.diff = np.abs(self.q - qprev)

        # self.wn = int((1-self.w) * self.M)
        # self.inlier = np.argpartition(self.P1, -self.wn)[-self.wn:]

    def update_normalize(self):
        self.s *= self.Xs/self.Ys 
        self.t = (self.t * self.Xs + self.Xm) - self.s * self.R @ self.Ym.T
        self.tform = self.Xf @ self.tmat @ np.linalg.inv(self.Yf)
        self.tforminv = np.linalg.inv(self.tform)
        self.update_transformer()
        self.TY = self.TY * self.Xs + self.Xm

    def get_transformer(self):
        return {'tform': self.tform, 's': self.s, 'R': self.R, 't':self.t }
