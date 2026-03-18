from builtins import super

from .pairwise_emregistration import pwEMRegistration
from .manifold_regularizers import lle_w, gl_w
from ...io._logger import logger

class pwAffineRegistration(pwEMRegistration):
    def __init__(self, *args, gamma1=None, delta=0.1, 
                 theta=0.1, kw=15, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'affine'
        self.B = self.xp.eye(self.D).to(self.device)
        self.t = self.xp.zeros(self.D).to(self.device)

        self.tol =  1e-9 if self.tol is None else self.tol
        self.delta = 0.1 if delta is None else delta
        self.gamma1 = 0 if gamma1 is None else gamma1
        self.kw = kw
        self.normal_ = 'each' if self.normal_ is None else self.normal_ #or global
        self.maxiter = self.maxiter or 150

        self.normal_XY()
        self.normal_features()
        self.constrained_pairs()
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

        YPYh = (Y_hat.transpose(1,0) * self.P1) @ Y_hat
        # if self.xp.det(YPYh) == 0:
        #     YPY = (self.Y.T * self.P1) @ self.Y
        #     XPY = self.PX.T @ self.Y
        #     error = True
        #     max_iter = 100
        #     iiter = 0
        #     while (error):
        #         YPT = self.xp.outer(self.t, self.xp.sum(self.Y.T * self.P1, 1))
        #         B_pre = self.B
        #         self.B = (XPY - YPT) @ self.xp.linalg.inv(YPY)
        #         self.t = muX - self.B @ muY
        #         iiter += 1
        #         error = (self.xp.linalg.norm(self.B - B_pre) > 1e-9) and (iiter < max_iter)
        # else:
        if self.gamma1 > 0: # TODO (P1WY)
            YtZY = (self.WY.transpose(1,0) * self.P1) @ self.WY
            YPYh.add_( (2 * self.gamma1 * self.sigma2) * YtZY )

        try:
            YPYhv = self.xp.linalg.inv(YPYh)
            self.B = A @ YPYhv
        except:
            B_pre = self.B
            YPYh.diagonal().add_(self.delta*self.sigma2)
            self.B = (A+self.delta*self.sigma2*B_pre) @ self.xp.linalg.inv(YPYh)

        self.t = muX - self.B @ muY
        self.TY = self.Y @ self.B.transpose(1,0) +  self.t
        trAB = self.xp.trace(A @ self.B.transpose(1,0))
        trXPX = self.xp.sum(self.Pt1 * self.xp.sum(X_hat * X_hat, 1))

        self.sigma2 = (trXPX - trAB) / (self.Np * self.D)
        if self.sigma2 < self.eps: 
            self.sigma2 = self.xp.clip(self.sigma_square(self.X, self.TY), min=self.sigma2_min)

        qprev = self.Q
        self.Q = self.D * self.Np/2 * (1+self.xp.log(self.sigma2))
        # q = (trXPX -  trAB ) / (2 * self.sigma2) + \
        #         self.D * self.Np/2 * self.xp.log(self.sigma2)
        self.diff = self.xp.abs(self.Q - qprev)

    def update_transformer(self, tmat=None, tmatinv=None):
        if not tmatinv is None:
            tmat = self.xp.linalg.inv( self.xp.asarray(tmatinv, dtype=self.floatx) )
        if not tmat is None:
            self.tmat = self.xp.asarray(tmat, dtype=self.floatx)
            self.tmatinv = self.xp.linalg.inv(self.tmat)
    
            #TODO
            self.B = self.tmat[:-1, :-1]
            self.t = self.tmat[:-1, -1]
        else:
            tmat = self.xp.eye(self.D+1, dtype=self.floatxx, device=self.device)
            tmat[:self.D, :self.D] = self.B
            tmat[:self.D, self.D] = self.t
            return tmat

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
        self.B = self.B.to(self.floatxx)
        self.t = self.t.to(self.floatxx)

        self.tmat = self.update_transformer()

        self.B *= (self.Xs/self.Ys)
        self.t = (self.t * self.Xs + self.Xm) - (self.Ym @ self.B.transpose(1,0)) #*(self.Xs/self.Ys) scaled in B
        self.s = self.Xs/self.Ys
        
        self.tform =self.update_transformer()
        # self.tform = self.Xf @ self.tmat @ self.xp.linalg.inv(self.Yf)
        # self.tforminv = self.xp.linalg.inv(self.tform)
        # TY1 = self.TY * self.Xs + self.Xm
        # self.TY  = self.Yr.to(self.device) @ self.B.T +  self.t
        self.TY = self.transform_point(self.Yr.to(self.device))

    def get_transformer(self):
        return { 'tform': self.tform, 'B': self.B, 't': self.t,}