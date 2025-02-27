from builtins import super
from .emregistration import EMRegistration

class SimilarityRegistration(EMRegistration):
    def __init__(self, *args, isoscale=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.isoscale = isoscale
        self.reg_core = 'isosimilarity' if isoscale else 'similarity'
        self.R = self.xp.eye(self.D) 
        self.t = self.xp.zeros(self.D)
        self.s = self.xp.ones(self.D)

        self.normal_ = 'each' if self.normal_ is None else self.normal_
        self.tol =  1e-9 if self.tol is None else self.tol

        self.normalXY()
        self.normalXYfeatures()
        self.update_transformer()

    def optimization(self):
        self.expectation()
        self.maximization()
        self.iteration += 1
    
    def maximization(self):
        muX = self.xp.divide(self.xp.sum(self.PX, 0), self.Np)
        muY = self.xp.divide(self.Y.transpose(1,0) @ self.P1, self.Np)

        X_hat = self.X - muX
        Y_hat = self.Y - muY

        A = self.PX.transpose(1,0) @ Y_hat - self.xp.outer(muX, self.P1 @ Y_hat)
        U, S, V = self.xp.linalg.svd(A, full_matrices=True)
        C = self.xp.eye(self.D, dtype=self.floatx, device=self.device)
        C[-1, -1] = self.xp.linalg.det(U @ V)
        self.R = U @ C @ V

        if self.xp.linalg.det(self.R) < 0:
            U[:, -1] = -U[:, -1]
            self.R = U @ C @ V
        trXPX = self.xp.sum(self.Pt1 * self.xp.sum(X_hat * X_hat, 1))

        if self.isoscale:
            trAR = self.xp.trace(A.transpose(1,0) @ self.R )
            trYPY = self.xp.sum(self.P1 * self.xp.sum(Y_hat * Y_hat, 1))
            self.s = self.xp.ones(self.D)*(trAR/trYPY)
            self.t = muX - self.s * (self.R @ muY)
            self.TY = self.s * (self.Y @ self.R.transpose(1,0)) +  self.t #.T
            self.sigma2 = (trXPX + self.s* self.s* trYPY - 2* self.s* trAR) / (self.Np * self.D)
        else:
            trYPY = self.xp.sum(self.P1 * self.xp.sum(Y_hat * Y_hat, 1))
            YPY = self.xp.sum( Y_hat * Y_hat *self.P1.unsqueeze(1), 0)
            YPY.masked_fill_(YPY == 0, self.eps)

            #TODO
            self.s0 = self.xp.diagonal(C)*S/YPY
            self.s = self.xp.diagonal(A.T @ self.R)/YPY
            ############

            self.t = muX -  (self.R * self.s) @ muY
            self.TY = self.Y @ (self.s.unsqueeze(1) * self.R.transpose(1,0) ) +  self.t #.T
    
            trARS = self.xp.trace(A.transpose(1,0) @ (self.R * self.s))
            trSSYPY = self.xp.sum((self.s **2) * YPY)
            self.sigma2 = (trXPX  - 2*trARS + trSSYPY) / (self.Np * self.D)

        if self.sigma2 < 0:
            # self.sigma2 = self.xp.asarray(1/self.iteration)
            self.sigma2 = self.xp.abs(self.sigma2) * 10 #TODO

        qprev = self.q
        self.q = self.D * self.Np/2 * (1+self.xp.log(self.sigma2))
        self.diff = self.xp.abs(self.q - qprev)

        # self.wn = int((1-self.w) * self.M)
        # self.inlier = np.argpartition(self.P1, -self.wn)[-self.wn:]

    def update_transformer(self, tmat=None, tmatinv=None):
        if not tmatinv is None:
            tmat = self.xp.linalg.inv( self.xp.asarray(tmatinv, dtype=self.floatx) )
        if not tmat is None:
            self.tmat = self.xp.asarray(tmat, dtype=self.floatx)
            self.tmatinv = self.xp.linalg.inv(self.tmat)
    
            # #TODO
            # B = self.tmat[:-1, :-1]
            # self.s = self.xp.linalg.det(B)**(1/(B.shape[0])) 
            # self.R = B/self.s
            # self.t = self.tmat[:-1, [-1]].T

        else:
            self.tmat = self.xp.eye(self.D+1, dtype=self.floatx, device=self.device)
            self.tmat[:self.D, :self.D] = self.R * self.s.unsqueeze(0) #R @ S
            self.tmat[:self.D,  self.D] = self.t
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

        self.s *= self.Xs/self.Ys 
        self.t = (self.t * self.Xs + self.Xm) - (self.R*self.s) @ self.Ym
        self.tform = self.Xf @ self.tmat @ self.xp.linalg.inv(self.Yf)
        self.tforminv = self.xp.linalg.inv(self.tform)
        self.TY = self.TY * self.Xs + self.Xm
        # TY = self.transform_point(self.Yr)
    
    def get_transformer(self):
        return {'tform': self.tform, 's': self.s, 'R': self.R, 't':self.t }
