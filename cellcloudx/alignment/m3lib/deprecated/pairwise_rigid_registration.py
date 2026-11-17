from builtins import super
from .pairwise_emregistration import pwEMRegistration

class pwRigidRegistration(pwEMRegistration):
    def __init__(self, *args, 
                   fix_s=True, s =1.0, s_clip=None,
                   **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'rigid'
        self.R = self.xp.eye(self.D) 
        self.t = self.xp.zeros(self.D)
        self.s = self.xp.tensor(s, dtype=self.floatx, device=self.device)
        self.s_clip = s_clip
        self.fix_s = fix_s
        self.normal_ = 'isoscale' if self.normal_ is None else self.normal_
        self.tol =  1e-9 if self.tol is None else self.tol

        self.maxiter = self.maxiter or 150
        self.normal_XY()
        self.normal_features()

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
        S = self.xp.diag(S)
        C = self.xp.eye(self.D, dtype=self.floatx, device=self.device)
        C[-1, -1] = self.xp.linalg.det(U @ V)
        self.R = U @ C @ V
        # self.R = U @ V

        if self.xp.linalg.det(self.R) < 0:
            U[:, -1] = -U[:, -1]
            self.R = U @ C @ V

        trAR = self.xp.trace(A.transpose(1,0) @ self.R)
        trXPX = self.xp.sum(self.Pt1 * self.xp.sum(X_hat * X_hat, 1))
        trYPY = self.xp.sum(self.P1 * self.xp.sum(Y_hat * Y_hat, 1))

        if self.fix_s is False:
            self.s = trAR/trYPY
        
        if self.s_clip is not None:
            self.s = self.xp.clip(self.s, *self.s_clip)

        self.t = muX - self.s * (self.R @ muY)

        self.TY = self.s * (self.Y @ self.R.transpose(1,0)) +  self.t #.T

        if self.fix_s is False:
            self.sigma2 = (trXPX  - self.s * trAR) / (self.Np * self.D)
            # self.sigma21 = (trXPX + self.s* self.s* trYPY - 2* self.s* trAR) / (self.Np * self.D)
        else:
            self.sigma2 = (trXPX + self.s* self.s* trYPY - 2* self.s* trAR) / (self.Np * self.D)

        if self.sigma2 < self.eps: 
            self.sigma2 = self.xp.clip(self.sigma_square(self.X, self.TY), min=self.sigma2_min)
    
        qprev = self.Q
        # self.Q = (trXPX - 2 * self.s * trAR + self.s * self.s * trYPY) / (2 * self.sigma2) \
        #           + self.D * self.Np/2 * self.xp.log(self.sigma2) 
        self.Q = (self.Np * self.D)/2*(1+self.xp.log(self.sigma2))
        self.diff = self.xp.abs(self.Q - qprev)

        # self.wn = int((1-self.w) * self.M)
        # self.inlier = np.argpartition(self.P1, -self.wn)[-self.wn:]

    def update_transformer(self, tmat=None, tmatinv=None):
        if not tmatinv is None:
            tmat = self.xp.linalg.inv( self.xp.asarray(tmatinv, dtype=self.floatxx) )
        if not tmat is None:
            self.tmat = self.xp.asarray(tmat, dtype=self.floatxx)
            self.tmatinv = self.xp.linalg.inv(self.tmat)
    
            #TODO
            B = self.tmat[:-1, :-1]
            self.s = self.xp.linalg.det(B)**(1/(B.shape[0])) 
            self.R = B/self.s
            self.t = self.tmat[:-1, [-1]].T

        else:
            tmat = self.xp.eye(self.D+1, dtype=self.floatxx, device=self.device)
            tmat[:self.D, :self.D] = self.R * self.s
            tmat[:self.D,  self.D] = self.t
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
        self.R = self.R.to(self.floatxx)
        self.t = self.t.to(self.floatxx)
        self.s = float(self.s)

        self.tmat = self.update_transformer()

        self.s *= self.Xs/self.Ys 
        self.t = (self.t * self.Xs + self.Xm) - self.s * self.R @ self.Ym
        
        self.tform =self.update_transformer()
        # self.tform = self.Xf @ self.tmat @ self.xp.linalg.inv(self.Yf)
        # TY = self.TY * self.Xs + self.Xm
        # self.TY  = self.s * (self.Yr.to(self.device) @ self.R.T) +  self.t
        self.TY = self.transform_point(self.Yr.to(self.device))
    
    def get_transformer(self):
        return {'tform': self.tform, 's': self.s, 'R': self.R, 't':self.t }
