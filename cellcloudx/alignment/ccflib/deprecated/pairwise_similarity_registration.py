from builtins import super
from .pairwise_emregistration import pwEMRegistration

class pwSimilarityRegistration(pwEMRegistration):
    def __init__(self, *args, isoscale=False,
                 fix_R=False, fix_t=False, fix_s=False, s_clip=None,
                 #R=None, t=None, s=None,  TODO
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.isoscale = isoscale
        self.fix_R = fix_R
        self.fix_t = fix_t
        self.fix_s = fix_s
        self.s_clip = s_clip
        self.R = None
        self.t = None
        self.s = None
        self.init_paras()

        self.normal_ =  'isoscale' if self.normal_ is None else self.normal_
        self.tol =  1e-9 if self.tol is None else self.tol
        self.maxiter = self.maxiter or 150
        self.normal_XY()
        self.normal_features()

    def init_paras(self): #TODO
        if self.s is not None:
            s = self.xp.tensor(self.s)
            if s.ndim==0:
                s = (self.xp.eye(self.D) * s).to(self.device)
            elif (s.ndim==1) and (s.shape[0]==self.D):
                s = (self.xp.diag(s)).to(self.device)
            # elif (s.ndim==2) and (s.shape==(self.D, self.D)):
            #     s = s.to(self.device)
            else:
                raise ValueError('s should be a scalar, 1-D or 2-D array')
            self.s = s
        else:
            self.s = self.xp.eye(self.D).to(self.device)

        if self.s_clip is not None:
            self.s = self.xp.clip( self.s.diagonal(), *self.s_clip).diag()

        if self.R is not None:
            R = self.xp.tensor(self.R)
            if R.ndim==0:
                R = (self.xp.eye(self.D) * R).to(self.device)
            elif (R.ndim==2) and (R.shape[0]==self.D):
                R = R.to(self.device)
            else:
                raise ValueError('R should be a  2-D array')
            self.R = R
        else:
            self.R = self.xp.eye(self.D).to(self.device)

        if self.t is not None:
            t = self.xp.tensor(self.t)
            if t.ndim==0:
                t = (self.xp.zeros(self.D) + t).to(self.device)
            elif (t.ndim==1) and (t.shape[0]==self.D):
                t = t.to(self.device)
            else:
                raise ValueError('t should be a scalar or 1-D array')
            self.t = t
        else:
            self.t = self.xp.zeros(self.D).to(self.device)

        reg_core= []
        if not self.fix_R:
            reg_core.append('R')
        if not self.fix_t:
            reg_core.append('T')
        if not self.fix_s:
            if self.isoscale:
                reg_core.append('Iso-S')
            else:
                reg_core.append('S')
        self.reg_core = '+'.join(reg_core)

    def optimization(self):
        self.expectation()
        self.maximization()
        self.iteration += 1
    
    def maximization(self):
        if not self.fix_t:
            muX = self.xp.divide(self.xp.sum(self.PX, 0), self.Np)
            muY = self.xp.divide(self.Y.transpose(1,0) @ self.P1, self.Np)

            X_hat = self.X - muX
            Y_hat = self.Y - muY
            A = self.PX.transpose(1,0) @ Y_hat - self.xp.outer(muX, self.P1 @ Y_hat)
        else:
            X_hat = self.X
            Y_hat = self.Y
            A = self.PX.transpose(1,0) @ Y_hat

        trXPX = self.xp.sum(self.Pt1 * self.xp.sum(X_hat * X_hat, 1))

        if self.isoscale:
            if not self.fix_R:
                U, S, V = self.xp.linalg.svd(A, full_matrices=True)
                C = self.xp.eye(self.D, dtype=self.floatx, device=self.device)
                C[-1, -1] = self.xp.linalg.det(U @ V)
                self.R = U @ C @ V
                if self.xp.linalg.det(self.R) < 0:
                    U[:, -1] = -U[:, -1]
                    self.R = U @ C @ V

            trAR = self.xp.trace(A.transpose(1,0) @ self.R )
            trYPY = self.xp.sum(self.P1 * self.xp.sum(Y_hat * Y_hat, 1))

            if not self.fix_s:
                s = trAR/trYPY
                self.s = self.xp.eye(self.D).to(self.device) * s
            
            if self.s_clip is not None:
                self.s = self.xp.clip( self.s.diagonal(), *self.s_clip).diag()
                
            if not self.fix_t:
                self.t = muX - self.R @ self.s @ muY

            self.TY = self.Y @ (self.R @ self.s).T +  self.t #.T
            self.sigma2 = (trXPX + self.s[0,0]* self.s[0,0]* trYPY - 2* self.s[0,0] * trAR) / (self.Np * self.D)
        else:
            YPY = self.xp.sum( Y_hat * Y_hat *self.P1.unsqueeze(1), 0)
            YPY.masked_fill_(YPY == 0, self.eps)
            if (not self.fix_s) and (not self.fix_R):
                max_iter = 70
                error = True
                iiter = 0
                C = self.xp.eye(self.D, dtype=self.floatx, device=self.device)
                self.R = C.clone()
                while (error):
                    U, S, V = self.xp.linalg.svd(A @ self.s, full_matrices=True) #A @ s.T
                    C[-1, -1] = self.xp.linalg.det(U @ V)
                    R_pre = self.R.clone()
                    self.R = U @ C @ V
                    if self.xp.linalg.det(self.R) < 0:
                        U[:, -1] = -U[:, -1]
                        self.R = U @ C @ V

                    self.s = self.xp.diagonal(A.T @ self.R)/YPY
                    if self.s_clip is not None:
                        self.s = self.xp.clip(self.s, *self.s_clip)
                    self.s = self.xp.diag(self.s) 
                    iiter += 1
                    error = (self.xp.dist(self.R, R_pre) > 1e-8) and (iiter < max_iter)

            elif (not self.fix_R) and self.fix_s:
                U, S, V = self.xp.linalg.svd(A @ self.s, full_matrices=True)
                C = self.xp.eye(self.D, dtype=self.floatx, device=self.device)
                C[-1, -1] = self.xp.linalg.det(U @ V)
                self.R = U @ C @ V
                if self.xp.linalg.det(self.R) < 0:
                    U[:, -1] = -U[:, -1]
                    self.R = U @ C @ V
    
            elif (not self.fix_s) and self.fix_R:
                self.s = self.xp.diagonal(A.T @ self.R)/YPY
                if self.s_clip is not None:
                    self.s = self.xp.clip(self.s, *self.s_clip)
                self.s = self.xp.diag(self.s) 

            if not self.fix_t:
                self.t = muX -  (self.R @ self.s) @ muY

            self.TY = self.Y @ (self.R @ self.s).transpose(1,0) +  self.t
            trARS = self.xp.trace(self.R @ self.s @ A.transpose(1,0))
            trSSYPY = self.xp.sum((self.s **2) * YPY)
            self.sigma2 = (trXPX  - 2*trARS + trSSYPY) / (self.Np * self.D)

        if self.sigma2 < self.eps: 
            self.sigma2 = self.xp.clip(self.sigma_square(self.X, self.TY), min=self.sigma2_min)
        qprev = self.Q
        self.Q = self.D * self.Np/2 * (1+self.xp.log(self.sigma2))
        self.diff = self.xp.abs(self.Q - qprev)

        # self.wn = int((1-self.w) * self.M)
        # self.inlier = np.argpartition(self.P1, -self.wn)[-self.wn:]

    def update_transformer(self):
        tmat = self.xp.eye(self.D+1, dtype=self.floatxx, device=self.device)
        tmat[:self.D, :self.D] = self.R @ self.s
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
        self.s = self.s.to(self.floatxx)

        self.tmat = self.update_transformer()

        self.s *= self.Xs/self.Ys

        if not self.fix_t:
            self.t = (self.t * self.Xs + self.Xm) - (self.R @ self.s) @ self.Ym

        # self.tform = self.Xf @ self.tmat @ self.xp.linalg.inv(self.Yf)
        # self.tforminv = self.xp.linalg.inv(self.tform)
    
        self.tform =self.update_transformer()

        # TY1 = self.TY * self.Xs + self.Xm
        # self.TY = self.Yr.to(self.device) @ (self.R @ self.s).T +  self.t #.T
        self.TY = self.transform_point(self.Yr.to(self.device))

    def get_transformer(self):
        return {'tform': self.tform, 's': self.s, 'R': self.R, 't':self.t }
        