from builtins import super
from .emregistration import EMRegistration

class ProjectiveRegistration(EMRegistration):
    def __init__(self, *args, gamma1=None, lr=0.005, lr_stepsize=None,
                 lr_gamma=0.5, opt='LBFGS', d=1.0,
                 opt_iter=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'projective'

        self.A = self.xp.nn.Parameter(
            self.xp.eye(self.D, device=self.device, dtype=self.floatx)
        )
        self.B = self.xp.nn.Parameter(
            self.xp.zeros(self.D, device=self.device, dtype=self.floatx)
        )
        self.t = self.xp.nn.Parameter(
            self.xp.zeros(self.D, device=self.device, dtype=self.floatx)
        )
        self.d = self.xp.tensor(d).to(self.device, dtype=self.floatx)
        self.lr = lr
        self.opt_iter = opt_iter
        self.opt = opt
        self.maxiter = self.maxiter or 150

        self.tol =  1e-9 if self.tol is None else self.tol
        self.gamma1 = 0 if gamma1 is None else gamma1
        self.normal_ = 'each' if self.normal_ is None else self.normal_ #or global

        self.normalXY()
        self.normalXYfeatures()
        self.update_transformer()
        # self.init_L( kw=self.kw, method=self.kd_method)

        self.sigma2 =  self.to_tensor(self.sigma2 or self.sigma_square(self.X, self.TY), 
                                      device=self.device)
        self.logsigma2 = self.xp.nn.Parameter(
            self.xp.log(self.to_tensor(self.sigma2 or 1.0, dtype=self.floatx, device=self.device)).clone()
        )
        self.optimizer = self.Optim()
        self.lr_stepsize = lr_stepsize
        if self.lr_stepsize:
            self.scheduler = self.xp.optim.lr_scheduler.StepLR(self.optimizer, 
                                                        step_size=lr_stepsize, 
                                                        gamma=lr_gamma)

    def Optim(self, ):
        params = [self.A, self.B, self.t, self.logsigma2]
        if self.opt == 'LBFGS':
            optimizer = self.xp.optim.LBFGS(params, 
                                             lr=self.lr,
                                             max_iter=self.opt_iter )
        elif self.opt == 'Adam':
            optimizer = self.xp.optim.Adam(params, lr=self.lr,)
        elif self.opt == 'RMSprop':
            optimizer = self.xp.optim.RMSprop(params, lr=self.lr,)
        elif self.opt == 'SGD':
            optimizer = self.xp.optim.SGD(params, momentum=0.3, lr=self.lr,)  
        elif self.opt == 'ASGD':
            optimizer = self.xp.optim.ASGD(params, lr=self.lr,)
        elif self.opt == 'AdamW':
            optimizer = self.xp.optim.AdamW(params, lr=self.lr,)
        elif self.opt == 'Adamax':
            optimizer = self.xp.optim.Adamax(params, lr=self.lr,)
        else:
            raise ValueError('Optimizer not supported')
        return optimizer

    def projective_transform(self):
        y_homo = self.Y @ self.A.T + self.t
        Y_base = self.Y @ self.B.unsqueeze(1) + self.d
        Y_base.masked_fill_(Y_base == 0, self.eps)
        return y_homo / Y_base

    def optimization(self):
        with self.xp.no_grad(): 
            self.expectation()
            trXPX =  self.xp.sum(self.Pt1 * self.xp.sum(self.X * self.X, 1))

        def closure():
            self.optimizer.zero_grad()
            TY = self.projective_transform()
            trXPY = self.xp.sum(self.PX * TY)
            trYPY = self.xp.sum(TY * TY, 1) @ self.P1
            res1 = (self.Np * self.D / 2) * self.logsigma2
            Q = (trXPX - 2*trXPY + trYPY)/(2*self.xp.exp(self.logsigma2)) + res1
            Q.backward(retain_graph=True)
            #TODO:
            #self.xp.exp(self.logsigma2) ==0
            return Q
        loss = self.optimizer.step(closure)
        if self.lr_stepsize:
            self.scheduler.step()

        with self.xp.no_grad():
            self.TY = self.projective_transform()
            self.sigma2 = self.xp.exp(self.logsigma2)
            qprev = self.q
            self.q = self.xp.tensor(loss.item())
            self.diff = self.xp.abs(qprev - self.q)
            if self.sigma2 < 0:
                self.sigma2 = self.xp.abs(self.sigma2) * 5
            self.iteration += 1

    def update_transformer(self):
        with self.xp.no_grad():
            self.tmat = self.xp.eye(self.D+1, dtype=self.floatx, device=self.device)
            self.tmat[:self.D, :self.D] = self.A
            self.tmat[:self.D, self.D] = self.t
            self.tmat[self.D, :self.D] = self.B
            self.tmat[self.D, self.D] = self.d
            self.tform = self.tmat

    def transform_point(self, Y=None):
        with self.xp.no_grad():
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
        with self.xp.no_grad():
            self.update_transformer()
            # self.transform_point()
            self.TY = self.projective_transform()

            self.tform = self.Xf @ self.tmat @ self.xp.linalg.inv(self.Yf)
            self.tform /= self.tform[-1,-1]*self.d

            self.A = (self.Xs*self.A + self.xp.outer(self.Xm, self.B))/self.Ys
            self.t = (self.t * self.Xs + self.Xm*self.d) - self.A @ self.Ym
            self.B = self.B / self.Ys
            self.d = (self.d - self.B @ self.Ym)/self.d #TODO inverse
            # self.update_transformer() #TODO

            self.TY = self.TY * self.Xs + self.Xm

    def get_transformer(self):
        return {'tform': self.tform, 'A': self.A, 'B': self.B, 't':self.t, 'd':self.d }