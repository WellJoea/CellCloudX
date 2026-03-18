from builtins import super
from .pairwise_emregistration import pwEMRegistration

class pwProjectiveRegistration(pwEMRegistration):
    def __init__(self, *args, gamma1=None, lr=0.005, lr_stepsize=None, B_scale=1e-4,
                 lr_gamma=0.5, opt='LBFGS',opt_iter=70, momentum=0.3,
                 A=None, B = None, d=1.0, t=None,
                  **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'projective'
        self.transformer = 'P'
        
        self.B_scale = B_scale
        self.tol =  -1.0 if self.tol is None else self.tol
        self.normal = 'isoscale' if self.normal is None else self.normal #or global
        self.maxiter = self.maxiter or 150

        self.tau2_decayto = self.scalar2vetor( 0.15 if self.tau2_decayto is None else self.tau2_decayto , L=self.L)   
        self.tau2_decaystop = self.scalar2vetor(0.8 if self.tau2_decaystop is None else self.tau2_decaystop, L=self.L)

        self.normal_XY()
        self.normal_features()
        self.init_sigma2(self.sigma2)
        self.init_transparas(lr=lr, opt_iter=opt_iter, momentum=momentum,
                              opt=opt, lr_stepsize=lr_stepsize, lr_gamma=lr_gamma, gamma1=gamma1)
        self.init_transformer( A = A, B=B, d=d, t=t)

    def init_transparas(self, **kargs):
        self.transparas = self.default_transparas()[self.transformer]
        self.transparas.update(kargs)
    
    def init_tmat_grad(self, **kargs):
        self.tmats = self.init_tmat(self.transformer, self.D, 
                        xp=self.xp, device=self.device, dtype=self.floatx, **kargs)
        #TODO check
        # if not kargs.get('A', None) is None:
        #     self.tmats['A'] = self.tmats['A']/self.Sy
        # if not kargs.get('t', None) is None:
        #     self.tmats['t'] = (self.tmats['t'] - self.My)/self.Sy

        self.A= self.xp.nn.Parameter( self.tmats['A'].to(self.device, dtype=self.floatx) )
        self.B = self.xp.nn.Parameter( self.tmats['B'].to(self.device, dtype=self.floatx)/self.B_scale )
        self.t = self.xp.nn.Parameter( self.tmats['t'].to(self.device, dtype=self.floatx) )
        self.d = self.to_tensor(self.tmats['d'], device=self.device, dtype=self.floatx)

    def init_transformer(self, **kargs):
        self.init_tmat_grad(**kargs)
        self.sigma2_min = self.to_tensor(self.sigma2_min, dtype=self.floatx, device=self.device)
        self.logsigma2 = self.xp.nn.Parameter(
            self.xp.log(self.to_tensor(self.sigma2, dtype=self.floatx, device=self.device)).clone()
        )
        self.optimizer = self.Optim(**self.transparas)
        if self.transparas['lr_stepsize']:
            self.scheduler = self.xp.optim.lr_scheduler.StepLR(self.optimizer, 
                                                        step_size=self.transparas['lr_stepsize'], 
                                                        gamma=self.transparas['lr_gamma'] )

    def Optim(self,  opt='LBFGS', opt_iter=70, momentum=0.3, lr=0.005, **kargs ):
        params = [self.A, self.B, self.t, self.logsigma2]
        if opt == 'LBFGS':
            optimizer = self.xp.optim.LBFGS(params, 
                                             lr=lr,
                                             max_iter=opt_iter,
                                             line_search_fn='strong_wolfe')
        elif opt == 'Adam':
            optimizer = self.xp.optim.Adam(params, lr=lr)
        elif opt == 'RMSprop':
            optimizer = self.xp.optim.RMSprop(params, lr=lr)
        elif opt == 'SGD':
            optimizer = self.xp.optim.SGD(params, momentum=momentum, lr=lr)  
        elif opt == 'ASGD':
            optimizer = self.xp.optim.ASGD(params, lr=lr)
        elif opt == 'AdamW':
            optimizer = self.xp.optim.AdamW(params, lr=lr)
        elif opt == 'Adamax':
            optimizer = self.xp.optim.Adamax(params, lr=lr)
        else:
            raise ValueError('Optimizer not supported')
        return optimizer

    def projective_transform1(self):
        Y = self.Y.to(self.floatxx)
        y_homo = Y @ self.tmats['A'].to(self.floatxx).T +self.tmats['t'].to(self.floatxx)
        Y_base = Y @ self.tmats['B'].to(self.floatxx).unsqueeze(1) + self.tmats['d'].to(self.floatxx)
        Y_base = self.xp.clamp(Y_base, min=1e-8, max=1e8)
        TY = y_homo / Y_base

        if self.xp.any(self.xp.isnan(TY)):
            self.init_sigma2()
            self.init_transformer()
            return self.Y
              
        return TY.to(self.floatx)

    def projective_transform0(self):
        y_homo = self.Y @ self.tmats['A'].T +self.tmats['t']
        Y_base = self.Y @ self.tmats['B'].unsqueeze(1) + self.tmats['d']
        Y_base = self.xp.clamp(Y_base, min=self.eps*10)
        return y_homo / Y_base

    def projective_transform(self):
        y_homo = self.Y @ self.A.T + self.t
        Y_base = self.Y @ (self.B.unsqueeze(1) * self.B_scale) + self.d
        Y_base = self.xp.clamp(Y_base, min=self.eps*10)

        self.tmats['A'] = self.A
        self.tmats['B'] = self.B * self.B_scale
        self.tmats['t'] = self.t
        self.tmats['d'] = self.d

        return y_homo / Y_base

    # def projective_transform(self):
    #     Y = self.Y.to(self.floatxx)
    #     y_homo = Y @ self.tmats['A'].to(self.floatxx).T +self.tmats['t'].to(self.floatxx)
    #     Y_base = Y @ self.tmats['B'].to(self.floatxx).unsqueeze(1) + self.tmats['d'].to(self.floatxx)
    #     Y_base = self.xp.clamp(Y_base, min=self.eps*10, max=1e8)
    #     TY = y_homo / Y_base
    #     return TY.to(self.floatx)