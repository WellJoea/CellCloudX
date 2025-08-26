from builtins import super
from .pairwise_emregistration import pwEMRegistration

class pwProjectiveRegistration(pwEMRegistration):
    def __init__(self, *args, gamma1=None, lr=0.005, lr_stepsize=None,
                 lr_gamma=0.5, opt='LBFGS',opt_iter=70, momentum=0.3,
                 A=None, B = None, d=1.0, t=None,
                  **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'projective'
        self.transformer = 'P'

        self.tol =  -1.0 if self.tol is None else self.tol
        self.normal = 'isoscale' if self.normal is None else self.normal #or global
        self.maxiter = self.maxiter or 150

        self.normal_XY()
        self.normal_features()
        self.init_sigma2(self.sigma2)
        self.init_transparas(lr=lr, opt_iter=opt_iter, momentum=momentum,
                              opt=opt, lr_stepsize=lr_stepsize, lr_gamma=lr_gamma, gamma1=gamma1)
        self.init_transformer( A = A, B=B, d=d, t=t)

    def init_transparas(self, **kargs):
        self.transparas = self.default_transparas()[self.transformer]
        self.transparas.update(kargs)

    def init_transformer(self, **kargs):
        self.tmats = self.init_tmat(self.transformer, self.D, 
                        xp=self.xp, device=self.device, dtype=self.floatx, **kargs)
        #TODO check
        # if not kargs.get('A', None) is None:
        #     self.tmats['A'] = self.tmats['A']/self.Sy
        # if not kargs.get('t', None) is None:
        #     self.tmats['t'] = (self.tmats['t'] - self.My)/self.Sy

        self.tmats['A'] = self.xp.nn.Parameter( self.tmats['A'].to(self.device, dtype=self.floatx) )
        self.tmats['B'] = self.xp.nn.Parameter( self.tmats['B'].to(self.device, dtype=self.floatx) )
        self.tmats['t'] = self.xp.nn.Parameter( self.tmats['t'].to(self.device, dtype=self.floatx) )
        self.tmats['d'] = self.to_tensor(self.tmats['d'], device=self.device, dtype=self.floatx)

        for i in 'ABtd':
            setattr(self, i, self.tmats[i])
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
        params = [self.tmats['A'], self.tmats['B'], self.tmats['t'], self.logsigma2]
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

    def projective_transform(self):
        y_homo = self.Y @ self.tmats['A'].T +self.tmats['t']
        Y_base = self.Y @ self.tmats['B'].unsqueeze(1) + self.tmats['d']
        Y_base = self.xp.clamp(Y_base, min=self.eps)
        return y_homo / Y_base