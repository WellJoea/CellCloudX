from .paired_emregistration import spEMRegistration
from ...io._logger import logger

class spRegistration(spEMRegistration):
    def __init__(self, *args, 
                 transformer = None,
                 tf = None,

                 A=None, t = None,
                 R = None, s = None, 
                 fix_s=None, s_clip=None,
                 isoscale=None, fix_R=None, fix_t=None, 

                lr=0.005, lr_stepsize=None, B_scale=1e-4,
                lr_gamma=0.5, opt='LBFGS',opt_iter=70, momentum=0.3,
                B = None, d=1.0,

                delta=0, 
                beta=1.5, alpha=3e2,
                gamma1=None, 
                gamma2=None, 
                gamma3=None,
                kw=15, kl=20,
                alpha_decayto = 0.2, 
                gamma_growto = 2.0,
                use_p1 = None,
                p1_thred = 0.3,
                low_rank= 3000,
                fast_low_rank =10000, num_eig=100,
                kd_method='sknn',
                low_rank_type = 'keops',

                transparas={},
                **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = tf or transformer or 'A'

        if self.transformer in 'P':
            self.tol =  -1 if self.tol is None else self.tol
        else:
            self.tol =  1e-9 if self.tol is None else self.tol
        
        if self.transformer in 'D':
            self.normal = 'each' if self.normal is None else self.normal #or global
        else:
            self.normal = 'isoscale' if self.normal is None else self.normal #or global
        self.maxiter = self.maxiter or 300

        self.tau2_decayto = self.scalar2vetor( 0.15 if self.tau2_decayto is None else self.tau2_decayto , L=self.FL)   
        self.tau2_decaystop = self.scalar2vetor(0.8 if self.tau2_decaystop is None else self.tau2_decaystop, L=self.FL)
        self.B_scale = B_scale or 1e-4

        self.normal_XY()
        self.normal_features()

        self.init_transformer( 
                A=A, t = t,
                R = R, s = s, 
                s_clip=s_clip,
                B = B, d= d or 1.0,
        )

        transparas.update(dict(
                fix_s=fix_s, s_clip=s_clip, fix_R=fix_R,
                fix_t=fix_t, isoscale=isoscale,

                lr= lr, 
                lr_stepsize=lr_stepsize,
                lr_gamma= lr_gamma, 
                opt= opt, opt_iter= opt_iter, 
                momentum= momentum,

                delta=delta, 
                beta=beta, alpha=alpha,
                gamma1=gamma1, 
                gamma2=gamma2, 
                gamma3=gamma3,
                kw=kw, kl=kl,
                alpha_decayto = alpha_decayto, 
                gamma_growto = gamma_growto,
                use_p1 = use_p1,
                p1_thred = p1_thred,
                low_rank= low_rank  ,
                low_rank_type = low_rank_type,
                fast_low_rank =fast_low_rank, 
                num_eig=num_eig,
                kd_method=kd_method ,
        )) #defluat
        self.init_transparas(transparas=transparas)

    def init_transformer(self, **kargs):
        #TODO check
        if (kargs.get('A', None)) is not None:
            kargs['A'] = kargs['A']/self.Sy
        if (kargs.get('t', None)) is not None:
            kargs['t'] = (kargs['t'] - self.My)/self.Sy
        if not kargs.get('s', None) is None:
            kargs['s'] = kargs['s']/self.Sy

        self.tmats = self.init_tmat(self.transformer, self.D, 
                        N = self.M,
                        xp=self.xp, device=self.device, dtype=self.floatx, 
                        **kargs)

        if self.transformer in 'P':
            self.A = self.xp.nn.Parameter( self.tmats['A'].to(self.device, dtype=self.floatx) )
            self.B = self.xp.nn.Parameter( self.tmats['B'].to(self.device, dtype=self.floatx)/self.B_scale )
            self.t = self.xp.nn.Parameter( self.tmats['t'].to(self.device, dtype=self.floatx) )
            self.d = self.to_tensor(self.tmats['d'], device=self.device, dtype=self.floatx)

            self.sigma2_min = self.to_tensor(self.sigma2_min, dtype=self.floatx, device=self.device)
            self.logsigma2 = self.xp.nn.Parameter(
                self.xp.log(self.to_tensor(self.sigma2 or 1.0, dtype=self.floatx, device=self.device)).clone()
            )

    def init_transparas(self, transparas):
        self.transparas = self.default_transparas()[self.transformer]
        # self.transparas.update(transparas)
        for k, v in self.transparas.items():
            iv = transparas.get(k, None)
            self.transparas[k] = iv if iv is not None else v

        for k,v in zip(['p1_thred', 'use_p1', 'gamma1', 'gamma3'], [0.3, False, 0, 0]):
            iv1 = self.transparas.get(k, None)
            iv2 =      transparas.get(k, None)
            self.transparas[k] = iv1 or iv2 or v

        if  self.transformer in 'DH':
            self.init_transparas_D()

        if  self.transformer in 'P':
            self.init_transparas_P()

        if (self.transparas.get('fix_s', False)) and (not self.transformer in 'HD'):
            self.transparas['gamma1'] = 0
            self.transparas['gamma3'] = 0
        
        if  self.transformer in 'D':
            if 'gamma3' in self.transparas:
                del self.transparas['gamma3']

        if not self.transformer in 'HD':
            if 'gamma2' in self.transparas:
                del self.transparas['gamma2']

    def init_transparas_D(self):
        ipara = self.transparas
        ipara['alpha_mu'] =  ipara['alpha_decayto'] **(1.0/ float(self.maxiter-1)) 
        ipara['gamma_nu'] =  ipara['gamma_growto']  **(1.0/ float(self.maxiter-1)) 
        for iarrt in ['alpha', 'gamma1', 'gamma2', 'p1_thred']:
            ipara[iarrt] = float(ipara[iarrt])

        #TODO :check gamma1, gamma2
        ipara['beta']  *= (self.scale_factor**2)
        ipara['alpha'] /= (self.scale_factor**2)
        ##########################

        ipara['use_low_rank'] = ( ipara['low_rank'] if type(ipara['low_rank']) == bool  
                                        else bool(self.N >= ipara['low_rank']) )
        ipara['use_fast_low_rank'] = ( ipara['fast_low_rank'] if type(ipara['fast_low_rank']) == bool  
                                        else bool(self.N >= ipara['fast_low_rank']) )
        ipara['fast_rank'] = ipara['use_low_rank'] or ipara['use_fast_low_rank']
        self.transparas = ipara

    def init_transparas_P(self):
        self.optimizer = self.Optim(**self.transparas)
        if self.transparas['lr_stepsize']:
            self.scheduler = self.xp.optim.lr_scheduler.StepLR(
                                    self.optimizer, 
                                    step_size=self.transparas['lr_stepsize'], 
                                    gamma=self.transparas['lr_gamma'] )

    def Optim(self, opt='LBFGS', opt_iter=70, momentum=0.3, lr=0.005, **kargs ):
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

    def projective_transform(self):
        y_homo = self.Y @ self.A.T + self.t
        Y_base = self.Y @ (self.B.unsqueeze(1) * self.B_scale) + self.d
        Y_base = self.xp.clamp(Y_base, min=self.eps*10)

        self.tmats['A'] = self.A
        self.tmats['B'] = self.B * self.B_scale
        self.tmats['t'] = self.t
        self.tmats['d'] = self.d

        return y_homo / Y_base
