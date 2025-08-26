from .pairwise_emregistration  import pwEMRegistration
from ...io._logger import logger

class pwDeformableRegistration:
    def __new__(cls, *args, **kwargs):
        df_version = kwargs.get('df_version', 1)
        if df_version == 0:
            bases = (DeformableRegistration_0,)
        elif df_version == 1:
            bases = (DeformableRegistration_1,)
        elif df_version == 2:
            bases = (DeformableRegistration_2,)
        elif df_version == 3:
            bases = (DeformableRegistration_3,)
        else:
            raise ValueError("Invalid df_version")

        new_cls = type('DeformableRegistration', (cls,) + bases, {})
        instance = object.__new__(new_cls)
        return instance

    def __init__(self, *args, **kwargs):
        self.df_version = kwargs.pop('df_version', 1)
        super().__init__(*args, **kwargs)
        self.each_ncols = 3
        self.reg_core = 'deformable'
        self.transformer = 'D'

        self.normal_XY()
        self.normal_features()

        self.init_transparas(**self.transparas)
        self.init_transformer()
        self.init_regularizer()

    def init_transformer(self, **kargs):
        self.tmats = self.init_tmat(self.transformer, self.D, N = self.M,
                        xp=self.xp, device=self.device, dtype=self.floatx, **kargs)
        
    def init_transparas(self, **kwargs):
        ipara = self.default_transparas()[self.transformer]
        ipara.update(kwargs)

        ipara['alpha_mu'] =  ipara['alpha_decayto'] **(1.0/ float(self.maxiter-1)) 
        ipara['gamma_nu'] =  ipara['gamma_growto']  **(1.0/ float(self.maxiter-1)) 
        for iarrt in ['alpha', 'gamma1', 'gamma2', 'p1_thred']:
            ipara[iarrt] = self.to_tensor(self.scalar2vetor(ipara[iarrt], self.L), 
                                          dtype=self.floatx, device=self.device)
        ipara['use_low_rank'] = ( ipara['low_rank'] if type(ipara['low_rank']) == bool  
                                        else bool(self.M >= ipara['low_rank']) )
        ipara['use_fast_low_rank'] = ( ipara['fast_low_rank'] if type(ipara['fast_low_rank']) == bool  
                                        else bool(self.M >= ipara['fast_low_rank']) )
        ipara['fast_rank'] = ipara['use_low_rank'] or ipara['use_fast_low_rank']
        self.transparas = ipara

    def init_regularizer(self):
        self.DR = self.pwdeformable_regularizer(**self.transparas, xp = self.xp)
        self.DR.compute(self.Y, device=self.device, dtype=self.floatx)
        self.clean_cache()

    def update_transform_v0(self):
        mu = self.alpha_mu**(self.iteration)
        nu = self.gamma_nu**(self.iteration)
        alpha = self.alpha * mu
        gamma1 = self.gamma1 * nu
        gamma2 = self.gamma2 * nu

        if self.fast_rank:
            B  = self.PX - self.xp.multiply(self.Y.T, self.P1).T
            B = 1.0/(self.sigma2*alpha) * B  # W = 1.0/(self.sigma2*alpha) * W
            Pg = 1.0/self.sigma2*(self.P1[:,None]* (self.U @ self.S))

            if (self.use_p1):
                P1 = self.P1.clone()
                P1[P1 < self.p1_thred] = 0
    
                if (self.gamma1>0):
                    LPLY = (self.DR.L.T @ (P1[:,None]*self.DR.LY))
                    LPLV = (self.DR.L.T @ (P1[:,None]*self.DR.LV))
                    B -=  gamma1 * self.sigma2 * LPLY
                    Pg += gamma1 * LPLV
    
                if (self.gamma2>0):
                    APAV = (self.DR.A.T @ (P1[:,None]*self.DR.AV))
                    Pg += gamma2 * APAV
            else:
                if (self.gamma1>0):
                    B -=  gamma1 * self.sigma2 * self.DR.QY
                    Pg += gamma1 * self.DR.J
                elif (self.gamma2>0):
                    Pg += gamma2 * self.DR.J
    
            Pg = 1.0/alpha * Pg
            W = B - Pg @ self.xp.linalg.solve(self.I + self.U.T @ Pg, self.U.T @ B)
            # W = 1.0/(self.sigma2*alpha) * W
        else:
            B  = self.PX - self.xp.multiply(self.Y.T, self.P1).T
            A = self.P1[:,None]*self.G 
            A.diagonal().add_(self.alpha * self.sigma2)

            if (self.use_p1):
                P1 = self.P1.clone()
                P1[P1 < self.p1_thred] = 0
    
                if (self.gamma1>0):
                    LPLY = (self.DR.L.T @ (P1[:,None]*self.DR.LY))
                    QG = (self.DR.L.T @ (P1[:,None]*self.DR.LG))
                    B -=  gamma1 * self.sigma2 * LPLY
                    A +=  gamma1 * self.sigma2 * QG
                if (self.gamma2>0):
                    RG = (self.DR.A.T @ (P1[:,None]*self.DR.AG))
                    A += gamma2 * self.sigma2 * RG
            else:
                if (self.gamma1>0):
                    B -=  gamma1 * self.sigma2 * self.DR.QY
                    A +=  gamma1 * self.sigma2 * self.DR.QG
                if (self.gamma2>0):
                    A += gamma2 * self.sigma2 * self.DR.RG
            W = self.xp.linalg.solve(A, B)
        self.W = W

class DeformableRegistration_0(pwEMRegistration):
    def __init__(self, *args, 
                 beta=None, alpha=None, gamma1=None, 
                gamma2=None, kw=15, kl=15,
                alpha_decayto = 1, 
                gamma_growto = 1,
                delta= 0,
                use_p1 = False,
                p1_thred = 0,
                low_rank= 3000,
                low_rank_type = 'keops',
                fast_low_rank =10000, num_eig=100,
                **kwargs):
        super().__init__(*args, **kwargs)

        self.tol =  1e-11 if self.tol is None else self.tol
        self.maxiter = self.maxiter or 150
        self.normal = 'each' if self.normal is None else self.normal

        self.transparas = dict(
                beta = beta or 1.0, 
                alpha= 0.5 if alpha is None else alpha,
                gamma1= 0 if gamma1 is None else gamma1,
                gamma2= 0 if gamma2 is None else gamma2,
                delta = delta or 0,
                p1_thred= 0.2 if p1_thred is None else p1_thred,
                low_rank=low_rank or 3000,
                fast_low_rank=fast_low_rank or 100000,
                num_eig=num_eig or 100,
                kw=kw or 15, kl=kl or 15,
                low_rank_type=low_rank_type,
                alpha_decayto=alpha_decayto or 1.0, 
                gamma_growto=gamma_growto or 1.0,
                use_p1=use_p1, 
        )
        self.pwexpectation = self.pwexpectation_df
        self.pwcompute_feauture = self.pwcompute_feauture_df

    def init_tau2(self, tau2, tau2_clip=None):
        self.DK = self.scalar2list(2, L=self.FL)
        self.tau2_auto = [bool(i) for i in self.tau2_auto]
        self.tau2_decayto = float(self.tau2_decayto or 1.0)
        self.tau2_grow = 1.0/self.decay_curve(self.tau2_decayto, self.tau2_decaystop, self.maxiter)
        self.tau2 = tau2
    
        for iL,fexist in enumerate(self.fexists):
            if fexist:
                for il in range(self.FL[iL]):
                    if (tau2[iL][il] is None) or (tau2[iL][il] <= 0):
                        if self.feat_normal[iL][il]in ['cos', 'cosine']:
                            # self.tau2[iL][il] = self.sigma_square_cos(self.XF[iL][il], self.YF[iL][il]) #TODO
                            self.tau2[iL][il] = self.sigma_square(self.XF[iL][il], self.YF[iL][il])*2.0 #TODO
                        else:
                            self.tau2[iL][il] = self.sigma_square(self.XF[iL][il], self.YF[iL][il])
                        self.DK[iL][il] = self.XF[iL][il].shape[1]
                    if tau2_clip is not None:
                        self.tau2[iL][il] = self.xp.clip(self.tau2[iL][il], *tau2_clip)

class DeformableRegistration_1(pwEMRegistration):
    def __init__(self, *args, beta=None, alpha=None, gamma1=None, 
                gamma2=None, kw=15, kl=15,
                alpha_decayto = 1, 
                gamma_growto = 1,
                delta=0,
                use_p1 = False,
                p1_thred = 0,
                low_rank= 3000,
                low_rank_type = 'keops',
                fast_low_rank =10000, num_eig=100, use_fg=False, beta_fg=None,
                **kwargs):
        super().__init__(*args, **kwargs)

        self.tol =  1e-11 if self.tol is None else self.tol
        self.maxiter = self.maxiter or 150
        self.normal = 'each' if self.normal is None else self.normal

        self.transparas = dict(
                beta = beta or 2.5, 
                alpha= 5e2 if alpha is None else alpha, #list or float
                gamma1= 0 if gamma1 is None else gamma1,#list or float
                gamma2= 0 if gamma2 is None else gamma2,#list or float
                p1_thred= 0.2 if p1_thred is None else p1_thred,#list or float
                delta = delta or 0,
                low_rank=low_rank or 3000,
                fast_low_rank=fast_low_rank or 100000,
                num_eig=num_eig or 100,
                kw=kw or 15, kl=kl or 15,
                low_rank_type=low_rank_type,
                alpha_decayto=alpha_decayto or 1.0, 
                gamma_growto=gamma_growto or 1.0,
                use_p1=use_p1, 
        )

class DeformableRegistration_2(DeformableRegistration_1):
    def __init__(self, *args, beta=3, alpha=6e2, gamma1=1e2, 
                gamma2=0, kw=15, kl=20,
                alpha_decayto = 0.2, 
                tau2_decayto = 0.15,
                gamma_growto = 2.0,
                use_p1 = False,
                p1_thred = 0.3,
                low_rank= 3000,
                low_rank_type = 'keops',
                fast_low_rank =10000, num_eig=100,
                **kwargs):
        super().__init__(*args, 
                        beta=beta or 3,
                         alpha= 5e2 if alpha is None else alpha,
                         gamma1= 1e3 if gamma1 is None else gamma1,
                         gamma2= 1e2 if gamma2 is None else gamma2,
                         p1_thred = 0.3 if p1_thred is None else p1_thred,
                         kw=kw or 15, kl=kl or 20,
                         alpha_decayto = alpha_decayto or 0.2,
                         tau2_decayto = tau2_decayto or 0.15, 
                         gamma_growto = gamma_growto or 2.0,
                         use_p1 = False if use_p1 is None else use_p1,
                         low_rank= low_rank,
                         low_rank_type = low_rank_type,
                         fast_low_rank =fast_low_rank, num_eig=num_eig,
                         **kwargs)

class DeformableRegistration_3(DeformableRegistration_1):
    def __init__(self, *args, beta=2.5, alpha=7e2, gamma1=1e3, 
                gamma2=1e2, kw=15, kl=20,
                alpha_decayto = 0.2, 
                tau2_decayto = 0.15,
                gamma_growto = 2.0,
                use_p1 = True,
                p1_thred = 0.4,
                low_rank= 3000,
                low_rank_type = 'keops',
                fast_low_rank = 10000, num_eig=100,
                **kwargs):
        super().__init__(*args, 
                         beta=beta or 2.5, 
                         alpha= 5e2 if alpha is None else alpha,
                         gamma1= 1e3 if gamma1 is None else gamma1,
                         gamma2= 1e2 if gamma2 is None else gamma2,
                         p1_thred = 0.4 if p1_thred is None else p1_thred,
                         kw=kw or 15, kl=kl or 20,
                         alpha_decayto = alpha_decayto or 0.2,
                         tau2_decayto = tau2_decayto or 0.15, 
                         gamma_growto = gamma_growto or 2.0,
                         use_p1 = True if use_p1 is None else use_p1,
                         low_rank= low_rank,
                         low_rank_type = low_rank_type,
                         fast_low_rank =fast_low_rank, num_eig=num_eig,
                         **kwargs)