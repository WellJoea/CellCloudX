from builtins import super
import numpy as np

from . import pairwise_emregistration
from .manifold_regularizers import deformable_regularizer
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
        self.normal_XY()
        self.normal_features()
        self.init_regularizer()

    @property
    def num_eig(self):
        return self.num_eig_

    def init_regularizer(self):
        self.DR = deformable_regularizer(
            beta =self.beta, kw=self.kw, kl=self.kl, num_eig=self.num_eig_,
            gamma1=self.gamma1, gamma2=self.gamma2, use_p1=self.use_p1,
            use_fast_low_rank = self.use_fast_low_rank, 
            use_low_rank=self.use_low_rank,
            low_rank_type = self.low_rank_type,
            use_unique=False, kd_method='sknn',
              xp = self.xp)
        self.DR.compute(self.Y, device=self.device, dtype=self.floatx)

        for ia in ['I', 'U', 'S', 'G',]:
            setattr(self, ia, getattr(self.DR, ia, None))
        self.clean_cache()

    def optimization(self):
        self.expectation()
        self.maximization()
        self.iteration += 1

    def maximization(self):
        self.update_transform()
        self.update_transformer()
        self.transform_point()
        self.update_variance()

    def update_transform(self):
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

    def update_transformer(self):
        if self.fast_rank is False:
            self.tmat = self.G @ self.W

        elif self.fast_rank is True:
            self.tmat = (self.U @ self.S) @ (self.U.T @ self.W)

    def transform_point(self, Y=None ): #TODO
        if Y is None:
            self.TY = self.Y + self.tmat
        else:
            return self.ccf_deformable_transform_point(
                        Y, Y=self.Y, Ym=self.Ym, Ys=self.Ys, 
                        Xm=self.Xm, Xs=self.Xs, beta=self.beta, 
                        W=self.W, G=self.G, U=self.U, S=self.S)

    def update_variance(self):
        qprev = self.sigma2        
        trxPx = self.xp.sum( self.Pt1 * self.xp.sum(self.X  * self.X, axis=1) )
        tryPy = self.xp.sum( self.P1 * self.xp.sum( self.TY * self.TY, axis=1))
        trPXY = self.xp.sum(self.TY * self.PX)

        self.sigma2 = (trxPx - 2 * trPXY + tryPy) / (self.Np * self.D)
        if self.sigma2 < self.eps: 
            self.sigma2 = self.xp.clip(self.sigma_square(self.X, self.TY), min=self.sigma2_min)

        self.diff = self.xp.abs(self.sigma2 - qprev)

    def update_normalize(self):
        for ia in ['G', 'U', 'S', 'W', 'Y', 'Ym', 'Ys', 'Xm', 'Xs', 'Yr']:
            if hasattr(self, ia) and getattr(self, ia) is not None:
                setattr(self, ia, getattr(self, ia).clone().to(self.device_pre, dtype=self.floatxx))
        self.s = self.Xs/self.Ys
    
        self.TY = self.TY.to(self.device_pre, dtype=self.floatxx) * self.Xs + self.Xm
        # TY2 = self.transform_point(self.Yr.to(self.device_pre))
        self.tform = self.TY - (self.Y * self.Ys + self.Ym)

    def get_transformer(self, attributes=None): # TODO
        if attributes is None:
            attributes = ['W', 'Xm', 'Xs', 'Xf', 'Ym', 'Ys', 'Yf', 'Y',
                           'beta', 'U', 'S', 'G',]    
        paras = {}            
        for attr in attributes:
            if hasattr(self, attr):
                paras[attr] = getattr(self, attr)
        return paras

class DeformableRegistration_0(pairwise_emregistration.pwEMRegistration):
    def __init__(self, *args, beta=None, alpha=None, gamma1=None, 
                gamma2=None, kw=15, kl=15,
                alpha_decayto = 1, 
                gamma_growto = 1,
                use_p1 = False,
                p1_thred = 0,
                low_rank= 3000,
                low_rank_type = 'keops',
                fast_low_rank =10000, num_eig=100, use_fg=False, beta_fg=None,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'deformable'
        self.tol =  1e-11 if self.tol is None else self.tol
        self.maxiter = self.maxiter or 150
        self.normal_ = 'each' if self.normal_ is None else self.normal_

        self.alpha = 0.5 if alpha is None else alpha
        self.beta = 1.0 if beta is None else beta
        self.gamma1 = 0 if gamma1 is None else gamma1
        self.gamma2 = 0 if gamma2 is None else gamma2

        self.alpha_decayto = alpha_decayto or 1.0
        self.alpha_mu = self.alpha_decayto**(1.0/ float(self.maxiter-1))
        self.gamma_growto = gamma_growto or 1.0
        self.gamma_nu = self.gamma_growto**(1.0/ float(self.maxiter-1))
        self.use_p1 = use_p1
        self.p1_thred = p1_thred

        self.fast_low_rank= fast_low_rank
        self.use_fast_low_rank =  (fast_low_rank if type(fast_low_rank) == bool  
                                    else (self.M >= fast_low_rank) )
        self.low_rank_type = low_rank_type
        self.low_rank = low_rank
        self.use_low_rank = (low_rank if type(low_rank) == bool  
                                    else (self.M >= low_rank) )
        self.fast_rank = (self.use_low_rank or self.use_fast_low_rank)
    
        self.num_eig_ = num_eig
        self.use_fg = use_fg
        self.beta_fg = beta_fg
        self.kw =kw
        self.kl =kl

        self.W = self.xp.zeros((self.M, self.D), dtype=self.floatx)
        self.expectation = self.pwexpectation_df
        self.compute_feauture = self.pwcompute_feauture_df

    def init_tau2(self, tau2, tau2_clip=None):
        tau2, self.tau2 = self.scalar2vetor(tau2, L=self.LF), self.xp.ones((self.LF))
        for lf in range(self.LF):
            if (tau2[lf] is None) or (tau2[lf] <= 0):
                if self.feat_normal[lf] in ['cos', 'cosine']:
                    self.tau2[lf] = self.sigma_square(self.XF[lf], self.YF[lf])*2.0 #TODO
                else:
                    self.tau2[lf] = self.sigma_square(self.XF[lf], self.YF[lf])
            else:
                self.tau2[lf] = tau2[lf]
    
            if tau2_clip[lf] is not None:
                self.tau2[lf] = self.xp.clip(self.tau2[lf], *tau2_clip[lf])

        self.tau2 = self.to_tensor(self.tau2, dtype=self.floatx, device=self.device)
        self.tau2_prediv = not self.tau2_auto
        self.tau2_decayto = float(self.tau2_decayto or 1.0)
        # self.tau2_decay = self.tau2_decayto** (-1.0 / float(self.maxiter-1))
        self.tau2_grow = self.xp.pow( self.tau2_decayto** (-1.0 / float(self.maxiter-1)) , self.xp.arange(self.maxiter))
        self.DK = [self.XF[lf].shape[1]/2.0 for lf in range(self.LF) ]

class DeformableRegistration_1(pairwise_emregistration.pwEMRegistration):
    def __init__(self, *args, beta=None, alpha=None, gamma1=None, 
                gamma2=None, kw=15, kl=15,
                alpha_decayto = 1, 
                gamma_growto = 1,
                use_p1 = False,
                p1_thred = 0,
                low_rank= 3000,
                low_rank_type = 'keops',
                fast_low_rank =10000, num_eig=100, use_fg=False, beta_fg=None,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'deformable'
        self.tol =  1e-11 if self.tol is None else self.tol
        self.maxiter = self.maxiter or 150
        self.normal_ = 'each' if self.normal_ is None else self.normal_

        self.alpha = float(5e2 if alpha is None else alpha)
        self.beta = 2.5 if beta is None else beta
        self.gamma1 = float(0 if gamma1 is None else gamma1)
        self.gamma2 = float(0 if gamma2 is None else gamma2)

        self.alpha_decayto = alpha_decayto or 1.0
        self.alpha_mu = self.alpha_decayto**(1.0/ float(self.maxiter-1))
        self.gamma_growto = gamma_growto or 1.0
        self.gamma_nu = self.gamma_growto**(1.0/ float(self.maxiter-1))
        self.use_p1 = use_p1
        self.p1_thred = p1_thred

        self.fast_low_rank= fast_low_rank
        self.low_rank_type = low_rank_type
        self.use_fast_low_rank =  (fast_low_rank if type(fast_low_rank) == bool  
                                    else (self.M >= fast_low_rank) )
        self.low_rank = low_rank
        self.use_low_rank = (low_rank if type(low_rank) == bool  
                                    else (self.M >= low_rank) )
        self.fast_rank = (self.use_low_rank or self.use_fast_low_rank)

        self.num_eig_ = num_eig
        self.use_fg = use_fg
        self.beta_fg = beta_fg
        self.kw =kw
        self.kl =kl

        self.W = self.xp.zeros((self.M, self.D), dtype=self.floatx)
        self.expectation = self.pwexpectation
        self.compute_feauture = self.pwcompute_feauture

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
        super().__init__(*args,  beta=beta or 3, alpha=alpha or 5e2, 
                         gamma1= 1e3 if gamma1 is None else gamma1,
                         gamma2= 1e2 if gamma2 is None else gamma2,
                         kw=kw or 15, kl=kl or 20,
                         alpha_decayto = alpha_decayto or 0.2,
                         tau2_decayto = tau2_decayto or 0.15, 
                         gamma_growto = gamma_growto or 2.0,
                         use_p1 = False if use_p1 is None else use_p1,
                         p1_thred = 0.3 if p1_thred is None else p1_thred,
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
        super().__init__(*args,  beta=beta or 2.5, alpha=alpha or 5e2, 
                         gamma1= 1e3 if gamma1 is None else gamma1,
                         gamma2= 1e2 if gamma2 is None else gamma2,
                         kw=kw or 15, kl=kl or 20,
                         alpha_decayto = alpha_decayto or 0.2,
                         tau2_decayto = tau2_decayto or 0.15, 
                         gamma_growto = gamma_growto or 2.0,
                         use_p1 = True if use_p1 is None else use_p1,
                         p1_thred = 0.4 if p1_thred is None else p1_thred,
                         low_rank= low_rank,
                         low_rank_type = low_rank_type,
                         fast_low_rank =fast_low_rank, num_eig=num_eig,
                         **kwargs)