from builtins import super
import numpy as np
import scipy.sparse as ssp
from .emregistration import EMRegistration
from .xmm import kernel_xmm, low_rank_eigen, lle_W, gl_w, low_rank_eigen_grbf, WoodburyB

class DeformableRegistration(EMRegistration):
    def __init__(self, *args, alpha=None, beta=None, gamma1=None, 
                 gamma2=None, kw=25, kl=25, rl_w=None,
                #  gamma_decay = 0.99, 
                #  gamma_decay_start = 150, 
                 low_rank_g =False,
                 use_fg=False, beta_fg=None, low_rank=True, num_eig=100,
                   **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'deformable'
        self.alpha = 0.5 if alpha is None else alpha
        self.beta_ = 2 if beta is None else beta
        self.gamma1 = 0 if gamma1 is None else gamma1
        self.gamma2 = 0 if gamma2 is None else gamma2
        self.tol =  1e-11 if self.tol is None else self.tol

        self.low_rank_g= low_rank_g
        if self.M > 5e5:
            #self.low_rank_g = True
            print('Both low_rank and low_rank_g should be set to True for large datasets.')
    
        self.low_rank_ = ((low_rank or low_rank_g) or (self.gamma1>0 or self.gamma2>0))
        self.num_eig_ = num_eig
        self.use_fg = use_fg
        self.beta_fg = beta_fg
        self.use_local =  (self.gamma1>0) or (self.gamma2>0)
        self.kw =kw
        self.kl =kl
        self.W = np.zeros((self.M, self.D))
        
        self.normal_ = 'each' if self.normal_ is None else self.normal_
        self.init_normalize()
        self.init_L( kw=self.kw, kl=self.kl, rl_w=rl_w, method='sknn')
        self.update_transformer()

    @property
    def beta(self):
        return self.beta_
    @property
    def num_eig(self):
        return self.num_eig_
    @property
    def low_rank(self):
        return self.low_rank_

    def init_L(self, kw=15, kl=20, rl_w=None, method='sknn'):
        # print('compute penalty matrix...')   
        if (self.gamma1>0):
            print('compute Y lle...')
            L = lle_W(self.Y, kw=kw, rl_w=rl_w, method=method, eps=self.eps)
            M = L.transpose().dot(L) #TODO check
            self.MY = self.gamma1 * M.dot(self.Y)
        else:
            self.MY = 0
    
        if (self.gamma2>0):
            print('compute Y gl...')
            A = gl_w(self.Y, kw=kl, method=method)
            # P = A.dot(A.transpose())
            P = A.transpose().dot(A) #TODO check

        if self.low_rank is True:
            print('compute G eigendecomposition...')
            if self.low_rank_g:
                self.Q, self.S  = low_rank_eigen_grbf(self.Y, np.float64(self.beta_**0.5), self.num_eig, sw_h=0)
                self.inv_S = np.diag(1./self.S)
                self.S = np.diag(self.S)
            else:
                G = kernel_xmm(self.Y, self.Y, sigma2=self.beta_, temp=1/2)[0]
                self.Q, self.S = low_rank_eigen(G, self.num_eig) #TODO
                self.inv_S = np.diag(1./self.S)
                self.S = np.diag(self.S)

            if self.use_local:
                K = 0
                if (self.gamma1>0):
                    K =  self.gamma1 * M + K
                if (self.gamma2>0):
                    K =  self.gamma2 * P + K

                # F = (self.alpha * self.xp.eye(self.M) + self.gamma1 * MG + self.gamma2 *PG)
                # self.Fv = np.linalg.inv(F)
                U = K.dot(self.Q)
                Av = 1 / self.alpha * self.xp.eye(self.M)
                Cv = self.inv_S
                V = self.Q.T
                self.Fv = WoodburyB(Av, U, Cv, V)
            else:
                self.Fv = 1 / self.alpha * self.xp.eye(self.M)
    
        else:
            print('compute G eigendecomposition...')
            G = kernel_xmm(self.Y, self.Y, sigma2=self.beta_, temp=1/2)[0]
            self.G = G
            self.Q = None
            self.S = None
    
            if self.use_fg: # TODO
                print('compute feature Y gmm...')
                G1 = kernel_xmm(self.Y_feat, self.Y_feat, sigma2=self.beta_fg/2)[0]
                self.G *= G1
        
            if self.use_local:
                self.MPG = 0
                if (self.gamma1>0):
                    self.MPG = self.gamma1 * M.dot(G) + self.MPG
                if (self.gamma2>0):
                    self.MPG =  self.gamma2 * P.dot(G) +self.MPG 

                self.F = (self.alpha * self.xp.eye(self.M) + self.MPG)
                # self.Fv = np.linalg.inv(F)
            else:
                self.F = self.alpha * self.xp.eye(self.M)
    
    def optimization(self):
        self.expectation()
        self.update_transform()
        self.update_transformer()
        self.transform_point()
        self.update_variance()
        self.iteration += 1

    def update_transform(self):

        if self.low_rank is False:
            if (self.use_local):
                A = self.xp.multiply(self.P1, self.G).T + self.sigma2 * self.F
                B = self.PX - self.xp.multiply(self.Y.T, self.P1).T - self.sigma2*self.MY
                self.W = self.xp.linalg.solve(A, B)
            else: # same
                # A = self.xp.dot(self.xp.diag(self.P1), self.G) + \
                #     self.alpha * self.sigma2 * self.xp.eye(self.M)
                # B = self.PX - self.xp.dot(self.xp.diag(self.P1), self.Y)
                A = self.xp.multiply(self.P1, self.G).T + self.sigma2 * self.F
                B = self.PX - self.xp.multiply(self.Y.T, self.P1).T

                self.W = self.xp.linalg.solve(A, B)

        elif self.low_rank is True:
            if (self.use_local):

                B = self.PX - self.xp.multiply(self.Y.T, self.P1).T - self.sigma2*self.MY
                dPQ = np.multiply(self.Q.T, self.P1).T
                Fvs2v = self.Fv / self.sigma2
                FB = Fvs2v @ B
                FQ = Fvs2v @ dPQ
                # Uv = np.linalg.inv(self.inv_S + self.Q.T @ FQ)
                # self.W = FB - (FQ @ Uv) @ (self.Q.T @ FB)
                del B, Fvs2v, dPQ
                self.W = FB - FQ @ np.linalg.solve(self.inv_S + self.Q.T @ FQ, self.Q.T @ FB)
    
            else: # same
                # dPQ = np.multiply(self.Q.T, self.P1).T
                # F = self.PX - np.multiply(self.Y.T, self.P1).T
                # self.W = 1 / (self.alpha * self.sigma2) * (F - np.matmul(dPQ, 
                #     (np.linalg.solve((self.alpha * self.sigma2 * self.inv_S + np.matmul(self.Q.T, dPQ)),
                #                     (np.matmul(self.Q.T, F))))))
    
                B = self.PX - self.xp.multiply(self.Y.T, self.P1).T #- self.sigma2*self.MY
                dPQ = np.multiply(self.Q.T, self.P1).T
                Fvs2v = self.Fv / self.sigma2
                FB = Fvs2v @ B
                FQ = Fvs2v @ dPQ
                del B, Fvs2v, dPQ
                self.W = FB - FQ @ np.linalg.solve(self.inv_S + self.Q.T @ FQ, self.Q.T @ FB)
    

    def update_transformer(self):
        if self.low_rank is False:
            self.tmat = self.xp.dot(self.G, self.W)

        elif self.low_rank is True:
            self.tmat = np.matmul(self.Q, np.matmul(self.S, np.matmul(self.Q.T, self.W)))

    def transform_point(self, Y=None ): #TODO
        if Y is None:
            self.TY = self.Y + self.tmat
        else:
            return self.ccf_deformable_transform_point(
                        Y, Y=self.Y, Ym=self.Ym, Ys=self.Ys, 
                        Xm=self.Xm, Xs=self.Xs, beta=self.beta, 
                        G=self.G, W=self.W, Q=self.Q, S=self.S)
            # Y_t = (Y -self.Ym)/self.Ys
            # if reset_G or (not np.array_equal(self.Y, Y.astype(self.Y.dtype))):
            #     G = self.kernal_gmm(Y_t, self.Y, sigma2=self.beta)[0]
            #     tmat = np.dot(G, self.W)
            # else:
            #     tmat = self.tmat
            # Y_n = (Y_t + tmat)* self.Xs + self.Xm
            # return Y_n

    def update_variance(self):
        qprev = self.sigma2

        # trxPx = np.dot(np.transpose(self.Pt1), 
        #              np.sum(np.multiply(self.X, self.X), axis=1))
        # tryPy = np.dot(np.transpose(self.P1),  
        #              np.sum(np.multiply(self.TY, self.TY), axis=1))
        
        trxPx = np.sum( self.Pt1.T * np.sum(np.multiply(self.tX, self.tX), axis=1) )
        tryPy = np.sum( self.P1.T * np.sum(np.multiply(self.TY, self.TY), axis=1))
        trPXY = np.sum(np.multiply(self.TY, self.PX))

        self.sigma2 = (trxPx - 2 * trPXY + tryPy) / (self.Np * self.D)
        if self.sigma2 <= 0:
            # self.sigma2 = self.tolerance / 10
            self.sigma2 = 1/self.iteration

        self.diff = np.abs(self.sigma2 - qprev)

    def update_normalize(self):
        # self.tmat = self.tmat * self.Xs + self.Xm
        self.TY = self.TY * self.Xs + self.Xm
        self.tform = self.TY - (self.Y * self.Ys + self.Ym)

    def get_transformer(self): # TODO
        paras = {'W': self.W}
        paras['Xm'] = self.Xm
        paras['Xs'] = self.Xs
        paras['Ym'] = self.Ym
        paras['Ys'] = self.Ys
        paras['Y'] = self.Y
        paras['beta'] = self.beta
        if self.low_rank:
            paras['Q'] = self.Q
            paras['S'] = self.S
        else:
            paras['G'] = self.G

        return paras
