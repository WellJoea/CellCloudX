from builtins import super
import numpy as np

from .emregistration import EMRegistration
from .xmm import kernel_xmm, low_rank_eigen, lle_W, low_rank_eigen_grbf, Woodbury

class DeformableRegistration(EMRegistration):
    def __init__(self, *args, alpha=None, beta=None, gamma=None, kw=25, rl_w=None,
                 gamma_decay = 0.99, 
                 gamma_decay_start = 150, low_rank_g =False,
                 use_fg=False, beta_fg=None, low_rank=True, use_lle=False, num_eig=100,
                   **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'deformable'
        self.alpha = 1 if alpha is None else alpha
        self.beta_ = 2 if beta is None else beta
        self.gamma = 1 if gamma is None else gamma
        self.tol =  1e-10 if self.tol is None else self.tol

        self.low_rank_g= low_rank_g
        if self.M > 5e5:
            self.low_rank_g = True
            print('Both low_rank and low_rank_g will be set to True for large datasets.')
    
        self.low_rank_ = ((low_rank or low_rank_g) and (self.gamma>0))
        self.num_eig_ = num_eig
        self.use_fg = use_fg
        self.beta_fg = beta_fg
        self.use_lle = use_lle and (self.gamma>0)
        self.use_mcl = (self.alpha>0)
        self.kw =kw
        self.W = np.zeros((self.M, self.D))
        
        self.normal_ = 'each' if self.normal_ is None else self.normal_
        self.init_normalize()
        self.init_L( kw=self.kw, rl_w=rl_w, method='sknn')
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

    def init_L(self, kw=15, rl_w=None, method='sknn'):
        # print('compute penalty matrix...')    
        if self.use_lle:
            print('compute Y lle...')
            L = lle_W(self.Y, kw=kw, rl_w=rl_w, method=method, eps=self.eps)
            M = L.transpose().dot(L)

        if self.low_rank is True:
            print('compute G eigendecomposition...')
            if self.low_rank_g:
                self.Q, self.S  = low_rank_eigen_grbf(self.Y, self.beta_, self.num_eig, sw_h=0)
                self.inv_S = np.diag(1./self.S)
                self.S = np.diag(self.S)
            else:
                G = kernel_xmm(self.Y, self.Y, sigma2=self.beta_, temp=1/2)[0]
                self.Q, self.S = low_rank_eigen(G, self.num_eig) #TODO
                self.inv_S = np.diag(1./self.S)
                self.S = np.diag(self.S)

            if self.use_lle:
                self.MY = M.dot(self.Y)
                try:
                    MG = M.dot(G)
                except:
                    MG = M.dot(self.Q).dot(self.S).dot(self.Q.T)

                if self.alpha == 0:
                    assert np.linalg.det(MG) >0
                    self.Fv = 1/self.gamma * np.linalg.inv(MG)
                else:
                    # F = (self.alpha * self.xp.eye(self.M) + self.gamma * MG)
                    # self.Fv = np.linalg.inv(F)
                    self.Fv = Woodbury(self.alpha * self.xp.eye(self.M),  
                                        M.dot(self.Q), self.gamma * self.S, self.Q.T)

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
        
            if self.use_lle:
                self.MG = M.dot(G)
                self.MY = M.dot(self.Y)

                if self.alpha == 0:
                    assert np.linalg.det(MG) >0
                    self.Fv = 1/self.gamma * np.linalg.inv(MG)
                else:
                    F = (self.alpha * self.xp.eye(self.M) + self.gamma * self.MG)
                    self.Fv = np.linalg.inv(F)

    def optimization(self):
        self.expectation()
        self.update_transform()
        self.update_transformer()
        self.transform_point()
        self.update_variance()
        self.iteration += 1

    def update_transform(self):
        self.use_lle = self.use_lle and (self.gamma>0)
        self.use_mcl = (self.alpha>0)

        if self.low_rank is False:
            if (self.use_lle):
                A = self.xp.multiply(self.P1, self.G).T + \
                    self.sigma2 * self.alpha * self.xp.eye(self.M) + \
                    self.sigma2 * self.gamma * self.MG
                B = self.PX - self.xp.multiply(self.Y.T, self.P1).T - self.sigma2*self.gamma*self.MY
                self.W = self.xp.linalg.solve(A, B)
            else:
                # A = self.xp.dot(self.xp.diag(self.P1), self.G) + \
                #     self.alpha * self.sigma2 * self.xp.eye(self.M)
                # B = self.PX - self.xp.dot(self.xp.diag(self.P1), self.Y)
                A = self.xp.multiply(self.P1, self.G).T + \
                    self.alpha * self.sigma2 * self.xp.eye(self.M)
                B = self.PX - self.xp.multiply(self.Y.T, self.P1).T

                self.W = self.xp.linalg.solve(A, B)

        elif self.low_rank is True:
            if (self.use_lle):
                # dPQ = np.multiply(self.Q.T, self.P1).T
                # Fvs2v = self.Fv / self.sigma2 
                # Uv = np.linalg.inv(self.inv_S + self.Q.T @ Fvs2v @ dPQ)
                # Av =  Fvs2v - Fvs2v @ dPQ @ Uv @ self.Q.T @ Fvs2v
                # B = self.PX - self.xp.multiply(self.Y.T, self.P1).T - self.sigma2*self.gamma*self.MY
                # self.W = Av @ B

                B = self.PX - self.xp.multiply(self.Y.T, self.P1).T - self.sigma2*self.gamma*self.MY
                dPQ = np.multiply(self.Q.T, self.P1).T
                Fvs2v = self.Fv / self.sigma2
                FB = Fvs2v @ B
                FQ = Fvs2v @ dPQ
                # Uv = np.linalg.inv(self.inv_S + self.Q.T @ FQ)
                # self.W = FB - (FQ @ Uv) @ (self.Q.T @ FB)
                self.W = FB - FQ @ np.linalg.solve(self.inv_S + self.Q.T @ FQ, self.Q.T @ FB)
    
            else:
                dPQ = np.multiply(self.Q.T, self.P1).T
                F = self.PX - np.multiply(self.Y.T, self.P1).T
                self.W = 1 / (self.alpha * self.sigma2) * (F - np.matmul(dPQ, 
                    (np.linalg.solve((self.alpha * self.sigma2 * self.inv_S + np.matmul(self.Q.T, dPQ)),
                                    (np.matmul(self.Q.T, F))))))

                # B = self.PX - self.xp.multiply(self.Y.T, self.P1).T
                # dPQ = np.multiply(self.Q.T, self.P1).T
                # Fvs2v = 1 / (self.alpha * self.sigma2)
                # FB = Fvs2v * B
                # FQ = Fvs2v * dPQ
                # Uv = np.linalg.solve(self.inv_S + self.Q.T @ FQ, self.Q.T @ FB)
                # # self.W = FB - (FQ @ Uv) @ (self.Q.T @ FB)
                # self.W = FB - FQ @ Uv
                # QtW = np.matmul(self.Q.T, self.W)
                # self.E = self.E + self.alpha / 2 * np.trace(np.matmul(QtW.T, np.matmul(self.S, QtW)))

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
