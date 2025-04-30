from scipy.sparse import issparse, csr_array, csc_array, diags, eye
from scipy.spatial import distance as scipy_distance
import scipy.sparse as ssp
from tqdm import tqdm
import numpy as np
import sys
from warnings import warn

from ...tools._neighbors import Neighbors, mtx_similarity
from .xmm import kernel_xmm, lle_W, kernel_xmm_k, kernel_xmm_p
from .utility import centerlize
from ._sswnn import sswnn_pair
from ...transform import homotransform_point, ccf_deformable_transform_point
from ...tools._neighbors import Neighbors, mtx_similarity

class EMRegistration(object):
    def __init__(self, X, Y, pairs=None, X_feat=None, Y_feat=None, labels=None,
                 gamma=0.9, gamma_clip=[1e-8, 0.95], maxiter=500, alpha=9e9,
                 theta=0.75, tol=1e-5, tol_r=1e-5, tol_s=1e-8,
                 minp=1e-5, rl_w=1e-5, unif=10, kw=15, normal=True,
                 feat_normal='l2', 
                 sigma2=None, sigma2f=None, verbose=1,
                 use_cuda=False, p_epoch=None, **kwargs):

        self.use_cuda = use_cuda
        if self.use_cuda:
            import cupy as cp
            self.xp = cp
            self.float = cp.float32
        else:
            self.xp = np
            self.float = np.float64
        self.eps = self.xp.finfo(self.xp.float64).eps
        self.reg_core = None
        self.maxiter = maxiter
        self.tol = tol
        self.tol_r = tol_r
        self.tol_s = tol_s

        self.alpha = alpha
        self.gamma = gamma
        self.gamma_clip = gamma_clip

        self.theta = theta

        self.sigma2 = sigma2
        self.sigma2f = sigma2f

        self.minp = minp
        self.unif = unif
        self.kw = kw
        self.rl_w = rl_w if self.kw > self.D else 0
        self.p_epoch = p_epoch
        self.iteration = 0
        self.normal = normal
        self.feat_normal = feat_normal
        self.verbose=verbose

        self.homotransform_point = homotransform_point
        self.ccf_deformable_transform_point=ccf_deformable_transform_point
        self.labels = labels
        self.D = X.shape[1]
        self.pairs = self.init_paris(X, Y, X_feat, Y_feat, pairs)
        self.init_XY( X, Y, X_feat, Y_feat)
        self.normalXY()
        self.normalXYfeatures()
    
    @property
    def Xm(self):
        return self.Xm_
    @property
    def Ym(self):
        return self.Ym_
    @property
    def Xs(self):
        return self.Xs_
    @property
    def Ys(self):
        return self.Ys_
    @property
    def Xf(self):
        return self.Xf_
    @property
    def Yf(self):
        return self.Yf_

    def init_paris(self, X, Y, X_feat, Y_feat, pairs, 
                    **kargs):
        fE = not ((X_feat is None) or (Y_feat is None))
        dargs = dict(
            kd_method='annoy', sp_method = 'sknn',
            use_dpca = False, dpca_npca = 60,
            m_neighbor=15, e_neighbor =30, s_neighbor =30,
            o_neighbor = None, 
            score_threshold = 0.4, max_pairs = None,
            lower = 0.01, upper=0.995,
        )
        dargs.update(kargs)

        if fE and (not pairs is None): # TODO
            print('coming soon...')
            raise NotImplementedError
        elif fE:
            pairs, mscore = sswnn_pair(X, Y, X_feat, Y_feat, **dargs)
        self.pairs = pairs

    def init_XY(self, X, Y, X_feat, Y_feat): #TODO
        assert X.shape[0] == Y.shape[0]
        self.fexist = not (X_feat is None or Y_feat is None)

        if self.pairs is None:
            assert X.shape == Y.shape
            self.uX, self.X_idx = np.unique(X, return_inverse=True)
            self.uY, self.Y_idx = np.unique(Y, return_inverse=True)

            self.uX = self.uX.astype(np.float64).copy()
            self.uY = self.uY.astype(np.float64).copy()
            self.X = X.astype(np.float64).copy()
            self.Y = Y.astype(np.float64).copy()

            self.pairs = np.vstack((self.X_idx, self.Y_idx)).T
        else:
            self.uX = X.astype(np.float64).copy()
            self.uY = Y.astype(np.float64).copy()
            self.X_idx = self.pairs[:, 0].copy()
            self.Y_idx = self.pairs[:, 1].copy()
            self.X = self.uX[self.X_idx].copy()
            self.Y = self.uY[self.Y_idx].copy()

            if self.fexist: #TODO
                assert X_feat.shape == Y_feat.shape
                assert X_feat.shape[0] == X.shape[0] 
                self.uXf = X_feat.astype(np.float64).copy()
                self.uYf = Y_feat.astype(np.float64).copy()

        self.N = self.X.shape[0]
        self.uN = self.uX.shape[0]
        assert self.X_idx.shape[0] == self.N

    def normalXY(self):
        if self.normal in [True, 'each']:
            _, self.Xm_, self.Xs_, self.Xf_ = centerlize(self.uX, Xm=None, Xs=None)
            _, self.Ym_, self.Ys_, self.Yf_ = centerlize(self.uY, Xm=None, Xs=None)

        elif self.normal == 'global':
            XY = np.concatenate((self.uX, self.uY), axis=0)
            _, XYm, XYs, XYf = centerlize(XY, Xm=None, Xs=None)

            self.Xm_, self.Ym_ = XYm, XYm
            self.Xs_, self.Ys_ = XYs, XYs
            self.Xf_, self.Yf_ = XYf, XYf

        elif self.normal == 'X':
            _, self.Xm_, self.Xs_, self.Xf_ = centerlize(self.uX, Xm=None, Xs=None)
            self.Ym_, self.Ys_, self.Yf_ = self.Xm_, self.Xs_, self.Xf_
        
        elif self.normal in [False, 'pass']:
            self.Xm_, self.Xs_, self.Xf_ = np.zeros(self.D), 1, np.eye(self.D+1)
            self.Ym_, self.Ys_, self.Yf_ = self.Xm_, self.Xs_, self.Xf_
        else:
            raise ValueError(
                "Unknown normalization method: {}".format(self.normal))

        self.X = centerlize(self.X, Xm=self.Xm_, Xs=self.Xs_)[0]
        self.Y = centerlize(self.Y, Xm=self.Ym_, Xs=self.Ys_)[0]
        self.TY = self.Y.copy()

    def normalXYfeatures(self): # TODO
        if self.fexist:
            if self.feat_normal == 'l2':
                l2x = self.xp.linalg.norm(self.uXf, ord=None, axis=1, keepdims=True)
                l2y = self.xp.linalg.norm(self.uYf, ord=None, axis=1, keepdims=True)
                l2x[l2x == 0] = 1
                l2y[l2y == 0] = 1
                self.uXf = self.uXf/l2x
                self.uYf = self.uYf/l2y
            elif self.feat_normal == 'zc': #TODO
                self.uXf = centerlize(self.uXf)[0]
                self.uYf = centerlize(self.uYf)[0]
            elif self.feat_normal == 'pcc': #TODO
                self.uXf = self.scale_array(self.uXf, anis_var=True)[0]
                self.uYf = self.scale_array(self.uYf, anis_var=True)[0]
            else:
                warn(f"Unknown feature normalization method: {self.feat_normal}")

    def adjustable_paras(self, **kargs):
        for key, value in kargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)
    def echo_paras(self, paras=None):
        if paras is None:
            paras = ['sigma2', 'sigma2f', 'K', 'KF', 'alpha', 'beta',  'gamma',
                      'use_lle', 'kw', 'beta_fg', 'use_fg', 'normal', 'low_rank','w']
        print('init parameters:')
        for ipara in paras:
            if hasattr(self, ipara):
                ivalue = getattr(self, ipara)
                try:
                    if ((type(ivalue) in [float]) or (np.issubdtype(ivalue, np.floating))):
                        print(f'{ipara} = {ivalue:.4e}')   
                    else:
                        print(f'{ipara} = {ivalue}')
                except:
                    print(f'{ipara} = {ivalue}')    

    def register(self, callback=lambda **kwargs: None):
        if self.fE:
            self.feature_P()
        if self.sigma2 is None:
            self.sigma2 = np.sum(np.square(self.X - self.Y)) / (self.N * self.D)
        self.Q = 1.0 + self.N * self.D * 0.5 * np.log(self.sigma2)

        pbar = tqdm(range(self.maxiter), total=self.maxiter, colour='red', desc='iterations')
        for char in pbar:
            self.optimization()
            if callable(callback):
                kwargs = {'iteration': self.iteration,
                          'error': self.Q, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)
            log_prt = {'iter': self.iteration,
                       'tol': f'{self.diff :.6e}',
                       'tol_r': f'{self.diff_r :.6e}',
                       'Q': f'{self.Q :.6f}',
                       'QZ': f'{self.QZ :.6f}',
                       'sigma2': f'{self.sigma2 :.4f}'}
            pbar.set_postfix(log_prt)
            if (self.p_epoch) and (self.iteration % self.p_epoch == 0):
                print("\033[K")
            if (self.diff <= self.tol) or (self.sigma2 <= self.tol_s and self.diff_r <= self.tol_r):
                print(f'Tolerance is lower than {self.tol}. Program ends early.')
                break
            # break
        pbar.close()
        if self.normal:
            self.update_normalize()

    def register(self, callback= None, **kwargs):
        self.adjustable_paras(**kwargs)
        if self.sigma2 is None:
            self.sigma2 = np.sum(np.square(self.X - self.Y)) / (self.N * self.D)
        self.Q = 1.0 + self.N * self.D * 0.5 * np.log(self.sigma2)

        if self.fE:
            self.feature_P()
        self.echo_paras()

        if self.verbose:
            pbar = tqdm(range(self.maxiter), total=self.maxiter, colour='red', desc='fllt')
            for i in pbar:
                self.optimization()
                if callable(callback):
                    kwargs = {'iteration': self.iteration,
                            'error': self.q, 'X': self.X, 'Y': self.TY}
                    callback(**kwargs)
                log_prt = {'iter': self.iteration,
                            'tol': f'{self.diff :.6e}',
                            'tol_r': f'{self.diff_r :.6e}',
                            'Q': f'{self.Q :.6f}',
                            'QZ': f'{self.QZ :.6f}',
                            'sigma2': f'{self.sigma2 :.4f}'}

                pbar.set_postfix(log_prt)
                if (self.p_epoch):
                    pbar.update(self.p_epoch)
                if self.diff <= self.tol:
                    print(f'Tolerance is lower than {self.tol}. Program ends early.')
                    break
            pbar.close()
        else:
            for i in range(self.maxiter):
                self.optimization()
                if self.diff <= self.tol: break
        self.update_normalize()

    def optimization(self):
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        dist2 = np.sum(np.square(self.X - self.TY), axis=1)
        P = self.xp.exp(-dist2 / (2 * self.sigma2))
        gs = ((2 * self.xp.pi * self.sigma2) ** (self.D / 2)) / self.gamma
        cs = (1 - self.gamma) / self.unif

        if self.fE:
            P = P * self.Pf
            gs = gs * self.gf
            cs = cs + self.cf
        C = gs * cs
        P = P / (P + C)  # N*1
        P = ssp.diags(P, shape=(P.shape[0], P.shape[0]))
        self.P = P
        # self.Pt1 = self.xp.sum(P, axis=0)
        # self.P1 = self.xp.sum(P, axis=1)
        self.Pt1 = self.xp.array(self.P.sum(0)).squeeze()
        self.P1 = self.xp.array(self.P.sum(1)).squeeze()
        self.Np = self.xp.sum(self.P1)
        self.preQ = self.Q

        self.Q = np.sum(self.P.diagonal() * dist2) / (2 * self.sigma2) + \
                 self.Np * self.D * np.log(self.sigma2) / 2 + \
                 - self.Np * np.log(self.gamma) - np.sum(1 - self.Pt1) * np.log(1 - self.gamma)
    
    def feature_P(self): #TODO
        # if self.sigma2f is None:
        #     sigma2f = self.sigma_square(self.X_feat, self.Y_feat,)

        self.Pf, self.gf, self.sigma2f = kernel_xmm(self.X_feat, self.Y_feat, 
                                                dfs=self.smm_dfs, 
                                                kernel=self.feat_model,
                                                sigma2=self.sigma2f)
        K = self.uN
            
        if self.reg_core in ['rigid', 'affine']:
            self.cdff = self.xp.sum(self.Pf, axis=0) / K
        else:
            self.cdff = self.xp.sum(self.Pf, axis=0) / K
        self.cf = 0

    def lle(self, use_unique=True, method='sknn'):
        if use_unique:
            self.LW = lle_W(self.uY, Y_index=self.Y_idx, kw=self.kw,  method=method)
        else:
            self.LW = lle_W(self.Y, kw=self.kw, method='sknn')

    def maximization(self):
        raise NotImplementedError(
            "maximization should be defined in child classes.")

    def update_normalize(self):
        raise NotImplementedError(
            "update_normalize should be defined in child classes.")

    def transform_point(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the source point cloud should be defined in child classes.")

class rigid_reg(EMRegistration):
    def __init__(self, *args, R=None, t=None, s=None, scale=True,
                 tform=None, tforminv=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'rigid'
        self.R = np.eye(self.D) if R is None else R
        self.t = np.zeros(self.D) if t is None else t
        self.s = 1 if s is None else s
        self.scale = scale
        self.update_transformer(tform=tform, tforminv=tforminv)
        if self.scale:
            self.lle(use_unique=True,  method='sknn')

    def maximization(self):
        PX = self.P.dot(self.X)
        PY = self.P.dot(self.Y)
        muX = self.xp.divide(self.xp.sum(PX, axis=0), self.Np)
        muY = self.xp.divide(self.xp.sum(PY, axis=0), self.Np)
        # muY = self.xp.divide(self.xp.dot(self.xp.transpose(self.Y), self.P1), self.Np)

        X_hat = self.X - muX
        Y_hat = self.Y - muY

        # self.A = np.dot(np.transpose(self.P.dot(X_hat)), Y_hat)
        # self.A = self.X.T @ PY - ( muX[:,None] @ muY[:,None].T ) * self.Np
        self.A = self.xp.dot(PX.T, Y_hat) - self.xp.outer(muX, self.xp.dot(self.P1.T, Y_hat))

        U, S, Vh = self.xp.linalg.svd(self.A, full_matrices=True)
        S = self.xp.diag(S)
        C = self.xp.eye(self.D)
        C[-1, -1] = self.xp.linalg.det(self.xp.dot(U, Vh))
        self.R = self.xp.dot(self.xp.dot(U, C), Vh)

        # trAR = self.xp.trace(self.xp.dot(self.A.T, self.R))
        # trYPY = np.trace(Y_hat.T @ np.diag(self.P1) @ Y_hat)
        # trYPY = np.sum(np.multiply(self.Y.T**2, self.P1)) - self.Np*(muY.T @ muY)
        self.trAR = np.trace(S @ C)
        self.trXPX = np.sum(self.Pt1.T * np.sum(np.multiply(X_hat, X_hat), axis=1))
        self.trYPY = np.sum(self.P1.T * np.sum(np.multiply(Y_hat, Y_hat), axis=1))

        ## LLE
        if self.scale is True:
            Z = self.LW.transpose().dot(self.P.dot(self.LW))
            YtZY = self.Y.T @ Z.dot(self.Y)
            self.s = self.trAR / (self.trYPY + 2 * self.alpha * self.sigma2 * np.trace(YtZY))
            self.QZ = self.alpha * self.xp.linalg.norm(self.s * np.sqrt(self.P) @ self.llW @ self.Y)
        else:
            self.QZ = 0

        self.t = muX - self.s * self.xp.dot(self.R, muY)
        self.Q += self.QZ
        self.diff = np.abs(self.Q - self.preQ)
        self.diff_r = np.abs(self.diff / self.Q)

        self.update_transformer()
        self.transform_point()
        V = np.square(self.X - self.TY)
        self.sigma2 = np.sum(V * self.P1[:, None]) / (self.D * self.Np)

        self.Ind = (self.P.diagonal() > self.theta)
        self.gamma = np.clip(self.Ind.sum() / self.N, *self.gamma_clip)

    def update_transformer(self, tmat=None, tmatinv=None):
        if not tmatinv is None:
            tmat = np.linalg.inv(np.float64(tmatinv))
        if not tmat is None:
            self.tmat = np.float64(tmat)
            self.tmatinv = np.linalg.inv(self.tmat)
    
            #TODO
            B = self.tmat[:-1, :-1]
            self.s = np.linalg.det(B)**(1/(B.shape[0])) 
            self.R = B/self.s
            self.t = self.tmat[:-1, [-1]].T

        else:
            self.tmat = np.eye(self.D+1, dtype=np.float64)
            self.tmat[:self.D, :self.D] = self.R * self.s
            self.tmat[:self.D,  self.D] = self.t
            self.tmatinv = np.linalg.inv(self.tmat)
            self.tform = self.tmat
    
    def transform_point(self, Y=None):
        if Y is None:
            self.TY = self.homotransform_point(self.Y, self.tmat, inverse=False)
            return
        else:
            return self.homotransform_point(Y, self.tmat, inverse=False)

    def update_normalize(self):
        self.s *= self.Xs/self.Ys 
        self.t = (self.t * self.Xs + self.Xm) - self.s * self.R @ self.Ym.T
        self.tform = self.Xf @ self.tmat @ np.linalg.inv(self.Yf)
        self.tforminv = np.linalg.inv(self.tform)
        self.update_transformer()
        self.TY = self.TY * self.Xs + self.Xm

    def get_transformer(self):
        return {'tform': self.tform, 's': self.s, 'R': self.R, 't':self.t }
    
class affine_reg(EMRegistration):
    def __init__(self, *args, B=None, t=None,
                 tform=None, tforminv=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.B = np.eye(self.D) if B is None else B
        self.t = np.zeros(self.D) if t is None else t

        self.update_transformer(tform=tform, tforminv=tforminv)
        self.lle(use_unique=True,  method='sknn')

    def maximization(self):
        PX = self.P.dot(self.X)
        PY = self.P.dot(self.Y)
        muX = self.xp.divide(self.xp.sum(PX, axis=0), self.Np)
        muY = self.xp.divide(self.xp.sum(PY, axis=0), self.Np)
        # muY = self.xp.divide(self.xp.dot(self.xp.transpose(self.Y), self.P1), self.Np)

        X_hat = self.X - muX
        Y_hat = self.Y - muY

        # self.A = np.dot(np.transpose(self.P.dot(X_hat)), Y_hat)
        self.A = self.X.T @ PY - (muX[:, None] @ muY[:, None].T) * self.Np
        # A = self.xp.dot(PX.T, Y_hat) - self.xp.outer(muX, self.xp.dot(self.P1.T, Y_hat))
        YPY = self.xp.dot(self.xp.dot(Y_hat.T, self.xp.diag(self.P1)), Y_hat)

        ## LLE
        Z = self.LW.transpose().dot(self.P.dot(self.LW))
        YtZY = self.Y.T @ Z.dot(self.Y)
        YtZY *= (2 * self.alpha * self.sigma2)

        self.B = self.xp.dot(self.A, self.xp.linalg.inv(YPY + YtZY))
        self.t = muX - self.xp.dot(self.B, muY)

        self.QZ = self.alpha * self.xp.linalg.norm(self.B @ self.Y.T @ np.sqrt(self.P) @ self.LW.T)
        self.Q += self.QZ
        self.diff = np.abs(self.Q - self.preQ)
        self.diff_r = np.abs(self.diff / self.Q)

        self.update_transformer()
        self.transform_point()
        V = np.square(self.X - self.TY)
        self.sigma2 = np.sum(V * self.P1[:, None]) / (self.D * self.Np)

        self.Ind = (self.P.diagonal() > self.theta)
        self.gamma = np.clip(self.Ind.sum() / self.N, *self.gamma_clip)

    def transform_point(self, Y=None):
        if Y is None:
            self.TY = self.homotransform_point(self.Y, self.tmat, inverse=False)
        else:
            return self.homotransform_point(Y, self.tmat, inverse=False)

    def update_transformer(self, tmat=None, tmatinv=None):
        if not tmatinv is None:
            tmat = self.xp.linalg.inv(self.xp.float64(tmatinv))
        if not tmat is None:
            self.tmat = self.xp.float64(tmat)
            self.tmatinv = self.xp.linalg.inv(self.tmat)
    
            #TODO
            self.B = self.tmat[:-1, :-1]
            self.t = self.tmat[:-1, [-1]].T
        else:
            self.tmat = self.xp.eye(self.D+1, dtype=self.xp.float64)
            self.tmat[:self.D, :self.D] = self.B
            self.tmat[:self.D, self.D] = self.t
            self.tmatinv = self.xp.linalg.inv(self.tmat)
            self.tform = self.tmat

    def update_normalize(self):
        self.B *= (self.Xs/self.Ys)
        self.t = (self.t * self.Xs + self.Xm) - self.B @ self.Ym.T
        self.tform = self.Xf @ self.tmat @ np.linalg.inv(self.Yf)
        self.tforminv = np.linalg.inv(self.tform)
        self.update_transformer()
        self.TY = self.TY * self.Xs + self.Xm

    def get_transformer(self):
        return { 'tform': self.tform, 'B': self.B, 't': self.t }

class deformable_reg(EMRegistration):
    def __init__(self, *args, seed=200504,
                 low_k= None, beta=1,
                 low_rank=False, num_eig=100, **kwargs):
        super().__init__(*args, **kwargs)

        self.low_rank = low_rank
        self.num_eig = num_eig
        self.beta = beta
        self.low_k = low_k
        self.seed = seed

        self.lle(use_unique=True, method='sknn')
        #self.kernal_K()

    def kernal_K(self,  low_k=None,):
        np.random.seed(self.seed)
        uniY = np.unique(self.Y, axis=0)
        if low_k is None:
            low_k = int(max([uniY.shape[0]**.35, 200]))

        ridx = np.random.choice(uniY.shape[0], low_k)
        # ridx = [65,857,894,570,904,1113,507,78,257,632,1135,1052,307,1154,930]
        # ridx = np.int64(ridx) -1
        ctrl_pts = uniY[ridx]
        self.K = kernel_xmm(ctrl_pts, ctrl_pts, sigma2=self.beta)[0]  # k*K
        self.U = kernel_xmm(self.Y, ctrl_pts, sigma2=self.beta)[0]  # M*k
        self.C = np.zeros((self.low_k, self.D))  # k * D

    def maximization(self):
        Z = self.LW.transpose().dot(self.P.dot(self.LW))
        EtP = (self.P.transpose().dot(self.U)).transpose()
        EtZ = self.alpha * self.sigma2 * ((Z.transpose().dot(self.U)).transpose())
        PQT = EtP @ self.U + EtZ @ self.U
        PYX = EtP @ self.X - EtP @ self.Y - EtZ @ self.Y
        self.C = np.linalg.solve(PQT, PYX)
        self.QZ = self.alpha / 2 * self.xp.linalg.norm(np.sqrt(self.P).dot(self.LW) @ self.TY)
        self.update_transformer()
        self.transform_point()

        self.Q += self.QZ
        self.diff = np.abs(self.Q - self.preQ)
        self.diff_r = np.abs(self.diff / self.Q)

        V = np.square(self.X - self.TY)
        self.sigma2 = np.sum(V * self.P1[:, None]) / (self.D * self.Np)

        self.Ind = (self.P.diagonal() > self.theta)
        self.gamma = np.clip(self.Ind.sum() / self.M, *self.gamma_clip)

    def update_transformer(self):
        if self.low_rank is False:
            self.tmat = self.xp.dot(self.U, self.C) 

        elif self.low_rank is True:
            self.tmat = np.matmul(self.Q, np.matmul(self.S, np.matmul(self.Q.T, self.W)))

    def transform_point(self, Y=None ): #TODO
        if Y is None:
            self.TY = self.Y + self.tmat
        else:
            return self.ccf_deformable_transform_point(
                        Y, Y=self.Y, Ym=self.Ym, Ys=self.Ys, 
                        Xm=self.Xm, Xs=self.Xs, beta=self.beta, 
                        G=self.U, W=self.C, Q=self.Q, S=self.S)
            # Y_t = (Y -self.Ym)/self.Ys
            # if reset_G or (not np.array_equal(self.Y, Y.astype(self.Y.dtype))):
            #     G = self.kernal_gmm(Y_t, self.Y, sigma2=self.beta)[0]
            #     tmat = np.dot(G, self.W)
            # else:
            #     tmat = self.tmat
            # Y_n = (Y_t + tmat)* self.Xs + self.Xm
            # return Y_n

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