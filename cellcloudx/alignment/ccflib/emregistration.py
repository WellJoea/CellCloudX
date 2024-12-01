import numpy as np
import numbers
from warnings import warn
import scipy as sci
from scipy.sparse import issparse, csr_array, csc_array, diags, linalg
import scipy.sparse as ssp
from scipy.spatial import distance as scipy_distance
from tqdm import tqdm

from .xmm import kernel_xmm, kernel_xmm_k, kernel_xmm_p, low_rank_eigen
from .utility import sigma_square, centerlize, scale_array
from ...transform import homotransform_point, ccf_deformable_transform_point
from ...tools._neighbors import Neighbors

class EMRegistration(object):
    """
    Expectation maximization point cloud registration.

    Attributes
    ----------
    X: numpy array
        NxD array of target points.

    Y: numpy array
        MxD array of source points.

    TY: numpy array
        MxD array of transformed source points.

    sigma2: float (positive)
        Initial variance of the Gaussian mixture model.

    N: int
        Number of target points.

    M: int
        Number of source points.

    D: int
        Dimensionality of source and target points

    iteration: int
        The current iteration throughout registration.

    maxiter: int
        Registration will terminate once the algorithm has taken this
        many iterations.

    tol: float (positive)
        Registration will terminate once the difference between
        consecutive objective function values falls within this tolerance.

    w: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).

    q: float
        The objective function value that represents the misalignment between source
        and target point clouds.

    diff: float (positive)
        The absolute difference between the current and previous objective function values.
    """

    def __init__(self, X, Y, X_feat=None, Y_feat=None, 
                 sigma2=None, sigma2f=None, maxiter=None, 
                 tol=None, w=None,  use_dpca=False,
                 normal=None, 
                 feat_normal='l2', 
                 feat_model = 'gmm',
                 smm_dfs = 2,
                 get_P = True,
                 use_cuda=False, 
                 w_clip = None, #w_clip=[1e-4, 1-1e-3],
                 K=None, KF=None, p_epoch=None, spilled_sigma2=1,
                 kd_method='sknn', 
                 verbose = 1, 
                ):
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError(
                "The target point cloud (X) must be at a 2D numpy array.")

        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError(
                "The source point cloud (Y) must be a 2D numpy array.")

        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError(
                "The source point cloud (Y) must be a 2D numpy array.")
        
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                "Both point clouds need to have the same number of dimensions.")

        if sigma2 is not None and (not isinstance(sigma2, numbers.Number) or sigma2 <= 0):
            raise ValueError(
                "Expected a positive value for sigma2 instead got: {}".format(sigma2))

        if maxiter is not None and (not isinstance(maxiter, numbers.Number) or maxiter < 0):
            raise ValueError(
                "Expected a positive integer for maxiter instead got: {}".format(maxiter))
        elif isinstance(maxiter, numbers.Number) and not isinstance(maxiter, int):
            warn("Received a non-integer value for maxiter: {}. Casting to integer.".format(maxiter))
            maxiter = int(maxiter)

        if w is not None and (not isinstance(w, numbers.Number) or w < 0 or w >= 1):
            raise ValueError(
                "Expected a value between 0 (inclusive) and 1 (exclusive) for w instead got: {}".format(w))
        if  kd_method=='annoy':
            Warning('annoy is slower than other knn methods.')
        if use_cuda:
            import cupy as cp
            self.xp = cp
        else:
            self.xp = np
        self.reg_core = None
        self.eps = self.xp.finfo(self.xp.float64).eps
        self.verbose = verbose

        self.normal_ = normal
        self.X = self.xp.array(X, dtype=self.xp.float64).copy()
        self.Y = self.xp.array(Y, dtype=self.xp.float64).copy()

        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape

        self.tol = tol
        self.w = np.clip(((self.N-self.M)/self.N)**2**0.5, 1e-3, 1-1e-3) if w is None else w
        self.w_clip = w_clip
        self.maxiter = 300 if maxiter is None else maxiter
        self.iteration = 0

        self.use_dpca=use_dpca

        self.get_P = get_P
        self.diff = np.inf
        self.q = np.inf
        self.kd_method = kd_method
        self.K = K
        self.KF = KF
        self.p_epoch= p_epoch
        
        self.sigma2 = sigma2
        self.sigma2f = sigma2f
        self.spilled_sigma2 = spilled_sigma2

        self.fE = not (X_feat is None or Y_feat is None)
        self.feat_normal = feat_normal
        self.smm_dfs = smm_dfs
        self.feat_model = feat_model
        if self.fE:
            assert X_feat.shape[1] == Y_feat.shape[1]
            self.X_feat = self.xp.asarray(X_feat)
            self.Y_feat = self.xp.asarray(Y_feat)
            self.Df = self.X_feat.shape[1]
        self.homotransform_point = homotransform_point
        self.ccf_deformable_transform_point = ccf_deformable_transform_point

    def init_normalize(self):
        if self.normal_ in [True, 'each']:
            self.X, self.Xm_, self.Xs_, self.Xf_ = centerlize(self.X,Xm=None, Xs=None)
            self.Y, self.Ym_, self.Ys_, self.Yf_ = centerlize(self.Y,Xm=None, Xs=None)

        elif self.normal_ == 'global':
            XY = np.concatenate((self.X, self.Y), axis=0)
            XY, XYm, XYs, XYf = centerlize(XY, Xm=None, Xs=None)
            self.X, self.Y = XY[:self.N,:], XY[self.N:,:]
            self.Xm_, self.Ym_ = XYm, XYm
            self.Xs_, self.Ys_ = XYs, XYs
            self.Xf_, self.Yf_ = XYf, XYf

        elif self.normal_ == 'X':
            self.X, self.Xm_, self.Xs_, self.Xf_ = centerlize(self.X,Xm=None, Xs=None)
            self.Y, self.Ym_, self.Ys_, self.Yf_ = centerlize(self.Y,Xm=self.Xm_, Xs=self.Xs_)
        
        elif self.normal_ in [False, 'pass']:
            self.X, self.Xm_, self.Xs_, self.Xf_ = centerlize(self.X, Xm=np.zeros(self.D), Xs=1)
            self.Y, self.Ym_, self.Ys_, self.Yf_ = centerlize(self.Y ,Xm=np.zeros(self.D), Xs=1)
        else:
            raise ValueError(
                "Unknown normalization method: {}".format(self.normal_))
        if self.fE:
            if self.feat_normal == 'l2':
                l2x = self.xp.linalg.norm(self.X_feat, ord=None, axis=1, keepdims=True)
                l2y = self.xp.linalg.norm(self.Y_feat, ord=None, axis=1, keepdims=True)
                l2x[l2x == 0] = 1
                l2y[l2y == 0] = 1
                self.X_feat = self.X_feat/l2x
                self.Y_feat = self.Y_feat/l2y
            elif self.feat_normal == 'zc': #TODO
                # self.X_feat = scale_array(self.X_feat)[0]
                # self.Y_feat = scale_array(self.Y_feat)[0]
                self.X_feat = centerlize(self.X_feat)[0]
                self.Y_feat = centerlize(self.Y_feat)[0]
            elif self.feat_normal == 'pcc': #TODO
                self.X_feat = scale_array(self.X_feat, anis_var=True)[0]
                self.Y_feat = scale_array(self.Y_feat, anis_var=True)[0]
            else:
                warn(f"Unknown feature normalization method: {self.feat_normal}")

    def feature_P(self):
        # if self.sigma2f is None:
        #     sigma2f = sigma_square(self.X_feat, self.Y_feat,)
        if self.KF:
            self.Pf, self.gf, self.sigma2f = kernel_xmm_k( self.X_feat, self.Y_feat, 
                                                       knn=self.KF, 
                                                        # method=self.kd_method,
                                                        method='annoy',
                                                        dfs=self.smm_dfs, 
                                                        kernel=self.feat_model,
                                                        sigma2=self.sigma2f)
            K = self.KF

        else:
            self.Pf, self.gf, self.sigma2f = kernel_xmm(self.X_feat, self.Y_feat, 
                                                    dfs=self.smm_dfs, 
                                                    kernel=self.feat_model,
                                                    sigma2=self.sigma2f)
            K = self.M
            
        if self.reg_core in ['rigid', 'affine']:
            self.cdff = self.xp.sum(self.Pf, axis=0) / K
        else:
            self.cdff = self.xp.sum(self.Pf, axis=0) / K
        self.cf = 0

    def adjustable_paras(self, **kargs):
        for key, value in kargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)

    def echo_paras(self, paras=None):
        if paras is None:
            paras = ['sigma2', 'sigma2f', 'K', 'KF', 'alpha', 'beta',  'gamma',
                      'use_lle', 'kw', 'beta_fg', 'use_fg', 'normal', 'low_rank', 'low_rank_g', 
                      'w']
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

    def register(self, callback= None, **kwargs):
        self.adjustable_paras(**kwargs)
        self.sigma2 = sigma_square(self.X, self.Y)
        self.q = 1.0 + self.N * self.D * 0.5 * np.log(sigma_square(self.X, self.Y))
        if self.fE:
            self.feature_P()
        self.TY = self.Y.copy()
        self.echo_paras()

        if self.verbose:
            pbar = tqdm(range(self.maxiter), total=self.maxiter, colour='red', desc='ccd')
            for i in pbar:
                self.optimization()
                if callable(callback):
                    kwargs = {'iteration': self.iteration,
                            'error': self.q, 'X': self.X, 'Y': self.TY}
                    callback(**kwargs)
                log_prt = {
                        'tol': f'{self.diff :.4e}', 
                        'Q': f'{self.q :.5e}', 
                            #'w': f'{self.w :.3e}',
                        # 'sigma2f': f'{self.sigma2f :.4e}',
                        'sigma2': f'{self.sigma2 :.4e}'}
                
                if not self.w_clip is None:
                    log_prt['w'] = f'{self.w :.3e}'

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
                if self.diff <= self.tol:
                    break
        if self.get_P:
           self.P, self.C = self.update_P(self.X, self.TY)
        self.update_normalize()

    def expectation(self):
        P, C = self.update_P(self.X, self.TY)
        P = P / C
        # P = (P * self.OM) / (C*(1-self.w))
        # P = P.tocsc()[:, self.inlier]
        # self.Pt1 = self.Pt1[self.inlier]
        self.tX = self.X #[self.inlier,:]

        self.Pt1 = self.xp.sum(P, axis=0)
        self.P1 = self.xp.sum(P, axis=1)
        self.Np = self.xp.sum(self.P1)
        self.PX = P.dot(self.tX)

        # if not self.reg_core in ['rigid', 'affine']:
        #     print(A1, B1, self.Np, self.N, self.M, 2222)
        #     import matplotlib.pyplot as plt
        #     fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        #     axs[0,0].hist(self.Pt1.flatten(), bins=100)
        #     axs[0,0].set_yscale('log')
        #     axs[0,1].hist(self.P1.flatten(), bins=100)
        #     axs[0,1].set_yscale('log')
        #     axs[1,0].hist(self.Pf.flatten(), bins=100)
        #     axs[1,0].set_yscale('log')
        #     axs[1,1].hist(self.cdff.flatten(), bins=100)
        #     axs[1,1].set_yscale('log')

        #     plt.show()

        #     Q = -np.sum(np.log(C)) + self.D*self.N*np.log(self.sigma2)/2
        #     M = np.trace(self.W.T @ self.G @ self.W)
        #     M1 = np.trace(self.W.T @ self.Q @ self.S @ self.Q.T @ self.W)
        #     print(Q, M, M1)

    def update_P(self, X, Y):
        if (not self.KF is None) and (not self.K is None): # PASS
            P, gs, self.sigma2 = kernel_xmm_k(X, Y, knn=self.K, 
                                               method=self.kd_method, 
                                               sigma2=self.sigma2,
                                               kernel='gmm',)
            if self.fE:
                self.Pf, self.gf, self.sigma2f = kernel_xmm_p(self.Y_feat, self.X_feat, 
                                        pairs=P.nonzero(), 
                                        dfs=self.smm_dfs, 
                                        kernel=self.feat_model,
                                        sigma2=None)
                Pe = P.astype('bool')*self.Pf
                Pa = Pe.sum(0)
                self.OM = Pe.astype('bool').sum(0)

        elif not self.KF is None: # Fast
            P, gs, self.sigma2 = kernel_xmm_p(Y, X, 
                                    pairs=self.Pf.nonzero(), 
                                    sigma2=self.sigma2,
                                    kernel='gmm',)
            self.OM = self.KF
            if self.fE:
                Pa = self.cdff
        elif not self.K is None: # Local bettwer
            P, gs, self.sigma2 = kernel_xmm_k(X, Y, knn=self.K, 
                                               method=self.kd_method, 
                                               sigma2=self.sigma2,
                                               kernel='gmm',)
            self.OM = self.K
            if self.fE:
                # self.Pf, self.gf, self.sigma2f = kernel_xmm_p(self.Y_feat, self.X_feat, 
                #                         pairs=P.nonzero(), 
                #                         dfs=self.smm_dfs, 
                #                         kernel=self.feat_model,
                #                         sigma2=None)
                # Pa = self.Pf.sum(0) # sigma2f change slightly
                # Pe = P.astype('bool')*self.Pf
                # Pa = Pe.sum(0)
                Pa = self.cdff
        
        else: # Best slow
            P, gs, self.sigma2 = kernel_xmm(X, Y, sigma2=self.sigma2, kernel='gmm',)
            self.OM = self.M
            if self.fE:
                Pa = self.cdff

        # print(gs, 0000, self.gf)
        gs = self.M /(1. - self.w)/gs
        cs = self.w/self.N
        cdfs = self.xp.sum(P, axis=0)

        # self.wn = int((1-self.w) * self.N)
        # self.inlier = np.argpartition(1.0 - np.divide(1, cdfs+gs*cs), -self.wn)[-self.wn:]
        # self.inlier = np.argpartition(cdfs, -self.wn)[-self.wn:]
        # if not self.w_clip is None:
        #     self.w = np.clip(1-self.Np/self.N, *self.w_clip)
        # inlier_prob = 1.0 - np.divide(1, cdfs+gs*cs)
        # P = P * inlier_prob
        # print( gs * cs, cdfs.sum(), P.sum(), 111)
        if self.fE:
            cdfs = self.xp.multiply(cdfs, Pa)
            gs = gs / self.gf
            cs = cs + self.cf*self.w
            P = P * self.Pf

        C = cdfs + max(gs * cs , self.eps)
        # print( gs * cs, cdfs.sum(), self.cdff.sum(), self.Pf.sum(),  P.sum(), 222222)
        return P, C

    def update_P1(self, X, Y):
        if not self.KF is None:
            P, gs, self.sigma2 = kernel_xmm_p(Y, X, 
                                    pairs=self.Pf.nonzero(), 
                                    sigma2=self.sigma2,
                                    kernel='gmm',)
            self.OM = self.KF
        elif not self.K is None:
            P, gs, self.sigma2 = kernel_xmm_k(X, Y, knn=self.K, 
                                               method=self.kd_method, 
                                               sigma2=self.sigma2,
                                               kernel='gmm',)
            self.OM = self.K
        else:
            P, gs, self.sigma2 = kernel_xmm(X, Y, sigma2=self.sigma2, kernel='gmm',)
            self.OM = self.M


        gs = self.OM /(1. - self.w)/gs
        cs = self.w/self.N

        if self.fE:
            gs = gs / self.gf
            P = P * self.Pf

        cdfs = self.xp.sum(P, axis=0)
        C = cdfs + gs * cs 
        return P, C
    
    def update_P0(self, X, Y): # TODO
        if not self.KF is None:
            P, gs, self.sigma2 = kernel_xmm_p(Y, X, 
                                    pairs=self.Pf.nonzero(), 
                                    sigma2=self.sigma2,
                                    kernel='gmm',)
            self.OM = self.KF
            if self.fE:
                Pa = self.cdff
        elif not self.K is None:
            P, gs, self.sigma2 = kernel_xmm_k(X, Y, knn=self.K, 
                                               method=self.kd_method, 
                                               sigma2=self.sigma2,
                                               kernel='gmm',)
            self.OM = self.K
            if self.fE:
                Pe = P.astype('bool')*self.Pf
                Pa = Pe.sum(0)
        else:
            P, gs, self.sigma2 = kernel_xmm(X, Y, sigma2=self.sigma2, kernel='gmm',)
            self.OM = self.M
            if self.fE:
                Pa = self.cdff

        P = P * self.Pf /Pa
        cs = self.w/(self.N*gs*(1-self.w))
        cdfs = self.xp.sum(P, axis=0)
        C = cdfs + cs #* Pa
        return P, C

    def optimization(self):
        raise NotImplementedError(
            "optimization should be defined in child classes.")

    def update_normalize(self):
        raise NotImplementedError(
            "update_normalize should be defined in child classes.")

    def transform_point(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the source point cloud should be defined in child classes.")

    @property
    def normal(self):
        return self.normal_
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
