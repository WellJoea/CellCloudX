import numpy as np
import numbers
from warnings import warn
from scipy.sparse import issparse, csr_array, csc_array, diags, linalg
import scipy.sparse as ssp
from scipy.spatial import distance as scipy_distance
from tqdm import tqdm

from .xmm import kernel_xmm, kernel_xmm_k, kernel_xmm_p
from ...tools._decomposition import dualPCA
from ...transform import homotransform_point, ccf_deformable_transform_point
from ...tools._neighbors import Neighbors, mtx_similarity
from ._sswnn import sswnn_pair

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
                 n_comps=60, get_P = True,
                 use_cuda=False, 
                 w_clip = None, #w_clip=[1e-4, 1-1e-3],
                 K=None, KF=None, p_epoch=None, spilled_sigma2=1,
                 kd_method='sknn', 
                 verbose = 1, 
                 **kwargs):
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
        if use_cuda:
            import cupy as cp
            self.xp = cp
        else:
            self.xp = np
        self.eps = self.xp.finfo(self.xp.float64).eps
        self.verbose = verbose
    
        self.normal_ = normal
        self.X = self.xp.array(X, dtype=self.xp.float64).copy()
        self.Y = self.xp.array(Y, dtype=self.xp.float64).copy()

        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape

        self.tol =  1e-8 if tol is None else tol
        self.w = 0.5 if w is None else w
        self.w_clip = w_clip
        self.maxiter = 300 if maxiter is None else maxiter
        self.iteration = 0

        self.use_dpca=use_dpca
        self.n_comps=n_comps 

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

    def pairs_match(self, sparse_sswnn = None, kd_method='annoy', 
                    sp_method = 'sknn', match_pairs=None,
                    m_neighbor=6, e_neighbor =30, s_neighbor =30,
                    o_neighbor = 30,
                    score_threshold = 0.3, **kargs):
        if sparse_sswnn is None:
            sparse_sswnn = sswnn_pair(
                self.X, self.Y, self.X_feat, self.Y_feat,
                kd_method=kd_method, sp_method = sp_method,

                m_neighbor=m_neighbor, e_neighbor =e_neighbor, 
                s_neighbor =s_neighbor, o_neighbor = o_neighbor,
                drawmatch=False, score_threshold=score_threshold, **kargs)
    
        mridx, mqidx = sparse_sswnn.nonzero()
        mscore = sparse_sswnn.data
        t_idx = (mscore>=score_threshold)
        mridx = mridx[t_idx]
        mqidx = mqidx[t_idx]
        mscore = mscore[t_idx]
        print(f'sswnn pairs: {mscore.shape[0]}...')
        if not match_pairs is None:
            match_pairs = min(match_pairs, len(mscore))
            ind = np.argpartition(mscore, -match_pairs)[-match_pairs:]
            mscore = mscore[ind]
            mridx = mridx[ind]
            mqidx = mqidx[ind]
        self.Xidx = mridx
        self.Yidx = mqidx

    def init_normalize(self):
        if self.normal_ in [True, 'each']:
            self.X, self.Xm_, self.Xs_, self.Xf_ = self.centerlize(self.X,Xm=None, Xs=None)
            self.Y, self.Ym_, self.Ys_, self.Yf_ = self.centerlize(self.Y,Xm=None, Xs=None)

        elif self.normal_ == 'global':
            XY = np.concatenate((self.X, self.Y), axis=0)
            XY, XYm, XYs, XYf = self.centerlize(XY, Xm=None, Xs=None)
            self.X, self.Y = XY[:self.N,:], XY[self.N:,:]
            self.Xm_, self.Ym_ = XYm, XYm
            self.Xs_, self.Ys_ = XYs, XYs
            self.Xf_, self.Yf_ = XYf, XYf

        elif self.normal_ == 'X':
            self.X, self.Xm_, self.Xs_, self.Xf_ = self.centerlize(self.X,Xm=None, Xs=None)
            self.Y, self.Ym_, self.Ys_, self.Yf_ = self.centerlize(self.Y,Xm=self.Xm_, Xs=self.Xs_)
        
        elif self.normal_ in [False, 'pass']:
            self.X, self.Xm_, self.Xs_, self.Xf_ = self.centerlize(self.X, Xm=np.zeros(self.D), Xs=1)
            self.Y, self.Ym_, self.Ys_, self.Yf_ = self.centerlize(self.Y ,Xm=np.zeros(self.D), Xs=1)
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
                # self.X_feat = self.scale_array(self.X_feat)[0]
                # self.Y_feat = self.scale_array(self.Y_feat)[0]
                self.X_feat = self.centerlize(self.X_feat)[0]
                self.Y_feat = self.centerlize(self.Y_feat)[0]
            elif self.feat_normal == 'pcc': #TODO
                self.X_feat = self.scale_array(self.X_feat, anis_var=True)[0]
                self.Y_feat = self.scale_array(self.Y_feat, anis_var=True)[0]
            else:
                warn(f"Unknown feature normalization method: {self.feat_normal}")
















    def feature_P(self):
        # if self.sigma2f is None:
        #     sigma2f = self.sigma_square(self.X_feat, self.Y_feat,)
        if self.KF:
            self.Pf, self.gf, self.sigma2f = kernel_xmm_k( self.X_feat, self.Y_feat, 
                                                       knn=self.KF, 
                                                        # method=self.kd_method,
                                                        method='annoy',
                                                        dfs=self.smm_dfs, 
                                                        kernel=self.feat_model,
                                                        sigma2=self.sigma2f)
        else:
            self.Pf, self.gf, self.sigma2f = kernel_xmm(self.X_feat, self.Y_feat, 
                                                    dfs=self.smm_dfs, 
                                                    kernel=self.feat_model,
                                                    sigma2=self.sigma2f)
        self.cdff = self.xp.sum(self.Pf, axis=0)
        self.cf = 0

    def adjustable_paras(self, **kargs):
        for key, value in kargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)

    def echo_paras(self, paras=None):
        if paras is None:
            paras = ['sigma2', 'sigma2f', 'K', 'KF', 'alpha', 'beta',  'gamma', 'use_fg', 
                     'beta_fg', 'use_lle', 'kw', 'normal', 'low_rank','w', 'kw']
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
        self.sigma2 = self.sigma_square(self.X, self.Y)
        self.q = 1.0 + self.N * self.D * 0.5 * np.log(self.sigma_square(self.X, self.Y))
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
        self.Pt1 = self.xp.sum(P, axis=0)
        self.P1 = self.xp.sum(P, axis=1)
        self.Np = self.xp.sum(self.P1)
        self.PX = P.dot(self.X)

        #Q = -np.sum(np.log(C)) + self.D*self.N*np.log(self.sigma2)/2
        if not self.w_clip is None:
            self.w = np.clip(1-self.Np/self.N, *self.w_clip)

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

        gs = self.OM /(1. - self.w)/gs
        cs = self.w/self.N
        cdfs = self.xp.sum(P, axis=0)

        if self.fE:
            cdfs = self.xp.multiply(cdfs, Pa)
            gs = gs / self.gf
            cs = cs + self.cf*self.w
            P = P * self.Pf

        C = cdfs + gs * cs 
        C = self.xp.clip(C, self.eps, None)
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
        # C = self.xp.clip(C, self.eps, None)
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
        # C = self.xp.clip(C, self.eps, None)
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

    @staticmethod
    def kernal_gmm(X_emb, Y_emb, sigma2=None, temp=1, shift=0, xp = None):
        if xp is None:
            xp = np
        assert X_emb.shape[1] == Y_emb.shape[1]
        (N, D) = X_emb.shape
        M = Y_emb.shape[0]
        Dist2 = scipy_distance.cdist(X_emb, Y_emb, "sqeuclidean").T
        if sigma2 is None:
            sigma2 = np.sum(Dist2) / (D*N*M) 
        P = np.exp( (shift-Dist2) / (2 * sigma2 * temp))
        return P, sigma2

    @staticmethod
    def kernal_gmm_k( X, Y, method='hnsw', metric='euclidean', 
                    knn=50, n_jobs=-1, temp=1, sigma2=None, **kargs):
        R,D = Y.shape
        Q = X.shape[0]

        snn = Neighbors(method=method, metric=metric, n_jobs=n_jobs)
        snn.fit(Y, **kargs)
        ckdout = snn.transform(X, knn=knn)
        nnidx = ckdout[1]

        src = nnidx.flatten('C')
        dst = np.repeat(np.arange(Q), knn)
        Dist2 = (ckdout[0].flatten('C'))**2

        if sigma2 is None:
            sigma2 = np.sum(Dist2) / (D*Q*knn)

        P = np.exp( (-Dist2) / (2 * sigma2 * temp))
        P = csr_array((P, (src, dst)), shape=(R, Q), dtype=np.float64)
        return P, sigma2
    
    @staticmethod
    def centerlize(X, Xm=None, Xs=None):
        X = np.array(X, dtype=np.float64)
        N,D = X.shape
        Xm = np.mean(X, axis=0) if Xm is None else Xm
        X -= Xm
        Xs = np.sqrt(np.sum(np.square(X))/N) if Xs is None else Xs  #(N*D) 
        X /= Xs
        Xf = np.eye(D+1, dtype=np.float64)
        Xf[:D,:D] *= Xs
        Xf[:D, D] = Xm
        return [X, Xm, Xs, Xf]
    
    @staticmethod
    def centerlize1(X, Xm=None, Xs=None):
        X = np.array(X, dtype=np.float64)
        N,D = X.shape
        Xm = np.mean(X, axis=0) if Xm is None else Xm
        X -= Xm
        Xs = np.sqrt(np.sum(np.square(X))/N) if Xs is None else Xs  #(N*D) 
        X /= Xs
        Xf = np.eye(D+1, dtype=np.float64)
        Xf[:D,:D] *= Xs
        Xf[:D, D] = Xm
        return [X, Xm, Xs, Xf]
    
    @staticmethod
    def scale_array( X,
                    zero_center = True,
                    anis_var = False,
                    axis = 0,
        ):
        if issparse(X):
            X = X.toarray()
        X = X.copy()
        N,D = X.shape

        mean = np.expand_dims(np.mean(X, axis=axis), axis=axis)

        if anis_var:
            std  = np.expand_dims(np.std(X, axis=axis, ddof=0), axis=axis)
            std[std == 0] = 1
        else:
            std = np.std(X)

        if zero_center:
            X -= mean
        X /=  std

        mean = np.squeeze(mean)
        std  = np.squeeze(std)
        Xf = np.eye(D+1, dtype=np.float64)
        Xf[:D,:D] *= std
        Xf[:D, D] = mean

        return X, mean, std, Xf

    def lle_W(self, Y= None):
        Y = self.Y if Y is None else Y
        
        self.point_kins(Y)
        ckdout = self.point_kdist(Y)
        ckdidx = ckdout[1][:,1:(self.knn+1)]
        src = ckdidx.flatten('C')
        dst = np.repeat(np.arange(ckdidx.shape[0]), ckdidx.shape[1])
        W = []
        for i in range(self.M):
            kn = ckdidx[i]
            z = (Y[kn] - Y[i]) #K*D
            G = z @ z.T # K*K
            G = G + np.eye(self.knn) * self.rl_w* np.trace(G)
            w = np.sum(np.linalg.inv(G), axis=1) #K*1
            w = w/ max(np.sum(w), self.eps)
            W.append(w)

        W = np.float64(W)
        # W = ssp.csr_array((W.flatten(), (dst, src)), shape=(self.M, self.M), dtype=np.float64)
        W = ssp.csr_array((W.flatten(), (src, dst)), shape=(self.M, self.M), dtype=np.float64) # CHECK TODO
        W = ssp.eye(self.M)- W
        self.llW = W

    @staticmethod
    def lle_M(Y, kw=15, rl_w=None,  method='annoy', eps=np.finfo(np.float64).eps):
        M, D = Y.shape
        eps = np.finfo(np.float64).eps
        if rl_w is None:
            rl_w = 1e-3 if(kw>D) else 0

        snn = Neighbors(method=method)
        snn.fit(Y)
        ckdout = snn.transform(Y, knn=kw+1)

        kdx = ckdout[1][:,1:]
        src = kdx.flatten('C')
        dst = np.repeat(np.arange(kdx.shape[0]), kdx.shape[1])

        L = []
        for i in range(M):
            kn = kdx[i]
            z = (Y[kn] - Y[i]) #K*D
            G = z @ z.T # K*K
            G = G +  np.eye(kw) * rl_w* np.trace(G)
            w = np.sum(np.linalg.inv(G), axis=1) #K*1
            w = w/ np.sum(w).clip(eps, None)
            L.append(w)

        L  = ssp.eye(M) - ssp.csr_array((np.array(L).flatten(), (dst, src)), shape=(M, M))
        LM =L.transpose().dot( L)
        return LM

    @staticmethod
    def sigma_square(X, Y):
        [N, D] = X.shape
        [M, D] = Y.shape
        # sigma2 = (M*np.trace(np.dot(np.transpose(X), X)) + 
        #           N*np.trace(np.dot(np.transpose(Y), Y)) - 
        #           2*np.dot(np.sum(X, axis=0), np.transpose(np.sum(Y, axis=0))))
        sigma2 = (M*np.sum(X * X) + 
                N*np.sum(Y * Y) - 
                2*np.dot(np.sum(X, axis=0), np.transpose(np.sum(Y, axis=0))))
        sigma2 /= (N*M*D)
        return sigma2

    @staticmethod
    def low_rank_eigen(G, num_eig):
        S, Q = np.linalg.eigh(G)
        eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
        Q = Q[:, eig_indices]  # eigenvectors
        S = S[eig_indices]  # eigenvalues.
        return Q, S

    @staticmethod
    def low_rank_eigen_sp(G, num_eig):
        k = min(G.shape[0]-1, num_eig)
        S, Q = linalg.eigs(G, k=k)

        eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
        Q = Q[:, eig_indices]  # eigenvectors
        S = S[eig_indices]  # eigenvalues.
        return np.real(Q), np.real(S)
    
    @staticmethod
    def Woodbury(A, U, C, V):
        Av = np.linalg.inv(A)
        Cv = np.linalg.inv(C)
        UCv = np.linalg.inv(Cv  + V @ Av @ U)
        return  Av - Av @ U @ UCv @ V @ Av