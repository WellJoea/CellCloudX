from __future__ import division
import numpy as np
import numbers
from warnings import warn
from scipy.sparse import issparse, csr_array, csc_array, diags
from scipy.spatial import distance as scipy_distance
from tqdm import tqdm

from ...tools._decomposition import dualPCA
from ...tools._neighbors import Neighbors, mtx_similarity
from ...transform import homotransform_point

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

    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th source point
        corresponds to the n-th target point.

    Pt1: numpy array
        Nx1 column array.
        Multiplication result between the transpose of P and a column vector of all 1s.

    P1: numpy array
        Mx1 column array.
        Multiplication result between P and a column vector of all 1s.

    Np: float (positive)
        The sum of all elements in P.

    """

    def __init__(self, X, Y, X_feat=None, Y_feat=None, sigma2=None, sigma2f=None, maxiter=None, 
                 tol=None, w=None, temp=1, use_dpca=False,
                 normal=None,  
                 feat_normal='l2', n_comps=60, get_P = True,
                 shift=0, use_cuda=False, w_clip=[1e-4, 1-1e-3],
                 K=None, KF=None, p_epoch=None, spilled_sigma2=1,
                 knn_method='hnsw', **kwargs):
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

        self.normal = normal
        self.X = self.xp.array(X, dtype=self.xp.float64).copy()
        self.Y = self.xp.array(Y, dtype=self.xp.float64).copy()

        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape

        self.tol =  1e-5 if tol is None else tol
        self.w = 0.5 if w is None else w
        self.w_clip = w_clip
        self.maxiter = 500 if maxiter is None else maxiter
        self.iteration = 0
        self.temp = self.xp.float64(temp)
        self.use_dpca=use_dpca
        self.n_comps=n_comps 
        self.shift= self.xp.float64(shift)

        self.get_P = get_P
        self.diff = np.inf
        self.q = np.inf
        self.knn_method = knn_method
        self.K = K
        self.KF = KF
        self.p_epoch= p_epoch
        self.tform = self.xp.eye(self.D + 1, dtype=self.xp.float64)
        
        self.sigma2 = sigma2
        self.sigma2f = sigma2f
        self.spilled_sigma2 = spilled_sigma2

        self.fE = not (X_feat is None or Y_feat is None)
        self.feat_normal = feat_normal
        if self.fE:
            assert X_feat.shape[1] == Y_feat.shape[1], "X_feat and Y_feat must have the same number of features"
            self.X_feat = self.xp.asarray(X_feat)
            self.Y_feat = self.xp.asarray(Y_feat)
            self.Df = self.X_feat.shape[1]
        self.homotransform_point = homotransform_point
        # self.transform_point() # TODO
        self.init_normalize()

    def init_normalize(self):
        if self.normal:
            self.X, self.Xm, self.Xs, self.Xf = self.znormal(self.X)
            self.Y, self.Ym, self.Ys, self.Yf = self.znormal(self.Y)
        self.TY = self.Y.copy()

        if self.fE:
            if self.feat_normal == 'l2':
                l2x = self.xp.linalg.norm(self.X_feat, ord=None, axis=1, keepdims=True)
                l2y = self.xp.linalg.norm(self.Y_feat, ord=None, axis=1, keepdims=True)
                l2x[l2x == 0] = 1
                l2y[l2y == 0] = 1
                # l2x = self.xp.clip(l2x, self.xp.finfo(self.X_feat.dtype).eps, None)
                # l2y = self.xp.clip(l2y, self.xp.finfo(self.Y_feat.dtype).eps, None)
                self.X_feat = self.X_feat/l2x
                self.Y_feat = self.Y_feat/l2y
            elif self.feat_normal == 'zscore':
                self.X_feat = self.scale_array(self.X_feat)
                self.Y_feat = self.scale_array(self.Y_feat)
            else:
                warn(f"Unknown feature normalization method: {self.feat_normal}")

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
                    knn=50, n_jobs=-1,  sigma2=None, **kargs):
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

        P = np.exp( (-Dist2) / (2 * sigma2 ))
        P = csr_array((P, (src, dst)), shape=(R, Q), dtype=np.float64)
        return P, sigma2
    

    @staticmethod
    def znormal(X):
        X = np.array(X, dtype=np.float64)
        N,D = X.shape
        Xm = np.mean(X, axis=0)
        X -= Xm
        Xs = np.sqrt(np.sum(np.square(X))/N)
        X /= Xs
        Xf = np.eye(D+1, dtype=np.float64)
        Xf[:D,:D] *= Xs
        Xf[:D, D] = Xm
        return [X, Xm, Xs, Xf]

    @staticmethod
    def scale_array( X,
                    zero_center = True,
                    anis_var = False,
                    max_value = None,
                    axis = 0,
                    verbose =1,
        ):
        if issparse(X):
            X = X.toarray()
        X = X.copy()

        if (verbose) and (not zero_center) and (max_value is not None):
            print( "... be careful when using `max_value` " "without `zero_center`.")

        if (verbose) and np.issubdtype(X.dtype, np.integer):
            print('... as scaling leads to float results, integer '
                'input is cast to float, returning copy.'
                'or you forget to normalize.')
            X = X.astype(float)

        mean = np.expand_dims(np.mean(X, axis=axis), axis=axis)

        if anis_var:
            std  = np.expand_dims(np.std(X, axis=axis), axis=axis)
            std[std == 0] = 1
        else:
            std = np.std(X)
            print(std)
    
        if zero_center:
            X -= mean
        X /= std

        # do the clipping
        if max_value is not None:
            (verbose>1) and print(f"... clipping at max_value {max_value}")
            X[X > max_value] = max_value

        return X

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
    
    def feature_P(self):
        if self.KF:
            self.Pf, self.sigma2f = self.kernal_gmm_k(self.X_feat, self.Y_feat, 
                                                       knn=self.KF, 
                                                      method=self.knn_method,
                                                      sigma2=self.sigma2f)
        else:
            self.Pf, self.sigma2f = self.kernal_gmm(self.X_feat, self.Y_feat, 
                                                    method=self.knn_method,
                                                    sigma2=self.sigma2f)
        self.cdff = self.xp.sum(self.Pf, axis=0)

        self.gf = (2 * np.pi * self.sigma2f) ** (0.5 * self.Df)

        # V_f = (self.den_f/self.M).var()
        # # self.c_f  = self.M * (2 * np.pi * self.sigma2f) ** (0.5 * (self.D + self.Df - 1))
        # # self.c_f  = self.M * ((2 * np.pi * self.sigma2f) **  (0.5 * (self.Df - 1)))
        # self.c_f  = self.M * ((2 * np.pi * self.sigma2f) ** (0.5 * self.Df))  / (1-self.w) 
        # self.c_f /= ((2 * np.pi * V_f) ** (0.5 * 1))
        # # self.c_f *= self.xp.exp(-self.xp.square(self.den_f) / (self.M * (2.0 * self.sigma2f)))
        # # self.c_f *= self.xp.exp(-self.xp.square(self.den_f/self.M) / (1 * (2.0 * self.sigma2f)))
        # # self.c_f *= np.exp(-1.0 / self.M * np.square(np.sum(self.Pf, axis=0)) / (2.0 * self.sigma2f))
        # self.c_f *= self.xp.exp(-self.xp.square(self.den_f/self.M) / (1 * (2.0 * V_f)))

        V = self.cdff/self.M
        U = V.mean()
        S2 = V.var()
        self.cf = ((2 * np.pi * S2) ** 0.5) * self.xp.exp(-(V - 0)**2 / (2.0 * S2))
        # self.cf = ((2 * np.pi * self.sigma2f) ** 0.5) * self.xp.exp(-(self.cdff)**2 / (2.0 * self.sigma2f * self.M))
        self.cf = 0
        import matplotlib.pyplot as plt
        try:
            plt.hist(self.Pf.data, bins=100)
        except:
            plt.hist(self.Pf.flatten(), bins=100)
        plt.title(f'Gmm P*Pf normal feature: {1111, self.sigma2f, self.cf}')
        plt.show()

    def register(self, callback=lambda **kwargs: None):
        self.q = 1.0 + self.N * self.D * 0.5 * np.log(self.sigma_square(self.X, self.Y))
        if self.fE:
            self.feature_P()

        pbar = tqdm(range(self.maxiter), total=self.maxiter, colour='red', desc='EM')
        for i in pbar:
            self.optimization()
            if callable(callback):
                kwargs = {'iteration': self.iteration,
                        'error': self.q, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)
            log_prt = {'iter': self.iteration, 
                       'tol': f'{self.diff :.4e}', 
                       'Q': f'{self.q :.4e}', 
                       'w': f'{self.w :.3e}',
                       'sigma2': f'{self.sigma2 :.4e}'}

            pbar.set_postfix(log_prt)
            if (self.p_epoch):
                pbar.update(self.p_epoch)
            if self.diff <= self.tol:
                print(f'Tolerance is lower than {self.tol}. Program ends early.')
                break
        pbar.close()

        if self.get_P:
           self.P, self.C = self.update_P(self.X, self.TY)
        if self.normal: 
            self.update_normalize()

    def expectation(self):
        P, C = self.update_P(self.X, self.TY)
        P = P / C
        self.Pt1 = self.xp.sum(P, axis=0)
        self.P1 = self.xp.sum(P, axis=1)
        self.Np = self.xp.sum(self.P1)
        self.PX = P.dot(self.X)
        if not self.w_clip is None:
            self.w = np.clip(1-self.Np/self.N, *self.w_clip)

    def update_P(self, X, Y):
        if self.K is None:
            P, self.sigma2 = self.kernal_gmm(X, Y, sigma2=self.sigma2)
            OM = self.M
        else:
            P, self.sigma2 = self.kernal_gmm_k(X, Y, knn=self.K, sigma2=self.sigma2)
            OM = self.K

        gs = ((2*np.pi*self.sigma2)**(self.D/2)) * OM /(1. - self.w)
        cs = self.w/self.N
        cdfs = self.xp.sum(P, axis=0)

        if self.fE:
            cdfs = self.xp.multiply(cdfs, self.cdff)
            gs = gs *self.gf
            cs = cs + self.cf*self.w
            P = P * self.Pf

        C = cdfs + gs * cs 
        return P, C

    def expectation1111(self):
        if self.K is None:
            P, self.sigma2 = self.kernal_gmm(self.X, self.TY,  sigma2=self.sigma2)
            OM = self.M
        else:
            P, self.sigma2 = self.kernal_gmm_k(self.X, self.TY,  knn=self.K, sigma2=self.sigma2)
            OM = self.K
        gs = ((2*np.pi*self.sigma2)**(self.D/2)) * OM /(1. - self.w)
        cs = self.w/self.N

        P = P * self.Pf
        cdfs = self.xp.sum(P, axis=0)

        if self.fE:
            #cdfs = self.xp.multiply(cdfs, self.cdff)
            gs = gs *self.gf
            cs = cs + self.cf*self.w
            #P = P * self.Pf

        C = cdfs + gs * cs 
        P = P / C

        self.Pt1 = self.xp.sum(P, axis=0)
        self.P1 = self.xp.sum(P, axis=1)
        self.Np = self.xp.sum(self.P1)
        self.PX = P.dot(self.X)
        self.w = np.clip(1-self.Np/self.N, *self.w_clip)

    def expectation222(self):
        P = self.kernal_gmm(self.X, self.TY,  sigma2=self.sigma2, temp=1, xp = self.xp)[0]
        gs = ((2*np.pi*self.sigma2)**(self.D/2)) * self.M /(1. - self.w)
        cs = self.w/self.N
        cdfs = self.xp.sum(P, axis=0, keepdims = True)

        if self.fE:
            cdfs = self.xp.multiply(cdfs, self.cdff)
            gs = gs *self.gf
            cs = cs + self.cf*self.w
            P = self.xp.multiply(P, self.Pf)

        C = cdfs + gs * cs 
        P = P / C

        self.Pt1 = self.xp.sum(P, axis=0)
        self.P1 = self.xp.sum(P, axis=1)
        self.Np = self.xp.sum(self.P1)
        self.PX = P.dot(self.X)
        self.w = 1-self.Np/self.N
        # if (self.iteration % self.p_epoch == 0):
        #     import matplotlib.pyplot as plt
        #     plt.hist(P.flatten(), bins=100)
        #     plt.title(f'Gmm P*Pf normal feature: {self.Np/self.N}')
        #     plt.show()

    def Gmm_feature2(self):
        self.Pf, self.sigma2f = Gmm(self.X_feat, self.Y_feat, norm=False,
                                     sigma2=self.sigma2f)
        print(self.sigma2f)
        import matplotlib.pyplot as plt
        plt.hist(self.Pf.flatten(), bins=100)
        plt.title('Gmm feature 2222222222')
        plt.show()  

        self.den_f = self.xp.sum(self.Pf, axis=0, keepdims = True)
        self.den_f = self.xp.clip(self.den_f, self.xp.finfo(self.X_feat.dtype).eps, None)
        self.c_f  = self.M * (2 * np.pi * self.sigma2f) ** (0.5 * (self.D + self.Df - 1))
        self.c_f *= self.xp.exp(-1.0 / self.M * self.xp.square(self.den_f) / (2.0 * self.sigma2f))
        # self.c_f *= np.exp(-1.0 / self.M * np.square(np.sum(self.Pf, axis=0)) / (2.0 * self.sigma2f))


    def Gmm_feature1(self):
        self.Pf, self.sigma2f = Gmm(self.X_feat, self.Y_feat, norm=False, 
                                    shift=2,
                                    sigma2=1)
        self.den_f = self.xp.sum(self.Pf, axis=0, keepdims = True)
        self.den_f = self.xp.clip(self.den_f, self.xp.finfo(self.X_feat.dtype).eps, None)
        # self.c_f  = self.M * (1) ** (0.5 * (self.D + self.Df - 1))
        # self.c_f *= self.xp.exp(-1.0 / (self.M **2) * self.xp.square(self.den_f) / 1)

        self.Pf /= self.den_f
        V = self.den_f/self.M
        U = V.mean()
        S2 = V.var()
        self.c_f  = self.M / (2 * np.pi * S2) ** 0.5 / (1-self.w)
        self.c_f *= self.xp.exp(-(V - 0)**2 / (2.0 * S2))

        print('sigma2f', self.sigma2f)
        import matplotlib.pyplot as plt
        plt.hist(self.Pf.flatten(), bins=100)
        plt.title('Gmm feature')
        plt.show()

        import matplotlib.pyplot as plt
        plt.hist(self.c_f.flatten(), bins=100)
        plt.title('Gmm feature c_f')
        plt.show()

    def expectation2(self):
        P = Gmm(self.X, self.TY, norm=False, sigma2=self.sigma2,
                 temp=1, xp = self.xp)[0]

        c = (2*np.pi*self.sigma2)**(self.D/2)
        c *= self.w/(1. - self.w)*self.M/self.N 
        den = self.xp.sum(P, axis = 0, keepdims = True)
        den = self.xp.clip(den, self.xp.finfo(self.X.dtype).eps, None) #+ c

        if self.fE:
            den = self.xp.multiply(den, self.den_f)
            den += self.c_f
            c *= (2.0 * np.pi * self.sigma2f) ** (self.Df * 0.5)
            P = self.xp.multiply(P, self.Pf)

        den += c

        P = self.xp.divide(P, den)
 
        self.Pt1 = self.xp.sum(P, axis=0)
        self.P1 = self.xp.sum(P, axis=1)
        self.Np = self.xp.sum(self.P1)
        self.PX = P.dot(self.X)

    def expectation1(self):
        P = Gmm(self.X, self.TY, norm=False, sigma2=self.sigma2,
                 temp=1, xp = self.xp)[0]
        P = self.xp.multiply(P, self.Pf)

        c = ((2*np.pi*self.sigma2)**(self.D/2))*self.w/(1. - self.w)*self.M/self.N
        den = self.xp.sum(P, axis = 0, keepdims = True)
        den = self.xp.clip(den, self.xp.finfo(self.X.dtype).eps, None)

        if self.fE:
            # den = self.xp.multiply(den, self.den_f)
            # den += self.c_f * (2*np.pi*self.sigma2)**(self.D/2)
            # # c *= (2.0 * np.pi * self.sigma2f) ** (self.Df * 0.5)
            # P = self.xp.multiply(P, self.Pf)
            #c *= self.den_f
            pass

        den += c
        if (self.iteration % self.p_epoch == 0):
            import matplotlib.pyplot as plt
            plt.hist(P.flatten(), bins=100)
            plt.title('expectation1: Gmm P*Pf feature')
            plt.show()

        P = self.xp.divide(P, den)
        if (self.iteration % self.p_epoch == 0):
            import matplotlib.pyplot as plt
            plt.hist(P.flatten(), bins=100)
            plt.title('Gmm P*Pf normal feature')
            plt.show()


        self.Pt1 = self.xp.sum(P, axis=0)
        self.P1 = self.xp.sum(P, axis=1)
        self.Np = self.xp.sum(self.P1)
        self.PX = P.dot(self.X)

    def expectation11(self):
        P = Gmm(self.X, self.TY, norm=False, sigma2=self.sigma2,
                 temp=1, xp = self.xp)[0]

        if (self.iteration % self.p_epoch == 0):
            import matplotlib.pyplot as plt
            plt.hist(P.flatten(), bins=100)
            plt.title('Gmm P feature')
            plt.show()


        P = self.xp.multiply(P, self.Pf)

        if (self.iteration % self.p_epoch == 0):
            import matplotlib.pyplot as plt
            plt.hist(P.flatten(), bins=100)
            plt.title('Gmm P*Pf feature')
            plt.show()
            print('P.sum', P.sum(),  P.sum(1), P.sum(0),  P.sum()/((2*np.pi*self.sigma2)**(self.D/2)) )

        c = ((2*np.pi*self.sigma2)**(self.D/2))*self.w/(1. - self.w)*self.M/self.N
        den = self.xp.sum(P, axis = 0, keepdims = True)
        den = self.xp.clip(den, self.xp.finfo(self.X.dtype).eps, None)

        if self.fE:
            # den = self.xp.multiply(den, self.den_f)
            c_f = self.c_f * (2*np.pi*self.sigma2)**(self.D/2)
            den += c_f
        den += c


        P = self.xp.divide(P, den)
        if (self.iteration % self.p_epoch == 0):
            import matplotlib.pyplot as plt
            plt.hist(P.flatten(), bins=100)
            plt.title('Gmm P*Pf normal feature')
            plt.show()
            print('den', den, c, self.c_f)
        self.Pt1 = self.xp.sum(P, axis=0)
        self.P1 = self.xp.sum(P, axis=1)
        self.Np = self.xp.sum(self.P1)
        self.PX = P.dot(self.X)

    def expectation_knn1(self):
        """
        Compute the expectation step of the EM algorithm.
        """
        # P = np.sum((self.X[None, :, :] - self.TY[:, None, :])**2, axis=2) # (1, N, D) - (M, 1, D)(sum2) =  (M, N)
        P = self.point_kdist(Y=self.TY)
        sP = np.exp(-P.data/(2*self.sigma2))

        if self.fmethod =='dist':
            fP = self.feature_ksimi(*P.nonzero())
        elif self.fmethod =='gmm':
            fP = self.feature_kgmm(*P.nonzero())
        P = csr_array((sP*fP, P.nonzero()), shape=P.shape)
        del sP
        del fP

        c = (2*np.pi*self.sigma2)**(self.D/2)*self.w/(1. - self.w)*self.M/self.N 
        #c *= np.sum(self.fP)
        den = np.sum(P, axis = 0)[None,:] # (1, N)
        den = np.clip(den, np.finfo(self.X.dtype).eps, None) + c

        # self.P = np.divide(P, den) #Pmn, (M, N)
        self.P = P.multiply(1/den) #Pmn, (M, N)
        self.Pt1 = np.sum(self.P, axis=0) # (1, N)
        self.P1 = np.sum(self.P, axis=1) #(M,1)
        self.Np = np.sum(self.P1) #(1,1)
        # self.PX = np.matmul(self.P, self.X) # (M,N) * (N,D)
        self.PX = self.P.dot(self.X) # (M,N) * (N,D)

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


    def point_kins(self,  method='hnsw', metric='euclidean', n_jobs=-1, **kargs):
        snn = Neighbors(method=method, metric=metric, n_jobs=n_jobs)
        snn.fit(self.X, **kargs)
        # from sklearn.neighbors import NearestNeighbors
        # snn = NearestNeighbors(n_neighbors=knn, algorithm='ball_tree')
        # snn.fit(self.X)
        self.snn = snn

    def feature_kins(self, method='hnsw', metric='euclidean', n_jobs=-1, **kargs):
        enn = Neighbors(method=method, metric=metric, n_jobs=n_jobs)
        enn.fit(self.X, **kargs)
        self.enn = enn

    def point_kdist(self, Y=None):
        Y = self.TY if Y is None else Y
        ckdout = self.snn.transform(Y, knn=self.knn)
        KZ0 = ckdout[0]
        KZ0[KZ0==0] = 1e-8
        ckdout[0] = KZ0 ** 2 #
        Dist_sp = self.snn.translabel(ckdout, rsize=self.N, return_type='sparse') # M,N
        # Dist_sp = csc_array(Dist_sp)
        # Dist_sp.eliminate_zeros()
        return Dist_sp

    def feature_simi(self):
        use_dpca=self.use_dpca
        temp=self.temp
        n_comps = self.n_comps
        shift = self.shift
        X_emb = self.X_feat
        Y_emb = self.Y_feat
        if use_dpca:
            n_comps = min(X_emb.shape[1], n_comps or 50, )
            X_emb1, Y_emb1 = dualPCA(X_emb, Y_emb, 
                            n_comps = n_comps,
                            scale=True,
                            axis=0,
                            zero_center=True)
        else:
            X_emb1, Y_emb1 = X_emb, Y_emb
        simi = mtx_similarity(X_emb1, Y_emb1).T
        Pf = np.exp( (simi-shift) / temp) # (M, N)
        #Pf = self.normalAdj(Pf)
        self.Pf = Pf
        import matplotlib.pyplot as plt
        plt.hist(self.Pf.flatten(), bins=100)
        plt.show()

    def feature_ksimi(self, col, row):
        use_dpca=self.use_dpca
        temp=self.temp
        n_comps = self.n_comps
        shift = self.shift
        X_emb = self.X_feat
        Y_emb = self.Y_feat
        if use_dpca:
            n_comps = min(X_emb.shape[1], n_comps or 50, )
            X_emb1, Y_emb1 = dualPCA(X_emb, Y_emb, 
                            n_comps = n_comps,
                            scale=True,
                            axis=0,
                            zero_center=True)
        else:
            X_emb1, Y_emb1 = X_emb, Y_emb
        simi = mtx_similarity(X_emb1, Y_emb1, pairidx=(row, col))
        Pf = np.exp( (simi-shift) / temp) # (M, N)
        #Pf = self.normalAdj(Pf)
        # import matplotlib.pyplot as plt
        # plt.hist(Pf.flatten(), bins=100)
        # plt.show()
        return Pf
