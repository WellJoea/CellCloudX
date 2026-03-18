from scipy.sparse import issparse, csr_array, csc_array, diags, eye
from scipy.spatial import distance as scipy_distance
import scipy.sparse as ssp
from tqdm import tqdm
import numpy as np

from ..tools._neighbors import Neighbors, mtx_similarity
from ..transform import homotransform_point

class EMRegistration(object):
    def __init__(self, X, Y, X_feat=None, Y_feat=None, match_pairs=None,
                 gamma=0.9,  gamma_clip = [0.0001, 0.99], maxiter=500, beta =0.1,
                 lambd = 9e9, theta=0.75,  tol=1e-5, tol_r = 1e-5, tol_s = 1e-8,
                 minp= 1e-5, rl_w = 1e-5, unif=10, knn = 15, normal=False,
                 use_cuda=False, p_epoch=None, **kwargs):
        
        self.use_cuda = use_cuda
        if self.use_cuda:
            import cupy as cp
            self.xp = cp
        else:
            self.xp = np
        
        self.normal = normal
        if self.normal:
            self.X, self.Xm, self.Xs, self.Xf = self.znormal(X)
            self.Y, self.Ym, self.Ys, self.Yf = self.znormal(Y)
        else:
            self.X = self.xp.array(X, dtype=self.xp.float64).copy()
            self.Y = self.xp.array(Y, dtype=self.xp.float64).copy()
        self.TY = self.Y.copy()

        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        
        self.tol=tol
        self.tol_r = tol_r
        self.tol_s = tol_s

        self.gamma = gamma
        self.gamma_clip = gamma_clip
        self.maxiter = maxiter
        self.beta = beta
        self.lambd = lambd
        self.theta = theta

        self.minp = minp
        self.unif = unif
        self.knn = knn
        self.rl_w = rl_w if self.knn > self.D else 0
  
        self.p_epoch = p_epoch      
        self.iteration = 0
        self.eps = self.xp.finfo(self.X.dtype).eps
        self.homotransform_point = homotransform_point

        if not match_pairs is None :
            self.match_pairs = self.xp.array(match_pairs, dtype=self.xp.int64)
        
    def point_kins(self, X=None, method='sknn', metric='euclidean', n_jobs=-1, **kargs):
        snn = Neighbors(method=method, metric=metric, n_jobs=n_jobs)
        X = self.X if X is None else X
        snn.fit(X, **kargs)
        self.snn = snn

    def point_kdist(self, Y=None, knn=None,):
        knn = self.knn if knn is None else knn
        Y = self.TY if Y is None else Y
        ckdout = self.snn.transform(Y, knn=(knn+1))
        return ckdout

    def lle_W(self, Y= None):
        Y = self.Y if Y is None else Y
        uni_Y, uni_I = np.unique(Y, return_index=True, axis=0)
        self.point_kins(uni_Y)
        ckdout = self.point_kdist(Y, knn=self.knn)
        ckdidx = ckdout[1][:,1:(self.knn+1)]
        W = []
        for i in range(self.M):
            kn = ckdidx[i]
            z = (uni_Y[kn,:] - Y[i,:]) #K*D
            G = z @ z.T # K*K
            G = G + np.eye(self.knn) * self.rl_w* np.trace(G)
            w = np.sum(np.linalg.inv(G), axis=1) #K*1
            w = w/ max(np.sum(w), self.eps)
            W.append(w)
        W = np.float64(W)
        src = uni_I[ckdidx.flatten('C')]
        dst = np.repeat(np.arange(ckdidx.shape[0]), ckdidx.shape[1])
        W = ssp.csr_array((W.flatten(), (dst, src)), shape=(self.M, self.M), dtype=np.float64) #src error
        W = ssp.eye(self.M)- W
        self.llW = W

    def lle_W0(self, Y= None):
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
        W = ssp.csr_array((W.flatten(), (dst, src)), shape=(self.M, self.M), dtype=np.float64)
        W = ssp.eye(self.M)- W
        self.llW = W

    @staticmethod
    def kernal_gmm(X_emb, Y_emb, norm=False, sigma2=None, temp=1, shift=0, xp = None):
        if xp is None:
            xp = np
        assert X_emb.shape[1] == Y_emb.shape[1]
        (N, D) = X_emb.shape
        M = Y_emb.shape[0]

        if norm:
            # X_emb = (X_emb - np.mean(X_emb, axis=0)) / np.std(X_emb, axis=0)
            # Y_emb = (Y_emb - np.mean(Y_emb, axis=0)) / np.std(Y_emb, axis=0)
            X_l2 =  X_emb/np.linalg.norm(X_emb, ord=None, axis=1, keepdims=True)
            Y_l2 =  Y_emb/np.linalg.norm(Y_emb, ord=None, axis=1, keepdims=True)
        else:
            X_l2 = X_emb
            Y_l2 = Y_emb
        
        Dist = scipy_distance.cdist(X_l2, Y_l2, "sqeuclidean")
        sigma2_ =np.sum(Dist) / (D*N*M)
        sigma2 = sigma2_ if sigma2 is None else sigma2
        P = np.exp( (shift-Dist) / (2 * sigma2 * temp))
        return P, sigma2_

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

    def register(self, callback=lambda **kwargs: None):
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
            if (self.diff <= self.tol) or (self.sigma2 <= self.tol_s or self.diff_r <= self.tol_r ):
                print(f'Tolerance is lower than {self.tol}. Program ends early.')
                break
            # break
        pbar.close()
        if self.normal: self.update_normalize()

    def expectation(self):
        dist2 = np.sum(np.square(self.X-self.TY), axis=1)
        P = self.xp.exp(-dist2/(2*self.sigma2))
        C = (2 * self.xp.pi * self.sigma2)**(self.D/2) * (1-self.gamma)/(self.gamma*self.unif)

        P = P/(P+C) # N*1
        P = self.xp.clip(P, self.eps, None)
        P = ssp.diags(P)
    
        self.P = P
        # self.Pt1 = self.xp.sum(P, axis=0)
        # self.P1 = self.xp.sum(P, axis=1)
        self.Pt1 = self.xp.array(self.P.sum(0)).squeeze()
        self.P1 = self.xp.array(self.P.sum(1)).squeeze()
        self.Np = self.xp.sum(self.P1)
        self.preQ = self.Q
        
        self.Q = np.sum(self.P.diagonal() * dist2)/(2*self.sigma2) + \
            self.Np * self.D * np.log(self.sigma2) / 2 + \
            - self.Np*np.log(self.gamma) - np.sum(1-self.Pt1)*np.log(1-self.gamma)

    def optimization(self):
        raise NotImplementedError(
            "optimization should be defined in child classes.")
    
    def update_normalize(self):
        raise NotImplementedError(
            "update_normalize should be defined in child classes.")

class rigid_reg(EMRegistration):
    def __init__(self, *args, R=None, t=None, s=None, scale=True, 
                 tform=None, tforminv=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.R = np.eye(self.D) if R is None else R
        self.t = np.zeros(self.D) if t is None else t
        self.s = 1 if s is None else s
        self.scale = scale
        self.update_transformer(tform=tform, tforminv=tforminv)

        self.sigma2 = np.sum(np.square(self.X - self.Y))/(self.N*self.D)
        self.lle_W()

    def optimization(self):
        self.expectation()
        self.update_transform()
        self.update_transformer()
        self.transform_point()
        self.update_variance()
        self.iteration += 1



    def update_transform(self):
        PX = self.P.dot(self.X)
        PY = self.P.dot(self.Y)
        muX = self.xp.divide(self.xp.sum(PX, axis=0), self.Np)
        muY = self.xp.divide(self.xp.sum(PY, axis=0), self.Np)
        #muY = self.xp.divide(self.xp.dot(self.xp.transpose(self.Y), self.P1), self.Np)

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
        #trYPY = np.trace(Y_hat.T @ np.diag(self.P1) @ Y_hat)
        #trYPY = np.sum(np.multiply(self.Y.T**2, self.P1)) - self.Np*(muY.T @ muY)
        self.trAR = np.trace(S @ C)
        self.trXPX = np.sum( self.Pt1.T * np.sum(np.multiply(X_hat, X_hat), axis=1))
        self.trYPY = np.sum( self.P1.T * np.sum(np.multiply(Y_hat, Y_hat), axis=1))
        
        ## LLE
        if self.scale is True:
            Z = self.llW.transpose().dot(self.P.dot(self.llW))
            YtZY = self.Y.T @ Z.dot(self.Y)
            self.s = self.trAR/ (self.trYPY + 2* self.lambd * self.sigma2 * np.trace(YtZY))
            self.QZ = self.lambd * self.xp.linalg.norm( self.s * np.sqrt(self.P) @ self.llW @ self.Y )
        else:
            self.QZ = 0

        self.t = muX - self.s * self.xp.dot(self.R, muY)

    def update_variance(self):
        self.Q += self.QZ

        self.diff = np.abs(self.Q - self.preQ)
        self.diff_r = np.abs(self.diff/self.Q)

        V = np.square(self.X-self.TY)
        self.sigma2 = np.sum( V * self.P1[:, None])/(self.D*self.Np)
        # if self.sigma2 <= 0:
        #     self.sigma2 = 0.1

        self.Ind = (self.P.diagonal() > self.theta)
        self.gamma = np.clip(self.Ind.sum()/self.M, *self.gamma_clip)

    def transform_point(self, Y=None):
        if Y is None:
            self.TY = self.homotransform_point(self.Y, self.tform, inverse=False)
            return
        else:
            return self.homotransform_point(Y, self.tform, inverse=False)

    def update_transformer(self, tform=None, tforminv=None):
        if not tforminv is None:
            tform = np.linalg.inv(np.float64(tforminv))
        if not tform is None:
            self.tform = np.float64(tform)
            self.tforminv = np.linalg.inv(self.tform)
    
            #TODO
            B = self.tform[:-1, :-1]
            self.s = np.linalg.det(B)**(1/(B.shape[0])) 
            self.R = B/self.s
            self.t = self.tform[:-1, [-1]].T

        else:
            self.tform = np.eye(self.D+1, dtype=np.float64)
            self.tform[:self.D, :self.D] = self.R * self.s
            self.tform[:self.D,  self.D] = self.t
            self.tforminv = np.linalg.inv(self.tform)

    def update_normalize(self):
        self.s *= self.Xs/self.Ys 
        self.t = (self.t * self.Xs + self.Xm) - self.s * self.R @ self.Ym.T
        self.H = self.Xf @ self.tform @ np.linalg.inv(self.Yf)
        self.update_transformer()
        self.TY = self.TY * self.Xs + self.Xm

    def update_variance1(self):
        self.sigma2 = (self.trXPX - self.s * self.trAR) / (self.Np * self.D)
        # if self.scale is True:
        #     self.sigma2 = (xPx - self.s * trAR) / (self.Np * self.D)
        # else:
        #     self.sigma2 = (xPx + self.YPY - self.s * trAR) / (self.Np * self.D)

        if self.sigma2 <= 0:
            # self.sigma2 = self.tolerance / 10
            self.sigma2 = 0.1

        qprev = self.Q
        self.Q = (self.trXPX - 2 * self.s * self.trAR + self.s * self.s * self.trYPY) / (2 * self.sigma2) 
        self.Q += self.D * self.Np/2 * np.log(self.sigma2)
        self.Q += (-self.Np*np.log(self.gamma) - (1-self.Np)*np.log(1-self.gamma))
        self.Q += self.QZ
        self.diff = np.abs(self.Q - qprev)

    def transform_point1(self):
        self.TY = self.s* self.R @ self.Y.T + self.t[:,None] #D*M
        self.TY = self.TY.T

class affine_reg(EMRegistration):
    def __init__(self, *args, B=None, t=None, 
                 tform=None, tforminv=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.B = np.eye(self.D) if B is None else B
        self.t = np.zeros(self.D) if t is None else t

        self.update_transformer(tform=tform, tforminv=tforminv)
        self.sigma2 = np.sum(np.square(self.X - self.Y))/(self.N*self.D)
        self.lle_W()

    def optimization(self):
        self.expectation()
        self.update_transform()
        self.update_transformer()
        self.transform_point()
        self.update_variance()
        self.iteration += 1

    def expectation(self):
        dist2 = np.sum(np.square(self.X-self.TY), axis=1)
        P = self.xp.exp(-dist2/(2*self.sigma2))
        C = (2 * self.xp.pi * self.sigma2)**(self.D/2) * (1-self.gamma)/(self.gamma*self.unif)

        P = P/(P+C) # N*1
        P = self.xp.clip(P, self.eps, None)
        P = ssp.diags(P)
    
        self.P = P
        # self.Pt1 = self.xp.sum(P, axis=0)
        # self.P1 = self.xp.sum(P, axis=1)
        self.Pt1 = self.xp.array(self.P.sum(0)).squeeze()
        self.P1 = self.xp.array(self.P.sum(1)).squeeze()
        self.Np = self.xp.sum(self.P1)
        self.preQ = self.Q
        
        self.Q = np.sum(self.P.diagonal() * dist2)/(2*self.sigma2) + \
            self.Np * self.D * np.log(self.sigma2) / 2 + \
            - self.Np*np.log(self.gamma) - np.sum(1-self.Pt1)*np.log(1-self.gamma)

    def update_transform(self):
        PX = self.P.dot(self.X)
        PY = self.P.dot(self.Y)
        muX = self.xp.divide(self.xp.sum(PX, axis=0), self.Np)
        muY = self.xp.divide(self.xp.sum(PY, axis=0), self.Np)
        #muY = self.xp.divide(self.xp.dot(self.xp.transpose(self.Y), self.P1), self.Np)

        X_hat = self.X - muX
        Y_hat = self.Y - muY
        
        # self.A = np.dot(np.transpose(self.P.dot(X_hat)), Y_hat) 
        self.A = self.X.T @ PY - ( muX[:,None] @ muY[:,None].T ) * self.Np
        # self.A = self.xp.dot(PX.T, Y_hat) - self.xp.outer(muX, self.xp.dot(self.P1.T, Y_hat))
        YPY = self.xp.dot(self.xp.dot(Y_hat.T, self.xp.diag(self.P1)), Y_hat)
        
        ## LLE
        Z = self.llW.transpose().dot(self.P.dot(self.llW))
        YtZY = self.Y.T @ Z.dot(self.Y)
        YtZY *= (2* self.lambd * self.sigma2)

        self.B = self.xp.dot(self.A, self.xp.linalg.inv(YPY+YtZY))
        self.t = muX - self.xp.dot(self.B, muY)

        self.QZ = self.lambd * self.xp.linalg.norm( self.B @ self.Y.T @ np.sqrt(self.P) @ self.llW.T)

    def update_variance(self):
        self.Q += self.QZ
        self.diff = np.abs(self.Q - self.preQ)
        self.diff_r = np.abs(self.diff/self.Q)

        V = np.square(self.X-self.TY)
        self.sigma2 = np.sum( V * self.P1[:, None])/(self.D*self.Np)


        self.Ind = (self.P.diagonal() > self.theta)
        self.gamma = np.clip(self.Ind.sum()/self.M, *self.gamma_clip)

    def transform_point(self, Y=None):
        if Y is None:
            self.TY = self.homotransform_point(self.Y, self.tform, inverse=False)
            return
        else: 
            return self.homotransform_point(Y, self.tform, inverse=False)

    def transform_point1(self):
        self.TY = self.B @ self.Y.T + self.t[:,None] #D*M
        self.TY = self.TY.T

    def update_transformer(self, tform=None, tforminv=None):
        if not tforminv is None:
            tform = self.xp.linalg.inv(self.xp.float64(tforminv))
        if not tform is None:
            self.tform = self.xp.float64(tform)
            self.tforminv = self.xp.linalg.inv(self.tform)
    
            #TODO
            self.B = self.tform[:-1, :-1]
            self.t = self.tform[:-1, [-1]].T
        else:
            self.tform = self.xp.eye(self.D+1, dtype=self.xp.float64)
            self.tform[:self.D, :self.D] = self.B
            self.tform[:self.D, self.D] = self.t
            self.tforminv = self.xp.linalg.inv(self.tform)

    def update_normalize(self):
        self.B *= (self.Xs/self.Ys)
        self.t = (self.t * self.Xs + self.Xm) - self.B @ self.Ym.T
        self.H = self.Xf @ self.tform @ np.linalg.inv(self.Yf)
        self.update_transformer()
        self.TY = self.TY * self.Xs + self.Xm

class deformable_reg(EMRegistration):
    def __init__(self, *args, seed = None,
                 low_k = 15,
                 low_rank=False, num_eig=100, **kwargs):
        super().__init__(*args, **kwargs)

        self.low_rank = low_rank
        self.num_eig = num_eig
        self.low_k=low_k
        self.seed=seed

        self.sigma2 = np.sum(np.square(self.X - self.Y))/(self.N*self.D)
        self.lle_W()
        self.kernal_K()

    def kernal_K(self, ):
        np.random.seed(self.seed)
        uniY = np.unique(self.Y, axis=0)
        ridx = np.random.choice(uniY.shape[0], self.low_k)
        ridx = [65,857,894,570,904,1113,507,78,257,632,1135,1052,307,1154,930]
        ridx = np.int64(ridx) -1
        ctrl_pts = uniY[ridx]
        self.K = self.kernal_gmm(ctrl_pts, ctrl_pts, sigma2= 0.5/self.beta)[0] #k*K
        self.U = self.kernal_gmm(self.Y, ctrl_pts, sigma2= 0.5/self.beta)[0]  #M*k
        self.C = np.zeros((self.low_k, self.D)) # k * D

    def optimization(self):
        self.expectation()
        self.update_transform()
        self.update_transformer()
        self.transform_point()
        self.update_variance()
        self.iteration += 1

    def expectation(self):
        dist2 = np.sum(np.square(self.X-self.TY), axis=1)
        P = self.xp.exp(-dist2/(2*self.sigma2))
        C = (2 * self.xp.pi * self.sigma2)**(self.D/2) * (1-self.gamma)/(self.gamma*self.unif)

        P = P/(P+C) # N*1
        P = self.xp.clip(P, self.eps, None)
        P = ssp.diags(P)
    
        self.P = P
        # self.Pt1 = self.xp.sum(P, axis=0)
        # self.P1 = self.xp.sum(P, axis=1)
        self.Pt1 = self.xp.array(self.P.sum(0)).squeeze()
        self.P1 = self.xp.array(self.P.sum(1)).squeeze()
        self.Np = self.xp.sum(self.P1)
        self.preQ = self.Q

        self.Q = np.sum(self.P.diagonal() * dist2)/(2*self.sigma2) + \
            self.Np * self.D * np.log(self.sigma2) / 2 + \
            - self.Np*np.log(self.gamma) - np.sum(1-self.Pt1)*np.log(1-self.gamma)

        self.QZ = self.lambd/2 * self.xp.linalg.norm( np.sqrt(self.P).dot(self.llW) @ self.TY)

    def update_transform(self):
        Z = self.llW.transpose().dot(self.P.dot(self.llW))
        EtP = (self.P.transpose().dot(self.U)).transpose()
        EtZ = self.lambd*self.sigma2*((Z.transpose().dot(self.U)).transpose())
        PQT = EtP @ self.U + EtZ @ self.U
        PYX = EtP @ self.X - EtP @ self.Y - EtZ @ self.Y
        self.C = np.linalg.solve(PQT, PYX)

    def update_transformer(self):
        self.tform = self.xp.dot(self.U, self.C) # (M,D) + (M,M)*(M,D)

    def transform_point(self, Y=None):
        if Y is None:
            self.TY = self.Y + self.tform
            return
        else:
            Y_n = (Y -self.Ym)/self.Ys*self.Xs if self.normal else Y 
            return Y_n + self.tform

    def update_variance(self):
        self.Q += self.QZ
        self.diff = np.abs(self.Q - self.preQ)
        self.diff_r = np.abs(self.diff/self.Q)

        V = np.square(self.X-self.TY)
        self.sigma2 = np.sum( V * self.P1[:, None])/(self.D*self.Np)

        self.Ind = (self.P.diagonal() > self.theta)
        self.gamma = np.clip(self.Ind.sum()/self.M, *self.gamma_clip)

    def update_normalize(self):
        self.tform = self.tform * self.Xs + self.Xm
        self.TY = self.TY * self.Xs + self.Xm


def lccd_reg( X, Y, transformer='affine', 
                    source_id = None, target_id= None,
                    callback=None, **kwargs):
    TRANS = {
        'rigid':rigid_reg, 
        'euclidean':rigid_reg,
        'similarity':rigid_reg, 
        'affine':affine_reg, 
        'deformable':deformable_reg,
        'constraineddeformable':deformable_reg,
    }
        
    fargs = {}
    if transformer in ['rigid', 'euclidean']:
        fargs.update({'scale': False})
    elif transformer in ['similarity']:
        fargs.update({'scale': True})
    elif transformer in ['deformable']:
        if (not source_id is None) and (not target_id is None):
            transformer = 'constraineddeformable'
    elif transformer in ['constraineddeformable']:
        assert (not source_id is None) and (not target_id is None)

    kwargs.update(fargs)
    model = TRANS[transformer](X, Y, **kwargs)
    model.register(callback)
    return model