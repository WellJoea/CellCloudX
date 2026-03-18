from tqdm import tqdm
import itertools
import numpy as np
import torch.multiprocessing as mp
import torch as th
import scipy.sparse as ssp

from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse.linalg import spsolve, cg

from .neighbors_ensemble import Neighbors
from ...utilis._clean_cache import clean_cache
from ...io._logger import logger

class deformable_regularizer(object):
    def __init__(self, beta =1.0, kw=15, kl=20, num_eig=100,
                gamma1=0.0, gamma2=0.0, use_p1=True,
                use_fast_low_rank = False, use_low_rank=False,
                low_rank_type = 'keops',
                use_unique=False, kd_method='sknn', xp = None, verbose=True, **kwargs):
        self.beta = beta
        self.kw = kw
        self.kl = kl
        self.low_rank_type = low_rank_type # keops or fgt
        self.use_fast_low_rank = bool(use_fast_low_rank)
        self.use_low_rank = bool(use_low_rank)
        self.use_fast = self.use_fast_low_rank or self.use_low_rank

        self.use_unique = bool(use_unique)
        self.kd_method = kd_method
        self.num_eig = num_eig
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.use_p1 = bool(use_p1)
        self.xp =  xp
        self.verbose = verbose
        for ia in ['G', 'U', 'S', 'L', 'A', 'I', 'LV', 'LY', 'AV', 'J']:
            setattr(self, ia, None)

    def compute(self, Y, device=None, dtype=None,):
        Y = Y.detach().clone().to(device, dtype=dtype)
        
        if self.verbose: logger.info(f'compute G EVD ({Y.shape[0]}): use fast/low_rank({self.low_rank_type})'
                                     f' = {self.use_fast_low_rank}/{self.use_low_rank}...')
        if self.use_fast:
            if self.low_rank_type == 'keops':
                # U, S = low_rank_evd_grbf_keops(Y, self.beta, self.num_eig) #TODO
                U, S  = low_rank_evd_grbf(Y, self.beta, self.num_eig, use_keops=True)
            elif self.low_rank_type == 'fgt':
                if self.use_fast_low_rank:
                    U, S  = low_rank_evd_grbf(Y, self.beta, self.num_eig, sw_h=0, use_keops=False)
                elif self.use_low_rank:
                    G = rbf_kernal(Y, Y, self.beta, temp=1.0, use_keops=False,)
                    U, S = low_rank_evd( G, self.num_eig) #(G+G.T)/2
            else:
                raise ValueError(f'low_rank_type {self.low_rank_type} not supported')
        else:
            G = rbf_kernal(Y, Y, self.beta, temp=1.0, use_keops=False,)

        if (self.gamma1>0):
            if self.verbose: logger.info('compute Y lle...')
            L = lle_w(Y, use_unique = self.use_unique, kw=self.kw, method=self.kd_method)

        if (self.gamma2>0):
            if self.verbose: logger.info('compute Y gl...')
            A = gl_w(Y, kw=self.kl, method=self.kd_method)

        if self.use_fast:
            self.G = None
            self.U = U
            self.S = self.xp.diag(S)
            self.I = self.xp.eye(self.num_eig, dtype=dtype, device=device)
            V = self.U @ self.S
            if (self.use_p1):
                if (self.gamma1>0):
                    self.L = L
                    self.LV = L @ V
                    self.LY = L @ Y
                if (self.gamma2>0):
                    self.A = A
                    self.AV = A @ V
            else:
                self.J = 0
                if (self.gamma2>0):
                    RV = A.T @ (A @ V)
                else:
                    RV = 0
            
                if (self.gamma1>0):
                    QV = L.T @ (L @ V)
                    self.J =  QV + RV*(self.gamma2/self.gamma1)
                    self.QY = L.T @ (L @ Y)
                else:
                    self.J = RV
                    self.QY = 0
        else:
            self.G = G
            self.U, self.S = None, None
            if (self.use_p1):
                if (self.gamma1>0):
                    self.L = L
                    self.LG = L @ G
                    self.LY = L @ Y
                if (self.gamma2>0):
                    self.A = A
                    self.AG = A @ G
            else:
                if (self.gamma2>0):
                    self.RG = A.T @ (A @ G)
                if (self.gamma1>0):
                    self.QG = L.T @ (L @ G)
                    self.QY = L.T @ (L @ Y)

class pwdeformable_regularizer(object):
    def __init__(self, beta =1.0, kw=15, kl=20, num_eig=100,
                gamma1=0.0, gamma2=0.0, use_p1=True,
                use_fast_low_rank = False, use_low_rank=False,
                low_rank_type = 'keops', #use_mrf=False, gamma_mrf=1.0,
                use_unique=False, kd_method='sknn', xp = None, verbose=True, **kwargs):
        self.beta = beta
        self.kw = kw
        self.kl = kl
        self.low_rank_type = low_rank_type # keops or fgt
        self.use_fast_low_rank = bool(use_fast_low_rank)
        # self.use_mrf = use_mrf
        # self.gamma_mrf = gamma_mrf

        self.use_low_rank = bool(use_low_rank)
        self.use_fast = self.use_fast_low_rank or self.use_low_rank

        self.use_unique = bool(use_unique)
        self.kd_method = kd_method
        self.num_eig = num_eig
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.use_p1 = bool(use_p1)
        self.xp =  xp
        self.verbose = verbose
        for ia in ['G', 'U', 'S', 'I', 'J1', 'J2', 'J3', 'E', 'F']:
            setattr(self, ia, None)

    def compute(self, Y, device=None, dtype=None,):
        Y = Y.detach().clone().to(device, dtype=dtype)
        # beta_sf = centerlize(Y)[2] #TODO
    
        if self.verbose: logger.info(f'compute G EVD ({Y.shape[0]}): use fast/low_rank({self.low_rank_type})'
                                     f' = {self.use_fast_low_rank}/{self.use_low_rank}...')
        if self.use_fast:
            if self.low_rank_type == 'thops':
                U, S  = low_rank_evd_grbf(Y, self.beta, self.num_eig, use_keops=True)
                # U, S  = low_rank_evd_grbf_thops(Y, self.beta, self.num_eig, niter=1, oversampling=20)
            elif self.low_rank_type == 'thops_':
                U, S  = low_rank_evd_grbf_thops(Y, self.beta, self.num_eig, niter=1, oversampling=20)
            elif self.low_rank_type == 'keops':
                # U, S = low_rank_evd_grbf_keops(Y, self.beta, self.num_eig) #TODO
                U, S  = low_rank_evd_grbf(Y, self.beta, self.num_eig, use_keops=True)
            elif self.low_rank_type == 'fgt':
                if self.use_fast_low_rank:
                    U, S  = low_rank_evd_grbf(Y, self.beta, self.num_eig, sw_h=0, use_keops=False)
                elif self.use_low_rank:
                    G = rbf_kernal(Y, Y, self.beta, temp=1.0, use_keops=False,)
                    U, S = low_rank_evd( G, self.num_eig) #(G+G.T)/2
            else:
                raise ValueError(f'low_rank_type {self.low_rank_type} not supported')
            
        else:
            G = rbf_kernal(Y, Y, self.beta, temp=1.0, use_keops=False,)

        if any(self.gamma1>0):
            if self.verbose: logger.info('compute Y lle...')
            L = lle_w(Y, use_unique = self.use_unique, kw=self.kw, method=self.kd_method)

        if any(self.gamma2>0):
            if self.verbose: logger.info('compute Y gl...')
            A = gl_w(Y, kw=self.kl, method=self.kd_method)

        if self.use_fast:
            self.G = None
            self.U = U
            self.S = self.xp.diag(S)
            self.I = self.xp.eye(self.num_eig, dtype=dtype, device=device)
            V = self.U @ self.S
            if (self.use_p1):
                if any(self.gamma1>0):
                    self.E = L
                    self.J1 = L @ V
                    self.J3 = L @ Y
                if any(self.gamma2>0):
                    self.F = A
                    self.J2 = A @ V
            else:
                if any(self.gamma2>0):
                    self.J2 = A.T @ (A @ V)
                else:
                    self.J2 = 0
            
                if any(self.gamma1>0):
                    self.J1 = L.T @ (L @ V)
                    self.J3 = L.T @ (L @ Y)
                else:
                    self.J1 = 0
                    self.J3 = 0

            # if self.use_mrf:
            #     D = U @ (S * U.T.sum(1))
            #     K = 1.0/(1.0 + self.gamma_mrf * D)
            #     C = U.T * K
            #     B = th.linalg.solve(C @ U - th.diag(1/(self.gamma_mrf*S)) , C.T,  left=False)
            #     self.K = K
            #     self.B = B

        else:
            self.G = G
            self.U, self.S = None, None
            if (self.use_p1):
                if any(self.gamma1>0):
                    self.E = L
                    self.J1 = L @ G
                    self.J3 = L @ Y
                if any(self.gamma2>0):
                    self.F = A
                    self.J2 = A @ G
            else:
                if any(self.gamma2>0):
                    self.J2 = A.T @ (A @ G)
                else:
                    self.J2 = 0
            
                if any(self.gamma1>0):
                    self.J1 = L.T @ (L @ G)
                    self.J3 = L.T @ (L @ Y)
                else:
                    self.J1 = 0
                    self.J3 = 0

            # if self.use_mrf:
            #     D = U @ (S * U.T.sum(1))
            #     K = 1.0/(1.0 + self.gamma_mrf * D)
            #     C = U.T * K
            #     B = th.linalg.solve(C @ U - th.diag(1/(self.gamma_mrf*S)) , C.T,  left=False)
            #     self.K = K
            #     self.B = B

class rfdeformable_regularizer(object):
    def __init__(self, beta =1.0, kw=15, kl=20, num_eig=100,
                gamma1=0.0, gamma2=0.0, use_p1=True,
                low_rank=3000, fast_low_rank = 5000, 
                use_fast_low_rank = False, use_low_rank=False,
                low_rank_type = 'keops',
                use_unique=False, kd_method='sknn', xp = None, verbose=True, **kwargs):
        self.beta = beta
        self.kw = kw
        self.kl = kl
        self.low_rank_type = low_rank_type # keops or fgt
        self.low_rank = low_rank
        self.fast_low_rank = fast_low_rank
        self.use_fast_low_rank = bool(use_fast_low_rank)
        self.use_low_rank = bool(use_low_rank)
        self.use_fast = self.use_fast_low_rank or self.use_low_rank

        self.use_unique = bool(use_unique)
        self.kd_method = kd_method
        self.num_eig = num_eig
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.use_p1 = bool(use_p1)
        self.xp =  xp
        self.verbose = verbose
        for ia in ['G', 'U', 'S', 'I', 'J1', 'J2', 'J3', 'E', 'F', ]:
            setattr(self, ia, None)

    def compute(self, Y, device=None, dtype=None,):
        Y = Y.detach().clone().to(device, dtype=dtype)
        
        if self.verbose: logger.info(f'compute G EVD ({Y.shape[0]}): use fast/low_rank({self.low_rank_type})'
                                     f' = {self.use_fast_low_rank}/{self.use_low_rank}...')
        if self.use_fast:
            if self.low_rank_type == 'thops':
                U, S  = low_rank_evd_grbf(Y, self.beta, self.num_eig, use_keops=True)
                # U, S  = low_rank_evd_grbf_thops(Y, self.beta, self.num_eig, niter=1, oversampling=20)
            elif self.low_rank_type == 'thops_':
                U, S  = low_rank_evd_grbf_thops(Y, self.beta, self.num_eig, niter=1, oversampling=20)
            elif self.low_rank_type == 'keops':
                # U, S = low_rank_evd_grbf_keops(Y, self.beta, self.num_eig) #TODO
                U, S  = low_rank_evd_grbf(Y, self.beta, self.num_eig, use_keops=True)
            elif self.low_rank_type == 'fgt':
                if self.use_fast_low_rank:
                    U, S  = low_rank_evd_grbf(Y, self.beta, self.num_eig, sw_h=0, use_keops=False)
                elif self.use_low_rank:
                    G = rbf_kernal(Y, Y, self.beta, temp=1.0, use_keops=False,)
                    U, S = low_rank_evd( G, self.num_eig) #(G+G.T)/2
            else:
                raise ValueError(f'low_rank_type {self.low_rank_type} not supported')
        else:
            G = rbf_kernal(Y, Y, self.beta, temp=1.0, use_keops=False,)

        if any(self.gamma1>0):
            if self.verbose: logger.info('compute Y lle...')
            L = lle_w(Y, use_unique = self.use_unique, kw=self.kw, method=self.kd_method)

        if any(self.gamma2>0):
            if self.verbose: logger.info('compute Y gl...')
            A = gl_w(Y, kw=self.kl, method=self.kd_method)

        if self.use_fast:
            self.G = None
            self.U = U
            self.S = self.xp.diag(S)
            self.I = self.xp.eye(self.num_eig, dtype=dtype, device=device)
            V = self.U @ self.S
            if (self.use_p1):
                if any(self.gamma1>0):
                    self.E = L
                    self.J1 = L @ V
                    self.J3 = L @ Y
                if any(self.gamma2>0):
                    self.F = A
                    self.J2 = A @ V
            else:
                if any(self.gamma2>0):
                    self.J2 = A.T @ (A @ V)
                else:
                    self.J2 = 0
            
                if any(self.gamma1>0):
                    self.J1 = L.T @ (L @ V)
                    self.J3 = L.T @ (L @ Y)
                else:
                    self.J1 = 0
                    self.J3 = 0
        else:
            self.G = G
            self.U, self.S = None, None
            if (self.use_p1):
                if any(self.gamma1>0):
                    self.E = L
                    self.J1 = L @ G
                    self.J3 = L @ Y
                if any(self.gamma2>0):
                    self.F = A
                    self.J2 = A @ G
            else:
                if any(self.gamma2>0):
                    self.J2 = A.T @ (A @ G)
                else:
                    self.J2 = 0
            
                if any(self.gamma1>0):
                    self.J1 = L.T @ (L @ G)
                    self.J3 = L.T @ (L @ Y)
                else:
                    self.J1 = 0
                    self.J3 = 0

    def compute_pair(self, Y1,  Y2, device=None, dtype=None,):
        Y1 = Y1.detach().clone().to(device, dtype=dtype)
        Y2 = Y2.detach().clone().to(device, dtype=dtype)

        if self.low_rank_type in ['thops_', 'thops', 'keops']:
            U, S, Vh  = low_rank_svd_grbf_thops(Y1, Y2, self.beta, self.num_eig, niter=1, oversampling=20)
        elif self.low_rank_type == 'keops':
            U, S, Vh  = low_rank_svd_grbf(Y1, Y2, self.beta, self.num_eig, use_keops=True)
        elif self.low_rank_type == 'fgt':
            try:
                U, S, Vh  = low_rank_svd_grbf(Y1, Y2, self.beta, self.num_eig, sw_h=0, use_keops=False)
            except:
                G = rbf_kernal(Y1, Y2, self.beta, temp=1.0, use_keops=False,)
                U, S, Vh = low_rank_evd( G, self.num_eig) #(G+G.T)/2
        return U, S, Vh

    def compute_pair0(self, Y1,  Y2, device=None, dtype=None,):
        Y1 = Y1.detach().clone().to(device, dtype=dtype)
        Y2 = Y2.detach().clone().to(device, dtype=dtype)

        use_low_rank = ( self.low_rank if type(self.low_rank) == bool  
                                else bool( (Y1.shape[0]*Y2.shape[0])**0.5 >= self.low_rank) )
        use_fast_low_rank = ( self.fast_low_rank if type(self.fast_low_rank) == bool  
                                else bool( (Y1.shape[0]*Y2.shape[0])**0.5 >= self.fast_low_rank) )
        use_fast = use_low_rank or use_fast_low_rank
        if use_fast:
            if self.low_rank_type == 'keops':
                U, S, Vh  = low_rank_svd_grbf(Y1, Y2, self.beta, self.num_eig, use_keops=True)
            elif self.low_rank_type == 'fgt':
                if use_fast_low_rank:
                    U, S, Vh  = low_rank_svd_grbf(Y1, Y2, self.beta, self.num_eig, sw_h=0, use_keops=False)
                elif use_low_rank:
                    G = rbf_kernal(Y1, Y2, self.beta, temp=1.0, use_keops=False,)
                    U, S, Vh = low_rank_evd( G, self.num_eig) #(G+G.T)/2
            else:
                raise ValueError(f'low_rank_type {self.low_rank_type} not supported')
            return U, S, Vh
        else:
            G = rbf_kernal(Y1, Y2, self.beta, temp=1.0, use_keops=False,)
            return G

class MRFSmoother:
    def __init__(self, Y, method='wgl', beta_mrf=0.8, alpha_mrf=0.5, 
                 num_eig=100,  use_keops=True, fast_rank=True,
                  knn=15,  kd_method ='sknn', use_sym=True, verbose=0,
                 U=None, S=None, G=None):

        self.M = Y.shape[0]
        self.verbose = verbose
        self.alpha = alpha_mrf

        self.fast_rank=fast_rank

        if method in ['gauss', 'gau', 'rbf', 'gaussian']:
            self.rbf_l(Y, beta_mrf, num_eig=num_eig,  use_keops=use_keops, U=U, S=S, G=G)
            self.smooth = self.rbf_smooth
        elif method == 'gl':
            self.gl_l( Y, knn=knn,  method=kd_method, use_sym=use_sym)
            self.smooth = self.gl_smooth
        elif method == 'wl':
            self.w_l(Y, beta_mrf, num_eig=num_eig,  use_keops=use_keops, U=U, S=S)
            self.smooth = self.w_smooth
        elif method == 'wgl':
            self.wgl_l(Y, beta_mrf, knn=knn)
            self.smooth = self.wgl_smooth
        else:
            raise ValueError('Unknown method type, use gauss, wl or gl')

    def rbf_l(self, Y, beta=1, num_eig=100, use_keops=True, U=None, S=None, G=None):
        if not G is None:
            self.fast_rank = False
            use_pre = True
            L = G.clone()
        elif (not U is None) and (not S is None):
            self.fast_rank = True
            use_pre = True
        else:
            use_pre = False
            Y = centerlize(Y)[0]

        if self.verbose: 
            logger.info(f'compute R EVD ({Y.shape[0]}): use fast = {self.fast_rank}...')
            
        if self.fast_rank:
            if not use_pre:
                U, S = low_rank_evd_grbf(Y, beta, num_eig, use_keops=use_keops)
            S = S **0.5
            D = U @ (S * U.T.sum(1))
            A = 1.0/(1.0 + self.alpha * D)
            C = U.T * A
            B = th.linalg.solve(C @ U - th.diag(1/(self.alpha*S)) , C.T,  left=False)

            self.A = A
            self.B = B
            self.C = C
        else:
            if not use_pre:
                L = rbf_kernal(Y, Y,  beta)
            D = L.sum(1)
            L.mul_(-self.alpha)
            L.diagonal().add_(self.alpha*D + 1) 

            # L0 = th.eye(G.shape[0]) + alpha*(th.diag(G.sum(1)) -G)
            Lv = th.linalg.inv(L)
            self.Lv = Lv

    def rbf_smooth(self, pi_raw):
        if self.fast_rank:
            pi_s = pi_raw * self.A - self.B @ (self.C @ pi_raw)
        else:
            pi_s = self.Lv @ pi_raw
        return pi_s 

    def w_l(self, Y, beta=1, num_eig=100, use_keops=True, U=None, S=None):
        if (not U is None) and (not S is None):
            use_pre = True
        else:
            use_pre = False
            Y = centerlize(Y)[0]

        if self.verbose: 
            logger.info(f'compute W EVD ({Y.shape[0]}): use fast = True...')
            
        if not use_pre:
            U, S = low_rank_evd_grbf(Y, beta, num_eig, use_keops=use_keops)
        S = S**self.alpha 
        D = (U @ (S * U.T.sum(1)))**(-0.5)
        self.U = U #*D[:,None]
        self.S = S

    def w_smooth(self, pi_raw):
        pi_s = (self.U * self.S) @ ( self.U.T @ pi_raw)
        return pi_s #* pi_raw

    def gl_l(self, Y, knn=15,  method='sknn', use_sym=True,):
        uY = Y.detach().cpu().numpy()
        M, D = uY.shape

        snn = Neighbors(method=method)
        snn.fit(uY)
        ckdout = snn.transform(uY, knn=knn+1)
        kdx = ckdout[1][:,1:]
        src = kdx.flatten('C')
        dst = np.repeat(np.arange(kdx.shape[0]), kdx.shape[1])

        A = np.ones_like(dst)
        W = ssp.csr_array((A, (dst, src)), shape=(M, M)) # not symmetric
        if use_sym:
            W = (W + W.T)/2.0
        
        self.L = ssp.diags(W.sum(1)) - W
        
    def gl_smooth(self, pi_raw,  use_cg=True):
        alpha=self.alpha
        if use_cg:
            def matvec(x):
                return x + alpha * self.L.dot(x)

            A_operator = LinearOperator(shape=(self.M, self.M), matvec=matvec, dtype=np.float64)
            diag_A = 1 + alpha * self.L.diagonal()
            M_diag = 1.0 / diag_A
            M_operator = LinearOperator(shape=(self.M, self.M), matvec=lambda x: M_diag * x, dtype=np.float64)

            pi_s, info = cg(A_operator, pi_raw, maxiter=1000, M=M_operator)
            if self.verbose:
                print(f"Warning: CG did not converge after 1000 iterations. Info: {info}")

        else:
            from scipy.linalg import cho_factor, cho_solve
            A = ssp.eye(self.M) + alpha * self.L
            try:
                c, low = cho_factor(A)
                pi_s = cho_solve((c, low), pi_raw)
            except Exception:
                pi_s = np.linalg.solve(A, pi_raw)
            return pi_s

    def wgl_l(self, Y,  beta=1, knn=15,  method='sknn', use_sym=False,):
        if self.verbose: 
            logger.info(f'compute G knn ({Y.shape[0]}): knn= {knn}...')
    
        uY = centerlize(Y)[0]
        uY = uY.detach().cpu().numpy()
        M, D = uY.shape

        snn = Neighbors(method=method)
        snn.fit(uY)
        ckdout = snn.transform(uY, knn=knn+1)

        src = ckdout[1].flatten('C')
        dst = np.repeat(np.arange(M), knn+1)

        dist = np.exp(-ckdout[0]**2/beta)
        dist = (dist/dist.sum(1, keepdims=True)).flatten('C')

        W = ssp.csr_array((dist, (dst, src)), shape=(M, M)) # not symmetric
        if use_sym:
            W = (W + W.T)/2.0
        self.W = spsparse_to_thsparse(W).to(Y.device, dtype=Y.dtype)
        
    def wgl_smooth(self, pi_raw):
        pi_s = self.W.matmul(pi_raw)
        return pi_s

    def project_to_simplex(self, v, S=1.0):
        """
        Euclidean projection of v onto {x >=0, sum x = S}
        Implements the O(n log n) algorithm (sorting).
        """
        if S < 0:
            raise ValueError("S must be non-negative")
        n = v.shape[0]
        # shift by mean for numerical stability
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho_candidates = u - (cssv - S) / (np.arange(1, n+1))
        rho = np.nonzero(rho_candidates > 0)[0]
        if rho.size == 0:
            # all project to zero vector with sum S => distribute evenly
            return np.full(n, S / n)
        rho = rho[-1]  # index
        theta = (cssv[rho] - S) / (rho + 1.0)
        w = np.maximum(v - theta, 0.0)
        # numerical guard
        w = np.maximum(w, 0.0)
        # fix tiny rounding errors to ensure exact sum S
        s = w.sum()
        if s <= 0:
            return np.full(n, S / n)
        return w * (S / s)

    def mrf_smooth0(self, pi_raw, L, gamma=1.0, S=None, tol=1e-8, use_cg=True):
        """
        Smooth pi_raw using Laplacian L with strength gamma, enforcing sum(pi)=S and nonnegativity via projection.
        L: csr_matrix (M x M)
        pi_raw: ndarray (M,)
        returns pi_smoothed (M,)
        Steps:
        Solve A pi + lambda 1 = pi_raw  (A = I + 2*gamma*L)
        Use elimination to compute lambda, then pi = A^{-1}(pi_raw - lambda 1)
        Finally, project to simplex sum S.
        """
        M = pi_raw.shape[0]
        if S is None:
            # default keep same total mass as raw
            S = float(pi_raw.sum())
        # Build A = I + 2*gamma*L
        A = eye(M, format='csr') + (2.0 * gamma) * L  # sparse
        # Solve A x = b for b = pi_raw and b = 1 to compute lambda
        # Use sparse solver
        if use_cg:
            # cg returns (x, info)
            x_b, info1 = cg(A, pi_raw, tol=1e-6, maxiter=500)
            x_1, info2 = cg(A, np.ones(M), tol=1e-6, maxiter=500)
            if info1 != 0 or info2 != 0:
                # fallback to direct solver
                b = pi_raw
                x_b = spsolve(A, b)
                x_1 = spsolve(A, np.ones(M))
        else:
            x_b = spsolve(A, pi_raw)
            x_1 = spsolve(A, np.ones(M))

        # compute lambda
        denom = np.dot(np.ones(M), x_1)
        if abs(denom) < 1e-12:
            lam = 0.0
        else:
            lam = (np.dot(np.ones(M), x_b) - S) / denom

        # compute pi = A^{-1}(pi_raw - lambda 1) = x_b - lambda * x_1
        pi_hat = x_b - lam * x_1

        # project to simplex to ensure non-negativity and exact sum S
        pi_proj = self.project_to_simplex(pi_hat, S=S)
        return pi_proj

class ParallelGsPair:
    def __init__(self, num_eig=100, low_rank_g=False, devices=None, verbose=0):
        self.num_eig = num_eig
        self.low_rank_g = low_rank_g
        self.devices = devices or [th.device('cuda' if th.cuda.is_available() else 'cpu')]
        self.verbose = verbose
        self.manager = mp.Manager()
        self.results = self.manager.dict()

        if self.verbose > 1:
            print(f"Initializing memory pools on {len(self.devices)} devices")
        self._init_memory_pools()

    def _init_memory_pools(self):
        self.mem_pools = {}
        for device in self.devices:
            if device.type == 'cuda':
                th.cuda.init()
                self.mem_pools[device] = th.cuda.CUDAPool(
                    device=device,
                    size=th.cuda.get_device_properties(device).total_memory * 0.8
                )

    def compute_pairs(self, Ys, L, h2s, mask=None):
        task_queue, total_tasks = self._preprocess_tasks(Ys, L, h2s, mask)

        ctx = mp.get_context('spawn')
        processes = []
        for device_idx, device in enumerate(self.devices):
            p = ctx.Process(
                target=self._worker,
                args=(task_queue, Ys, h2s, device, device_idx)
            )
            p.start()
            processes.append(p)
        
        self._monitor_progress(processes, total_tasks)
        
        self._postprocess_results(L)

    def _preprocess_tasks(self, Ys, L, h2s, mask):
        task_queue = mp.Queue()
        task_count = 0
        
        device_data = self._preallocate_data(Ys)
        
        for i, j in itertools.product(range(L), range(L)):
            if i == j:
                task_queue.put(('eigen', i, j))
                task_count += 1
            else:
                if mask is not None and mask[i,j] == 0:
                    continue
                # 对称性检查
                if self._check_symmetry(i, j, h2s):
                    self._handle_symmetry(i, j)
                    continue
                task_queue.put(('svd', i, j))
                task_count += 1
        return task_queue, task_count

    def _preallocate_data(self, Ys):
        device_data = {}
        for device in self.devices:
            device_data[device] = [
                Y.to(device, non_blocking=True) 
                for Y in Ys
            ]
        return device_data

    def _check_symmetry(self, i, j, h2s):
        return (f'US{j}{i}' in self.results 
                and h2s[i] == h2s[j]
                and f'Vh{j}{i}' in self.results)

    def _handle_symmetry(self, i, j):
        self.results[f'US{i}{j}'] = self.results[f'Vh{j}{i}'].T
        self.results[f'Vh{i}{j}'] = self.results[f'US{j}{i}'].T

    def _worker(self, queue, Ys, h2s, device, device_idx):
        th.cuda.set_device(device) if device.type == 'cuda' else None
        
        # 启用内存池
        if device in self.mem_pools:
            th.cuda.set_per_process_memory_fraction(0.8, device=device)
            th.cuda.set_allocator(self.mem_pools[device].allocator)
        
        while not queue.empty():
            try:
                task_type, i, j = queue.get_nowait()
                X = Ys[i].to(device, non_blocking=True)
                h2 = h2s[i]
                
                if task_type == 'eigen':
                    Q, S = self._compute_evd(X, h2, device)
                    self.results[f'Q{i}{j}'] = Q.cpu()
                    self.results[f'S{i}{j}'] = S.cpu()
                else:
                    Y = Ys[j].to(device, non_blocking=True)
                    U, S, Vh = self._compute_svd(X, Y, h2, device)
                    self.results[f'US{i}{j}'] = (U * S).cpu()
                    self.results[f'Vh{i}{j}'] = Vh.cpu()
                    
            except Exception as e:
                print(f"Device {device} error: {str(e)}")
                queue.put((task_type, i, j)) 

    def _compute_svd(self, X, Y, h2, device):
        with th.cuda.device(device):
            if (X.shape[0] * Y.shape[0]) >= (self.low_rank_g / 100):
                return self._low_rank_svd_grbf(X, Y, h2**0.5)
            else:
                G = th.cdist(X, Y)
                G.pow_(2).div_(-h2).exp_()
                return self._low_rank_evd(G)

    def _compute_evd(self, X, h2, device):
        with th.cuda.device(device):
            if (X.shape[0] ** 2) >= self.low_rank_g:
                return self._low_rank_evd_grbf(X, h2**0.5)
            else:
                G = th.cdist(X, X)
                G.pow_(2).div_(-h2).exp_()
                return self._low_rank_evd(G)

    def _monitor_progress(self, processes, total_tasks):
        with tqdm(total=total_tasks, desc="Parallel Computing", 
                 disable=not self.verbose) as pbar:
            while any(p.is_alive() for p in processes):
                current = total_tasks - self._count_remaining_tasks(processes)
                pbar.update(current - pbar.n)

        for p in processes:
            if p.exitcode != 0:
                raise RuntimeError(f"Process {p.pid} exited with code {p.exitcode}")

    def _count_remaining_tasks(self, processes):
        return sum(p.is_alive() for p in processes)

    def _postprocess_results(self, L):
        for key in self.results:
            if key.startswith('Q') or key.startswith('S'):
                setattr(self, key, self.results[key])
            elif key.startswith('US'):
                setattr(self, key, self.results[key])
            elif key.startswith('Vh'):
                setattr(self, key, self.results[key])

class Gspair(object):
    def __init__(self,  num_eig=100, use_ifgt = False, xp = None, njob=None, verbose=0):
        if xp is None:
            import torch as xp
        self.xp = xp
        self.njob = njob
        self.use_ifgt = use_ifgt
        self.num_eig = num_eig
        self.verbose = verbose

    def compute_pairs(self, Ys, L, h2s, mask=None):
        import itertools
        with tqdm(total=L*L, 
                    desc="Gs E/SVD",
                    colour='#AAAAAA', 
                    disable=(self.verbose==0)) as pbar:
            for i, j in itertools.product(range(L), range(L)):
                pbar.set_postfix(dict(i=int(i), j=int(j)))
                pbar.update()
                if i==j:
                    (Q, S) = compute_evd(Ys[i], h2s[i], self.num_eig, 
                                            use_ifgt=self.use_ifgt, xp=self.xp)
                    setattr(self, 'Q'+str(i)+str(j), Q)
                    setattr(self, 'S'+str(i)+str(j), S)
                else:
                    if (mask is not None) and (mask[i,j]==0):
                        continue
                    if hasattr(self, f'US{j}{i}') and h2s[i] == h2s[j]:
                        for ia, ib in zip(['US', 'Vh'], ['Vh', 'US']):
                            setattr(self, ia+str(i)+str(j), getattr(self, ib+str(j)+str(i)).T)
                    else:
                        (U, S, Vh) = compute_svd(Ys[i], Ys[j], h2s[i], self.num_eig, 
                                                 use_ifgt=self.use_ifgt/100, xp=self.xp)
                        setattr(self, 'US'+str(i)+str(j), U * S)
                        setattr(self, 'Vh'+str(i)+str(j), Vh)

    def compute_pairs_multhread(self, Ys, L, h2s, mask=None):
        import concurrent.futures
        tasks = []
        for i, j in itertools.product(range(L), range(L)):
            if i == j:
                tasks.append(('eigen', i, j))
            else:
                if mask is not None and mask[i, j] == 0:
                    continue
                if hasattr(self, f'US{j}{i}') and h2s[i] == h2s[j]:
                    # Handle symmetric cases without computation
                    for src_suffix, dst_suffix in zip(['Vh', 'US'], ['US', 'Vh']):
                        src_attr = f"{src_suffix}{j}{i}"
                        dst_attr = f"{dst_suffix}{i}{j}"
                        if hasattr(self, src_attr):
                            setattr(self, dst_attr, getattr(self, src_attr).T)
                tasks.append(('svd', i, j))

        with tqdm(total=len(tasks), 
                desc="Gs E/SVD",
                colour='#AAAAAA', 
                disable=(self.verbose==0)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.njob) as executor:
                futures = []
                for task_type, i, j in tasks:
                    if task_type == 'eigen':
                        future = executor.submit(
                            self._process_eigen_task, Ys[i], h2s[i], i, j
                        )
                    else:
                        future = executor.submit(
                            self._process_svd_task, Ys[i], Ys[j], h2s[i], i, j
                        )
                    future.add_done_callback(lambda fn: pbar.update())
                    futures.append(future)
                # Wait for all tasks to complete
                concurrent.futures.wait(futures)

    def _process_eigen_task(self, Y, h2, i, j):
        (Q, S) = compute_evd(Y, h2, self.num_eig, 
                                use_ifgt=self.use_ifgt, xp=self.xp)
        setattr(self, f'Q{i}{j}', Q)
        setattr(self, f'S{i}{j}', S)

    def _process_svd_task(self, X, Y, h2, i, j):
        (U, S, Vh) = compute_svd(X, Y, h2, self.num_eig, 
                            use_ifgt=self.use_ifgt/100, xp=self.xp)
        setattr(self, f'US{i}{j}', U * S)
        setattr(self, f'Vh{i}{j}', Vh)

def rbf_kernal(X, Y, h2, use_keops=False,
                     device=None, dtype=None,
                    temp=1.0 ):
    device = X.device if device is None else device
    dtype  = X.dtype if dtype is None else dtype
    h = (temp/th.asarray(h2, device=device, dtype=dtype))**0.5
    if use_keops:
        d2f = kodist2( X.to(device, dtype=dtype) * h,
                       Y.to(device, dtype=dtype) * h)
        # d2f = d2f*(-1.0/float(tau2)/temp)
        d2f = d2f*(-1.0)
        d2f = d2f.exp()
    else:
        try:
            d2f = thdist2(X.to(device, dtype=dtype) * h, 
                          Y.to(device, dtype=dtype) * h)
            # d2f.mul_(-1.0/float(tau2)/temp)
            d2f.mul_(-1.0)
            d2f.exp_()
        except:
            clean_cache()
            raise('Memory Error in computing d2f')
    return d2f

def kodist2(X, Y):
    import pykeops
    pykeops.set_verbose(False)
    from pykeops.torch import LazyTensor

    x_i = LazyTensor(X[:, None, :])
    y_j = LazyTensor(Y[None, :, :])
    return ((x_i - y_j)**2).sum(dim=2)

def thdist2(X, Y):
    import torch as th
    D = th.cdist(X, Y, p=2)
    D.pow_(2)

    # D1 = (( X[:, None, :] - Y[None,:, :] )**2).sum(-1)
    return D

def lle_w(Y, kw=15, use_unique=False, rl_w=None,  method='sknn'): #TODO 
    #D2 =np.sum(np.square(Y[:, None, :]- Y[None, :, :]), axis=-1)
    #D3 = D2 + np.eye(D2.shape[0])*D2.max()
    #cidx = np.argpartition(D3, self.knn)[:, :self.knn]

    if hasattr(Y, 'detach'):
        uY = Y.detach().cpu().numpy()
        is_tensor = True
        device = Y.device
        dtype = Y.dtype
    else:
        uY = Y
        is_tensor = False

    if use_unique:
        uY, Yidx = np.unique(uY, return_inverse=True,  axis=0)
    eps = np.finfo(uY.dtype).eps
    M, D = uY.shape
    Mr =  Y.shape[0]

    if rl_w is None:
        rl_w = 1e-6 if(kw>D) else 0

    snn = Neighbors(method=method)
    snn.fit(uY)
    ckdout = snn.transform(uY, knn=kw+1)
    kdx = ckdout[1][:,1:]
    L = []
    for i in range(M):
        kn = kdx[i]
        z = (uY[kn] - uY[i]) #K*D
        G = z @ z.T # K*K
        Gtr = np.trace(G)
        if Gtr != 0:
            G = G +  np.eye(kw) * rl_w* Gtr
        else:
            G = G +  np.eye(kw) * rl_w
        w = np.sum(np.linalg.inv(G), axis=1) #K*1
        #w = solve(G, v, assume_a="pos")
        w = w/ np.sum(w).clip(eps, None)
        L.append(w)
    src = kdx.flatten('C')
    dst = np.repeat(np.arange(kdx.shape[0]), kdx.shape[1])
    L = ssp.csr_array((np.array(L).flatten(), (dst, src)), shape=(M, M))
    if use_unique:
        L = L[Yidx][:, Yidx]
    L  = ssp.eye(Mr) - L
    if is_tensor:
        L = spsparse_to_thsparse(L).to(dtype).to(device)
    return L

def gl_w(Y, Y_feat=None, kw=15, use_unique=False, rl_w=None,  method='sknn'): #TODO 
    if hasattr(Y, 'detach'):
        uY = Y.detach().cpu().numpy()
        is_tensor = True
        device = Y.device
        dtype = Y.dtype
    else:
        uY = Y
        is_tensor = False

    if use_unique:
        uY, Yidx = np.unique(uY, return_inverse=True,  axis=0)

    M, D = uY.shape

    if rl_w is None:
        rl_w = 1e-3 if(kw>D) else 0

    snn = Neighbors(method=method)
    snn.fit(uY)
    ckdout = snn.transform(uY, knn=kw+1)
    kdx = ckdout[1][:,1:]
    src = kdx.flatten('C')
    dst = np.repeat(np.arange(kdx.shape[0]), kdx.shape[1])

    if not Y_feat is None: #TODO
        pass
    else:
        A = np.ones_like(dst)

    L = ssp.csr_array((A, (dst, src)), shape=(M, M)) # not symmetric
    if use_unique:
        L = L[Yidx][:, Yidx]
    
    # symmetric
    # L = L.maximum(L.T)

    D = ssp.diags((L.sum(1) )**(-0.5)) # TODO
    # D[D == 0] = 1

    A = D @ L @ D
    K  = ssp.eye(A.shape[0]) - A

    if is_tensor:
        K = spsparse_to_thsparse(K).to(dtype).to(device)
    return K

def low_rank_evd_grbf(X, h2, num_eig, sw_h=0.0, eps=1e-10, use_keops=True):
    if hasattr(X, 'detach'):
        is_tensor = True
        device = X.device
        dtype = X.dtype
    else:
        is_tensor = False
        device = None
        dtype = None

    M, D  = X.shape
    k = min(M-1, num_eig)

    if use_keops:
        import pykeops
        import torch as th
        pykeops.set_verbose(False)
        from pykeops.torch import Genred, LazyTensor

        # genred = Genred(f'Exp(-SqDist(Xi, Xj) * H ) * Vj',
        #             [f'Xi = Vi({D})', f'Xj = Vj({D})', 
        #             'H = Pm(1)', 'Vj = Vj(1)' ],
        #             reduction_op='Sum', axis=1, 
        #             # dtype_acc='float64',
        #              use_double_acc=True,
        #             )
        genred = Genred(f'Exp(-SqDist(Xi, Xj) ) * Vj',
                    [f'Xi = Vi({D})', f'Xj = Vj({D})', 'Vj = Vj(1)'],
                    reduction_op='Sum', axis=1, 
                    # dtype_acc='float64',
                     use_double_acc=True,
                    )
        H = (1.0/th.asarray(h2, device=device, dtype=dtype))**0.5
        X = th.asarray(X, dtype=dtype, device=device)*H

        # x_i = LazyTensor(X[:, None, :])
        # x_j = LazyTensor(X[None, :, :])
        # G  = (-((x_i - x_j) ** 2).sum(2) ).exp()
    
        def matvec(x):
            K = genred(X, X, th.tensor(x, dtype=dtype, device=device)).squeeze(1)
            # K = genred(X, X, H, th.tensor(x, dtype=dtype, device=device)).squeeze(1)
            # K = G @ th.tensor(x, dtype=dtype, device=device)
            return K.detach().cpu().numpy()

    else:
        from ...third_party._ifgt_warp import GaussTransform, GaussTransform_fgt
        H = (1.0/np.array(h2))**0.5
        X = X.detach().cpu().numpy()*H
        trans = GaussTransform_fgt(X, 1.0, sw_h=sw_h, eps=eps) #XX*s/h, s
        def matvec(x):
            return trans.compute(X, x.T)

    lo = ssp.linalg.LinearOperator((M,M), matvec)
    S, Q = ssp.linalg.eigsh(lo, k=k, which='LM') # speed limitation

    eig_indices = np.argsort(-np.abs(S))[:k]
    Q = np.real(Q[:, eig_indices])  # eigenvectors
    S = np.real(S[eig_indices])  # eigenvalues.

    if is_tensor:
        import torch as th
        Q = th.tensor(Q, dtype=dtype, device=device)
        S = th.tensor(S, dtype=dtype, device=device)
    return Q, S

def low_rank_evd_grbf_keops(X, h2, num_eig, device=None, eps=0): #TODO
    import torch as xp
    import cupyx.scipy.sparse.linalg as cussp_lg
    import cupy as cp
    import pykeops
    pykeops.set_verbose(False)
    from pykeops.torch import Genred, LazyTensor

    device = xp.device(X.device if device is None else device)
    thdtype = X.dtype
    # cpdtype = eval(f"cp.{str(thdtype).split('.')[-1]}")

    M, D= X.shape
    k = min(M-1, num_eig)
    genred = Genred(f'Exp(-SqDist(Xi, Xj) * H ) * Vj',
                     [f'Xi = Vi({D})', f'Xj = Vj({D})', 
                      'H = Pm(1)', 'Vj = Vj(1)' ],
                     reduction_op='Sum', axis=1, 
                     dtype_acc='float64',
                    #  use_double_acc=True,
                       )
    H = xp.tensor([1.0/h2], device=device, dtype=thdtype)

    with cp.cuda.Device(device.index):
    # cp.cuda.Device(device.index).use()
        def matvec(x):
            K = genred(X, X, H, xp.as_tensor(x)).squeeze(1)
            return cp.asarray(K)

        lo = cussp_lg.LinearOperator((M,M), matvec,  dtype=cp.float64)
        S, Q = cussp_lg.eigsh(lo, k=k, which='LM', maxiter = M*10, tol=eps)

        eig_indices = cp.argsort(-cp.abs(S))[:k]
        Q = cp.real(Q[:, eig_indices])  # eigenvectors
        S = cp.real(S[eig_indices])  # eigenvalues.

    Q = xp.as_tensor(Q)
    S = xp.as_tensor(S)
    return Q, S

def low_rank_evd_grbf_thops(X, h2, num_eig, n_iter=2, oversampling=20):
    import torch as th
    import pykeops
    pykeops.set_verbose(False)
    from pykeops.torch import Genred
    
    device = X.device if isinstance(X, th.Tensor) else 'cuda' if th.cuda.is_available() else 'cpu'
    dtype = X.dtype if isinstance(X, th.Tensor) else th.float32
    
    X = th.as_tensor(X, dtype=dtype, device=device)
    N, D = X.shape

    k = min(N - 1, num_eig)
    L = min(k + oversampling, N)
    
    H = (1.0 / th.tensor(h2, device=device, dtype=dtype))**0.5
    X_scaled = X * H  


    genred = Genred(f'Exp(-SqDist(Xi, Xj)) * Vj',
        [f'Xi = Vi({D})', f'Xj = Vj({D})', f'Vj = Vj({L})'],
        reduction_op='Sum', axis=1, 
        use_double_acc=True,
    )

    def matvec(v):
        return genred(X_scaled, X_scaled, v)
    
    # Initialize random matrix for randomized SVD
    R = th.randn(N, L, device=device, dtype=dtype)
    R = th.linalg.qr(R).Q
    J = matvec(R)
    Q = th.linalg.qr(J).Q
    
    # Power iterations to improve approximation
    for _ in range(n_iter):
        J = matvec(Q)
        Q = th.linalg.qr(J).Q
    
    # Form the low-rank matrix and compute its eigenvectors
    B = Q.T @ matvec(Q)
    S, V = th.linalg.eigh(B)
    Q = Q @ V
    # Q = th.linalg.qr(Q).Q
    
    # Select top k eigenvalues and eigenvectors
    indices = th.argsort(S, descending=True)[:k]
    S = S[indices]
    Q = Q[:, indices]
    
    return Q, S

def low_rank_svd_grbf_thops(X, Y, h2, num_eig, oversampling=30, niter=2):
    """
    Compute the low-rank SVD for a Gaussian Radial Basis Function (GRBF) kernel matrix
    using randomized methods similar to svd_lowrank.
    
    Args:
        X (Tensor): Input tensor of shape (N, D)
        Y (Tensor): Input tensor of shape (M, D)
        h2 (float): Bandwidth parameter for the GRBF kernel
        num_eig (int): Number of eigenvalues to compute
        oversampling (int): Additional number of random vectors for basis approximation
        niter (int): Number of subspace iterations to conduct
        
    Returns:
        U (Tensor): Left singular vectors of shape (N, num_eig)
        S (Tensor): Singular values of length num_eig
        Vh (Tensor): Right singular vectors of shape (num_eig, M)
    """
    import pykeops
    pykeops.set_verbose(False)
    from pykeops.torch import Genred

    device = X.device if isinstance(X, th.Tensor) else 'cuda' if th.cuda.is_available() else 'cpu'
    dtype = X.dtype if isinstance(X, th.Tensor) else th.float32
    
    X = th.as_tensor(X, dtype=dtype, device=device)
    Y = th.as_tensor(Y, dtype=dtype, device=device)
    
    N, D = X.shape
    M, D2 = Y.shape
    assert D == D2, "X and Y must have the same feature dimension"

    k = min(M-1, N-1, num_eig)
    L = min(k + oversampling, N, M)
    
    H = (1.0 / th.tensor(h2, device=device, dtype=dtype))**0.5
    X = X * H
    Y = Y * H
    
    genred = Genred(f'Exp(-SqDist(Xi, Xj) ) * Vj',
        [f'Xi = Vi({D})', f'Xj = Vj({D})', f'Vj = Vj({L})' ],
        reduction_op='Sum', axis=1, 
        # dtype_acc='float64',
        use_double_acc=True,
    )
    
    R = th.randn(M, L, dtype=dtype, device=device)
    J = genred(X, Y, R)
    Q = th.linalg.qr(J).Q

    for i in range(niter):
        J = genred(Y, X, Q)
        Q = th.linalg.qr(J).Q
        J = genred(X, Y, Q)
        Q = th.linalg.qr(J).Q

    B = genred(Y, X, Q)
    U_b, S, Vh = th.linalg.svd(B.T, full_matrices=False)
    U = Q @ U_b

    U = U[:, :k]
    S = S[:k]
    Vh = Vh[:k, :]
    
    return U, S, Vh

def low_rank_svd_grbf(X, Y, h2, num_eig, sw_h=0.0, use_keops=True, 
                      eps =1e-10):
    if hasattr(X, 'detach'):
        is_tensor = True
        device = X.device
        dtype = X.dtype
    else:
        is_tensor = False
        device = None
        dtype = None

    # X = X.copy().astype(np.float32)
    N, (M, D) = X.shape[0], Y.shape
    k = min(M-1, N-1, num_eig)

    if use_keops:
        import pykeops
        import torch as th
        pykeops.set_verbose(False)
        from pykeops.torch import Genred
        genred = Genred(f'Exp(-SqDist(Xi, Xj) ) * Vj',
            [f'Xi = Vi({D})', f'Xj = Vj({D})', 'Vj = Vj(1)' ],
            reduction_op='Sum', axis=1, 
            # dtype_acc='float64',
            use_double_acc=True,
        )
        H = (1.0/th.asarray(h2, device=device, dtype=dtype))**0.5
        X = th.asarray(X, dtype=dtype, device=device) * H
        Y = th.asarray(Y, dtype=dtype, device=device) * H
        def matvec(x):
            mu = genred(X, Y, th.tensor(x, dtype=dtype, device=device)).squeeze(1)
            return mu.detach().cpu().numpy()
        def rmatvec(y):
            nu = genred(Y, X, th.tensor(y, dtype=dtype, device=device)).squeeze(1)
            return nu.detach().cpu().numpy()
    else:
        from ...third_party._ifgt_warp import GaussTransform_fgt
        H = (1.0/np.array(h2))**0.5
        X = X.detach().cpu().numpy() * H
        Y = Y.detach().cpu().numpy() * H
        tran1 = GaussTransform_fgt(X, 1.0, sw_h=sw_h, eps=eps)
        tran2 = GaussTransform_fgt(Y, 1.0, sw_h=sw_h, eps=eps)
        def matvec(x):
            return tran2.compute(X, x.T)
        def rmatvec(y):
            return tran1.compute(Y, y.T)


    lo = ssp.linalg.LinearOperator((N,M), matvec, rmatvec)
    U, S, Vh = ssp.linalg.svds(lo, k=k, which='LM')

    eig_indices = np.argsort(-np.abs(S))[:k]
    U = np.real(U[:, eig_indices])
    S = np.real(S[eig_indices])
    Vh = np.real(Vh[eig_indices,:])
    if is_tensor:
        import torch as th
        U = th.tensor(U, dtype=dtype, device=device)
        S = th.tensor(S, dtype=dtype, device=device)
        Vh = th.tensor(Vh, dtype=dtype, device=device)
    return U, S, Vh

def low_rank_evd(G, num_eig, xp=None):
    if hasattr(G, 'detach'):
        import torch as th
        is_tensor = True
        device = G.device
        dtype = G.dtype
        xp = th
    else:
        is_tensor = False
        xp = np if xp is None else xp
    if G.shape[0] == G.shape[1]:
        S, Q = xp.linalg.eigh(G) #only for symmetric 
        eig_indices = xp.argsort(-xp.abs(S))[:num_eig]
        Q = Q[:, eig_indices]
        S = S[eig_indices]
        if is_tensor:
            Q = Q.clone().to(dtype).to(device)
            S = S.clone().to(dtype).to(device)
        return Q, S
    else:
        U, S, Vh = xp.linalg.svd(G)
        eig_indices = xp.argsort(-xp.abs(S))[:num_eig]
        U = U[:, eig_indices]
        S = S[eig_indices]
        Vh = Vh[eig_indices,:]
        if is_tensor:
            U = U.clone().to(dtype).to(device)
            S = S.clone().to(dtype).to(device)
            Vh = Vh.clone().to(dtype).to(device)
        return U, S, Vh

def compute_svd( X, Y, h2, num_eig, use_ifgt = 1e7, use_keops=True, xp=None):
    if use_keops:
        U, S, Vh = low_rank_svd_grbf(X, Y, h2, num_eig, use_keops=use_keops)
    else:
        if (X.shape[0] * Y.shape[0]) >= use_ifgt:
            U, S, Vh = low_rank_svd_grbf(X, Y, h2, num_eig, use_keops=use_keops)
        else:
            H = xp.tensor(h2)**0.5
            G = xp.cdist(X/H, Y/H)
            G.pow_(2).mul_(-1.0)
            G.exp_()
            U, S, Vh = low_rank_evd(G, num_eig)
        return U, S, Vh

def compute_evd(X, h2, num_eig, use_ifgt = 1e9, use_keops=True, xp=None):
    if use_keops:
        Q, S = low_rank_evd_grbf(X, h2, num_eig, use_keops=use_keops)
    else:
        if (X.shape[0] **2) >= use_ifgt:
            Q, S = low_rank_evd_grbf(X, h2, num_eig, use_keops=use_keops)
        else:
            H = xp.tensor(h2)**0.5
            G = xp.cdist(X/H, X/H)
            G.pow_(2).mul_(-1.0)
            G.exp_()
            Q, S = low_rank_evd(G, num_eig)
        return Q, S

def WoodburyC(Av, U, Cv, V, xp=np):
    UCv = xp.linalg.inv(Cv  + Av * (V @ U))
    Fc = -(Av * Av) * (U @ UCv @ V)
    Fc.diagonal().add_(Av)
    return Fc

def WoodburyB(Av, U, Cv, V, xp=np):
    UCv = xp.linalg.inv(Cv  + V @ Av @ U)
    return  Av - (Av @ U) @ UCv @ (V @ Av)

def WoodburyA(A, U, C, V, xp=np):
    Av = xp.linalg.inv(A)
    Cv = xp.linalg.inv(C)
    UCv = xp.linalg.inv(Cv  + V @ Av @ U)
    return  Av - (Av @ U) @ UCv @ (V @ Av)

def spsparse_to_thsparse(X):
    import torch as th
    XX = X.tocoo()
    values = XX.data
    indices = np.vstack((XX.row, XX.col))
    i = th.LongTensor(indices)
    v = th.tensor(values, dtype=th.float64)
    shape = th.Size(XX.shape)
    return th.sparse_coo_tensor(i, v, shape)

def thsparse_to_spsparse(X):
    XX = X.to_sparse_coo().coalesce()
    values = XX.values().detach().cpu().numpy()
    indices = XX.indices().detach().cpu().numpy()
    shape = XX.shape
    return ssp.csr_array((values, indices), shape=shape)

def centerlize(X, Xm=None, Xs=None, device=None, xp = th):
    device = X.device if device is None else device
    if X.is_sparse: 
        X = X.to_dense()

    X = X.clone().to(device)
    N,D = X.shape
    Xm = xp.mean(X, 0) if Xm is None else Xm.to(device)

    X -= Xm
    Xs = xp.sqrt(xp.sum(xp.square(X))/(N*D/2)) if Xs is None else Xs.to(device) # N
    X /= Xs
    Xf = xp.eye(D+1, dtype=X.dtype, device=device)
    Xf[:D,:D] *= Xs
    Xf[:D, D] = Xm
    return [X, Xm, Xs, Xf]