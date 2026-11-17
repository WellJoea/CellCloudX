import numpy as np
# import torch.multiprocessing as mp
import torch as th
import scipy.sparse as ssp

from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse.linalg import spsolve, cg

from .neighbors_ensemble import Neighbors
from ...io._logger import logger
from .manifold_regularizers import (low_rank_evd_grbf, centerlize, rbf_kernal, spsparse_to_thsparse)


def uv_solver(Kv, a, h, mu, epsv, rho_y, tau_y, solver='uot', r0=None, the=1.0, eps=1e-8):
    if solver == 'uot':
        u = (a / Kv).pow(the)
        r = None
    elif solver == 'closed':
        r = imbids_marginal_closed(Kv, a, h, mu, epsv, rho_y, eps=eps)
        u = (r / Kv)
    elif solver == 'bisect':
        r = imbids_marginal_bisect(Kv, a, h, mu, epsv, rho_y, tau_y, eps=eps)
        u = (r / Kv)
    elif solver == 'newton':
        r = imbids_marginal_newton(Kv, a, h, mu, epsv, rho_y, tau_y, eps=eps)
        u = (r / Kv)
    elif solver == 'linear':
        z0 = (mu - r0).clamp_min(eps) if r0 is not None else h
        r = imbids_marginal_linear(Kv, a, h, mu, z0, epsv, rho_y, tau_y, eps=eps)
        u = (r / Kv)
    else:
        raise ValueError(f'Unknown solver_mode: {solver}')
    return r, u

def imbids_marginal_closed(hat, a, h, m, epsv, rho, eps=1e-8):
    """
    Closed-form split marginal update with residual weight epsv + rho.

        rb = hat^{eps/(eps+rho)} a^{rho/(eps+rho)}
        r  = m * rb / (rb + h)
    """
    A = epsv + rho
    # rb = ((epsv / A) * th.log(hat) + (rho / A) * th.log(a)).exp()
    rb = (hat ** (epsv / A)) * (a ** (rho / A))
    return (m * rb / (rb + h))

def imbids_marginal_linear(hat, a, h, m, z0, epsv, rho, tau, eps=1e-8):
    """Linearized residual KL update. Fast approximation for arbitrary tau."""
    h = h.view(-1).clamp_min(eps)
    A = epsv + rho
    r = (hat.pow(epsv / A) * a.pow(rho / A) * z0.pow(tau / A)) / h.pow(tau / A)
    return r.clamp(eps, m - eps)

    ell = -tau * th.log(z0 / h)
    r = ((epsv * th.log(hat) + rho * th.log(a) - ell) / A).exp()
    return r.clamp(eps, m - eps)


def imbids_marginal_bisect(hat, a, h, m, epsv, rho, tau, eps=1e-8, n_iter=30):
    """Exact split marginal update for arbitrary residual weight tau."""
    hat = hat.reshape(-1).clamp_min(eps)
    a   = a.reshape(-1).clamp_min(eps)
    h   = h.reshape(-1).clamp_min(eps)
    m   = m.reshape(-1).clamp_min(10 * eps)

    rho = float(rho)
    tau = float(tau)
    epsv = float(epsv)
    A = epsv + rho

    if tau <= 0:
        return th.exp((epsv * th.log(hat) + rho * th.log(a)) / A)

    rhs = epsv * th.log(hat) + rho * th.log(a) - tau * th.log(h)

    lo = th.full_like(m, eps)
    hi = m - eps

    for _ in range(n_iter):
        x = 0.5 * (lo + hi)
        f = A * th.log(x) - tau * th.log(m - x) - rhs
        lo = th.where(f < 0, x, lo)
        hi = th.where(f >= 0, x, hi)
    return 0.5 * (lo + hi)

def imbids_marginal_newton(hat, a, h, m, epsv, rho, tau, eps=1e-8, n_iter=10):
    hat = hat.reshape(-1).clamp_min(eps)
    a   = a.reshape(-1).clamp_min(eps)
    h   = h.reshape(-1).clamp_min(eps)
    m   = m.reshape(-1).clamp_min(10 * eps)

    rho = float(rho)
    tau = float(tau)
    epsv = float(epsv)
    A = epsv + rho

    x = th.exp((epsv * th.log(hat) + rho * th.log(a)) / A)
    # x = x.clamp_min(eps, m - eps)
    x = x.clamp_min(eps)

    if tau <= 0:
        return x

    rhs = epsv * th.log(hat) + rho * th.log(a) - tau * th.log(h)

    lo = th.full_like(m, eps)
    hi = m - eps

    for _ in range(n_iter):
        f  = A * th.log(x) - tau * th.log(m - x) - rhs
        fp = A / x + tau / (m - x)

        lo = th.where(f < 0, x, lo)
        hi = th.where(f >= 0, x, hi)

        xn = x - f / fp
        x = th.where((xn > lo) & (xn < hi), xn, 0.5 * (lo + hi))
    return x

def qnorm(E, loq=0.01, hiq=0.995, eps=1e-8):
    E = E.view(-1).clamp_min(eps)
    lo = th.quantile(E.detach(), float(loq))
    hi = th.quantile(E.detach(), float(hiq))
    if (hi - lo) < 10.0 * eps:
        return th.ones_like(E)
    return ((E - lo) / (hi - lo)).clamp(0.0, 1.0)

def imbids_split_budget(E, m, w, gamma=1.0, z=None, zqt=0.75, smoother=None,
                        lambda_s=0.0, loq=0.01, hiq=0.995, eps=1e-8):
    """
    Direct-KL evidence budget split.

    E : pre-budget evidence, e.g. K @ p or K.T @ q
    m : total marginal capacity
    w : overlap prior

    return a, h, alpha, q, s
        a     : inlier budget
        h     : residual/outlier budget
        alpha : inlier posterior
        q     : inlier shape
        s     : residual shape
    """
    E = E.view(-1).clamp_min(eps) # E *sf
    m = m.view(-1).clamp_min(eps)

    L = qnorm(E, loq=loq, hiq=hiq, eps=eps).pow(float(gamma))
    if z is None:
        z = th.quantile(L.detach(), float(zqt)).clamp_min(eps)
    alpha = (w * L) / (w * L + (1.0 - w) * z) # alpha \in (0,1)
    if smoother is not None and lambda_s > 0:
        logit = False
        if logit: #logit smoother
            lgt = th.log(alpha) - th.log1p(-alpha)
            lgt = (1.0 - float(lambda_s)) * lgt + float(lambda_s) * smoother.smooth(lgt)
            alpha = th.sigmoid(lgt).clamp(eps, 1.0 - eps)
        else:
            alpha = smoother.smooth(alpha)

    a = m * alpha
    h = m * (1.0 - alpha)
    # q = a * sf / a.sum()
    # s = h * sf / h.sum()
    return a, alpha, h

def imbids_qt_budget(E, w, gamma=1.0, zqt=0.75, smoother=None,
                        lambda_s=0.0, loq=0.01, hiq=0.995, eps=1e-8):

    L = qnorm(E, loq=loq, hiq=hiq, eps=eps).pow(float(gamma))
    z = th.quantile(L.detach(), float(zqt)).clamp_min(eps)
    alpha = (w * L) / (w * L + (1.0 - w) * z) # alpha \in (0,1)
    if smoother is not None and lambda_s > 0:
        logit = False
        if logit: #logit smoother
            lgt = th.log(alpha) - th.log1p(-alpha)
            lgt = (1.0 - float(lambda_s)) * lgt + float(lambda_s) * smoother.smooth(lgt)
            alpha = th.sigmoid(lgt).clamp(eps, 1.0 - eps)
        else:
            alpha = smoother.smooth(alpha)
    return alpha


def exbids_split_budget(E, q, w=0.5, gamma=1.0, zqt=0.75,
                        loq=0.01, hiq=0.995, eps=1e-8):
    L = qnorm(E, loq=loq, hiq=hiq, eps=eps).pow(float(gamma)).clamp_min(eps)
    z = th.quantile(L.detach(), float(zqt))

    alpha = (w * L) / (w * L + (1.0 - w) * z )
    chi = 1.0 - alpha
    # chi = chi * (sf / ((q * chi).sum() ))
    chi = chi  * (q.sum() / (q * chi).sum() )
    
    qK = q * alpha
    qK = qK / (qK.sum() )
    return chi, qK

def imbids_otm_refresh_budget(r, a=None, bm='fixed', sm='none', kappa =None,
                         rho =None, gamma=0.7, dtmp=1.0,
                         smoother=None,
                         eps=1e-8):
    """Budget refresh used by implicit Bi-DS-OTM."""
    if bm == 'fixed':
        assert a is not None, 'a must be provided for fixed budget_mode'
        return a

    rr = r.clone() #** dtmp
    if a is None:
        qbar = th.full_like(r, 1.0 / (r.shape[0]))
        qsum = 1.0
    else:
        qbar = a
        qsum = a.sum()

    rho = 1.0 - gamma
    if bm in ['slack_free', 'slack_fixed']: # r, rho, gamma,  kappa
        if sm in ('mrf', 'evidence'):
            rr = smoother.smooth(rr)
        # q = (rho * rr + gamma * qbar)/ (rho * rr.sum() + gamma* qsum)
        a = (kappa* rho/rr.sum() * rr + kappa* gamma/qsum * qbar)/ (rho  + gamma)

    elif  bm == 'free': # r, rho, gamma,
        if sm in ('mrf', 'budget'):
            rr = smoother.smooth(rr)
        a_n = (rho * rr + gamma * qbar)/ (rho + gamma)
        if sm in ('budget'):
            a_n = smoother.smooth(a, pi_n=a_n)
        a = a_n
    else:
        raise ValueError(f'Unknown implicit budget_mode: {bm}')
    return a


class MRFSmoother:
    def __init__(self, Y, sm='gl', st ='prop_gl', alpha=0.5, 
                 beta_mrf=0.8,
                 num_eig=100,  use_keops=True, fast_rank=True,
                 p_step = 1, knn=15,  kd_method ='sknn', use_sym=True, 
                 verbose=0,
                 U=None, S=None, G=None):

        self.M = Y.shape[0]
        self.verbose = verbose
        self.alpha = alpha
        self.eta = alpha/(1.0 + alpha)
        self.sm = sm
        self.st = st
        self.p_step = p_step

        self.fast_rank=fast_rank

        if self.sm in ['gauss', 'gau', 'rbf', 'gaussian']:
            self.rbf_l(Y, beta_mrf, num_eig=num_eig,  use_keops=use_keops, U=U, S=S, G=G)
            self.smooth = self.rbf_smooth
        elif self.sm == 'wl':
            self.w_l(Y, beta_mrf, num_eig=num_eig,  use_keops=use_keops, U=U, S=S)
            self.smooth = self.w_smooth
        elif self.sm == 'gl':
            self.gl_l( Y, knn=knn,  method=kd_method, use_sym=use_sym)
            self.smooth = self.gl_smooth
        elif self.sm == 'wgl':
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
        D = W.sum(1)

        if self.st  == 'prop_rw':
            D[D==0] = 1
            S = W / D[:, None]
            self.L = spsparse_to_thsparse(S).to(Y.device, dtype=Y.dtype)
        elif self.st  == 'prop_gl':
            D[D==0] = 1
            D_half = np.power(D, -0.5)
            S = (W * D_half.reshape(-1, 1)) * D_half.reshape(1, -1)
            self.L = spsparse_to_thsparse(S).to(Y.device, dtype=Y.dtype)
        else:
            self.L = ssp.diags(D) - W

    def gl_smooth(self, pi_raw, pi_n=None):
        alpha=self.alpha
        if self.st == 'cg':
            def matvec(x):
                return x + alpha * self.L.dot(x)

            A_operator = LinearOperator(shape=(self.M, self.M), matvec=matvec, dtype=np.float64)
            diag_A = 1 + alpha * self.L.diagonal()
            M_diag = 1.0 / diag_A
            M_operator = LinearOperator(shape=(self.M, self.M), matvec=lambda x: M_diag * x, dtype=np.float64)

            pi_s, info = cg(A_operator, pi_raw, maxiter=1000, M=M_operator)
            if self.verbose:
                print(f"Warning: CG did not converge after 1000 iterations. Info: {info}")
        elif self.st == 'cho':
            from scipy.linalg import cho_factor, cho_solve
            A = ssp.eye(self.M) + alpha * self.L
            try:
                c, low = cho_factor(A)
                pi_s = cho_solve((c, low), pi_raw)
            except Exception:
                pi_s = np.linalg.solve(A, pi_raw)
            return pi_s
        elif self.st == 'exact':
            A = ssp.eye(self.M) + alpha * self.L
            pi_s = spsolve(A, pi_raw)

        elif self.st in ['prop_gl', 'prop_rw' ]:
            A = ssp.eye(self.M) + alpha * self.L
            if pi_n is None:
                pi_s = pi_raw
                for _ in range(self.p_step):
                    pi_s = (1- self.eta) * pi_raw + (self.eta * pi_s) @ self.L
            else:
                pi_s = (1- self.eta) * pi_raw + (self.eta * pi_n) @ self.L
        else:
            raise ValueError('Unknown method type, use cg, cho, exact or prop')

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
