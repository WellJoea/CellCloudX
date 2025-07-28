from tqdm import tqdm
import numpy as np
import torch as th


from .operation_expectation import pwExpection
from .operation_th import thopt
from .shared_wnn import swnn
from .xmm import kernel_xmm, lle_W, kernel_xmm_k, kernel_xmm_p

from .shared_wnn import swnn
from ...transform import homotransform_point, ccf_deformable_transform_point
from .operation_expectation import rigid_outlier
from ...io._logger import logger

class EMRegistration(thopt, pwExpection):
    def __init__(self, X, Y,  X_feat=None, Y_feat=None,
                 pairs=None,
                 lable=None,
                 maxiter=500,
                 tol = 1e-9,
                 normal='isoscale',

                 sigma2=None,
                 sigma2_min = 1e-7,
                 tau2=None, 
                 tau2_auto=False,
                 tau2_clip= [5e-3, 5.0],
                 tau2_decayto = 0.15,
                 feat_normal='cos', 
                 
                 w=0.1, 
                 wa=0.99,
                 w_clip=[0.05, 1.0-1e-8],
            
                 record = None,

                #  alpha=9e9,
                #  theta=0.75, tol=1e-5, tol_r=1e-5, tol_s=1e-8,
                #  minp=1e-5, rl_w=1e-5, unif=10, kw=15, normal=True,

                   verbose=2,
                 eps = 1e-8, seed=200504,
                 floatx = 'float32', floatxx='float64',
                 device=None, pargs={},
                 **kwargs):
        super().__init__(device=device, device_pre=None, eps=eps,
                          floatx=floatx, floatxx=floatxx, seed=seed)
        self.verbose = verbose

        self.reg_core = None
        self.maxiter = maxiter
        self.tol = tol

        self.sigma2 = sigma2
        self.sigma2_min = sigma2_min


        # self.tol_r = tol_r
        # self.tol_s = tol_s
        # self.alpha = alpha
        # self.theta = theta
        # self.minp = minp
        # self.unif = unif
        # self.kw = kw
        # self.rl_w = rl_w if self.kw > self.D else 0
        # self.p_epoch = p_epoch
        # self.iteration = 0
        self.normal = normal
        self.verbose=verbose

        self.record = [] if record is None else record
        self.records = {}

        self.homotransform_point = homotransform_point
        self.ccf_deformable_transform_point=ccf_deformable_transform_point

        self.lable=lable
        self.init_XY( X, Y, X_feat, Y_feat)
        if self.fexist:
            self.feat_normal = self.scalar2vetor(feat_normal, L=self.LF)
            self.tau2 = self.scalar2vetor(tau2, L=self.LF)
            self.tau2_decayto = self.scalar2vetor(tau2_decayto, L=self.LF)
            self.tau2_clip = self.scalar2vetor(tau2_clip, L=self.LF, force=True)
        else:
            self.feat_normal = None
            self.tau2_clip = None
        self.init_pairs(pairs, pargs=pargs)

        self.normal_XY()
        self.features_distance()
        self.init_outlier(w, w_clip)
        self.w_clip = w_clip
        self.wa = wa

    def init_XY(self, X, Y, X_feat, Y_feat):
        self.Xr = self.to_tensor(X, dtype=self.floatxx, device='cpu')
        self.Yr = self.to_tensor(Y, dtype=self.floatxx, device='cpu')
        assert self.Xr.shape[1] == self.Yr.shape[1]
        self.D = self.Xr.shape[1]
    
        self.fexist = not (X_feat is None or Y_feat is None)
        if self.fexist:
            self.XFr = self.init_feature(X_feat, self.Xr.shape[0])
            self.YFr = self.init_feature(Y_feat, self.Yr.shape[0])
            assert len(self.XFr) == len(self.YFr), "Features must have the same length"
            self.LF = len(self.XFr)
            self.DFs = []
            for il in range(self.LF):
                assert self.XFr[il].shape[1] == self.YFr[il].shape[1], "Features must have the same dimension"
                self.DFs.append(self.XFr[il].shape[1])

    def init_feature(self, Fs, N):
        if isinstance(Fs, (list, tuple)):
            LF = len(Fs)
            Fa = []
            for l in range(LF):
                assert N == len(Fs[l]), "Features must have the same points number with X,Y"
                iF = self.to_tensor(Fs[l], dtype=self.floatxx, device='cpu').clone()
                Fa.append(iF)
        else:
            Fa = [self.to_tensor(Fs, dtype=self.floatxx, device='cpu').clone()]
        return Fa

    def init_outlier(self, w, w_clip):
        if w is None:
            w = rigid_outlier(self.Xr, self.Yr)
        self.w = self.to_tensor(w, dtype=self.floatx, device=self.device)
        if w_clip is not None:
            self.w = self.xp.clip(self.w, *w_clip)
    
    def swnn_pairs(self, X, Y, X_feat, Y_feat, pargs={}):
        dargs = dict(
            kd_method='keops', sp_method = 'sknn',
            mnn=10, snn=30, fnn=50,
            scale_locs=True, scale_feats=True, use_soft=False,
            lower = 0.01, upper=0.995,
            min_score= 0.35,
            max_pairs=5e4,
            min_pairs=200,
            swnn_version=1, device=self.device, dtype=None, verbose=0)
        dargs.update(pargs)
        pairs = []
        for iL in range(len(X_feat)):
            ipair, score = swnn(X, Y, X_feat[iL], Y_feat[iL], **dargs)
            pairs.append(ipair)
        pairs = self.xp.cat(pairs, dim=0)
        pairs = self.xp.unique(pairs, dim=0)
        return pairs
    
    def init_pairs(self, pairs, pargs={}):
        if pairs is None:
            if self.fexist:
                self.pairs = self.swnn_pairs(self.Xr, self.Yr, self.XFr, self.YFr, pargs)
            else:
                assert self.Xr.shape == self.Yr.shape, 'X.shape != Y.shape, pairs indices are required.'
                self.pairs = self.xp.stack([ self.xp.arange(self.Xr.shape[0], dtype=self.xp.int64), 
                                             self.xp.arange(self.Xr.shape[0], dtype=self.xp.int64) ],
                                    axis=1)
        else:
            self.pairs = pairs
        self.pairs = self.to_tensor(self.pairs, dtype=self.xp.int64, device='cpu')

    def normal_XY(self):
        Xa = [self.Xr, self.Yr]
        L = len(Xa)
        M, S = self.xp.zeros((L,self.D)), self.xp.ones((L))

        if self.normal in ['each', 'isoscale', True]:
            for l in range(L):
                iX = self.to_tensor(Xa[l], dtype=self.floatx, device=self.device)
                M[l], S[l] = self.centerlize(iX, Xm=None, Xs=None)[1:3]
            if self.normal in ['isoscale', True]:
                S = self.xp.mean(S).expand(L).clone()
    
        elif self.normal  in ['global']:
            XX = self.xp.concat(Xa, 0).to(self.device, dtype=self.floatx)
            iXm, iXs = self.centerlize(XX, Xm=None, Xs=None)[1:3]
            M = iXm.expand(L, -1).clone()
            S = iXs.expand(L).clone()

        elif self.normal == 'X':
            iX = self.to_tensor(Xa[0], dtype=self.floatx, device=self.device)
            iXm, iXs = self.centerlize(iX, Xm=None, Xs=None)[1:3]
            M = iXm.expand(L, -1).clone()
            S = iXs.expand(L).clone()
        
        elif self.normal in [False, 'pass']:
            pass
        else:
            raise ValueError(
                "Unknown normalization method: {}".format(self.normal))
        Xn, Tf = [], []
        for l in range(L):
            iX = self.to_tensor(Xa[l], dtype=self.floatx, device=self.device)
            iT = self.centerlize(iX, Xm=M[l], Xs=S[l])
            Xn.append(iT[0])
            Tf.append(iT[3].to(dtype=self.floatxx))
        self.X, self.Y = Xn
        self.Xf, self.Yf = Tf
        self.Xm, self.Ym = M.to(self.device, dtype=self.floatxx)
        self.Xs, self.Ys = S.to(self.device, dtype=self.floatxx)

    def features_distance(self):
        self.Fs = []
        if self.fexist:
            for il in range(self.LF):
                inorm = self.feat_normal[il]
                iFx = self.to_tensor(self.XFr[il][self.pairs[:,0]], dtype=self.floatx, device=self.device)
                iFy = self.to_tensor(self.YFr[il][self.pairs[:,1]], dtype=self.floatx, device=self.device)

                if inorm in ['cosine', 'cos'] :
                    iF = self.cos_distance(iFx, iFy, use_pair=True)
                elif inorm in ['kl', 'kl_div']:
                    iF = self.kl_distance(iFx, iFy, use_pair=True, negative_handling ='softmax',)
                elif inorm in ['sym_kl', 'sym_kl_div']:
                    iF1 = self.kl_distance(iFx, iFy, use_pair=True, negative_handling ='softmax',)
                    iF2 = self.kl_distance(iFy, iFx, use_pair=True, negative_handling ='softmax',)
                    iF = (iF1 + iF2) / 2
                elif inorm in ['euc', 'euclidean']:
                    iF = self.euclidean_distance(iFx, iFy, use_pair=True)
                elif inorm in ['soft_label']: #TODO
                    iF = self.label_distance(iFx, iFy, use_pair=True)
                elif inorm in ['label']: #TODO
                    iF = self.label_distance(iFx, iFy, use_pair=True)
                else:
                    logger.warning(f"Unknown feature normalization method: {inorm}")
                self.Fs.append(iF)

    def init_tau2(self,):
        if self.fexist:
            tau2 = self.xp.ones((self.LF))
            for il in range(self.LF):
                if (self.tau2 is None) or (self.tau2[il] is None):
                    tau2[il] = self.Fs[il].mean()
                else:
                    tau2[il] = self.tau2[il]
            
                if self.tau2_clip[il] is not None:
                    tau2[il] = self.xp.clip(tau2[il], *self.tau2_clip[il])
            
            self.tau2 = self.to_tensor(tau2, dtype=self.floatx, device=self.device)  
            self.tau2_decayto = self.to_tensor(self.tau2_decayto, dtype=self.floatx, device=self.device)  
            # self.tau2_decay = self.tau2_decayto** (-1.0 / float(self.maxiter-1))
            self.tau2_grow = [ self.xp.pow( idecay** (-1.0 / float(self.maxiter-1)) , self.xp.arange(self.maxiter))
                               for idecay in self.tau2_decayto ]

    def init_params(self, **kargs):
        self.sigma2 =  self.to_tensor(self.sigma2 or self.sigma_square(self.X, self.Y), 
                                device=self.device)
        self.init_tau2()
        self.N = self.X.shape[0]
        self.TY = self.Y.clone()
    
        self.Q = 1.0 + self.N * self.D * 0.5 * self.xp.log(self.sigma2)
        self.diff = self.Q

    def echo_infors(self, paras=None):
        paras = ['sigma2', 'sigma2f', 'K', 'KF', 'alpha', 'beta',  'gamma',
                    'use_lle', 'kw', 'beta_fg', 'use_fg', 'normal', 'low_rank','w']
        self.echo_paras(paras=paras, dvinfo=[self.get_memory(self.device)] )

    def register(self, callback=lambda **kwargs: None):
        self.adjustable_paras(**kwargs)
        self.init_params()
        self.echo_infors()

        pbar = tqdm(range(self.maxiter), total=self.maxiter, colour='red', desc=f'{self.reg_core}', disable=(self.verbose <1))
        for char in pbar:
            self.optimization()

            if callable(callback):
                kwargs = {'iteration': self.iteration,
                          'error': self.Q, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)

            if len(self.record):
                for ird in self.record:
                    try: #TODO
                        value = getattr(self, ird).clone()
                    except:
                        value = getattr(self, ird)

                    if ird in self.records:
                        self.records[ird].append(value)
                    else:
                        self.records[ird] = [value]

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


    def optimization(self):
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        dist2 = self.xp.sum(self.xp.square(self.X - self.TY), axis=1)
        P = dist2 * (-1.0/ (2.0*self.sigma2) )
        if self.fexist:
            for iL in range(self.LF):
                P.add_( 1.0/(self.tau2[iL] * (self.tau2_grow[iL][self.iteration])) * self.Fs[iL] )
        P.exp_()

        if not self.lable is None:
            P = P * self.lable_score


        gs = ((2 * self.xp.pi * self.sigma2) ** (self.D / 2)) / (1 - self.w)
        cs = self.w / self.unif
        C = P + gs * cs
        C.masked_fill_(C == 0, 1.0)

        P = P / C 
        self.P = P
        self.Pt1 = P
        self.P1 = P
        self.Np = self.xp.sum(self.P)
        self.preQ = self.Q

        self.Q = self.xp.sum(self.P * dist2) / (2 * self.sigma2) + \
                 self.Np * self.D * self.xp.log(self.sigma2) / 2 + \
                 - self.Np * self.xp.log(1- self.w) - (1 - self.Np) * self.xp.log(self.w)

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

    def maximization(self, P, Np, X, Y):
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

        self.TY = self.s * self.R @ self.Y.T + self.t

        V = np.square(self.X - self.TY)
        self.sigma2 = np.sum(V * self.P1[:, None]) / (self.D * self.Np)

        self.Ind = (self.P.diagonal() > self.theta)
        self.gamma = np.clip(self.Ind.sum() / self.N, *self.gamma_clip)


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
        self.TY = self.B @ self.Y.T + self.t

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