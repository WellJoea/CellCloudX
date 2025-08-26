from builtins import super
import numpy as np
from tqdm import tqdm

from .operation_th import thopt
from ...utilis._arrays import list_iter
from .operation_expectation import (neighbor_weight,
                                    expectation_ko, expectation_xp, default_transparas)
from .operation_maximization import init_tmat

from .utility import split_bins_by_bps
from .reference_operation import rfRegularizer_Dataset, rfCompute_Feauture, rfOptimization
from ...io._logger import logger

class rfEMRegistration(thopt, rfOptimization):
    def __init__(self, X, Y, X_feat=None, Y_feat=None, 
                 transformer='E',
                 transparas=None,
                 maxiter = 200,
                 inneriter = 1,
                 tol = 1e-9,

                 floatx = None,
                 floatxx = None,
                 device = None,
                 device_pre = 'cpu',

                 delta =0, 
                 zeta = 0,
                 eta = 0,
                 penal_term = 'both',

                 w=0.5,
                 wa=0.995,
                 w_clip=[0, 0.999],

                 sigma2=None, 
                 sigma2_min = 1e-6,
                 tau2=None, 
                 tau2_auto=False,
                 tau2_clip= [5e-3, 100.0],
                 tau2_decayto = 0.15,
                 tau2_decaystop = 1.0,

                 normal='isoscale', #'isoscale',
                 scale_factor = None,
                 feat_normal='cos', 
                 
                 use_projection=False,
                 zspace= 0.1,
                 use_keops=5e7,
                 record = None,

                 min_points = 100,
                 seed = 491001,
                 verbose = 2,
                 **kargs
                ):
        super().__init__(device=device, device_pre=device_pre,
                          floatx=floatx, floatxx = floatxx, seed=seed)
        self.verbose = verbose

        self.maxiter = maxiter or 300
        self.inneriter = inneriter or 15
        assert self.maxiter % self.inneriter == 0, "maxiter must be multiple of inneriter"

        self.iteration = 0
        self.tol = tol

        self.init_XY(X, Y)
        self.init_feature(X_feat, Y_feat)

        self.normal = normal
        self.scale_factor=scale_factor or 1.0
        self.normal_XY()
        self.normal_features()

        self.delta = self.init_penalty(delta, w_term=penal_term,)
        self.zeta = self.init_penalty(zeta, w_term=penal_term)
        self.eta = self.init_penalty(eta, w_term=penal_term)
        self.zspace= self.init_zspace(zspace)

        self.ws, self.was = self.init_ws(w, wa, w_clip)
        self.w_clip = w_clip
        self.use_keops = use_keops

        self.sigma2=sigma2
        self.sigma2_min = sigma2_min
        self.tau2 = tau2
        self.tau2_auto= tau2_auto
        self.tau2_decayto = tau2_decayto
        self.tau2_decaystop = tau2_decaystop

        if self.fexist:
            self.feat_normal = self.scalar2vetor(feat_normal, L=self.LF)
            self.tau2_clip = self.scalar2vetor(tau2_clip, L=self.LF, force=True)
        else:
            self.feat_normal = None
            self.tau2_clip = None
    
        self.init_transformer(transformer)
        self.init_transparas(transparas)
        
        self.min_points = min_points
        self.use_projection = use_projection
        self.record =  [] if record is None else record
        self.records = {}

    def init_XY(self, X, Y,):
        self.D = X.shape[1]
        assert X.shape[1] == self.D, 'X should be of shape (N, 3)'
        assert Y.shape[1] == self.D, 'Y should be of shape (M, 3)'
        self.Xr = self.to_tensor(X, dtype=self.floatx, device='cpu').clone()
        self.Yr = self.to_tensor(Y, dtype=self.floatx, device='cpu').clone()

        self.Zid, self.Ms = self.xp.unique(self.Yr[:,self.D-1], sorted=True,  return_counts=True)
        self.Yins = [  self.xp.where(self.Yr[:,self.D-1] == zid)[0] for zid in self.Zid]
        self.N, self.M = self.Xr.shape[0], self.Yr.shape[0]
        self.L = self.Zid.shape[0]

    def init_feature(self, X_feat, Y_feat):
        self.fexist = not (X_feat is None or Y_feat is None)
        if self.fexist:
            self.XFr = self.check_feature(X_feat, self.N)
            self.YFr = self.check_feature(Y_feat, self.M)
            assert len(self.XFr) == len(self.YFr), "Features must have the same length"
            self.LF = len(self.XFr)
            self.DFs = []
            for il in range(self.LF):
                assert self.XFr[il].shape[1] == self.YFr[il].shape[1], "Features must have the same dimension"
                self.DFs.append(self.XFr[il].shape[1])

    def check_feature(self, Fs, N):
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
    
        S = S/float(self.scale_factor)
        Xn, Tf = [], []
        for l in range(L):
            iX = self.to_tensor(Xa[l], dtype=self.floatx, device=self.device)
            iT = self.centerlize(iX, Xm=M[l], Xs=S[l])
            Xn.append(iT[0])
            Tf.append(iT[3].to(dtype=self.floatxx))
        self.X, self.Y = Xn
        self.Hx, self.Hy = Tf
        self.Mx, self.My = M.to(self.device, dtype=self.floatxx)
        self.Sx, self.Sy = S.to(self.device, dtype=self.floatxx)

    def normal_features(self):
        if self.fexist:
            Fr, Fs = [self.XFr, self.YFr], []
            for n in range(len(Fr)):
                iFs = []
                for il in range(self.LF):
                    inorm = self.feat_normal[il]
                    iF = self.to_tensor(Fr[n][il], dtype=self.floatx, device=self.device)
                    if inorm in ['cosine', 'cos'] :
                        iF = self.normalize(iF)
                    elif inorm == 'zc':
                        iF = self.centerlize(iF)[0]
                    elif inorm == 'pcc':    
                        iF = self.scaling(iF, anis_var=True)
                    elif inorm == 'pass':    
                        pass
                    else:
                        logger.warning(f"Unknown feature normalization method: {inorm}")
                    iFs.append(iF)
                Fs.append(iFs)
            self.XF, self.YF = Fs
        else:
            self.XF, self.YF =  None, None

    def init_ws(self, w, wa, w_clip):
        ws = self.to_tensor(self.scalar2vetor(w, self.L), dtype=self.floatx, device=self.device)
        ws = ws.clip(*w_clip)
        was = self.to_tensor(self.scalar2vetor(wa, self.L), dtype=self.floatx, device=self.device)
        return ws, was

    def init_penalty(self, omega, w_term='both'):
        omega = neighbor_weight(omega, L= self.L, w_term=w_term, xp =self.xp)
        omega = self.to_tensor(omega, dtype=self.floatx, device=self.device)
        return omega

    def init_zspace(self, zspace):
        zspace = self.xp.tensor(zspace)
        if zspace.ndim == 0:
            zspace = self.xp.ones(self.L-1) * zspace
        elif zspace.ndim == 1:
            zspace = self.xp.asarray(zspace)
        else:
            raise ValueError(f'zspace should be a scalar or a vector of length {self.L-1}')
        assert  all(zspace < 1), "zspace should be smaller than 1"
        return zspace.to(self.device, dtype=self.floatx)

    def init_Xzbins(self, X, X_feats=None):
        bps = self.Zid.to(self.device)
        lpos, rpos, Xs, XFs = split_bins_by_bps(X, bps, X_feats=X_feats, 
                                    min_points=self.min_points, axis = self.D-1, 
                                    zspace=self.zspace, xp=self.xp )
        if self.fexist:
            XFs =  [ ilfs.to(self.device) for ifs in XFs for ilfs in ifs ]
        return Xs, XFs

    def init_Yzbins(self, Y,  Y_feats=None):
        # mask = xp.ones(D, dtype=bool)
        # mask[K] = False
        Ys, YFs = [], []
        for idx in self.Yins:
            Ys.append(Y[idx].to(self.device))
            # TODO
            # if self.use_YZ: 
            #     Ys.append(Y[idx][:, :self.D].to(self.device))
            # else:
            #     self.Ys.append(self.Y[idx][:, :self.D-1].to(self.device))
            if self.fexist:
                iyfs = [ iyf[idx].to(self.device) for iyf in Y_feats ]
            else:
                iyfs = None
            YFs.append(iyfs)
        return Ys,  YFs

    def init_tmat(self, T,  N= None, s_clip=None,  device=None, dtype=None):
        xp = self.xp
        D = self.D
        itam = {}
        itam['tc'] = xp.zeros(1, device=device, dtype=dtype).clone()
        if T in ['E']:
            itam['R'] = xp.eye(D-1, device=device, dtype=dtype).clone()
            itam['t'] = xp.zeros(D-1, device=device, dtype=dtype).clone()
            itam['s'] = xp.tensor(1, device=device, dtype=dtype).clone()
            if s_clip is not None:
                itam['s'] = xp.clip(itam['s'], *s_clip)
        elif T in 'SRTILO':
            itam['R'] = xp.eye(D-1, device=device, dtype=dtype).clone()
            itam['t'] = xp.zeros(D-1, device=device, dtype=dtype).clone()
            itam['s'] = xp.ones(D-1, device=device, dtype=dtype).clone()
            if s_clip is not None:
                itam['s'] = xp.clip(itam['s'], *s_clip).diag().diag()
        elif T in ['A']:
            itam['A'] = xp.eye(D-1, device=device, dtype=dtype).clone()
            itam['t'] = xp.zeros(D-1, device=device, dtype=dtype).clone()
        elif T in ['D']:
            itam['W'] = xp.zeros((N,D-1), device=device, dtype=dtype).clone()
        return itam

    def init_transformer(self, transformer):
        self.alltype = 'ESADRTILO'
        if (type(transformer) == str) and len(transformer.replace(' ', '')) == self.L:
            self.transformer =  list(transformer.replace(' ', ''))
        else:
            self.transformer =  self.scalar2vetor(transformer, self.L)
        for i in self.transformer:
            if i not in self.alltype:
                raise ValueError(f'transformer {i} is not supported, should be one of {self.alltype}')
        if ('D' in self.transformer):
            assert sum( itf == 'D' for itf in self.transformer) == len(self.transformer)
            self.reg_core = 'rf-deformable'
        else:
            self.reg_core = 'rf-linear'

        self.tmats = {}
        for iL in range(self.L):
            self.tmats[iL] =  self.init_tmat(self.transformer[iL], self.Ms[iL], device=self.device, dtype=self.floatx)

    def init_transparas(self, nparas, alltype='ESADRTILO'): 
        dparas = default_transparas({'D':  dict(fast_low_rank = 5000)})
        nparas = {} if nparas is None else nparas
        self.transparas = {}
        for iL in range(self.L):
            itrans = self.transformer[iL]

            ipara = {**dparas[itrans], 
                     **nparas.get(itrans,{}),
                     **nparas.get(iL, {}).get(itrans,{}),
                     **nparas.get(iL, {}) #TODO
                     } 
            if itrans == 'D':
                ipara['alpha_mu'] = ipara['alpha_decayto']**(1.0/ float(self.maxiter-1)) 
                ipara['gamma_nu'] = ipara['gamma_growto']**(1.0/ float(self.maxiter-1))
            
                ipara['use_low_rank'] = ( ipara['low_rank'] if type(ipara['low_rank']) == bool  
                                                else bool(self.Ms[iL] >= ipara['low_rank']) )
                ipara['use_fast_low_rank'] = ( ipara['fast_low_rank'] if type(ipara['fast_low_rank']) == bool  
                                                else bool(self.Ms[iL] >= ipara['fast_low_rank']) )
                ipara['fast_rank'] = ipara['use_low_rank'] or ipara['use_fast_low_rank']
            self.transparas[iL] = ipara
    
    def init_sigma2(self, sigma2,  simga2_clip = [1, None]):
        if sigma2 is None:
            sigma2 = []
            for l in range(self.L):
                sigma2.append(self.sigma_square(self.Xs[l], self.TYs[l]))
        else:
            sigma2 = self.xp.asarray(sigma2, dtype=self.floatx, device=self.device)
            if sigma2.ndim == 0:
                sigma2 = self.xp.ones(self.L, device=self.device)*sigma2
            elif sigma2.ndim == 1:
                assert sigma2.shape[0] == self.L, f'sigma2 should be a scalar or a vector of length {self.L}'

        self.sigma2 = self.to_tensor(sigma2, dtype=self.floatx, device=self.device)
        self.sigma2 = self.sigma2.clip(*simga2_clip)

    def decay_curve(self, decayto, decaystop, maxiter):
        assert 0 <= decaystop <= 1.0
        decaystop = int(decaystop * maxiter)
        decayrate = self.xp.ones((maxiter), device=self.device, dtype=self.floatx)
        decayrate[:decaystop] = self.xp.linspace(1.0, decayto, decaystop)
        decayrate[decaystop:] = decayto
        return decayrate

    def init_tau2(self, tau2, tau2_clip=None, mask=None):
        if self.fexist:
            if tau2 is None:
                tau2 = self.xp.zeros((self.L, self.LF), dtype=self.floatx)
                for l in range(self.L):
                    for lf in range(self.LF):
                        if self.feat_normal[lf] in ['cos', 'cosine']:
                            tau2[l,lf] = self.sigma_square_cos(self.XFs[l][lf], self.YFs[l][lf])
                        else:
                            tau2[l,lf] = self.sigma_square(self.XFs[l][lf], self.YFs[l][lf])
            else:
                tau2 = self.xp.asarray(tau2, dtype=self.floatx, device=self.device)
                if tau2.ndim == 0:
                    tau2 = self.xp.ones((self.L, self.LF), dtype=self.floatx)*tau2
                elif tau2.ndim == 1:
                    if tau2.shape[0] == self.LF:
                        tau2 = tau2.expand(self.L, -1).clone()
                    elif tau2.shape[0] == self.L:
                        tau2 = tau2.expand(self.LF, -1).T.clone()
                elif tau2.ndim == 2:
                        assert tau2.shape == (self.L, self.LF)
            self.tau2 = self.to_tensor(tau2, dtype=self.floatx, device=self.device)
            for lf in range(self.LF):
                if self.feat_normal[lf] in ['cos', 'cosine']:
                    itau_mean = self.sigma_square_cos(self.XF[lf], self.YF[lf])
                else:
                    itau_mean = self.sigma_square(self.XF[lf], self.YF[lf])
                self.tau2[:,lf] = self.tau2[:,lf].clip(itau_mean, None)
        
            if tau2_clip is not None:
                self.tau2 = self.tau2.clip(*tau2_clip)

            self.tau2_prediv = self.scalar2vetor( not self.tau2_auto, L=self.L)
            self.tau2_grow = 1.0/self.decay_curve(self.tau2_decayto, self.tau2_decaystop, self.maxiter)
            self.DK = [2 if self.feat_normal[lf] in ['cos', 'cosine'] else self.XF[lf].shape[1]
                            for lf in range(self.LF) ]
        else:
            self.tau2 = self.xp.zeros((self.L), dtype=self.floatx)
            self.tau2_prediv = False
            self.tau2_grow = self.xp.ones(self.maxiter)
            self.DK = [1]

    def init_params(self):
        self.delta = self.to_tensor(self.delta, dtype=self.floatx, device=self.device)
        self.mask  = self.delta >0
        self.zeta = self.to_tensor(self.zeta, dtype=self.floatx, device=self.device)*self.mask
        self.eta = self.to_tensor(self.eta, dtype=self.floatx, device=self.device)*self.mask

        self.Ys,  self.YFs = self.init_Yzbins(self.Y, Y_feats=self.YF)
        self.TYs = [ self.Ys[i].clone() for i in range(self.L)]
        # self.Yrs = [ self.Yr[self.Yins[i]] for i in range(self.L)]
        self.Xs, self.XFs = self.init_Xzbins(self.X,  X_feats=self.XF)

        self.Ns = self.xp.asarray([i.shape[0] for i in self.Xs])
        self.Ms = self.xp.asarray([i.shape[0] for i in self.Ys])
        self.init_sigma2(self.sigma2)
        self.init_tau2(self.tau2, tau2_clip=self.tau2_clip)

        self.diff = 1e8
        self.Q = 1e8
        self.Qs = self.xp.ones(self.L, device=self.device)*1e8
    
        if ('D' in self.transformer):
            self.Ws = [ self.xp.zeros_like(i[:,:-1]) for i in self.Ys]
            self.tcs =  self.xp.hstack([i[:, self.D-1].min() for i in self.Xs])

            self.Ws_tmp = [ self.xp.zeros_like(i[:,:-1]) for i in self.Ys]
            self.tcs_tmp =  self.xp.hstack([i[:, self.D-1].min() for i in self.Xs])
        else:
            self.As = self.xp.zeros((self.L, self.D-1, self.D-1,), 
                                    device=self.device, dtype=self.floatx)
            self.tcs =  self.xp.hstack([i[:, self.D-1].min() for i in self.Xs])
            self.tabs = self.xp.zeros((self.L, self.D-1), 
                                      device=self.device, dtype=self.floatx)

            self.As_tmp = self.xp.zeros((self.L, self.D-1, self.D-1), 
                                        device=self.device, dtype=self.floatx)
            self.tcs_tmp =  self.xp.hstack([i[:, self.D-1].min() for i in self.Xs])
            self.tabs_tmp = self.xp.zeros((self.L, self.D-1), 
                                          device=self.device, dtype=self.floatx)

    def register(self, callback= None, **kwargs):
        self.init_params()
        self.echo_paras()

        self.DR = rfRegularizer_Dataset(self.Ys, self.delta, self.transparas, self.transformer,
                                        device=self.device, dtype=self.floatx, verbose=self.verbose)
        self.FDs = rfCompute_Feauture(keops_thr=self.use_keops, xp=self.xp, verbose=self.verbose)
        self.FDs.compute_pairs(self.XFs, self.YFs, self.tau2, tau2_prediv=self.tau2_prediv, 
                               device=self.device, dtype=self.floatx)

        pbar = tqdm(range(self.maxiter), total=self.maxiter, colour='red', 
                    desc=f'{self.reg_core}', disable=(self.verbose<1))
        for i in pbar:
            self.optimization()
            pbar.set_postfix(self.postfix())

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

            if (self.diff <= self.tol):
                pbar.close()
                logger.info(f'Tolerance is lower than {self.tol :.3e}. Program ends early.')
                break
            elif (self.xp.all(self.xp.abs(self.sigma2) <= self.sigma2_min*10)):
                pbar.close()
                logger.info(f'All of the \u03C3^2 is lower than {self.sigma2_min*10  :.3e}. Program ends early.')
                break
        pbar.close()

        self.update_normalize()
        self.del_cache_attributes()
        self.detach_to_cpu(to_numpy=True)

    def expectation0(self, iL, X, TY, XF, YF):
        D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]

        isigma2, tau2, tau2_alpha  = self.sigma2[iL], self.tau2[iL], self.tau2_grow[self.iteration]
        gs = Mi/Ni*self.ws[iL]/(1-self.ws[iL])
        d2f= getattr(self.fds, f'f{iL}')

        if type(self.use_keops) in [bool, np.bool]:
            use_keops= self.use_keops
        else:
            use_keops = True if Ni * Mi > self.use_keops else False
        iexpectation = expectation_ko if use_keops else expectation_xp

        iPt1, iP1, iPX, iNp, tau2s = \
            iexpectation(X, TY, isigma2, gs, xp=self.xp,
                                  d2f=d2f, tau2=tau2, 
                                  tau2_auto=self.tau2_auto, 
                                  eps=self.eps,
                                  tau2_alpha=tau2_alpha, DF=self.DK, 
                                  feat_normal=self.feat_normal, XF=XF, YF=YF,
                                  device=self.device)

        w = (1- iNp/Ni).clip(*self.w_clip)
        self.ws[iL] = self.wa*self.ws[iL] + (1-self.wa)*w

        if self.fexist and self.tau2_auto:
            self.tau2[iL] = tau2s
        return iPt1, iP1, iNp, iPX

    def update_normalize(self):
        device, dtype = 'cpu', self.floatxx
        Sx, Sy  = self.Sx.to(device, dtype=dtype), self.Sy.to(device, dtype=dtype)
        Tx, Ty  = self.Mx.to(device, dtype=dtype), self.My.to(device, dtype=dtype)
        Sr = Sx/Sy
        D, self.TY = self.D, self.xp.zeros((self.M, self.D), device=device, dtype=dtype)
        for iL in range(self.L):
            itf = self.transformer[iL]
            iTM = self.tmats[iL]
            iTM = { k: v.to(device, dtype=dtype) for k,v in iTM.items() }

            if itf in 'ESARTILO':
                if itf in ['E']:
                    A = iTM['R'] * iTM['s'] * Sr
                    iTM['s'] *= Sr
                elif itf in 'SRTILO':
                    A = iTM['R'] * iTM['s']* Sr
                    iTM['s'] *= Sr
                elif itf in ['A']:
                    A = iTM['A']* Sr
                    iTM['A'] = A
                else:
                    raise(f"Unknown transformer: {itf}")

                if not self.transparas[iL].get('fix_t' , False):
                    t = iTM['t']*Sx + Tx[:-1] - Ty[:-1] @ A.T
                    tc = iTM['tc']*Sx + Tx[-1]
                else: #TODO
                    t = iTM['t']*Sx + Tx[:-1] - Ty[:-1] @ A.T
                    tc = iTM['tc']*Sx + Tx[-1]
                
                H = self.xp.eye(D+1, D+1, device=device, dtype=dtype)
                H[:D-1, :D-1] = A
                H[:D-1,  D] = t
                H[D-1, D] = tc
                H[D-1, D-1] = 0 # drop Yz

                TY = self.TYs[iL].to(device, dtype=dtype) * Sx + Tx
                # TY1 = self.Yrs[iL].to(device, dtype=dtype) @ H[:-1,:-1].T + self.xp.hstack([t, tc]) #TODO check
                # TY2 = self.homotransform_point(self.Yrs[iL], H)
                # print(iL, H, TY - TY1, TY1-TY2)               

                iTM['t'] = t
                iTM['tc'] = tc
                iTM['tform'] = H

            elif itf in ['D']:
                for ia in ['G', 'U', 'S']:
                    iv =  getattr(self.DR, f'{iL}_{ia}', None)
                    if iv is not None:
                        iTM[ia] = iv.to(device, dtype=dtype)
                    else:
                        iTM[ia] = iv

                iTM['Y'] = self.Ys[iL][:, :-1].to(device, dtype=dtype)
                iTM['Ym'] = Ty[:-1]
                iTM['Ys'] = Sy
                iTM['Xm'] = Tx[:-1]
                iTM['Xs'] = Sx
                iTM['beta'] = self.transparas[iL]['beta']
                iTM['tc'] = iTM['tc']*Sx + Tx[-1]

                if not iTM['G'] is None:
                    H = iTM['G']  @  iTM['W']
                else:
                    H = iTM['U'] @ iTM['S'] @ (iTM['U'].T @  iTM['W'])
        
                TY = self.TYs[iL].to(device, dtype=dtype) * Sx + Tx
                # TY1 = self.xp.hstack(
                #         [ ((self.Yrs[iL] - Ty)[:,:-1]/Sy + H)* iTM['Xs'] + iTM['Xm'],
                #          iTM['tc'].expand(self.Yrs[iL].shape[0], 1) ]
                # )
                # print(iL, self.xp.dist(TY, TY1), TY, TY1  )

            iTM['transformer'] = itf
            self.tmats[iL] = iTM
            self.TY[self.Yins[iL]] = TY
            # self.TYs[iL] = TY
        # TY1 = self.transform_point(self.Yr).to(device, dtype=dtype)
        # print(self.xp.dist(self.TY, TY1), self.TY - TY1)

    def transform_point(self, Yr, tmats = None, device=None, dtype=None,  **kargs):
        device = self.device if device is None else device
        dtype = self.floatxx if dtype is None else dtype

        tmats = self.tmats if tmats is None else tmats
        Yins = [ Yr[:,self.D-1] == i for i in self.Zid ]
        Ys =   [ Yr[i] for i in Yins ]
        TYS = self.transform_points(Ys, tmats, device=device, dtype=dtype, **kargs)
        T = self.xp.zeros(Yr.shape, device=device, dtype=dtype)
        for ity, idx in zip(TYS, Yins):
            T[idx] = ity
        return T

    def transform_points(self, Ys, tmats, device=None, dtype=None,  use_keops=True,):
        assert len(Ys )== len(tmats)
        L = len(Ys)
        device = self.device if device is None else device
        dtype = self.floatxx if dtype is None else dtype
        Ys = [ self.xp.asarray(iY, device=device, dtype=dtype) for iY in Ys]

        TYs = []
        for iL in range(L):
            iTM = tmats[iL]
            itf = iTM['transformer']

            if itf == 'D':
                iTY = self.ccf_deformable_transform_point(Ys[iL][:,:-1],
                         dtype=dtype, device=device, use_keops=use_keops, **iTM)
                iTYc = self.xp.ones((Ys[iL].shape[0], 1), device=device, dtype=dtype) * float(iTM['tc'])
                iTY = self.xp.hstack([iTY, iTYc])
            else:
                iTY = self.homotransform_point(Ys[iL], iTM['tform'],
                        xp=self.xp,  dtype=dtype, device=device,)
            TYs.append(iTY)
        return TYs

    def echo_paras(self, paras=None,paras1=None, maxrows=10, ncols = 2):
        if paras is None:
            paras = ['N', 'M', 'L', 'maxiter', 'inneriter', 'tol', 'floatx', 'floatxx', 'sigma2_min', 
                     'normal', 'verbose', 'device', 'device_pre', 'use_keops', 'fexist', 'scale_factor',
                      ]
        # if self.use_sample:
        #     paras += [ 'sample_min', 'sample_stopiter', ]
        if self.fexist:
            paras += [ 'feat_normal', 'tau2_decayto', 'tau2_decaystop', 'tau2_auto',]
        if paras1 is None:
            paras1 = ['transformer', ]
        logpara = [] 
        for ipara in paras:
            if hasattr(self, ipara):
                ivalue = getattr(self, ipara)
                try:
                    if ((type(ivalue) in [float]) or
                         (self.xp.is_floating_point(ivalue))
                        ):
                        logpara.append([ipara, f'{ivalue:.3e}'])   
                    elif (ivalue is None) or (type(ivalue) in [bool, str, int, type(None)]) :
                        logpara.append([ipara, f'{ivalue}'])
                    elif type(ivalue) in [list, tuple]:
                        logpara.append([ipara, f'{ivalue}'])
                    else:
                        logpara.append([ipara, f'{ivalue}'])
                except:
                    logpara.append([ipara, f'{ivalue}'])
        logpara = sorted(logpara, key=lambda x: x[0])
        lpadsize = max([len(ipara) for ipara, ivalue in logpara])
        rpadsize = max([len(str(ivalue)) for ipara, ivalue in logpara])
        logpara = [f'{ipara.ljust(lpadsize)} = {ivalue.ljust(rpadsize)}' for ipara, ivalue in logpara]

        logpara2 = []
        if len(paras1)>0:
            for ipara1 in paras1:
                if hasattr(self, ipara1):
                    ivalue = getattr(self, ipara1)
                    if ipara1 == 'transformer':
                        ivalue = ''.join(ivalue)
                    # ivalue = (f'+ {ipara1} = {ivalue}').ljust(headsize-2) + ' +'
                    ivalue = f'{ipara1} = {ivalue}'
                    logpara2.append(ivalue)
            logpara2 = sorted(logpara2)
            minlen2 = max([len(ipara) for ipara in logpara2])
            padsize = (lpadsize + rpadsize)*ncols + 3*(ncols -1)
            if padsize < minlen2:
                ipadlen = np.ceil(minlen2/ncols).astype(np.int64) #not strict
                logpara = [ f'{ipara0.ljust(ipadlen)}' for ipara0 in logpara]

        if len(logpara)>maxrows:
            nrows = int(np.ceil(len(logpara)/ncols))
        else:
            nrows = len(logpara)
        logpara1 = []
        for i in range(nrows):
            ilog = []
            for j in range(ncols):
                idx = i + j*nrows
                if idx < len(logpara):
                    ilog.append(logpara[idx])
            ilog = '+ ' + ' + '.join(ilog) + ' +'
            logpara1.append(ilog)

        headsize= len(logpara1[0])
        headinfo = 'init parameters:'.center(headsize, '-')
        dvinfo = ' '.join(sorted(set(self.dvinfo)))
        dvinfo = dvinfo.center(headsize, '-')

        logpara2  = [ '+ ' + ipara2.ljust(headsize-4) + ' +' for ipara2 in logpara2]
        logpara1 = '\n' + '\n'.join([headinfo] + logpara1 + logpara2 + [dvinfo])

        if self.verbose > 1:
            logger.info(logpara1)

    def del_cache_attributes(self, attributes=None):
        if attributes is None:
            attributes = [ 'Pf', 'P1', 'Pt1', 'cdff', 
                          'PX', 'MY', 'Av', 'AvUCv', 
                          'd2f', 'dpf', 'fds', # 'DR', 'FDs',
                           #Xr, Yr,
                          'XF', 'YF',  'I', 'TYs_tmp',
                          'VAv', 'Fv', 'F', 'MPG' ,]                
        for attr in attributes:
            if hasattr(self, attr):
                delattr(self, attr)
        self.clean_cache()

    def detach_to_cpu(self, attributes=None, to_numpy=True):
        if attributes is None:
            attributes = ['R', 'A', 'B', 't', 'd', 's',
                          'Xm', 'Xs', 'Xf', 'X', 'Ym', 'Ys', 'Y', 'Yf', 
                          'beta', 'G', 'Q', 'U', 'S', 'W', 'inv_S', 'Qs', 'Ls',
                          'tmat', 'tmatinv', 'tform', 'tforminv',
                          'TY',  'TYs',  'Ys', 'Xs', 'Xa', 'P', 'C', 
                          'As_tmp',  'tabs_tmp', 'tcs_tmp', 'Ws_tmp', 
                          'As',  'tabs', 'tcs', 'Ws', 
                          ] 
        for a in attributes:
            if hasattr(self, a):
                value = getattr(self, a)
                if to_numpy:
                    issp, spty = self.is_sparse(value)
                    if (spty=='torch'):
                        if issp:
                            value = self.thsparse_to_spsparse(value)
                        else:
                            value = value.detach().cpu().numpy()
                        setattr(self, a, value)
                    elif type(value) in [list]:
                        value = [ v.detach().cpu().numpy() if self.xp.is_tensor(v) else v
                                  for v in value ]
                        setattr(self, a, value)
                else:
                    value = value.detach().cpu()
                    setattr(self, a, value)

    def postfix(self, **kargs):
        iargs = {'tol': f'{self.diff :.3e}', 
                'Q': f'{self.Q :.3e}', }
        iargs.update(kargs)
        return iargs