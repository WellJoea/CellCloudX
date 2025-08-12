import numpy as np
from tqdm import tqdm

from ..operation_th import thopt
from ....utilis._arrays import list_iter
from ..operation_expectation import (neighbor_weight, rigid_outlier, gwcompute_feauture,
                                    expectation_ko, expectation_xp, similarity_paras)
from ..operation_maximization import gwMaximization
from ....io._logger import logger

class gwEMRegistration(thopt, gwMaximization):
    def __init__(self, Xs, Fs=None, 
                 maxiter=None,
                 inneriter=None, 
                 tol=None, 
                 
                 omega=1,
                 omega_term = 'both',
                 kappa = 0,
                 root = None, 
                 graph_strategy ='sync',

                 sigma2=None, 
                 sigma2_min = 1e-6,
                 sigma2_sync = False,
                 tau2=None, 
                 tau2_auto=False,
                 tau2_clip= [5e-3, 5.0],
                 tau2_decayto = 0.15,

                 normal=None,
                 scale_factor=None,
                 feat_normal='cos', 

                 floatx = None,
                 floatxx = None,
                 device = None,
                 device_pre = 'cpu',
                 use_keops=5e7,

                 w=None, c=None,
                 wa=0.99,
                 w_clip=[0, 0.9],

                 K=None, KF=None, p_epoch=None,
                 kd_method='sknn', 
                 kdf_method='annoy',

                 seed = 491001,
                 sample = None,
                 sample_min = 5000,
                 
                 record = None,
                 verbose = 2, 
                 **kargs
                ):
        super().__init__(device=device, device_pre=device_pre,
                          floatx=floatx, floatxx=floatxx, seed=seed)
        self.verbose = verbose
        self.reg_core = None
        self.normal = normal
        self.scale_factor=scale_factor or 1.0
        self.graph_strategy = graph_strategy or 'sync'
        assert self.graph_strategy in ['sync', 'async'], "graph_strategy must be 'sync' or 'async'"

        self.maxiter = maxiter
        self.inneriter = inneriter

        self.tol = tol
        self.iteration = 0

        self.multi_points(Xs)
        self.multi_feature(Fs)

        self.sigma2 = sigma2
        self.sigma2_min = sigma2_min
        self.sigma2_sync = sigma2_sync
        self.tau2 = tau2
        self.tau2_auto =  tau2_auto 
        self.tau2_decayto = 1.0 if tau2_decayto is None else tau2_decayto 
        
        if np.any(self.fexist):
            self.feat_normal = self.scalar2vetor(feat_normal, L=self.LF)
            self.tau2_clip = self.scalar2vetor(tau2_clip, L=self.LF, force=True)
        else:
            self.feat_normal = None
            self.tau2_clip = None

        self.sample = sample
        self.sample_min = sample_min
        self.use_sample = True if (not self.sample is None) and (self.sample <1) else False
        
        self.root = root
        self.kappa = self.scalar2vetor(kappa, L=self.L)
        self.init_omega(omega, omega_term=omega_term, root=root)
        self.init_keops(use_keops)
        self.w = w
        self.w_clip = w_clip
        self.wa = wa

        self.record =  [] if record is None else record
        self.records = {}
        self.fds = gwcompute_feauture(keops_thr=self.keops_thr, xp=self.xp)
        self.similarity_paras = similarity_paras

    def default_transparas(self):
        self.dparas = {
            'E':dict(
                fix_s=True, s_clip=None,
            ),
            'S':dict(isoscale=False,
                    fix_R=False, fix_t=False, fix_s=False,
                    s_clip=None),
            'A':dict(delta=0.1),
            'D':dict( beta=4.0, alpha=5e2, 
                  low_rank= 3000,
                  low_rank_type = 'keops',
                  fast_low_rank = 10000, num_eig=100, 
                    gamma1=0, gamma2=0,  kw= 15, kl=15,
                    alpha_decayto = 0.5,  use_p1=False, p1_thred = 0,
                    gamma_growto = 1, kd_method='sknn'),
            **self.similarity_paras()
        }

    def multi_points(self, Xs):
        assert isinstance(Xs, list), "Xs must be a list of arrays"
        for i in Xs[1:]:
            assert Xs[0].shape[1] == i.shape[1], "Xs must have the same dimension"
        self.L = len(Xs)
        self.D = Xs[0].shape[1]
        self.Ns = self.to_tensor([i.shape[0] for i in Xs], device=self.device, dtype=self.xp.int64)
        self.N = self.xp.sum(self.Ns)
        self.Xr = [ self.to_tensor(ix, device='cpu', dtype=self.floatxx).clone() for ix in Xs ]
    
    def multi_feature(self, Fs):
        self.Fr = []
        self.fexist = np.zeros(self.L, dtype=bool)

        if not Fs is None:
            assert isinstance(Fs, list), "Fs must be a list"
            assert self.L == len(Fs), "Xs and Fs must have the same length"

            Fins = [i for i in range(self.L) if not (Fs[i] is None or Fs[i] is [])]
            if len(Fins) > 0 :
                Iref = Fins[0]
                Fref = Fs[Iref]

                for l in range(self.L):
                    if (Fs[l] is None) or len(Fs[l]) == 0:
                        self.fexist[l] = False
                        self.Fr.append(None)
                    else:
                        assert type(Fref) == type(Fs[l]), f"F{l} must be the same type with F{Iref}."
                        if isinstance(Fs[l], (list)):
                            assert len(Fref) == len(Fs[l]), "F{l} must have the same feature number with F{Iref}."
                            iF = [ self.to_tensor(i, dtype=self.floatxx, device='cpu').clone() for i in Fs[l] ]
                        else:
                            iF = [ self.to_tensor(Fs[l], dtype=self.floatxx, device='cpu').clone() ]
                        self.fexist[l] = True
                        self.Fr.append(iF)
            
                        for ii in range(len(iF)):
                            assert iF[ii].shape[0] == self.Ns[l], f"Fs {l}_{ii} must have the same points number with Xs"
                            assert iF[ii].shape[1] == self.Fr[Iref][ii].shape[1], "F{l} must have the same dimension in each feature"

                self.LF = len(self.Fr[Iref])
                self.DFs = self.xp.asarray([ l.shape[1] for l in self.Fr[Iref] ])

    def normal_Xs(self):
        Xm, Xs, Xa = self.xp.zeros((self.L,self.D)), self.xp.ones(self.L), []
        if self.normal in ['each', 'isoscale', True]:
            for l in range(self.L):
                iX = self.to_tensor(self.Xr[l], dtype=self.floatx, device=self.device)
                Xm[l], Xs[l]= self.centerlize(iX, Xm=None, Xs=None)[1:3]
            if self.normal in ['isoscale', True]:
                Xs = self.xp.mean(Xs).expand(self.L).clone()

        elif self.normal in [ 'global']:
            XX = self.xp.concat(self.Xr, 0).to(self.device, dtype=self.floatx)
            iXm, iXs = self.centerlize(XX, Xm=None, Xs=None)[1:3]
            Xm = iXm.expand(self.L, -1).clone()
            Xs = iXs.expand(self.L).clone()
        elif self.normal in [False, 'pass']:
            pass
        else:
            raise ValueError(f"normal must be in ['each', 'global', 'isoscale', True, False, 'pass']")
        
        Xs = Xs/float(self.scale_factor)
        for l in range(self.L):
            iX = self.to_tensor(self.Xr[l], dtype=self.floatx, device=self.device)
            Xa.append(self.centerlize(iX, Xm=Xm[l], Xs=Xs[l])[0])
        
        self.Xa = Xa
        self.Xm = Xm
        self.Xs = Xs

    def normal_Fs(self):
        if np.any(self.fexist):
            Fa = []
            for l in range(self.L):
                if self.fexist[l]:
                    iFs = []
                    for n in range(self.LF):
                        inorm = self.feat_normal[n]
                        iF = self.to_tensor(self.Fr[l][n], dtype=self.floatx, device=self.device_pre)
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
                else:
                    iFs = None
                Fa.append(iFs)
            self.Fa = Fa
        else:
            self.Fa = [None] * self.L

    def init_omega(self, omega, omega_term='both', root = None):
        if not root is None:
            assert root < self.L
        self.root = root
        self.omega = neighbor_weight(omega, L= self.L, w_term=omega_term, xp =self.xp, root=root)
        # self.omega.fill_diagonal_(0)
        self.omega = self.to_tensor(self.omega, dtype=self.floatx, device=self.device)

    def init_outlier(self, w,  w_clip=None, mask=None):
        if w is None:
            ws = self.xp.zeros((self.L, self.L), dtype=self.floatx)
            for l in range(self.L):
                for ll in range(self.L):
                    if (mask is not None) and (mask[l,ll]==0):
                        continue
                    if l != ll: 
                        ws[l, ll] = rigid_outlier(self.Xr[l], self.Xr[ll],)
                    else:
                        ws[l, ll] = 0
        else:
            w = self.xp.tensor(w)
            if w.ndim == 0:
                ws = self.xp.ones((self.L, self.L), dtype=self.floatx)*w
            elif w.ndim == 1:
                if w.shape[0] == self.L:
                    ws = w.expand(self.L, -1).clone()
                else:
                    raise ValueError("w must be a 1D array with length L or a 2D array with shape (L, L)")
            elif w.ndim == 2:
                if w.shape == (self.L, self.L):
                    ws = w
                else:
                    raise ValueError("w must be a 2D array with shape (L, L)")
        if w_clip is not None:
            ws = self.xp.clip(ws, w_clip[0], w_clip[1])
        self.ws = self.to_tensor(ws, dtype=self.floatx, device=self.device).clone()
        if mask is not None:
            self.ws = self.ws*(mask>0)

    def init_keops(self, use_keops ):
        self.keops_thr = self.xp.zeros((self.L, self.L), dtype=self.xp.int8)
        if type(use_keops) in [bool]:
            self.keops_thr += int(use_keops)
        elif type(use_keops) in [float, int]:    
            for l in range(self.L):
                for ll in range(l, self.L):
                    ithr = self.Ns[l]*self.Ns[ll] >= use_keops
                    self.keops_thr[l, ll] = ithr
                    self.keops_thr[ll, l] = ithr
        else:
            raise TypeError('use_keops must be bool or int or float.')

        if self.keops_thr.sum() == 0:
            self.use_keops = False
        elif self.keops_thr.sum() == self.L*self.L:
            self.use_keops = True
        else:
            self.use_keops = use_keops

        if not self.use_keops is False:
            try:
                import pykeops
                pykeops.set_verbose(False)
                from pykeops.torch import LazyTensor
                self.LazyTensor = LazyTensor
            except:
                raise ImportError('pykeops is not installed, `pip install pykeops`')
    
    def init_sigma2(self, sigma2,  simga2_clip = [1, None], mask=None):
        if sigma2 is None:
            self.sigma2 = self.xp.zeros((self.L, self.L), dtype=self.floatx)
            for l in range(self.L):
                for ll in range(self.L):
                    if (mask is not None) and (mask[l,ll]==0):
                        continue
                    self.sigma2[l, ll] = self.sigma_square(self.Xa[l], self.Xa[ll])
                    # if l != ll:
                    #     self.sigma2[l, ll] = self.sigma_square(self.Xa[l], self.Xa[ll])
                    # else:
                    #     self.sigma2[l, ll] = 1.0
        else:
            sigma2 = self.xp.asarray(sigma2, dtype=self.floatx)
            if sigma2.ndim == 0:
                self.sigma2 = self.xp.ones((self.L, self.L), dtype=self.floatx)*sigma2
            elif sigma2.ndim == 1:
                if sigma2.shape[0] == self.L:
                    self.sigma2 = sigma2.expand(self.L, -1).clone()
                else:
                    raise ValueError("sigma2 must be a scalar or 1D array with length L or a 2D array with shape (L, L)")
            elif sigma2.ndim == 2:
                assert sigma2.shape == (self.L, self.L), "sigma2 must be a scalar or 1D array with length L or a 2D array with shape (L, L)"
                self.sigma2 = sigma2
    
        self.sigma2 = self.to_tensor(self.sigma2, dtype=self.floatx, device=self.device).clip(*simga2_clip)
        if mask is not None:
            self.sigma2 = self.sigma2*(mask>0)

    def init_tau2(self, tau2, tau2_clip=None, mask=None):
        if np.any(self.fexist):
            if tau2 is None:
                self.tau2 = self.xp.zeros((self.L, self.L, self.LF), dtype=self.floatx)
                for l in range(self.L):
                    for ll in range(self.L):
                        if (mask is not None) and (mask[l,ll]==0):
                            continue
                        if (self.fexist[l]) and (self.fexist[ll]):
                            for lf in range(self.LF):
                                if self.feat_normal[lf] in ['cos', 'cosine']:
                                    self.tau2[l, ll, lf] = self.sigma_square_cos(self.Fa[l][lf], self.Fa[ll][lf]) #TODO
                                else:
                                    self.tau2[l, ll, lf] = self.sigma_square(self.Fa[l][lf], self.Fa[ll][lf])
                        else:
                            self.tau2[l, ll] = 0
            else:
                tau2 = self.xp.asarray(tau2, dtype=self.floatx)
                if tau2.ndim == 0:
                    self.tau2 = self.xp.ones((self.L, self.L, self.LF), dtype=self.floatx)*tau2
                elif tau2.ndim == 1:
                    if tau2.shape[0] == self.LF:
                        self.tau2 = tau2.expand(self.L, self.L, -1).clone()
                    else:
                        raise ValueError("tau2 must be a scalar or 1D array with length LF or a 3D array with shape (L, L, LF)")
                elif tau2.ndim == 2:
                    if tau2.shape == (self.L, self.LF):
                        self.tau2 = tau2.expand(self.L, -1, -1).clone()
                    else:
                        raise ValueError("tau2 must be a scalar or 2D array with shape (L, LF) or a 3D array with shape (L, L, LF)")
                elif tau2.ndim == 3:
                    assert tau2.shape == (self.L, self.L, self.LF)
                    self.tau2 = tau2
            if (tau2_clip is not None):
                for lf in range(self.LF):
                    if (tau2_clip[lf] is not None):
                        self.tau2[:, :, lf] = self.xp.clip(self.tau2[:, :, lf], *tau2_clip[lf])

            self.tau2 = self.to_tensor(self.tau2, dtype=self.floatx, device=self.device)
            if mask is not None:
                self.tau2 = self.tau2*(mask[...,None]>0)

            self.tau2_auto = self.scalar2vetor(self.tau2_auto, L=self.L)
            self.tau2_prediv = [ not i for i in  self.tau2_auto ]
            self.tau2_decayto = self.scalar2vetor(self.tau2_decayto, L=self.L)
            self.tau2_decay = [ float(i) ** (-1.0 / float(self.maxiter-1)) for i in self.tau2_decayto]
            self.DK = [2 if self.feat_normal[lf] in ['cos', 'cosine'] else self.Ns[lf]
                            for lf in range(self.LF) ]
        else:
            self.tau2 = self.xp.zeros((self.L, self.L), dtype=self.floatx)
            self.tau2_auto = self.scalar2vetor(False, L=self.L)
            self.tau2_prediv = self.scalar2vetor(False, L=self.L)
            self.tau2_decayto = self.scalar2vetor(1.0, L=self.L)
            self.tau2_decay = self.scalar2vetor(1.0, L=self.L)
            self.DK = [1]
    
    def init_params(self, ):
        self.omega = self.to_tensor(self.omega, dtype=self.floatx, device=self.device)
        omegasum1 = self.omega.sum(1, keepdim=True)
        omegasum1[omegasum1==0] = 1.0
        self.omega = self.omega/omegasum1

        self.Ns = self.Ns.to(self.device, dtype=self.floatx) #mean()
        self.Nf = self.Ns/self.Ns.mean()
        # self.omega_tmp = self.omega.clone()/self.Nf[None,:]

        self.OM = [ self.xp.where(self.omega[l]>0)[0].tolist() for l in range(self.L) ]
        self.OL = (range(self.L) if self.root is None 
                    else [*range(self.root, -1, -1), *range(self.root+1,self.L)])
        self.init_outlier(self.w, w_clip=self.w_clip, mask=self.omega)
        self.init_sigma2(self.sigma2, mask=self.omega)
        self.init_tau2(self.tau2, tau2_clip=self.tau2_clip, mask=self.omega)

        self.Q = 1e8
        self.diff = self.Q
        self.Qs = self.xp.zeros((self.L), dtype=self.floatx).to(self.device)
        self.Ls = self.xp.zeros((self.L, self.L), dtype=self.floatx).to(self.device)
        self.TYs = [ self.Xa[i].clone() for i in range(self.L)]
        self.TYs_tmp = [ self.Xa[i].clone() for i in range(self.L)]

    def register(self, callback= None, **kwargs):
        self.adjustable_paras(**kwargs)
        self.init_params()
        self.echo_paras()

        try:
            self.init_regularizer()
        except:
            self.DR.__dict__.clear()
            self.clean_cache()
    
        if np.any(self.fexist):
            try:
                self.fds.compute_pairs(self.Fa, self.tau2, mask=self.omega, 
                                       tau2_prediv=self.tau2_prediv, fexist=self.fexist,
                                       device=self.device, dtype=self.floatx)
            except:
                self.fds.__dict__.clear()
                self.clean_cache()
                raise ValueError('Failed to compute the feature distance matrix. Check the Memory.')

        pbar = tqdm(range(self.maxiter), total=self.maxiter, colour='red', desc=f'{self.reg_core}', disable=(self.verbose<1))
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

    def random_index(self, il, ins, iter):
        rng = np.random.default_rng(seed=[il, ins, iter])
        return rng.choice( self.Ns[ins], replace=False,
                            size= max(self.Ns[ins]*self.sample,
                                       min(self.sample_min, self.Ns[ins])))

    def expectation(self, iL, Xs, TYs, Ys,):
        yid, xids = iL, self.OM[iL]
        # wl = (self.omega[iL]/self.Nf)[self.OM[iL]] # TODO v0
        wl = self.omega[iL][self.OM[iL]] # v1
        wl = wl/self.xp.sum(wl)

        Pt1s, P1s, PXs, Nps, ols = [],[],[],[],[]
        iTY, iY, iXs = TYs[yid], Ys[yid], []

        for wlj, xid in zip(wl, xids):
            iX = Xs[xid] #TODO sample
            D, Ni, Mi = iX.shape[1], iX.shape[0], iTY.shape[0]

            gs = Mi/Ni*self.ws[yid, xid]/(1-self.ws[yid, xid])
            d2f= getattr(self.fds, f'f{yid}_{xid}')
    
            #check:
            if d2f is None:
                assert (self.fexist[yid] != True or self.fexist[xid] != True)
            else:
                assert (self.fexist[yid] == True and self.fexist[xid] == True)
        
            iexpectation = expectation_ko if self.keops_thr[yid, xid] else expectation_xp
            iPt1, iP1, iPX, iNp, tau2s = iexpectation(
                    iX, iTY, self.sigma2[yid, xid], gs, 
                    xp = self.xp, d2f=d2f, tau2=self.tau2[yid, xid], 
                    tau2_auto=self.tau2_auto[iL], eps=self.eps, 
                    tau2_alpha=self.tau2_decay[iL]**self.iteration, DF=self.DK,
                    feat_normal=self.feat_normal, XF=self.Fa[xid], YF=self.Fa[yid], device=self.device,)

            if self.tau2_auto[iL] and self.fexist[iL]:
                self.tau2[yid, xid] = tau2s

            w = (1- iNp/Ni).clip(*self.w_clip)
            self.ws[yid, xid] = self.wa*self.ws[yid, xid] + (1-self.wa)*w
            olj = wlj/self.sigma2[yid, xid]

            Pt1s.append(iPt1)
            P1s.append(iP1)
            PXs.append(iPX)
            Nps.append(iNp)
            ols.append(olj)
            iXs.append(iX)
        return Pt1s, P1s, Nps, PXs, ols, iY, iXs

    def adjustable_paras(self, **kargs):
        for key, value in kargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)

    def echo_paras(self, paras=None,paras1=None, maxrows=10, ncols = 2):
        if paras is None:
            paras = ['N', 'M', 'D', 'L', 'LF', 'K', 'KF', 
                     'device', 'device_pre', 'feat_model', 'use_keops', 'floatx', 'feat_normal', 
                     'maxiter', 'reg_core', 'tol', 'graph_strategy', 'root', 'inneriter', 'sigma2_min',
                     'sigma2_sync','omega_term',
                      ]
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
                        logpara.append([ipara, f'{ivalue:.4e}'])   
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

    def postfix(self, **kargs):
        iargs = {'tol': f'{self.diff :.3e}', 
                 'Q': f'{self.Q :.3e}',}
        iargs.update(kargs)
        return iargs

    def del_cache_attributes(self, attributes=None):
        if attributes is None:
            attributes = [ 'Pf', 'P1', 'Pt1', 'cdff', 
                          'PX', 'MY', 'Av', 'AvUCv', 
                         'd2f', 'dpf', 'fds',
                          #Xr, Yr,
                          'XF', 'YF',  'I', 'TYs_tmp',
                          'VAv', 'Fv', 'F', 'MPG' , 'DR',
]                
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
                          'TY',  'TYs', 'Xa', 'P', 'C'] 
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
