import numpy as np
from tqdm import tqdm

from .operation_th import thopt
from .operation_expectation import (neighbor_weight, rigid_outlier)
from .groupwise_operation import gwEM_core, gwcompute_feauture
from ...io._logger import logger

class gwEMRegistration(thopt, gwEM_core):
    def __init__(self, Xs, Fs=None, 
                 Rs=None, Gs=None, #Pass
                 maxiter=None,
                 inneriter=None, 
                 tol=None, 
                
                 omega=1,
                 omega_term = 'both',
                 omega_normal = True,
                 kappa = 0,
                 root = None, 
                 graph_strategy ='sync',

                 sigma2=None, 
                 sigma2_min = 1e-6,
                 sigma2_sync = False,
                 tau2=None, 
                 tau2_auto=False,
                 tau2_clip= [5e-3, 100.0],
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
                 wa=0.995,
                 w_clip=[0, 0.999],

                 seed = 491001,
                 sample = None,
                 sample_growto = 1.0,
                 sample_min = 5000,
                 sample_stopiter = 0.75,

                 p_version = 'v2',
                 delta = 0,  #v2
                 zeta = 0,   #v1
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
        self.Fr, self.fexist, self.LF, self.DFs = self.multi_feature(Fs, Ns =self.Ns)

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
        self.sample_growto = sample_growto
        self.sample_min = sample_min
        self.sample_stopiter = sample_stopiter

        self.root = root
        self.kappa = self.scalar2vetor(kappa, L=self.L)
        self.init_omega(omega, omega_term=omega_term, root=root)
        self.omega_normal = omega_normal

        self.init_keops(use_keops)
        self.init_outlier( w, wa, w_clip=w_clip )

        self.p_version = p_version
        self.delta = self.scalar2vetor(delta, L=self.L) 
        self.zeta = self.scalar2vetor(zeta, L=self.L)
        
        self.record =  [] if record is None else record
        self.records = {}
        self.fds = gwcompute_feauture(keops_thr=self.keops_thr, xp=self.xp)
        self.dparas = self.default_transparas()

    def multi_points(self, Xs):
        assert isinstance(Xs, list), "Xs must be a list of arrays"
        for i in Xs[1:]:
            assert Xs[0].shape[1] == i.shape[1], "Xs must have the same dimension"
        self.L = len(Xs)
        self.D = Xs[0].shape[1]
        self.Ns = self.to_tensor([i.shape[0] for i in Xs], device=self.device, dtype=self.xp.int64)
        self.N = self.xp.sum(self.Ns)
        self.Xr = [ self.to_tensor(ix, device='cpu', dtype=self.floatxx).clone() for ix in Xs ]
    
    def multi_points_rf(self, Rs): #PASS
        if (Rs is None) or (len(Rs) == 0):
            self.rexist = False
        else:
            self.rexist = True
            assert isinstance(Rs, list), "Rs must be a list of arrays"
            assert len(Rs) == self.L, "Rs must have the same length with Xs"

            for i in Rs:
                assert i.shape[1] == self.D, "Rs must have the same dimension with Xs"
            self.Rr = [ self.to_tensor(ir, device='cpu', dtype=self.floatxx).clone() for ir in Rs ]
            self.Nrs = self.to_tensor([i.shape[0] for i in Rs], device=self.device, dtype=self.xp.int64)
    
    def multi_feature(self, Fs, Ns=None):
        Fr, DFs, LF = [], [], 0
        fexist = np.zeros(self.L, dtype=bool)

        if not Fs is None:
            assert isinstance(Fs, list), "Fs must be a list"
            assert self.L == len(Fs), "Xs and Fs must have the same length"

            Fins = [i for i in range(self.L) if not (Fs[i] is None or Fs[i] is [])]
            if len(Fins) > 0 :
                Iref = Fins[0]
                Fref = Fs[Iref]

                for l in range(self.L):
                    if (Fs[l] is None) or len(Fs[l]) == 0:
                        fexist[l] = False
                        Fr.append(None)
                    else:
                        assert type(Fref) == type(Fs[l]), f"F{l} must be the same type with F{Iref}."
                        if isinstance(Fs[l], (list)):
                            assert len(Fref) == len(Fs[l]), "F{l} must have the same feature number with F{Iref}."
                            iF = [ self.to_tensor(i, dtype=self.floatxx, device='cpu').clone() for i in Fs[l] ]
                        else:
                            iF = [ self.to_tensor(Fs[l], dtype=self.floatxx, device='cpu').clone() ]
                        fexist[l] = True
                        Fr.append(iF)
            
                        for ii in range(len(iF)):
                            if not Ns is None:
                                assert iF[ii].shape[0] == Ns[l], f"Fs {l}_{ii} must have the same points number with Xs"
                            assert iF[ii].shape[1] == Fr[Iref][ii].shape[1], "F{l} must have the same dimension in each feature"

                LF = len(Fr[Iref])
                DFs = self.xp.asarray([ l.shape[1] for l in Fr[Iref] ])
        return  Fr, fexist, LF, DFs

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
        self.omega = self.to_tensor(self.omega, dtype=self.floatx, device=self.device)

    def init_outlier(self, w, wa, w_clip=None, mask=None):
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

        self.w_clip = w_clip
        self.ws = self.to_tensor(ws, dtype=self.floatx, device=self.device).clone()
        self.was = self.to_tensor(wa, dtype=self.floatx, device=self.device)
        self.was = self.was * self.xp.ones((self.L, self.L), dtype=self.floatx,  device=self.device)
        if mask is not None:
            self.ws  = self.ws*(mask>0)
            self.was = self.was*(mask>0)

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

    def init_sample(self ):
        self.use_sample = False
        if (not self.sample is None):
            assert self.sample > 0 and self.sample <=1, "sample must be in (0, 1]"
            if (0< self.sample <1) :
                self.use_sample = True

        self.sample_stopiter = (int(self.sample_stopiter) if (self.sample_stopiter >1) 
                                    else int(self.maxiter*self.sample_stopiter))
        self.sample_grow = self.sample_growto ** (-1.0 / float(self.sample_stopiter-1))

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
        mask = self.sigma2>0
        self.sigma2_expect = (self.sigma2*mask).sum(1)/mask.sum(1)

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
            self.tau2_decayto = self.to_tensor(self.scalar2vetor(self.tau2_decayto, L=self.L))
            self.tau2_decay = [ float(i) ** (-1.0 / float(self.maxiter-1)) for i in self.tau2_decayto]
            self.DK = [2 if self.feat_normal[lf] in ['cos', 'cosine'] else self.Ns[lf]
                            for lf in range(self.LF) ]
        else:
            self.tau2 = self.xp.zeros((self.L, self.L), dtype=self.floatx)
            self.tau2_auto = self.scalar2vetor(False, L=self.L)
            self.tau2_prediv = self.scalar2vetor(True, L=self.L)
            self.tau2_decayto = self.to_tensor(self.scalar2vetor(1.0, L=self.L))
            self.tau2_decay = self.scalar2vetor(1.0, L=self.L)
            self.DK = [1]
    
    def default_transparas(self, kargs={}):
        dp= {
            'E':dict(
                fix_s=True, s_clip=None,
            ),
            'S':dict(isoscale=False,
                    fix_R=False, fix_t=False, fix_s=False, s_clip=[0.8, 1.2]),
            'A':dict(delta=0.1),
            'D':dict( beta=2.5, alpha=5e2, 
                    low_rank= 3000,
                    low_rank_type = 'keops',
                    fast_low_rank = 7000, num_eig=100, 
                    gamma1=0, gamma2=0,  kw= 15, kl=15,
                    alpha_decayto = 0.5,  use_p1=False, p1_thred = 0,
                    gamma_growto = 1, kd_method='sknn'),
            'P':dict(gamma1=None, lr=0.005, lr_stepsize=None,
                    lr_gamma=0.5, opt='LBFGS', d=1.0,
                    opt_iter=70),
            'T': dict(isoscale=True, 
                                fix_R=True,
                                fix_t=False,
                                fix_s=True,),
            'R':dict(isoscale=True, 
                                fix_R=False,
                                fix_t=True,
                                fix_s=True,),
            'I': dict(isoscale=True, 
                                fix_R=True,
                                fix_t=False,
                                fix_s=False,),
            'L': dict(isoscale=False,
                                fix_R=True,
                                fix_t=False,
                                fix_s=False,),
            'O': dict(isoscale=True,
                                fix_R=False,
                                fix_t=False,
                                fix_s=False,),
        }
        for k,v in kargs.items():
            dp[k].update(v)
        return dp

    def init_params(self, ):
        self.Ns = self.Ns.to(self.device, dtype=self.floatx) #mean()
        self.Nf = self.Ns/self.Ns.mean()

        self.omega = self.to_tensor(self.omega, dtype=self.floatx, device=self.device)
        if self.omega_normal:
            # self.omega = self.omega/self.Nf[None,:] #TODO
            omegasum1 = self.omega.sum(1, keepdim=True)
            omegasum1[omegasum1==0] = 1.0
            self.omega = self.omega/omegasum1

        self.OM = [ self.xp.where(self.omega[l]>0)[0].tolist() for l in range(self.L) ]
        self.OL = (range(self.L) if self.root is None 
                    else [*range(self.root, -1, -1), *range(self.root+1,self.L)])
        self.init_sample()
        self.init_sigma2(self.sigma2, mask=self.omega)
        self.init_tau2(self.tau2, tau2_clip=self.tau2_clip, mask=self.omega)
        self.sigma2[ self.sigma2>0 ] = self.sigma2[ self.sigma2>0 ].max() # trick 
    
        self.Q = 1e8
        self.diff = self.Q
        self.Qs = self.xp.zeros((self.L), dtype=self.floatx).to(self.device)
        self.Ls = self.xp.zeros((self.L, self.L), dtype=self.floatx).to(self.device)
        self.TYs = [ self.Xa[i].clone() for i in range(self.L)]
        self.TYs_tmp = [ self.Xa[i].clone() for i in range(self.L)]

        if self.p_version == 'v1':
            self.Af = self.xp.eye(self.D, dtype=self.floatx).to(self.device)
            self.tf = self.xp.zeros((self.D,), dtype=self.floatxx).to(self.device)

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

    def sample_index(self, iL, ins, itera):
        gr = self.xp.Generator() #device=self.device
        gr.manual_seed( int(f'{iL+1}{ins}{itera}' ))
        N = int(self.Ns[ins])
        sample = min(1.0, self.sample*(self.sample_grow**itera))
        perm = self.xp.randperm(N, generator=gr)
        n_samples = int(max(N*sample, min(self.sample_min, N)))
        return perm[:n_samples]

    def sample_index_np(self, iL, ins, itera):
        rng = np.random.default_rng(seed=[iL, ins, itera])
        N = int(self.Ns[ins])
        sample = min(1.0, self.sample*(self.sample_grow**itera))
        idx = rng.choice( N, replace=False,
                         size= int(max(N*sample, min(self.sample_min, N))))
        # idx = self.xp.tensor(idx, dtype=self.xp.int64, device=self.device)
        return idx

    def adjustable_paras(self, **kargs):
        for key, value in kargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)

    def echo_paras(self, paras=None,paras1=None, maxrows=10, ncols = 2):
        if paras is None:
            paras = ['N', 'M', 'D', 'L', 'LF', 'K', 'KF', 'scale_factor',
                     'device', 'device_pre', 'feat_model', 'use_keops', 'floatx', 'feat_normal', 'normal',
                     'maxiter', 'reg_core', 'tol', 'graph_strategy', 'root', 'inneriter', 'sigma2_min',
                     'sigma2_sync','omega_term', 'omega_normal', 'sample', 'sample_growto', 'p_version',
                      ]
        if self.use_sample:
            paras += [ 'sample_min', 'sample_stopiter', ]
    
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
                          'XF', 'YF',  'I', #'TYs_tmp',
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
                    setattr(self, a, self.tensor2numpy(value))
                else:
                    setattr(self, a, self.tensordetach(value))