import numpy as np
from tqdm import tqdm

from .operation_th import thopt
from .shared_wnn import swnn
from ...transform import homotransform_point, ccf_deformable_transform_point
from ...io._logger import logger
from .operation_expectation import pwExpection, similarity_paras

class pwEMRegistration(thopt, pwExpection):
    def __init__(self, X, Y, X_feat=None, Y_feat=None, 
                 maxiter=None, 
                 tol=None, 
    
                 normal=None, 
                 scale_factor = None,
                 sigma2=None, 
                 sigma2_min = 1e-7,

                 tau2=None, 
                 tau2_auto=False,
                 tau2_clip= [5e-3, 100.0],
                 tau2_decayto = 0.15,
                 tau2_decaystop = 1.0,
                 feat_normal='cos', 
                 feat_model = 'rbf',
                 smm_dfs = 5,

                 pairs=None,
                 c_alpha=1,
                 c_threshold=0.65,
                 use_swnn = False, 
                 swnn_args = {},

                 get_final_P = False,
    
                 floatx = None,
                 floatxx = None,
                 device = None,
                 device_pre = 'cpu',
                 use_keops=8e8,

                 w=None, c=None,
                 wa=0.995,
                 w_clip=[0, 0.999],

                 K=None, KF=None, p_epoch=None, 
                 kd_method='sknn', 
                 kdf_method='annoy',
                 
                 eps=1e-8,
                 seed = 491001,
                 record = None,
                 each_ncols= None,
                 verbose = 2, 
                 **kargs
                ):
        super().__init__(device=device, device_pre=device_pre, eps=eps,
                          floatx=floatx, floatxx=floatxx, seed=seed)
        self.verbose = verbose
        self.each_ncols = each_ncols or 2
        self.reg_core = None

        self.maxiter = maxiter
        self.iteration = 0
        self.tol = tol

        self.init_XY(X, Y, X_feat, Y_feat)
        self.init_keops(use_keops)
        self.init_outlier(w, w_clip)
        self.w_clip = w_clip
        self.wa = wa

        self.normal_ = normal
        self.scale_factor=scale_factor or 1.0
        self.feat_model = feat_model
        self.smm_dfs = smm_dfs

        self.sigma2 = sigma2
        self.sigma2_min = sigma2_min
        self.tau2 = tau2
        self.tau2_decayto = tau2_decayto 
        self.tau2_decaystop = tau2_decaystop
        self.tau2_auto = tau2_auto

        if self.fexist:
            self.feat_normal = self.scalar2vetor(feat_normal, L=self.FL)
            self.tau2_clip = self.scalar2vetor(tau2_clip, L=self.FL, force=True)
        else:
            self.feat_normal = None
            self.tau2_clip = None

        self.get_final_P = get_final_P
        self.pairs=pairs
        self.use_swnn = use_swnn
        self.swnn_args = swnn_args
        self.c_alpha = c_alpha
        self.c_threshold = c_threshold
        # self.constrained_pairs()

        self.kd_method = kd_method
        self.kdf_method = kdf_method
        self.K = K
        self.KF = KF

        self.record = [] if record is None else record
        self.records = {}

        self.expectation_func = self.expectation_ko if self.use_keops else self.expectation_xp
        self.expectation_df_func = self.expectation_ko_df if self.use_keops else self.expectation_xp_df
        self.expectation = self.pwexpectation
        self.compute_feauture = self.pwcompute_feauture
        self.homotransform_point = homotransform_point
        self.ccf_deformable_transform_point = ccf_deformable_transform_point
        self.similarity_paras = similarity_paras

    def swnn_pairs(self, ):
        dargs = dict(
                 mnn=6, snn=30, fnn=60, temp=1.0,
                 scale_locs=False, scale_feats=False,
                 lower = 0.01, upper=0.995,
                 min_score= 0.35,
                 max_pairs=5e4,
                 min_pairs=100)
        dargs.update(self.swnn_args)
        pairs = swnn(self.X, self.Y, self.XF, self.YF, **dargs)[0]
        return pairs

    def constrained_pairs(self):
        if self.use_swnn:
            pairs = self.swnn_pairs()
        if self.pairs is None:
            if self.use_swnn:
                self.pairs = pairs
        else:
            if self.use_swnn:
                self.pairs = self.xp.stack([self.pairs, pairs], 0)

    def init_XY(self, X, Y, X_feat, Y_feat):
        self.Xr = self.to_tensor(X, dtype=self.floatxx, device='cpu')
        self.Yr = self.to_tensor(Y, dtype=self.floatxx, device='cpu')

        self.N, self.Dx = self.Xr.shape
        self.M, self.Dy = self.Yr.shape
        self.D = self.Dx

        self.fexist = not (X_feat is None or Y_feat is None)
        if self.fexist:
            self.XFr = self.init_feature(X_feat, self.N)
            self.YFr = self.init_feature(Y_feat, self.M)
            assert len(self.XFr) == len(self.YFr), "Features must have the same length"
            self.FL = len(self.XFr)
            self.FDs = []
            for il in range(self.FL):
                assert self.XFr[il].shape[1] == self.YFr[il].shape[1], "Features must have the same dimension"
                self.FDs.append(self.XFr[il].shape[1])

    def init_feature(self, Fs, N):
        if isinstance(Fs, (list, tuple)):
            FL = len(Fs)
            Fa = []
            for l in range(FL):
                assert N == len(Fs[l]), "Features must have the same points number with X,Y"
                iF = self.to_tensor(Fs[l], dtype=self.floatxx, device='cpu').clone()
                Fa.append(iF)
        else:
            Fa = [self.to_tensor(Fs, dtype=self.floatxx, device='cpu').clone()]
        return Fa
        
    def init_keops(self, use_keops ):
        if type(use_keops) in [bool]:
            self.use_keops = use_keops
        elif type(use_keops) in [float, int]:    
            self.use_keops = (True if self.N*self.M >= use_keops else False)
        else:
            raise TypeError('use_keops must be bool or int or float.')
        if self.use_keops:
            try:
                import pykeops
                pykeops.set_verbose(False)
                from pykeops.torch import LazyTensor
                self.LazyTensor = LazyTensor
            except:
                raise ImportError('pykeops is not installed, `pip install pykeops`')
            
    def init_outlier(self, w, w_clip):
        if w is None:
            w = self.rigid_outlier(self.Xr, self.Yr)
        self.w = self.to_tensor(w, dtype=self.floatx, device=self.device)
        if w_clip is not None:
            self.w = self.xp.clip(self.w, *w_clip)

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
        self.Xf, self.Yf = Tf
        self.Xm, self.Ym = M.to(self.device, dtype=self.floatxx)
        self.Xs, self.Ys = S.to(self.device, dtype=self.floatxx)

    def normal_features(self):
        if self.fexist:
            Fr, Fs = [self.XFr, self.YFr], []
            for n in range(len(Fr)):
                iFs = []
                for il in range(self.FL):
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
            self.XF, self.YF = None, None

    def decay_curve(self, decayto, decaystop, maxiter):
        assert 0 <= decaystop <= 1.0
        decaystop = int(decaystop * maxiter)
        decayrate = self.xp.ones((maxiter), device=self.device, dtype=self.floatx)
        decayrate[:decaystop] = self.xp.linspace(1.0, decayto, decaystop)
        decayrate[decaystop:] = decayto
        return decayrate
        
    def init_tau2(self, tau2, tau2_clip=None):
        if self.fexist:
            tau2, self.tau2 = self.scalar2vetor(tau2, L=self.FL), self.xp.ones((self.FL))
            for lf in range(self.FL):
                if (tau2[lf] is None) or (tau2[lf] <= 0):
                    if self.feat_normal[lf] in ['cos', 'cosine']:
                        self.tau2[lf] = self.sigma_square_cos(self.XF[lf], self.YF[lf]) #TODO
                    else:
                        self.tau2[lf] = self.sigma_square(self.XF[lf], self.YF[lf])
                else:
                    self.tau2[lf] = tau2[lf]
        
                if tau2_clip[lf] is not None:
                    self.tau2[lf] = self.xp.clip(self.tau2[lf], *tau2_clip[lf])

            self.tau2 = self.to_tensor(self.tau2, dtype=self.floatx, device=self.device)
            self.tau2_prediv = not self.tau2_auto
            self.tau2_decayto = float(self.tau2_decayto or 1.0)
            self.tau2_grow = 1.0/self.decay_curve(self.tau2_decayto, self.tau2_decaystop, self.maxiter)
            # self.tau2_grow = self.xp.pow( self.tau2_decayto** (-1.0 / float(self.maxiter-1)) , self.xp.arange(self.maxiter))
            self.DK = [2 if self.feat_normal[lf] in ['cos', 'cosine'] else self.XF[lf].shape[1]
                            for lf in range(self.FL) ]
        else:
            self.tau2 = None
            self.tau2_prediv = False
            self.tau2_decayto = 1.0
            self.tau2_grow = self.xp.ones(self.maxiter)
            self.DK = [1]

    def init_params(self):
        self.sigma2 =  self.to_tensor(self.sigma2 or self.sigma_square(self.X, self.Y), 
                                      device=self.device)
        self.init_tau2(self.tau2, getattr(self, 'tau2_clip', None))
        self.TY = self.Y.clone()
    
        self.Q = 1.0 + self.N * self.D * 0.5 * self.xp.log(self.sigma2)
        self.gs = (self.M/self.N)*self.w/(1-self.w)
        self.Np = self.N
        self.diff = self.Q

    def register(self, callback= None, **kwargs):
        self.adjustable_paras(**kwargs)
        self.init_params()
        self.echo_paras(ncols=self.each_ncols)
        self.compute_feauture()

        pbar = tqdm(range(self.maxiter), total=self.maxiter, colour='red', desc=f'{self.reg_core}', disable=(self.verbose <1))
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
            elif (self.xp.abs(self.sigma2) < self.eps):
                pbar.close()
                logger.info(f'\u03C3^2 is lower than {self.eps :.3e}. Program ends early.')
                break
        pbar.close()

        if self.get_final_P:
           self.P = self.update_P_xp(self.X, self.TY,self.sigma2, self.gs, xp=self.xp,
                                    d2f=self.d2f, tau2=self.tau2, 
                                    tau2_auto=self.tau2_auto, eps=self.eps,
                                    tau2_alpha=self.tau2_grow[self.iteration], DF=self.DK)
           self.P = self.P.detach().cpu()
        self.update_normalize()
        self.del_cache_attributes()
        self.detach_to_cpu(to_numpy=True)

    @property
    def normal(self):
        return self.normal_

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

    def postfix(self, **kargs):
        iargs = {'tol': f'{self.diff :.3e}', 
                 'Q': f'{self.Q :.3e}', 
                #  'np': f'{self.Np :.3e}',
                 '\u03C3^2': f'{self.sigma2 :.3e}'}
        if self.fexist and self.tau2_auto:
            for il in range(self.FL):
                iargs.update({f'\u03C4^2_{il}': f'{self.tau2[il] :.3e}'})
        iargs.update(kargs)
        return iargs

    def del_cache_attributes(self, attributes=None):
        if attributes is None:
            attributes = [ 'Pf', 'P1', 'Pt1', 'cdff', 
                          'PX', 'MY', 'Av', 'AvUCv', 
                         'd2f', 'dpf', 'fds',
                          #Xr, Yr,
                          'XF', 'YF',  'I',
                          'VAv', 'Fv', 'F', 'MPG' , 'DR',
                          'L', 'LV', 'LY',  'AV', 'J', 'QY', 'LG', 
                          'AG', 'RG', 'QG', 'QY' ]                
        for attr in attributes:
            if hasattr(self, attr):
                delattr(self, attr)
        self.clean_cache()

    def detach_to_cpu(self, attributes=None, to_numpy=True):
        if attributes is None:
            attributes = ['R', 'A', 'B', 't', 'd', 's',
                          'Xm', 'Xs', 'Xf', 'X', 'Ym', 'Ys', 'Y', 'Yf', 
                          'beta', 'G', 'Q', 'U', 'S', 'W', 'inv_S',
                          'tmat', 'tmatinv', 'tform', 'tforminv', 'tform_d',
                          'TY',  'TYs', 'P', 'C']    
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
