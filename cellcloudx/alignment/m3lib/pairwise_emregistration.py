import numpy as np
import torch as th
from tqdm import tqdm

from .shared_wnn import swnn

from .label_counts import label_onehots
from ...io._logger import logger

from .operation_th import thopt
from .pairwise_operation import pwEM_core

class pwEMRegistration(thopt, pwEM_core):
    def __init__(self, X, Y, X_feat=None, Y_feat=None, 
                 X_label=None, Y_label =None, S_lable=None,
                 maxiter=None, 
                 tol=None, 
    
                 normal=None, 
                 omega = None,
                 omega_normal = False,
                 scale_factor = None,
                 p_model = None,  #['LGM', 'OTM]

                 sigma2=None, 
                 sigma2_min = None,
                 sigma2_sync = False,

                 tau2=None, 
                 tau2_auto=False,
                 tau2_clip= [5e-3, 50.0],
                 tau2_decayto = None,
                 tau2_decaystop = None,
                 feat_normal=None, 

                omega_l=None, 
                lable_normal=None, 
                knn=None, radius=None, knn_method = 'sknn', 
                fuzzy_global=None, 
                fuzzy_unique=None,
                knn_qt=None,
                delta_l = None,
                use_gating = None,
                lowm_l = None,
                
                 floatx = None,
                 floatxx = None,
                 device = None,
                 device_pre = None,
                 use_keops=8e8,
                
                 wo=None, wx =None,  wy=None,
                 w_dr=0.988,
                 g_dr=0.6,
            
                 use_stable=False,
                 wm_model = None,
                 use_mrf = False,
                 a0 = None, 
                 a1 = None,
                 di = None,
                 w_clip=[0.005, 0.995],
                 U = None,
                 qt=None,
                 dtmp=None,

                otm_eps = None, 
                otm_rho_x= None, otm_rho_y=None, 
                otm_rho_o =None,
                otm_the_x=None, otm_the_y=None, 
                otm_uv_iter=None,
                otm_tol = None, otm_reg = None,
                otm_ab_iter=None,  otm_DSM =  None,
                otm_solver =None, 
                otm_kss = None,

                dustbin_mode=None,
                smooth_mode=None,
                budget_gamma = None,
                prebudget=None,
                budget_eta = None,
                smooth_alpha = None,
                budget_mode = None,
                
                get_match = False,

                #  K=None, KF=None, p_epoch=None, 
                #  kd_method='sknn', 
                #  kdf_method='annoy',

                #  pairs=None,
                #  c_alpha=1,
                #  c_threshold=0.65,
                #  use_swnn = False, 
                #  swnn_args = {},

                 eps=1e-8,
                 seed = 491001,
                 sample = None,
                 sample_growto = 1.0,
                 sample_min = 5000,
                 sample_stop = 0.75,

                 record = None,
                 each_ncols= None,
                 header='',
                 verbose = 3, 
                 mrf_paras={},
                 **kargs
                ):
        super().__init__(device=device, device_pre=device_pre, eps=eps,
                          floatx=floatx, floatxx=floatxx, seed=seed)
        self.verbose = verbose
        self.each_ncols = each_ncols or 3
        self.reg_core = None

        self.transformer='E'
        self.maxiter = maxiter
        self.iteration = 0 
        self.tol = tol
        self.p_model = (p_model or 'LGM').upper() #['LGM', 'OTM]
        

        self.init_points(X, Y, X_feat, Y_feat, X_label, Y_label)
        self.init_keops(use_keops)

        self.record = [] if record is None else record
        self.records = {}

        self.sample = self.scalar2vetor(sample, L=self.L)       
        self.sample_growto = sample_growto
        self.sample_min = sample_min
        self.sample_stop = sample_stop

        self.init_PModeParas(p_model, wo =wo, wx=wx, wy=wy, w_clip=w_clip, 
                             w_dr=w_dr, g_dr=g_dr, 
                             use_stable=use_stable, U=U, qt = qt, dtmp=dtmp,
                             a0=a0, a1=a1, di=di, wm_model=wm_model, use_mrf=use_mrf,
                            otm_eps = otm_eps,  otm_rho_x= otm_rho_x, otm_rho_y=otm_rho_y,
                            otm_rho_o =otm_rho_o,
                            otm_the_x= otm_the_x, otm_the_y = otm_the_y,
                            otm_uv_iter=otm_uv_iter, otm_tol=otm_tol, otm_reg=otm_reg,
                            otm_ab_iter=otm_ab_iter, otm_DSM=otm_DSM,  otm_kss=otm_kss,
                            otm_solver=otm_solver, dustbin_mode=dustbin_mode,
                            smooth_mode=smooth_mode, budget_gamma=budget_gamma, budget_eta=budget_eta,
                            smooth_alpha=smooth_alpha, prebudget=prebudget,
                            budget_mode=budget_mode, mrf_paras=mrf_paras,
                            )

        self.omega_normal = omega_normal
        self.init_omega(omega)

        self.normal = normal
        self.scale_factor=scale_factor or 1.0
        self.feat_normal = self.scalar2list( 'ccos' if feat_normal is None else feat_normal, L=self.FL)
    
        self.init_LabelParas(S_lable=S_lable, omega_l=omega_l, lable_normal=lable_normal, 
                            knn=knn, radius=radius, knn_method = knn_method, 
                            fuzzy_global=fuzzy_global, fuzzy_unique=fuzzy_unique, knn_qt=knn_qt,
                            delta_l = delta_l,use_gating = use_gating, lowm_l = lowm_l,
                            )
        self.normal_lables()

        self.sigma2 = self.scalar2vetor(sigma2, L=self.L)  
        self.sigma2_min = sigma2_min or 1e-7
        self.sigma2_sync = sigma2_sync

        self.tau2 =  self.scalar2list(tau2, L=self.FL)
        self.tau2_clip = tau2_clip
        # self.tau2_decayto = self.scalar2list(tau2_decayto, L=self.FL)   
        # self.tau2_decaystop = self.scalar2list(tau2_decaystop, L=self.FL) 
        self.tau2_decayto = tau2_decayto
        self.tau2_decaystop = tau2_decaystop or 0.7
        self.tau2_auto = self.scalar2vetor(tau2_auto, L=self.L)  

        self.get_match = get_match
        self.header = header
        '''
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

        self.expectation_func = self.expectation_ko if self.use_keops else self.expectation_xp
        self.expectation_df_func = self.expectation_ko_df if self.use_keops else self.expectation_xp_df
        self.expectation = self.pwexpectation
        self.compute_feauture = self.pwcompute_feauture
        self.homotransform_point = homotransform_point
        self.similarity_paras = similarity_paras
        '''

    def swnn_pairs(self, ): #TODO
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

    def init_points(self, X, Y, X_feat, Y_feat,  X_label, Y_label):
        '''
        Xs = [ th.rand(10000,3), th.rand(20000,3) ]
        XFs = [[th.rand(10000,30), th.rand(10000,50)], [th.rand(20000,35)], ]
        Y = th.rand(8000, 3)
        YFs = [[th.rand(8000,30), th.rand(8000,50)], None, ]
        '''
        self.Yr = self.to_tensor(Y, dtype=self.floatxx, device='cpu')
        self.M, self.D = self.Yr.shape

        self.Xr, self.Ns = [], []
        if not isinstance(X, (list, tuple)):
            X = [X]
        for iX in X:
            try:
                iX = self.to_tensor(iX, dtype=self.floatxx, device='cpu')
                assert iX.shape[1] == self.D, "X must have the same dimension with Y"
                self.Ns.append(iX.shape[0])
                self.Xr.append(iX)
            except:
                raise TypeError('X must be tensor or array or list of tensors or arrays')
        self.L = len(self.Xr)
        self.Na = self.to_tensor(self.Ns, dtype=self.floatx, device=self.device)
        self.XFr, self.YFr, self.FL, self.FDs, self.fexists = [], [], [], [], []
      
        if not isinstance(X_feat, (list, tuple)):
            X_feat = [X_feat]
        if not isinstance(Y_feat, (list, tuple)):
            Y_feat = [Y_feat]

        if (self.L == 1) and  (len(X_feat) == len(Y_feat)) and (len(X_feat) >1):
            X_feat = [X_feat]
            Y_feat = [Y_feat]

        assert len(X_feat) == self.L, "X_feat must have the same length with X"
        assert len(Y_feat) == self.L, "Y_feat must have the same length with Y"

        for iL in range(self.L):
            XFl = self.init_feature(X_feat[iL], self.Ns[iL], iL)
            YFl = self.init_feature(Y_feat[iL], self.M, iL)
            if XFl is None or YFl is None:
                fexist = False
                FL = 0
                FDl = 0
            else:
                assert len(XFl) == len(YFl), "Features must have the same length"
                fexist = True
                FL = len(XFl)
                FDl = []
                for il in range(FL):
                    assert XFl[il].shape[1] == YFl[il].shape[1], "Features must have the same dimension"
                    FDl.append(XFl[il].shape[1])
            self.XFr.append(XFl)
            self.YFr.append(YFl)
            self.FL.append(FL)
            self.FDs.append(FDl)
            self.fexists.append(fexist)
 
        if not isinstance(X_label, (list, tuple)):
            X_label = [X_label]
        if not isinstance(Y_label, (list, tuple)):
            Y_label = [Y_label]

        
        if (self.L == 1) and  (len(X_label) == len(Y_label)) and (len(X_label) >1):
            X_label = [X_label]
            Y_label = [Y_label]

        assert len(X_label) == self.L, "X_label must have the same length with X"
        assert len(Y_label) == self.L, "Y_label must have the same length with Y"

        self.XLr, self.YLr, self.LL, self.lexists = [], [], [], []
        for iL in range(self.L):
            XLl = self.init_lable(X_label[iL], self.Ns[iL], iL)
            YLl = self.init_lable(Y_label[iL], self.M, iL)
            if XLl is None or YLl is None:
                lexist = False
                LL = 0
            else:
                assert len(XLl) == len(YLl), "Lables must have the same length"
                lexist = True
                LL = len(XLl)
            self.XLr.append(XLl)
            self.YLr.append(YLl)
            self.LL.append(LL)
            self.lexists.append(lexist)

    def init_feature(self, Fs, N, L):
        if self.is_null(Fs):
            return None
        if isinstance(Fs, (list, tuple)):
            FL = len(Fs)
            Fa = []
            for l in range(FL):
                # assert not Fs[l] is None
                assert N == len(Fs[l]), "Features must have the same points number with X,Y"
                iF = self.to_tensor(Fs[l], dtype=self.floatxx, device='cpu').clone()
                Fa.append(iF)
        else:
            assert N == len(Fs), "Features must have the same points number with X,Y"
            Fa = [self.to_tensor(Fs, dtype=self.floatxx, device='cpu').clone()]
        for j ,ifa in enumerate(Fa):
            if (ifa.dim() == 1) or (ifa.dim() == 2 and min(ifa.shape) == 1):
                Fa[j] = Fa[j].squeeze().unsqueeze(-1)
                if (self.is_int_tensor(ifa)):
                    logger.warning(f"Feature {L}-{j} may be label type, please check!")
                    # if  (not 'lable' in self.feat_normal[L][j]):
                    #     logger.warning(f"feat_normal {L}-{j} isnot label type, please check!")
        return Fa

    def init_lable(self, Ls, N, L):
        if self.is_null(Ls):
            return None
        if isinstance(Ls, (list, tuple)):
            Ls = Ls
        else:
            Ls = [Ls]
        La = []
        for j ,ila in enumerate(Ls):
            iL = np.asarray(ila).copy()
            if (len(ila.shape) == 1) or ((len(ila.shape) == 2) and min(ila.shape) == 1):
                iL = np.expand_dims(np.squeeze(iL), axis=-1)
                assert N == iL.shape[0], f"Lable {L}-{j} must have the same points number with X,Y"
                # if (self.np_type(iL) in ['str', 'int']) and (not 'lable' in self.lable_normal[L][j]):
                #     logger.warning(f"lable_normal {L}-{j} may be label type, please check!")
                # if  (self.np_type(iL) in ['float']) and (not 'prob' in self.lable_normal[L][j]):
                #     logger.warning(f"lable_normal {L}-{j} may be prob type, please check!")
            La.append(iL)
        return La

    def init_keops(self, use_keops ):
        if type(use_keops) in [bool]:
            self.use_keops = [use_keops] * self.L
        elif type(use_keops) in [float, int]:    
            self.use_keops = [ True if N*self.M >= use_keops else False for N in self.Ns ]
        else:
            raise TypeError('use_keops must be bool or int or float.')
        if np.any(self.use_keops):
            try:
                import pykeops
                pykeops.set_verbose(False)
                from pykeops.torch import LazyTensor
                self.LazyTensor = LazyTensor
            except:
                raise ImportError('pykeops is not installed, `pip install pykeops`')

    def init_PModeParas(self, p_model, wo=None, wx =None, wy=None, 
                        w_clip = None,
                        w_dr = None, g_dr = None, 
                        use_stable = None, U=None, a0=None, a1=None, qt = None,
                        dtmp=None, di = None, wm_model = None, use_mrf=None,
                        otm_eps = None,  otm_rho_x= None, otm_rho_y=None,
                        otm_the_x=None, otm_the_y=None, otm_rho_o = None,
                        otm_uv_iter=None, otm_tol=None, otm_reg=None,
                        otm_ab_iter=None, otm_DSM=None, otm_solver=None, otm_kss=None,
                        dustbin_mode=None, prebudget=None,
                        smooth_mode=None, budget_gamma=None, budget_eta=None, smooth_alpha=None,
                        budget_mode=None, mrf_paras={},         
        ):
        L, n2v = self.L, self.n2v
        p_model = (p_model or 'LGM').upper()
        ppas ={
            'L':L, 
            'p_model': p_model,
            'wo': wo,
            'w_clip':  [1e-3, 0.999] if w_clip is None else w_clip,
            'w_dr': w_dr or 0.985,
            'g_dr': g_dr or 0.6,
            'use_stable': False if use_stable is None else use_stable,  
        }

        otm_paras=dict(
            otm_eps=n2v(otm_eps, 1.0),  otm_rho_x= otm_rho_x or 19.0, otm_rho_y= otm_rho_y or 19.0, 
            otm_the_x = otm_the_x or 0.95, otm_the_y= otm_the_y or 0.95,
            otm_uv_iter=otm_uv_iter or 50,
            otm_tol = otm_tol or 1e-3, otm_reg = otm_reg or 'entropy',
            otm_ab_iter= otm_ab_iter or 1,  otm_DSM = (otm_DSM or 'explicit').lower(),
            budget_mode = budget_mode or 'slack_free',)

        if p_model  == 'LGM':
            lgm_paras={
                        'wm_model': 'm1' if wm_model is None else wm_model, 
                    }
            assert lgm_paras['wm_model'] in ['m0', 'm1', 'm2', 'm3'], "wm_model must be m0, m1, m2 or m3"
            lgm_paras['use_mrf'] = (False if lgm_paras['wm_model'] in ['m0'] else True) & (False if use_mrf is None else use_mrf )

            if lgm_paras['wm_model'] in ['m1']:
                a0 = self.scalar2vetor(n2v(a0, 1.1), L= L)
                a1 = self.scalar2vetor(n2v(a1, 1.4), L= L)
                a1 = [ max((self.M - a0[il]- self.Na[il] * (self.sample[il] or 1) + 1)/self.M+0.01, 
                            a1[il]) for il in range(L) ]

                lgm_paras.update({
                    'a0': a0, 
                    'a1': a1,
                    'U' : n2v(U, None),
                })
            if lgm_paras['wm_model'] in ['m2', 'm3']:
                lgm_paras.update({
                    'di': n2v(di, 0.01), 
                    'dtmp': n2v(dtmp, 0.75), 
                })
            if lgm_paras['wm_model'] in ['m2']:
                lgm_paras['U'] =  n2v(U, None)
            if lgm_paras['wm_model'] in ['m3']:
                lgm_paras['qt'] =  n2v(qt, 0.85)

            ppas.update(lgm_paras)
            ppas['wms'] = None
    
        elif p_model == 'OTM':
            ppas.update(otm_paras)
            assert ppas['otm_DSM'] in ['explicit', 'implicit'], "otm_DSM must be explicit or implicit"
            
            ppas['us'] = None
            ppas['vs']  = None
            ppas['as']  = None
            ppas['bs']  = None

            ppas['rx']  = None #TODO
            ppas['ry']  = None #TODO
            
            ppas['qt'] =  n2v(qt, 0.85)
                
            if ppas['otm_DSM'] == 'implicit':
                ppas['otm_solver'] = otm_solver or 'uot'
                ppas['otm_rho_o'] = n2v(otm_rho_o, 1.0)

                ppas['budget_eta'] = budget_eta or  0.5
                ppas['smooth_alpha'] = smooth_alpha or 0.5
                ppas['budget_gamma'] = budget_gamma or 0.5
                ppas['smooth_mode'] = smooth_mode or 'none'
       
            if ppas['otm_DSM'] == 'explicit':
                ppas['dustbin_mode'] = dustbin_mode or 'kxy'
                ppas['wx']  =  wx 
                ppas['wy']  =  wy
                ppas['otm_kss']  = n2v(otm_kss, 0)
                ppas['prebudget'] = n2v(prebudget, False)
                ppas['kxs']  = None
                ppas['kys']  = None
                ppas['uds']  = None
                ppas['vds']  = None
        else:
            raise ValueError('p_model must be LGM or OTM')

        tolist = ['wo', 'wx', 'wy', 'a0', 'a1', 'U', 'qt', 'di', 'dtmp', 'wms'] + \
                 ['otm_eps', 'otm_rho_x', 'otm_rho_y',  'otm_solver', 'otm_rho_o', 'otm_the_x', 'otm_the_y', 
                  'otm_reg', 'otm_DSM', 'dustbin_mode', 'smooth_mode', 'budget_eta', 
                  'budget_gamma', 'prebudget',
                  'budget_mode', 'us', 'vs', 'uds', 'vds', 
                  'otm_kss', 'kxs', 'kys',
                  'as', 'bs', 'ry', 'rx']

        for iv in tolist:
            if iv in ppas.keys():
                ppas[iv] = self.scalar2vetor(ppas.get(iv), L= L)

        #TODO
        if p_model == 'OTM':
            for il in range(L):
                epsv = float(ppas['otm_eps'][il])
                if ppas['otm_reg'][il] != 'entropy':
                    raise NotImplementedError("OTM currently implements otm_reg='entropy' only.")
                for axis in ('x', 'y'):
                    rk, tk = 'otm_rho_' + axis, 'otm_the_' + axis
                    rho, theta = ppas[rk][il], ppas[tk][il]
                    if theta is None:
                        if rho is None:
                            theta = 0.95
                        else:
                            rho = float(rho)
                            if not np.isfinite(rho) or rho < 0:
                                raise ValueError("otm_rho must be finite and non-negative.")
                            theta = rho / (rho + epsv)
                    theta = float(theta)
                    if not np.isfinite(theta) or not 0 <= theta < 1:
                        raise ValueError("otm_the must be in [0, 1).")
                    ppas[tk][il] = theta
                    ppas[rk][il] = epsv * theta / (1 - theta)

        for il in range(L):
            if ppas['wo'][il] is None:
                ppas['wo'][il] = self.rigid_overlap(self.Xr[il], self.Yr)
            if 'wx' in ppas.keys() and ppas['wx'][il] is None:
                ppas['wx'][il] = 1 - ppas['wo'][il]
            if 'wy' in ppas.keys() and ppas['wy'][il] is None:
                ppas['wy'][il] = 1 - ppas['wo'][il]
                
            if  'otm_solver' in ppas:
                solver = ppas['otm_solver'][il]
                rho_o = ppas['otm_rho_o'][il]
                epsv = ppas['otm_eps'][il]
                rho_y = ppas['otm_rho_y'][il]
                rho_x = ppas['otm_rho_x'][il]
                if solver in ['uot']:
                    ppas['otm_rho_o'][il] = 0.0
                
                cx_error = rho_o* rho_x - epsv - rho_x
                cy_error = rho_o* rho_y - epsv - rho_y
                if ((solver in ['bisect', 'newton' ])
                    and (abs(cx_error) < 1e-5)
                    and (abs(cy_error) < 1e-5)):
                    ppas['otm_solver'][il] = 'closed'


        totensor= ['wo', 'wx', 'wy', 'a0', 'a1', 'otm_kss']
        for iv in totensor:
            if iv in ppas.keys():
                ppas[iv] = self.to_tensor(ppas[iv], dtype=self.floatx, device=self.device)
        
        for iv in ['wo', 'wx', 'wy']:
            if iv in ppas.keys():
                ppas[iv] =  self.xp.clip(ppas[iv], * ppas['w_clip'])

        for ik in ['otm_the_x', 'otm_the_y']:
            if ik in ppas.keys():
                ivs = ppas[ik]
                for iv in ivs:
                    assert 0 <= iv < 1, f"{ik} must be in [0, 1)."
        for ik in ['w_dr', 'g_dr']:
            if ik in ppas.keys():
                iv = ppas[ik]
                assert 0 < iv < 1, f"{ik} must be in (0, 1)."

        for ik in ['budget_eta']:
            if ik in ppas.keys():
                ivs = ppas[ik]
                for iv in ivs:
                    assert 0 <=  iv <= 1, f"{ik} must be in (0, 1)."
       
        self.ppas = ppas
        self.mrf_paras = dict(
                sm='gl', st ='prop_gl', alpha=0.5, 
                beta_mrf=0.8,
                num_eig=100,  use_keops=True, fast_rank=True,
                p_step = 1, knn=15,  kd_method ='sknn', use_sym=True, 
                U=None, S=None, G=None)
        self.mrf_paras.update(mrf_paras)
    
    def init_LabelParas(self, S_lable=None, omega_l=None, lable_normal=None,  knn=None, radius=None, 
                        knn_method = 'sknn', fuzzy_global=None, knn_qt=None,  fuzzy_unique=None,
                        delta_l = None,
                        use_gating = None,
                        lowm_l = None,):
        self.lexist  = any(self.list_flatten(self.lexists))
        if self.lexist :
            n2v = self.n2v
            self.lable_normal = self.scalar2list( 'prob' if lable_normal is None else lable_normal, L=self.LL)
            self.omega_l= self.scalar2listv1( n2v(omega_l, 1.0), L= self.LL)
            self.S_lable = self.scalar2list(S_lable, L= self.LL)
            self.lpas =dict(
                knn = self.scalar2list(knn or 0, L= self.LL),
                radius = self.scalar2list(radius or None, L= self.LL),
                knn_method = knn_method,
                fuzzy_global = self.scalar2listv1( n2v(fuzzy_global, 0), L= self.LL),
                fuzzy_unique = self.scalar2listv1( n2v(fuzzy_unique, 0), L= self.LL),
                knn_qt = self.scalar2list(knn_qt or 0.98, L= self.LL),
                delta_l = self.scalar2vetor(delta_l or 0.0, L= self.L),
                use_gating = self.scalar2vetor(n2v(use_gating, True), L= self.L),
                lowm_l = self.scalar2vetor(n2v(lowm_l, False), L= self.L),
            )

    def init_omega(self, omega,):
        if omega is None: 
            omega = 1.0
        self.omega = self.scalar2vetor(omega, L= self.L)
        self.omega = self.to_tensor(self.omega, dtype=self.floatx, device=self.device)

    def init_outlier(self, w, w_clip, wa, wma=None, a0=1.0, a1=1.0, U=None):
        ws = self.scalar2vetor(w, L= self.L)
        for l in range(self.L):
            if ws[l] is None:
                ws[l] = self.rigid_outlier(self.Xr[l], self.Yr)
        self.ws = self.to_tensor(ws, dtype=self.floatx, device=self.device)
        self.w_clip = w_clip
        self.wa = wa
        self.wma = wma or wa
        self.a0 = self.to_tensor(self.scalar2vetor(a0, L= self.L), dtype=self.floatx, device=self.device)
        self.a1 = self.to_tensor(self.scalar2vetor(a1, L= self.L), dtype=self.floatx, device=self.device)
        self.U = self.scalar2vetor(U, L= self.L)
        self.wms = self.scalar2vetor(None, L= self.L)
    
        if w_clip is not None:
            self.ws = self.xp.clip(self.ws, *w_clip)
        self.w = self.ws

    def normal_XY(self):
        Xa = [ self.xp.vstack(self.Xr), self.Yr ]
        L = len(Xa)
        M, S = self.xp.zeros((L,self.D)), self.xp.ones((L))

        if self.normal in ['each', 'isoscale', True]:
            for l in range(L):
                iX = self.to_tensor(Xa[l], dtype=self.floatxx, device=self.device)
                M[l], S[l] = self.centerlize(iX, Xm=None, Xs=None)[1][:2]
            if self.normal in ['isoscale', True]:
                S = self.xp.mean(S).expand(L).clone()

        elif self.normal  in ['global']:
            XX = self.xp.concat(Xa, 0).to(self.device, dtype=self.floatxx)
            iXm, iXs = self.centerlize(XX, Xm=None, Xs=None)[1][:2]
            M = iXm.expand(L, -1).clone()
            S = iXs.expand(L).clone()

        elif self.normal == 'X':
            iX = self.to_tensor(Xa[0], dtype=self.floatxx, device=self.device)
            iXm, iXs = self.centerlize(iX, Xm=None, Xs=None)[1][:2]
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
            iX = self.to_tensor(Xa[l], dtype=self.floatxx, device=self.device)
            iT = self.centerlize(iX, Xm=M[l], Xs=S[l])
            Xn.append(iT[0].to(dtype=self.floatx))
            Tf.append(iT[1][2].to(dtype=self.floatxx))
        self.Xs, self.Y = Xn
        self.Xs = self.xp.split(self.Xs, self.Ns, dim=0)

        self.Hx, self.Hy = Tf
        self.Mx, self.My = M.to(self.device, dtype=self.floatxx)
        self.Sx, self.Sy = S.to(self.device, dtype=self.floatxx)

        # self.Xf, self.Yf = Tf
        # self.Xm, self.Ym = M.to(self.device, dtype=self.floatxx)
        # self.Xs, self.Ys = S.to(self.device, dtype=self.floatxx)

    def normal_features0(self):
        Fr, Fs = [self.XFr, self.YFr], []
        for n in range(len(Fr)):
            nFs = []
            for iL, fexist in enumerate(self.fexists):
                if fexist:
                    iFs = []
                    for il in range(self.FL[iL]):
                        inorm = self.feat_normal[iL][il]
                        iF = self.to_tensor(Fr[n][iL][il], dtype=self.floatx, device=self.device)
                        if inorm in ['cosine', 'cos'] :
                            iF = self.normalize(iF)
                        elif inorm == 'zc':
                            iF = self.centerlize(iF)[0]
                        elif inorm in ['ccosine', 'ccos']:     
                            iF = self.center_normalize(iF)
                        elif inorm == 'pcc':    
                            iF = self.scaling(iF, anis_var=True)
                        elif inorm == 'label':
                            iF = iF * 2  # 0-2 
                        elif inorm in ['pass', 'restc', 'resta']:    
                            iF = iF
                        else:
                            logger.warning(f"Unknown feature normalization method: {inorm}")
                        iFs.append(iF)
                else:
                    iFs = None
                nFs.append(iFs)
            Fs.append(nFs)
        self.XF, self.YF = Fs

    def normal_features(self):
        Fr, Fs = [self.XFr, self.YFr], []
        imageFs = {}

        for n in range(len(Fr)):
            nFs = []
            for iL, fexist in enumerate(self.fexists):
                if fexist:
                    iFs = []
                    for il in range(self.FL[iL]):
                        inorm = self.feat_normal[iL][il]

                        if inorm in ['image', 'i2p', 'img']:
                            key = (iL, il)
                            if key not in imageFs:
                                jFs = [
                                    self.to_tensor(
                                        Fr[k][iL][il],
                                        dtype=self.floatx,
                                        device=self.device,
                                    )
                                    for k in range(len(Fr))
                                ]
                                imageFs[key] = self.image_normalize(*jFs)[0]
                            iF = imageFs[key][n]
                        else:
                            iF = self.to_tensor(
                                Fr[n][iL][il],
                                dtype=self.floatx,
                                device=self.device,
                            )

                            if inorm in ['cosine', 'cos']:
                                iF = self.normalize(iF)
                            elif inorm == 'zc':
                                iF = self.centerlize(iF)[0]
                            elif inorm in ['ccosine', 'ccos']:
                                iF = self.center_normalize(iF)[0]
                            elif inorm == 'pcc':
                                iF = self.scale_normalize(iF)[0]
                            elif inorm in ['zscore', 'std']:
                                iF = self.scaling( iF, anis_var=True, )[0]
                            elif inorm == 'label':
                                iF = iF * 2  # 0-2
                            elif inorm in ['pass', 'restc', 'resta']:
                                iF = iF
                            else:
                                logger.warning(f"Unknown feature normalization method: {inorm}" )
                        iFs.append(iF)
                else:
                    iFs = None
                nFs.append(iFs)
            Fs.append(nFs)
        self.XF, self.YF = Fs
    

    def normal_lables(self,):
        self.LDs = self.scalar2vetor(0, L=self.L)  
        if self.lexist:
            Ws = self.omega_l 
            lpas = self.lpas
            As, Bs, LDs = [], [], []
            for iL, ilexist in enumerate(self.lexists):
                iXL, iYL = self.XLr[iL], self.YLr[iL]
                if ilexist:
                    iAs, iBs = [], []
                    iWs = np.asarray(Ws[iL], dtype=float)
                    iWs = iWs / iWs.sum()
                    for il in range(self.LL[iL]):
                        lnormal = self.lable_normal[iL][il]
                        if lnormal == 'lable':
                            if (not self.np_type(iXL[il]) in ['str', 'int'])  or (self.np_type(iYL[il]) in ['float']) :
                                logger.warning(f"lable_normal {iL}-{il} may be prob type, please check!")
                            # if (self.np_type(iXL[il]) in ['str', 'int']) and (not 'lable' in lnormal):
                            #     logger.warning(f"lable_normal {iL}-{il} may be label type, please check!")
                            # if  (self.np_type(iL) in ['float']) and (not 'prob' in self.lable_normal[L][j]):
                            #     logger.warning(f"lable_normal {L}-{j} may be prob type, please check!")
                            iA, iB =  label_onehots( [iXL[il], iYL[il] ] , 
                                                    Xs = [self.Xr[iL].detach().cpu().numpy(), 
                                                            self.Yr.detach().cpu().numpy() ], 
                                                    knn = lpas['knn'][iL][il], 
                                                    radius=lpas['radius'][iL][il] , 
                                                    knn_method =lpas['knn_method'] ,
                                                    normal='row', 
                                                    fuzzy_global=lpas['fuzzy_global'][iL][il] , 
                                                    fuzzy_unique=lpas['fuzzy_unique'][iL][il] ,
                                                    knn_qt = lpas['knn_qt'][iL][il] )[0]
                        else:
                            if (not self.np_type(iXL[il]) in ['float'])  or (self.np_type(iYL[il]) in ['str', 'int']) :
                                logger.warning(f"lable_normal {iL}-{il} may be label type, please check!")
                            iA, iB = iXL[il], iYL[il]

                        iS = self.S_lable[iL][il]
                        if iS is not None:
                            iB = iB @ iS
                            # rs = iB.sum(axis=1, keepdims=True)
                            # iB = iB  / rs
                        iB = iWs[il] * iB
                        iBs.append(iB)
                        iAs.append(iA)
                    iBs = self.to_tensor(np.concat(iBs, axis=1), dtype=self.floatx, device=self.device)
                    iAs = self.to_tensor(np.concat(iAs, axis=1), dtype=self.floatx, device=self.device)
                    As.append(iAs)
                    Bs.append(iBs)
                    self.LDs[iL] = iAs.shape[1]
                else:
                    As.append(None)
                    Bs.append(None)
                    LDs.append(0)
            self.As, self.Bs= As, Bs

    def decay_curve(self, decayto, decaystop, maxiter):
        assert 0 <= decaystop <= 1.0
        decaystop = int(decaystop * maxiter)
        decayrate = self.xp.ones((maxiter), device=self.device, dtype=self.floatx)
        decayrate[:decaystop] = self.xp.linspace(1.0, decayto, decaystop)
        decayrate[decaystop:] = decayto
        return decayrate
    
    def init_sigma2(self, sigma2=None, sigma2_clip=[1, None]):
        if sigma2 is None:
            sigma2 = self.scalar2vetor(sigma2, L=self.L)  

        for iL in range(self.L):
            if sigma2[iL] is None or sigma2[iL] <= 0:
                sigma2[iL] = self.sigma_square(self.Xs[iL], self.Y)
    
            if sigma2_clip is not None:
                sigma2[iL] = self.xp.clip(sigma2[iL], *sigma2_clip)

        self.sigma2 = self.to_tensor(sigma2, dtype=self.floatx, device=self.device)
        self.sigma2_exp = self.sigma2.mean()

    def init_tau2(self, tau2, tau2_clip=None):
        self.DK = self.scalar2list(2, L=self.FL)
        self.tau2_auto = [ bool(i) for i in self.tau2_auto ]
        self.tau2_grow = [ 1.0/self.decay_curve(self.tau2_decayto[i], self.tau2_decaystop[i], self.maxiter) for i in range(self.L) ]
        self.tau2 = tau2

        for iL,fexist in enumerate(self.fexists):
            if fexist:
                for il in range(self.FL[iL]):
                    if (tau2[iL][il] is None) or (tau2[iL][il] <= 0):
                        if self.feat_normal[iL][il]in ['cos', 'cosine', 'ccosine', 'ccos', 'restc']:
                            self.tau2[iL][il] = self.sigma_square_cos(self.XF[iL][il], self.YF[iL][il]) #TODO
                            self.DK[iL][il] = 2
                        elif self.feat_normal[iL][il]in ['label']:
                            self.tau2[iL][il] = 1.0
                            self.DK[iL][il] = 2
                        else:
                            self.tau2[iL][il] = self.sigma_square(self.XF[iL][il], self.YF[iL][il])
                            self.DK[iL][il] = self.XF[iL][il].shape[1]
                    else:
                        tau2[iL][il] = self.to_tensor(tau2[iL][il], dtype=self.floatx, device=self.device)

                    if tau2_clip is not None:
                        self.tau2[iL][il] = self.xp.clip(self.tau2[iL][il], *tau2_clip)

    def init_sample(self ): 
        for iL in range(self.L):
            if self.sample[iL]:
                assert self.sample[iL] > 0 and self.sample[iL] <=1, "sample must be in (0, 1]"

        self.sample_stop = (int(self.sample_stop) if (self.sample_stop >1) 
                                    else int(self.maxiter*self.sample_stop))
        self.sample_grow = self.sample_growto ** (-1.0 / float(self.sample_stop-1))

    def init_params(self):
        self.init_sample()
        self.init_sigma2(self.sigma2)
        self.init_tau2(self.tau2, getattr(self, 'tau2_clip', None))
        self.TY = self.Y.clone()
        if self.transformer in 'H':
             self.Yh = self.Y.clone()

        # if self.ppas['p_model']  == 'LGM':
        #     if self.ppas['use_wm']:
        #         self.ppas['a1'] = [ max((self.M - self.a0[il]- self.Na[il] * (self.sample[il] or 1) + 1)/self.M+0.01, 
        #                         self.ppas['a1'][il])
        #                 for il in range(self.L) ]
        #         self.ppas['a1'] = self.to_tensor(self.ppas['a1'], dtype=self.floatx, device=self.device)
        #     if self.ppas['use_mrf']:
        #         self.transparas = {
        #             'beta_mrf': 0.8,
        #             'num_eig': 100,
        #             'fast_rank': True,
        #             'mrf_method': 'wgl',
        #             'knn_mrf': 15,
        #             **self.transparas
        #         }
 
        self.Q = sum(self.Ns) * self.D * 0.5 * (1 + self.xp.log(self.sigma2)).sum()
        self.diff = self.Q

    def renew_paras(self):
        if self.omega_normal:
            self.Nf = self.Na/self.Na.mean()
            self.omega = self.omega/self.Nf #TODO check
            # self.omega = self.omega * 2/self.omega.sum()

    def register(self, callback= None, **kwargs):
        self.adjustable_paras(**kwargs)
        self.init_params()
        self.renew_paras()
        self.echo_paras(ncols=self.each_ncols, verbose=self.verbose)
        self.pwcompute_feauture()
        self.pwcompute_regularizer()
        self.pwcompute_MRFSmoother()

        if self.p_model == 'LGM':
            if self.ppas['wm_model'] == 'm0':
                self.pwexpectation_func = self.pwexpectation_lgm_m0
            elif self.ppas['wm_model'] == 'm1':
                self.pwexpectation_func = self.pwexpectation_lgm_m1
            elif self.ppas['wm_model'] == 'm2':
                self.pwexpectation_func = self.pwexpectation_lgm_m2
            elif self.ppas['wm_model'] == 'm3':
                self.pwexpectation_func = self.pwexpectation_lgm_m3
            else:
                raise ValueError(f'error wm_model {self.p_model}')

        elif self.p_model == 'OTM':
            self.pwexpectation_func = self.pwexpectation_otm
        else:
            raise ValueError(f'p_model must be LGM or OTM, but got {self.p_model}')

        pbar = tqdm(range(self.maxiter), total=self.maxiter, colour='red', 
                    desc=f'{self.transformer}', disable=(self.verbose <1))
        for i in pbar:
            self.optimization()
            pbar.set_postfix(self.postfix())

            if callable(callback):
                kwargs = {'iteration': self.iteration,
                            'error': self.Q, 'X': self.Xs, 'Y': self.TY}
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
            elif (self.xp.all(self.xp.abs(self.sigma2)) < self.eps):
                pbar.close()
                logger.info(f'\u03C3^2 is lower than {self.eps :.3e}. Program ends early.')
                break
        pbar.close()

        # if self.get_match:
        #    self.P = self.update_P_xp(self.X, self.TY,self.sigma2, self.gs, xp=self.xp,
        #                             d2f=self.d2f, tau2=self.tau2, 
        #                             tau2_auto=self.tau2_auto, eps=self.eps,
        #                             tau2_alpha=self.tau2_grow[self.iteration], DF=self.DK)
        #    self.P = self.P.detach().cpu()
        self.update_normalize()
        self.update_records()
        self.del_cache_attributes()
        self.detach_to_cpu(to_numpy=True)

    def postfix(self, **kargs):
        iargs = {'tol': f'{self.diff.item() :.3e}', 
                 'Q': f'{self.Q.item() :.3e}', 
                 #  'np': f'{self.Np :.3e}',
                 '\u03C3^2': '_'.join([ f'{isig.item() :.3e}' for isig in self.sigma2])
                }
        # if any(self.fexists) and any(self.tau2_auto):
        #     tau2 = ';'.join([ f'{i :.2e}'for i in (self.flatten_list(self.tau2)) ])
        #     iargs.update({f'\u03C4^2': tau2})
        iargs.update(kargs)
        return iargs

    def adjustable_paras(self, **kargs):
        for key, value in kargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)

    def echo_value(self, ivalue, q=3):
        try:
            if ivalue is None:
                nval = 'None'
            elif self.xp.is_tensor(ivalue):
                if ivalue.ndim == 0:
                    nval = f'{ivalue.item():.{q}e}'
                else:
                    if self.xp.is_floating_point(ivalue) and ivalue.ndim == 1:
                        nval = f'{[ round(float(x), q) for x in ivalue]}'
                    else:
                        nval =  f'{ivalue.cpu().flatten().tolist()}'
            elif (type(ivalue) in [float]):
                nval = f'{ivalue:.{q}e}'
            elif (type(ivalue) in [list, tuple]):
                ival = list(self.flatten_list(ivalue))
                iq = max(q-1, 1) if len(ival) >1 else q
                ival = [ self.echo_value(i, iq) for i in ival]
                nval =  f'{ival}'.replace("'", "")
                # if len(ival)==1:
                #     nval = ival[0] 
                # else:
                #     nval =  f'{ival}'
            else:
                nval = f'{ivalue}'
        except:
            nval = f'{ivalue}'
        return nval

    def echo_paras(self, paras=None, maxrows=10, ncols = 2, dvinfo=None, verbose=3):
        if paras is None:
            paras = ['Ns', 'M', 'D', 'L', 'FDs', 'FL', 'fexists', 'lexist', 'use_keops', 
                     'omega', 'omega_normal', 'w','wa', 'normal', 'scale_factor',
                     'feat_normal',  'p_model',
                     'use_stable', 'eps',
                     'sigma2', 'sigma2_min',  'sigma2_sync',
                     'tau2', 'tau2_clip', 'tau2_decayto', 'tau2_decaystop', 'tau2_auto',  
                     'record',  'df_version','get_match',
                      'maxiter', 'transformer',  'tol', 'verbose',
                      'device', 'device_pre',  'floatx', 'floatxx', 
                      'pair_num',
            ]
        paras = sorted(paras)

        if self.is_null(self.sample):
            paras += ['sample']
        else:
            paras += ['sample', 'sample_growto', 'sample_min', 'sample_stop' ]

        drop_paras = [
            'use_fast_low_rank', 'use_low_rank',
        ]
        logpara = [] 
        for ipara in paras:
            if hasattr(self, ipara):
                ivalue = self.echo_value(getattr(self, ipara))
                logpara.append([ipara, ivalue])
        # logpara = sorted(logpara, key=lambda x: x[0])

        transpara_items = self.transparas.items()
        if self.transformer in {'K', 'U', 'V', 'M'}:
            # pwDiffeoRegistration keeps all four parameter families in one
            # mapping.  Echo only the family consumed by the active method.
            def preferred_key(specific, common):
                return specific if specific in self.transparas else common

            common_keys = [
                'kernel_scales', 'kernel_weights',
                'alpha', 'alpha_decayto',
            ]

            if self.transformer == 'K':
                rkhs_model = self.transparas.get('rkhs_model', 'rff')
                method_keys = ['delta', 'rkhs_model', 'n_stages', 'rkhs_lipschitz_max']
                if rkhs_model == 'rff':
                    method_keys += [
                        'rff_features', 'rkhs_inner_steps',
                        'rkhs_inner_tol', #'random_state',
                    ]
                elif rkhs_model == 'control_white':
                    method_keys += [
                        'n_controls', 'control_method',
                        'kernel_jitter', #'random_state',
                    ]
                elif rkhs_model == 'full_nonwhite':
                    method_keys += [
                        'rkhs_solver', 'rkhs_backend',
                        'rkhs_inner_steps', 'rkhs_inner_tol',
                        'rkhs_direct_max_points', 'rkhs_use_keops',
                    ]

            elif self.transformer == 'U':
                control_keys = [
                    preferred_key('svf_n_controls', 'n_controls'),
                    preferred_key('svf_control_method', 'control_method'),
                    # preferred_key('svf_random_state', 'random_state'),
                ]
                method_keys = control_keys + [
                    'svf_steps','svf_max_steps', 'svf_inner_steps', 'svf_lr',
                    'svf_tol', 'svf_patience', 'svf_use_keops',
                ]

            elif self.transformer == 'V':
                control_keys = [
                    preferred_key('tvvf_n_controls', 'n_controls'),
                    preferred_key('tvvf_control_method', 'control_method'),
                    # preferred_key('tvvf_random_state', 'random_state'),
                ]
                method_keys = control_keys + [
                    'tvvf_n_time', 'tvvf_substeps', 'tvvf_max_substeps',
                    'tvvf_inner_steps', 'tvvf_lr',
                    'tvvf_tol', 'tvvf_patience', 'tvvf_use_keops',
                ]

            else:  # self.transformer == 'M'
                control_keys = [
                    'lddmm_n_controls',
                    preferred_key('lddmm_control_method', 'control_method'),
                    # preferred_key('lddmm_random_state', 'random_state'),
                ]
                method_keys = control_keys + [
                    'lddmm_method', 'lddmm_steps',  'lddmm_max_steps', 'lddmm_lipschitz_max',
                    'lddmm_inner_steps', 'lddmm_warm_steps',
                    'lddmm_lr', 'lddmm_tol', 'lddmm_patience',
                    'lddmm_chunk_size',
                ]
                if self.transparas.get('lddmm_method') == 'rff':
                    method_keys += ['lddmm_rff_features']
                else:
                    method_keys += ['lddmm_use_keops']

            transpara_keys = list(dict.fromkeys(common_keys + method_keys))
            transpara_items = [
                (key, self.transparas[key])
                for key in transpara_keys
                if key in self.transparas
            ]

        logpara1 = []
        for key, val in transpara_items:
            ivalue = self.echo_value(val)
            logpara1.append([key, ivalue])
        if len(logpara1)>0:
            if self.transformer not in {'K', 'U', 'V', 'M'}:
                logpara1 = sorted(logpara1, key=lambda x: x[0])
            logpara += logpara1

        if verbose >1:
            epas = sorted(
                    ['wo', 'wx', 'wy',  'w_clip',  'w_dr', 'g_dr',
                    'a0', 'a1', 'U', 
                    'qt', 'di', 'dtmp','use_stable', 'wm_model', 'use_mrf'] + \
                    ['otm_eps', 'otm_the_x', 'otm_the_y', 'otm_uv_iter', 
                     'otm_rho_o', #'otm_rho_x', 'otm_rho_y',
                     'otm_kss',
                    'otm_solver', 'otm_ab_iter', 'otm_reg', 'otm_DSM', 'otm_tol', 'dustbin_mode', 
                    'prebudget', 'budget_gamma', 'budget_eta', 'budget_mode']
            )
            for ipa in epas:
                if ipa in self.ppas.keys():
                    ivalue = self.echo_value(self.ppas[ipa])
                    logpara.append([ipa, ivalue])
            
            if self.lexist:
                for key in ['LDs', 'omega_l']:
                    ivalue = self.echo_value(getattr(self, key))
                    logpara.append([key, ivalue])
                lpas = ['lable_normal', 'knn', 'knn_method', 'fuzzy_global', 'fuzzy_unique', 'knn_qt', 
                        'delta_l', 'use_gating', 'lowm_l' ]
                for ipa in lpas:
                    if ipa in self.lpas.keys():
                        ivalue = self.echo_value(self.lpas[ipa])
                        logpara.append([ipa, ivalue])

        logpara = [ ipar for ipar in logpara if ipar[0] not in drop_paras ]

        lpadsize = max([len(ipara) for ipara, ivalue in logpara])
        rpadsize = max([len(str(ivalue)) for ipara, ivalue in logpara])
        logpara = [f'{ipara.ljust(lpadsize)} = {ivalue.ljust(rpadsize)}' for ipara, ivalue in logpara]

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
        headinfo = ( f'{self.header}init parameters:').center(headsize, '-')
        if dvinfo is None:
            dvinfo = ' '.join(sorted(set(self.dvinfo)))
            dvinfo = dvinfo.center(headsize, '-')
        elif dvinfo is False:
            dvinfo = ''
        else:
            dvinfo = ' '.join(sorted(set(dvinfo)))
            dvinfo = dvinfo.center(headsize, '-')
        logpara1 = '\n' + '\n'.join([headinfo] + logpara1 + [dvinfo])
        
        if verbose > 1:
            logger.info(logpara1)

    def del_cache_attributes(self, attributes=None):
        if attributes is None:
            attributes = [ 'Pf', 'P1', 'Pt1', 'cdff', 
                          'PX', 'MY', 'Av', 'AvUCv', 
                         'd2f', 'dpf', 'fds',
                          #Xr, Yr,
                           'XFr', 'YFr',
                          'XF', 'YF',  'I',
                          'VAv', 'Fv', 'F', 'MPG' , 'DR',
                          'L', 'LV', 'LY',  'AV', 'J', 'QY', 'LG', 'Yh',
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
                          'us', 'vs',
                          'TY',  'TYs', 'P', 'C'] 

        for a in attributes:
            if hasattr(self, a):
                value = getattr(self, a)
                if to_numpy:
                    setattr(self, a, self.tensor2numpy(value))
                else:
                    setattr(self, a, self.tensordetach(value))
