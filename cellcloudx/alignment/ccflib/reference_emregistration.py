from builtins import super
import numpy as np
from tqdm import tqdm

from .operation_th import thopt
from .operation_expectation import (neighbor_weight, init_tmat,
                                    expectation_ko, expectation_xp, default_transparas)

from .operation_utility import split_bins_by_bps, get_pos_by_bps
from .reference_operation import rfRegularizer_Dataset, rfCompute_Feauture, rfEM_core
from ...io._logger import logger
from ...plotting._colors import colrows, color_palette

class rfEMRegistration(thopt, rfEM_core):
    def __init__(self, X, Y, X_feat=None, Y_feat=None, 
                 maxiter=None, 
                 tol=None, 
                 inneriter = 1,

                 transformer='E',
                 transparas=None,

                 floatx = None,
                 floatxx = None,
                 device = None,
                 device_pre = None,

                 delta =0, 
                 zeta = 0,
                 eta = 0,
                 penal_term = 'both',

                 w=0.5,
                 wa=0.995,
                 w_clip=[0, 0.999],

                 sigma2=None, 
                 sigma2_min = 1e-6,
                 sigma2_sync=False,
                 tau2=None, 
                 tau2_auto=False,
                 tau2_clip= [5e-3, 100.0],
                 tau2_decayto = 0.15,
                 tau2_decaystop = 1.0,

                 normal='isoscale', #'isoscale',
                 omega = None,
                 omega_normal = False,
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
        self.tol = tol or 1e-8

        self.init_points(X, Y, X_feat, Y_feat)
        self.omega_normal = omega_normal
        self.omega = self.scalar2matrix( 1.0 if omega is None else omega,
                                         (self.L, self.XL), 
                                        device=self.device, dtype=self.floatx )

        self.use_keops = use_keops
        self.normal = normal
        self.scale_factor=scale_factor or 1.0
        self.normal_XY()

        self.feat_normal = self.scalar2list(feat_normal, L=self.FL)
        self.normal_features()

        self.delta = self.init_penalty(delta, w_term=penal_term,)
        self.zeta = self.init_penalty(zeta, w_term=penal_term)
        self.eta = self.init_penalty(eta, w_term=penal_term)
        self.zspace= self.init_zspace(zspace)

        self.ws, self.was = self.init_ws(w, wa, w_clip)
        self.w_clip = w_clip

        self.sigma2= self.scalar2matrix( 0 if sigma2 is None else sigma2, 
                                        (self.L, self.XL), 
                                        device=self.device, dtype=self.floatx )
        self.sigma2_min = sigma2_min
        self.sigma2_sync = sigma2_sync

        self.tau2 =  [  self.scalar2list(tau2, self.FL) for iL in range(self.L) ]
        self.tau2_auto=  self.scalar2vetor( tau2_auto, self.L)
        self.tau2_clip = tau2_clip
        self.tau2_decayto = tau2_decayto
        self.tau2_decaystop = tau2_decaystop
    
        self.init_transformer(transformer)
        self.init_transparas(transparas)
        
        self.min_points = min_points
        self.use_projection = use_projection
        self.record =  [] if record is None else record
        self.records = {}
 
    def init_points(self, X, Y, X_feat, Y_feat):
        '''
        Xs = [ th.rand(10000,3), th.rand(20000,3) ]
        XFs = [[th.rand(10000,30), th.rand(10000,50)], [th.rand(20000,35)], ]
        Y = th.rand(8000, 3)
        YFs = [[th.rand(8000,30), th.rand(8000,50)], None, ]
        '''
        self.Yr = self.to_tensor(Y, dtype=self.floatxx, device='cpu')
        self.M, self.D = self.Yr.shape

        self.Zid, self.Ms = self.xp.unique(self.Yr[:,self.D-1], sorted=True,  return_counts=True)
        self.Yins = [  self.xp.where(self.Yr[:,self.D-1] == zid)[0] for zid in self.Zid]
        self.L = self.Zid.shape[0]

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
        self.XL = len(self.Xr)

        self.XFr, self.YFr, self.FL, self.FDs, self.fexists = [], [], [], [], []
        if X_feat is None:
            X_feat = [None] * self.XL
        elif not isinstance(X_feat, (list, tuple)):
            X_feat = [X_feat]
        
        if Y_feat is None:
            Y_feat = [None] * self.XL
        elif not isinstance(Y_feat, (list, tuple)):
            Y_feat = [Y_feat]

        assert len(X_feat) == self.XL, "X_feat must have the same length with X"
        assert len(Y_feat) == self.XL, "Y_feat must have the same length with Y"

        for iL in range(self.XL):
            XFl = self.init_feature(X_feat[iL], self.Ns[iL])
            YFl = self.init_feature(Y_feat[iL], self.M)
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

    def init_feature(self, Fs, N):
        if Fs is None or ( isinstance(Fs, (list, tuple)) and  len(Fs) == 0 ) or (Fs.shape[0]==0):
            Fa =  None
        elif isinstance(Fs, (list, tuple)):
            FL = len(Fs)
            Fa = []
            for l in range(FL):
                # assert not Fs[l] is None
                assert N == len(Fs[l]), "Features must have the same points number with X,Y"
                iF = self.to_tensor(Fs[l], dtype=self.floatxx, device='cpu').clone()
                Fa.append(iF)
        else:
            Fa = [self.to_tensor(Fs, dtype=self.floatxx, device='cpu').clone()]
        return Fa

    def normal_XY(self):
        Xa = [ self.xp.vstack(self.Xr), self.Yr ]
        L = len(Xa)
        M, S = self.xp.zeros((L,self.D)), self.xp.ones((L))

        if self.normal in ['each', 'isoscale', True]:
            for l in range(L):
                iX = self.to_tensor(Xa[l], dtype=self.floatxx, device=self.device_pre)
                M[l], S[l] = self.centerlize(iX, Xm=None, Xs=None)[1:3]
            if self.normal in ['isoscale', True]:
                S = self.xp.mean(S).expand(L).clone()

        elif self.normal  in ['global']:
            XX = self.xp.concat(Xa, 0).to(self.device_pre, dtype=self.floatxx)
            iXm, iXs = self.centerlize(XX, Xm=None, Xs=None)[1:3]
            M = iXm.expand(L, -1).clone()
            S = iXs.expand(L).clone()

        elif self.normal == 'X':
            iX = self.to_tensor(Xa[0], dtype=self.floatxx, device=self.device_pre)
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
            iX = self.to_tensor(Xa[l], dtype=self.floatxx, device=self.device)
            iT = self.centerlize(iX, Xm=M[l], Xs=S[l])
            Xn.append(iT[0].to(self.device_pre, dtype=self.floatxx))
            Tf.append(iT[3].to(dtype=self.floatxx))
        self.Xs, self.Y = Xn
        self.Xs = list(self.xp.split(self.Xs, self.Ns, dim=0))

        self.Hx, self.Hy = Tf
        self.Mx, self.My = M.to(self.device, dtype=self.floatxx)
        self.Sx, self.Sy = S.to(self.device, dtype=self.floatxx)

    def normal_features(self):
        Fr, Fs = [self.XFr, self.YFr], []
        for n in range(len(Fr)):
            nFs = []
            for iL, fexist in enumerate(self.fexists):
                if fexist:
                    iFs = []
                    for il in range(self.FL[iL]):
                        inorm = self.feat_normal[iL][il]
                        iF = self.to_tensor(Fr[n][iL][il], dtype=self.floatx, device=self.device_pre)
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
                        iFs.append(iF.to(self.device_pre))
                else:
                    iFs = None
                nFs.append(iFs)
            Fs.append(nFs)
        self.XF, self.YF = Fs

    def init_ws(self, w, wa, w_clip):
        ws =  self.scalar2matrix(w, (self.L, self.XL), 
                                    device=self.device, dtype=self.floatx )
        ws = ws.clip(*w_clip)
        was = self.scalar2matrix(wa, (self.L, self.XL), 
                                    device=self.device, dtype=self.floatx )
        return ws, was

    def init_penalty(self, weight, w_term='both'):
        weight = neighbor_weight(weight, L= self.L, w_term=w_term, xp =self.xp)
        weight = self.to_tensor(weight, dtype=self.floatx, device=self.device)
        return weight

    def init_zspace(self, zspace):
        if zspace is None:
            zspace = 0
        zspace = self.scalar2matrix(zspace, (self.L-1, self.XL), device=self.device_pre, dtype=self.floatxx)
        assert  self.xp.all(zspace < 1), "zspace should be smaller than 1"
        return zspace

    def init_Xzbins0(self, Xs, X_feats):
        X = self.xp.cat(Xs, 0)
        bps = self.Zid.to(X.device)
        axis = -1
        lpos, rpos = get_pos_by_bps(X, bps, axis = axis, zspace=self.zspace, xp=self.xp)
        Xa, XFa = [], []
        for l,r in zip(lpos, rpos):
            idx = (X[:,axis] >= l) & (X[:,axis] <= r)
            n_points = idx.sum()
            while n_points < self.min_points:
                zl = (r - l)*0.1
                l -= zl
                r += zl
                idx = (X[:,axis] >= l) & (X[:,axis] <= r)
                n_points = idx.sum()
            XsL, XfL = [], []
            for iL in range(len(Xs)):
                iX = Xs[iL]
                idx = (iX[:,axis] >= l) & (iX[:,axis] <= r)

                if self.FL[iL]:
                    ixf = [ ixf[idx].to(self.device, dtype=self.floatx) for ixf in X_feats[iL] ]
                else:
                    ixf = None
                XfL.append(ixf)
                XsL.append(iX[idx].to(self.device, dtype=self.floatx))

            Xa.append(XsL)
            XFa.append(XfL)
        return Xa, XFa

    def init_Xzbins(self, Xs, X_feats):
        minpts = self.xp.zeros(self.L)
        axis = -1
        LPs, RPs = [], []
        for xl in range(self.XL):
            X = Xs[xl]
            bps = self.Zid.to(X.device)
            lpos, rpos = get_pos_by_bps(X, bps, axis = axis, zspace=self.zspace[:, xl], xp=self.xp)
            LPs.append(lpos)
            RPs.append(rpos)
            for iL in range(self.L):
                l,r = lpos[iL], rpos[iL]
                idx = (X[:,axis] >= l) & (X[:,axis] <= r)
                n_points = idx.sum()
                minpts[iL] += float(n_points)

        Xa, XFa = [], []
        for iL in range(self.L):
            mps = minpts[iL]
            if mps < self.min_points:
                logger.warning(f"Bin {iL} has less than {self.min_points} points, will expand the range")
                aps = mps
                while aps < self.min_points:
                    aps = 0
                    for xl in range(self.XL):
                        X = Xs[xl]
                        l,r = LPs[xl][iL], RPs[xl][iL]
                        zl = (r - l)*0.05
                        l -= zl
                        r += zl
                        idx = (X[:,axis] >= l) & (X[:,axis] <= r)
                        aps += idx.sum()
                        LPs[xl][iL], RPs[xl][iL] = l,r
            XsL, XfL = [], []
            for xl in range(self.XL):
                iX = Xs[xl]
                l,r = LPs[xl][iL], RPs[xl][iL]
                idx = (iX[:,axis] >= l) & (iX[:,axis] <= r)

                if self.FL[xl]:
                    ixf = [ ixf[idx].to(self.device, dtype=self.floatx) for ixf in X_feats[xl] ]
                else:
                    ixf = None              
                
                XfL.append(ixf)
                XsL.append(iX[idx].to(self.device, dtype=self.floatx))
            Xa.append(XsL)
            XFa.append(XfL)
        return Xa, XFa

    def init_Yzbins(self, Y,  Y_feats):
        Ya, YFa = [], []
        for idx in self.Yins:
            Ya.append(Y[idx].to(self.device, dtype=self.floatx))
            # TODO
            # if self.use_YZ: 
            #     Ys.append(Y[idx][:, :self.D].to(self.device))
            # else:
            #     self.Ys.append(self.Y[idx][:, :self.D-1].to(self.device))
            iYF = []
            for iL, fexist in enumerate(self.fexists):
                if fexist:
                    iyf = [ iyf[idx].to(self.device, dtype=self.floatx) for iyf in Y_feats[iL] ]
                else:
                    iyf = None
                iYF.append(iyf)
            YFa.append(iYF)
        return Ya,  YFa

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

                for iarrt in ['alpha', 'gamma1', 'gamma2', 'p1_thred']:
                    ipara[iarrt] = self.to_tensor(self.scalar2vetor(ipara[iarrt], self.XL), 
                                                dtype=self.floatx, device=self.device)
            self.transparas[iL] = ipara

    def init_sigma2(self, sigma2, sigma2_clip=[1, None]):
        self.sigma2_exp = self.xp.zeros(self.L)

        for iL in range(self.L):
            for ixl in range(self.XL):
                if (sigma2[iL][ixl] <=0):
                    if not (self.is_null(self.Xa[iL][ixl]) or self.is_null(self.Ya[iL])):
                        sigma2[iL][ixl] = self.sigma_square(self.Xa[iL][ixl], self.Ya[iL])
        
        self.sigma2 = self.to_tensor(sigma2, dtype=self.floatx, device=self.device)
        if sigma2_clip is not None:
            self.sigma2 = self.xp.clip(self.sigma2, *sigma2_clip)
        self.sigma2_exp = self.sigma2.sum(1) /(self.sigma2>0).sum(1)
    
    def decay_curve(self, decayto, decaystop, maxiter):
        assert 0 <= decaystop <= 1.0
        decaystop = int(decaystop * maxiter)
        decayrate = self.xp.ones((maxiter), device=self.device, dtype=self.floatx)
        decayrate[:decaystop] = self.xp.linspace(1.0, decayto, decaystop)
        decayrate[decaystop:] = decayto
        return decayrate

    def init_tau2(self, tau2, tau2_clip=None):
        self.DK = self.scalar2list(2, L=self.FL)
        self.tau2_prediv = [not bool(i) for i in self.tau2_auto]
        self.tau2_decayto = float(self.tau2_decayto or 1.0)
        self.tau2_grow = 1.0/self.decay_curve(self.tau2_decayto, self.tau2_decaystop, self.maxiter)
        self.tau2 = tau2
    
        for iL in range(self.L):
            for ifl, fl  in enumerate(self.FL):
                for iil in range(fl):
                    if (tau2[iL][ifl][iil] is None or tau2[iL][ifl][iil] <=0) and \
                         ((not self.XFa[iL][ifl][iil] is None) and (len(self.XFa[iL][ifl][iil]) > 0)) :
                        if self.feat_normal[ifl][iil] in ['cos', 'cosine']:
                            self.tau2[iL][ifl][iil] = self.sigma_square_cos(self.XFa[iL][ifl][iil], self.YFa[iL][ifl][iil]) #TODO
                            self.DK[ifl][iil] = 2
                        else:
                            self.tau2[iL][ifl][iil] = self.sigma_square(self.XFa[iL][ifl][iil], self.YFa[iL][ifl][iil])
                            self.DK[ifl][iil] = self.XFa[iL][ifl][iil].shape[1]
                
                        if tau2_clip is not None:
                            self.tau2[iL][ifl][iil] = self.xp.clip(self.tau2[iL][ifl][iil], *tau2_clip)

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

    def init_transmatrix(self):
        for iL in range(self.L):
            self.tmats[iL] =  self.init_tmat(self.transformer[iL], self.Ms[iL], device=self.device,
                                              dtype=self.floatx)
        self.tcs =  self.xp.hstack([ min([iixa[:, self.D-1].min() for iixa in ixa if iixa.shape[0]>0]) for ixa in self.Xa])
        self.tcs_tmp =  self.tcs.clone()

        if ('D' in self.transformer):
            self.Ws = [ self.xp.zeros_like(i[:,:-1]) for i in self.Ya]
            self.Ws_tmp = [ self.xp.zeros_like(i[:,:-1]) for i in self.Ya]
        else:
            self.As = self.xp.zeros((self.L, self.D-1, self.D-1,), 
                                    device=self.device, dtype=self.floatx)
            self.tabs = self.xp.zeros((self.L, self.D-1), 
                                      device=self.device, dtype=self.floatx)
            self.As_tmp = self.xp.zeros((self.L, self.D-1, self.D-1), 
                                        device=self.device, dtype=self.floatx)
            self.tabs_tmp = self.xp.zeros((self.L, self.D-1), 
                                          device=self.device, dtype=self.floatx)

    def init_keops(self):
        self.keops_thr = self.xp.ones(self.L, dtype=self.xp.bool)
        for iL in range(self.L):
            iNM = self.Na[iL].sum() * self.Ma[iL]
            if iNM < self.use_keops:
                self.keops_thr[iL] = False
            
    def init_params(self):
        self.delta = self.to_tensor(self.delta, dtype=self.floatx, device=self.device)
        self.mask  = self.delta >0
        self.zeta = self.to_tensor(self.zeta, dtype=self.floatx, device=self.device)*self.mask
        self.eta = self.to_tensor(self.eta, dtype=self.floatx, device=self.device)*self.mask

        self.Ya,  self.YFa = self.init_Yzbins(self.Y, Y_feats=self.YF)
        self.TYs = [ self.Ya[i].clone() for i in range(self.L)]
        # self.Yrs = [ self.Yr[self.Yins[i]] for i in range(self.L)]
        self.Xa, self.XFa = self.init_Xzbins(self.Xs,  X_feats=self.XF)

        self.Na = self.xp.asarray([ [iixa.shape[0] for iixa in ixa ] for ixa in self.Xa])
        self.Ma = self.xp.asarray([i.shape[0] for i in self.Ya])
        self.init_keops()
        self.init_sigma2(self.sigma2)
        self.init_tau2(self.tau2, tau2_clip=self.tau2_clip)
        self.init_transmatrix()

        if self.omega_normal:
            self.Nf  = self.Na.sum(1)[:,None]/self.Na.clamp(1,None)
            # self.omega = self.omega/self.omega.sum()
            # self.omega = self.omega*self.Nf.to(self.device, dtype=self.floatx) #TODO check

        self.diff = 1e8
        self.Q = 1e8
        self.Qs = self.xp.ones(self.L, device=self.device)*1e8
    
    def register(self, callback= None, **kwargs):
        self.init_params()
        self.echo_paras()

        self.DR = rfRegularizer_Dataset(self.Ya, self.delta, self.transparas, self.transformer,
                                        device=self.device, dtype=self.floatx, verbose=self.verbose)
        self.FDs = rfCompute_Feauture(keops_thr=self.keops_thr, xp=self.xp, verbose=self.verbose)
        self.FDs.compute_pairs(self.XFa, self.YFa, self.tau2, tau2_prediv=self.tau2_prediv, 
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

    def echo_paras(self, paras=None,paras1=None, maxrows=10, ncols = 2):
        if paras is None:
            paras = ['D', 'M', 'L', 'XL', 'Ns', 'FL', 'FDs', 'fexists',  'penal_term', 'w_clip',
                     'maxiter', 'inneriter', 'tol', 'floatx', 'floatxx', 
                     'normal', 'verbose', 'use_keops',  'scale_factor', 'sigma2_sync',
                     'feat_normal', 'tau2_decayto', 'tau2_decaystop', 'sigma2_min', 
                     'omega_normal', 'min_points',
                      ]
        # if self.use_sample:
        #     paras += [ 'sample_min', 'sample_stopiter', ]

        if paras1 is None:
            paras1 = ['transformer', ]
        logpara = [] 
        for ipara in paras:
            if hasattr(self, ipara):
                ivalue = self.echo_value(getattr(self, ipara))
                logpara.append([ipara, ivalue])

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

    def show_reg(self, fscale = 5.0, size=0.5, werror=0, herror=0, nrow=None, ncol=None, colors=None):
        import matplotlib.pyplot as plt

        XL, L, T =self.XL, self.L, self.transformer
        nrow, ncol = colrows(L, nrows=nrow, ncols=ncol)
        if colors is None or len(colors) ==0:
            # if XL == 1:
            #     colors = ['red', 'blue']
            # else:
            colors = color_palette(XL+1)
        sizes = self.scalar2vetor(size, XL+1)
        fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey=True, 
                                figsize=((fscale+werror)*ncol,(fscale+herror)*nrow))

        for i in range(L):
            ax = axs[i//ncol, i%ncol]
            for xl in range(XL):
                ax.scatter( self.Xa[i][xl][:,0], 
                            self.Xa[i][xl][:,1], c=colors[xl], edgecolors='None', s=sizes[xl])
            ax.scatter( self.TYs[i][:,0],
                        self.TYs[i][:,1], c=colors[-1], edgecolors='None', s=sizes[-1])
            ax.set_title( f'{i}_{T[i]}')
            ax.set_aspect('equal')
        
        if nrow*ncol - L >0:
            for j in range(nrow*ncol - L):
                fig.delaxes(axs.flatten()[-j-1])
        fig.tight_layout()
        plt.show()
        
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
                          'TY',  'TYs',  'Ys', 'Xs', 'Xa', 'Ya', 'XFa', 'YFa', 'P', 'C', 
                          'As_tmp',  'tabs_tmp', 'tcs_tmp', 'Ws_tmp', 
                          'As',  'tabs', 'tcs', 'Ws', 
                          ] 
        for a in attributes:
            if hasattr(self, a):
                value = getattr(self, a)
                if to_numpy:
                    setattr(self, a, self.tensor2numpy(value))
                else:
                    setattr(self, a, self.tensordetach(value))

    def postfix(self, **kargs):
        iargs = {'tol': f'{self.diff :.3e}', 
                'Q': f'{self.Q :.3e}', }
        iargs.update(kargs)
        return iargs