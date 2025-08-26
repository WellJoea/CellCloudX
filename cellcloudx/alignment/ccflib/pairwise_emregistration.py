import numpy as np
import torch as th
from tqdm import tqdm

from .shared_wnn import swnn
from ...io._logger import logger

from .operation_th import thopt
from .pairwise_operation import pwEM_core

class pwEMRegistration(thopt, pwEM_core):
    def __init__(self, X, Y, X_feat=None, Y_feat=None, 
                 maxiter=None, 
                 tol=None, 
    
                 normal=None, 
                 omega = None,
                 omega_normal = False,
                 scale_factor = None,

                 sigma2=None, 
                 sigma2_min = 1e-7,
                 sigma2_sync = False,

                 tau2=None, 
                 tau2_auto=False,
                 tau2_clip= [5e-3, 50.0],
                 tau2_decayto = 0.15,
                 tau2_decaystop = 1.0,
                 feat_normal='cos', 

                 pairs=None,
                 c_alpha=1,
                 c_threshold=0.65,
                 use_swnn = False, 
                 swnn_args = {},
                 get_final_P = False,
    
                 floatx = None,
                 floatxx = None,
                 device = None,
                 device_pre = None,
                 use_keops=8e8,

                 w=None, c=None,
                 wa=0.995,
                 w_clip=[0, 0.999],

                 K=None, KF=None, p_epoch=None, 
                 kd_method='sknn', 
                 kdf_method='annoy',
                 
                 eps=1e-8,

                 seed = 491001,
                 sample = None,
                 sample_growto = 1.0,
                 sample_min = 5000,
                 sample_stop = 0.75,

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

        self.transformer='E'
        self.maxiter = maxiter
        self.iteration = 0
        self.tol = tol

        self.init_points(X, Y, X_feat, Y_feat)
        self.init_keops(use_keops)
        self.init_outlier(w, w_clip, wa)
        
        self.omega_normal = omega_normal
        self.init_omega(omega)

        self.normal = normal
        self.scale_factor=scale_factor or 1.0
        self.feat_normal = self.scalar2list(feat_normal, L=self.FL)

        self.sigma2 = self.scalar2vetor(sigma2, L=self.L)  
        self.sigma2_min = sigma2_min
        self.sigma2_sync = sigma2_sync

        self.tau2 =  self.scalar2list(tau2, L=self.FL)
        self.tau2_clip = tau2_clip
        self.tau2_decayto = tau2_decayto #self.scalar2list(tau2_decayto, L=self.FL)   
        self.tau2_decaystop = tau2_decaystop #self.scalar2list(tau2_decaystop, L=self.FL)   
        self.tau2_auto = self.scalar2vetor(tau2_auto, L=self.L)  

        self.record = [] if record is None else record
        self.records = {}

        self.sample = self.scalar2vetor(sample, L=self.L)    
        self.sample_growto = sample_growto
        self.sample_min = sample_min
        self.sample_stop = sample_stop

        self.get_final_P = get_final_P

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

    def init_points(self, X, Y, X_feat, Y_feat):
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

        self.XFr, self.YFr, self.FL, self.FDs, self.fexists = [], [], [], [], []
      
        if not isinstance(X_feat, (list, tuple)):
            X_feat = [X_feat]
        if not isinstance(Y_feat, (list, tuple)):
            Y_feat = [Y_feat]

        assert len(X_feat) == self.L, "X_feat must have the same length with X"
        assert len(Y_feat) == self.L, "Y_feat must have the same length with Y"

        for iL in range(self.L):
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
        if self.is_null(Fs):
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

    def init_omega(self, omega,):
        if omega is None: 
            omega = 1.0
        self.omega = self.scalar2vetor(omega, L= self.L)
        self.omega = self.to_tensor(self.omega, dtype=self.floatx, device=self.device)

    def init_outlier(self, w, w_clip, wa):
        ws = self.scalar2vetor(w, L= self.L)
        for l in range(self.L):
            if ws[l] is None:
                ws[l] = self.rigid_outlier(self.Xr[l], self.Yr)
        self.ws = self.to_tensor(ws, dtype=self.floatx, device=self.device)
        self.w_clip = w_clip
        self.wa = wa
        if w_clip is not None:
            self.ws = self.xp.clip(self.ws, *w_clip)

    def normal_XY(self):
        Xa = [ self.xp.vstack(self.Xr), self.Yr ]
        L = len(Xa)
        M, S = self.xp.zeros((L,self.D)), self.xp.ones((L))

        if self.normal in ['each', 'isoscale', True]:
            for l in range(L):
                iX = self.to_tensor(Xa[l], dtype=self.floatxx, device=self.device)
                M[l], S[l] = self.centerlize(iX, Xm=None, Xs=None)[1:3]
            if self.normal in ['isoscale', True]:
                S = self.xp.mean(S).expand(L).clone()

        elif self.normal  in ['global']:
            XX = self.xp.concat(Xa, 0).to(self.device, dtype=self.floatxx)
            iXm, iXs = self.centerlize(XX, Xm=None, Xs=None)[1:3]
            M = iXm.expand(L, -1).clone()
            S = iXs.expand(L).clone()

        elif self.normal == 'X':
            iX = self.to_tensor(Xa[0], dtype=self.floatxx, device=self.device)
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
            Xn.append(iT[0].to(dtype=self.floatx))
            Tf.append(iT[3].to(dtype=self.floatxx))
        self.Xs, self.Y = Xn
        self.Xs = self.xp.split(self.Xs, self.Ns, dim=0)

        self.Hx, self.Hy = Tf
        self.Mx, self.My = M.to(self.device, dtype=self.floatxx)
        self.Sx, self.Sy = S.to(self.device, dtype=self.floatxx)

        # self.Xf, self.Yf = Tf
        # self.Xm, self.Ym = M.to(self.device, dtype=self.floatxx)
        # self.Xs, self.Ys = S.to(self.device, dtype=self.floatxx)

    def normal_features(self):
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
                        elif inorm == 'pcc':    
                            iF = self.scaling(iF, anis_var=True)
                        elif inorm == 'pass':    
                            pass
                        else:
                            logger.warning(f"Unknown feature normalization method: {inorm}")
                        iFs.append(iF)
                else:
                    iFs = None
                nFs.append(iFs)
            Fs.append(nFs)
        self.XF, self.YF = Fs

    def decay_curve(self, decayto, decaystop, maxiter):
        assert 0 <= decaystop <= 1.0
        decaystop = int(decaystop * maxiter)
        decayrate = self.xp.ones((maxiter), device=self.device, dtype=self.floatx)
        decayrate[:decaystop] = self.xp.linspace(1.0, decayto, decaystop)
        decayrate[decaystop:] = decayto
        return decayrate
    
    def init_sigma2(self, sigma2, sigma2_clip=[1, None]):
        for iL in range(self.L):
            if sigma2[iL] is None or sigma2[iL] <= 0:
                sigma2[iL] = self.sigma_square(self.Xs[iL], self.Y)
    
            if sigma2_clip is not None:
                sigma2[iL] = self.xp.clip(sigma2[iL], *sigma2_clip)

        self.sigma2 = self.to_tensor(sigma2, dtype=self.floatx, device=self.device)
        self.sigma2_exp = self.sigma2.mean()

    def init_tau2(self, tau2, tau2_clip=None):
        self.DK = self.scalar2list(2, L=self.FL)
        self.tau2_auto = [bool(i) for i in self.tau2_auto]
        self.tau2_decayto = float(self.tau2_decayto or 1.0)
        self.tau2_grow = 1.0/self.decay_curve(self.tau2_decayto, self.tau2_decaystop, self.maxiter)
        self.tau2 = tau2

        for iL,fexist in enumerate(self.fexists):
            if fexist:
                for il in range(self.FL[iL]):
                    if (tau2[iL][il] is None) or (tau2[iL][il] <= 0):
                        if self.feat_normal[iL][il]in ['cos', 'cosine']:
                            self.tau2[iL][il] = self.sigma_square_cos(self.XF[iL][il], self.YF[iL][il]) #TODO
                            self.DK[iL][il] = 2
                        else:
                            self.tau2[iL][il] = self.sigma_square(self.XF[iL][il], self.YF[iL][il])
                            self.DK[iL][il] = self.XF[iL][il].shape[1]
                        
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
    
        self.Nf = self.to_tensor(self.Ns, dtype=self.floatx, device=self.device)
        self.Nf = self.Nf/self.Nf.mean()
        if self.omega_normal:
            self.omega = self.omega/self.Nf #TODO check
            # self.omega = self.omega/self.omega.sum()

        self.Q = sum(self.Ns) * self.D * 0.5 * (1 + self.xp.log(self.sigma2)).sum()
        self.diff = self.Q

    def register(self, callback= None, **kwargs):
        self.adjustable_paras(**kwargs)
        self.init_params()
        self.echo_paras(ncols=self.each_ncols)
        self.pwcompute_feauture()

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

        if self.get_final_P:
           self.P = self.update_P_xp(self.X, self.TY,self.sigma2, self.gs, xp=self.xp,
                                    d2f=self.d2f, tau2=self.tau2, 
                                    tau2_auto=self.tau2_auto, eps=self.eps,
                                    tau2_alpha=self.tau2_grow[self.iteration], DF=self.DK)
           self.P = self.P.detach().cpu()
        self.update_normalize()
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

    def echo_paras(self, paras=None, maxrows=10, ncols = 2, dvinfo=None,):
        if paras is None:
            paras = ['Ns', 'M', 'D', 'L', 'FDs', 'FL', 'fexists',  'use_keops', 
                     'omega', 'omega_normal', 'ws','wa', 'normal', 'scale_factor',
                     'feat_normal', 
                     'sigma2', 'sigma2_min',  'sigma2_sync',
                     'tau2', 'tau2_clip', 'tau2_decayto', 'tau2_decaystop', 'tau2_auto',  
                     'record',  'df_version',
                      
                      'maxiter', 'transformer',  'tol', 'verbose',
                      'device', 'device_pre',  'floatx', 'floatxx', 
                     'sample', 'sample_growto', 'sample_min', 'sample_stop', 'pair_num',
            ]
        drop_paras = [
            'use_fast_low_rank', 'use_low_rank',
        ]
        logpara = [] 
        for ipara in paras:
            if hasattr(self, ipara):
                ivalue = self.echo_value(getattr(self, ipara))
                logpara.append([ipara, ivalue])
        logpara = sorted(logpara, key=lambda x: x[0])

        logpara1 = []
        for key, val in self.transparas.items():
            ivalue = self.echo_value(val)
            logpara1.append([key, ivalue])
        if len(logpara1)>0:
            logpara1 = sorted(logpara1, key=lambda x: x[0])
            logpara += logpara1    
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
        headinfo = 'init parameters:'.center(headsize, '-')
        if dvinfo is None:
            dvinfo = ' '.join(sorted(set(self.dvinfo)))
            dvinfo = dvinfo.center(headsize, '-')
        elif dvinfo is False:
            dvinfo = ''
        else:
            dvinfo = ' '.join(sorted(set(dvinfo)))
            dvinfo = dvinfo.center(headsize, '-')
        logpara1 = '\n' + '\n'.join([headinfo] + logpara1 + [dvinfo])
        
        if self.verbose > 1:
            logger.info(logpara1)

    def del_cache_attributes(self, attributes=None):
        if attributes is None:
            attributes = [ 'Pf', 'P1', 'Pt1', 'cdff', 
                          'PX', 'MY', 'Av', 'AvUCv', 
                         'd2f', 'dpf', 'fds',
                          #Xr, Yr,
                           'XFr', 'YFr'
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
                    setattr(self, a, self.tensor2numpy(value))
                else:
                    setattr(self, a, self.tensordetach(value))