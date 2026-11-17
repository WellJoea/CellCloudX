import numpy as np
import torch as th
from tqdm import tqdm

from .shared_wnn import swnn
from ...io._logger import logger

from .operation_th import thopt
from .paired_operation import spEM_core

class spEMRegistration(thopt, spEM_core):
    def __init__(self, X, Y, X_feat=None, Y_feat=None, 
                 pairs = None,
        
                 maxiter=None, 
                 tol=None, 
    
                 normal=None, 
                 scale_factor = None,

                 sigma2=None, 
                 sigma2_min = 1e-7,

                 tau2=None, 
                 tau2_auto=False,
                 tau2_delta = 1,
                 tau2_clip= [5e-3, 50.0],
                 tau2_decayto = None,
                 tau2_decaystop = None,
                 feat_normal=None, 
                 st_ratio =1.0,
    
                 floatx = None,
                 floatxx = None,
                 device = None,
                 device_pre = None,
                #  use_keops=8e8,
                
                 w=None, c=None,
                 theta = 0.75, wso = 5, wto=5,
                 wa=0.997, w_clip=[0.05, 0.95],

                 U = None,     
                 eps=1e-8,

                 seed = 491001,

                 record = None,
                 each_ncols= None,
                 header='',
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
        self.init_points(X, Y, X_feat, Y_feat,  pairs=pairs)
        self.init_outlier(w, wa, w_clip, wso, wto, theta, st_ratio)

        # self.use_keops = use_keops
        self.normal = normal
        self.scale_factor=scale_factor or 1.0

        self.sigma2 = sigma2  
        self.sigma2_min = sigma2_min

        self.feat_normal = self.scalar2vetor( 'cos' if feat_normal is None else feat_normal, L=self.FL)
        self.tau2 =  self.scalar2vetor(tau2, L=self.FL)
        self.tau2_clip = tau2_clip
        self.tau2_delta = self.scalar2vetor(tau2_delta, L=self.FL)
        self.tau2_decayto = self.scalar2vetor(tau2_decayto or 1.0, L=self.FL)   
        self.tau2_decaystop = self.scalar2vetor(tau2_decaystop or 1.0, L=self.FL) 
        self.tau2_auto = self.scalar2vetor(tau2_auto, L=self.FL)

        self.record = [] if record is None else record
        self.records = {}

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

    # def swnn_pairs(self, ):
    #     dargs = dict(
    #              mnn=6, snn=30, fnn=60, temp=1.0,
    #              scale_locs=False, scale_feats=False,
    #              lower = 0.01, upper=0.995,
    #              min_score= 0.35,
    #              max_pairs=5e4,
    #              min_pairs=100)
    #     dargs.update(self.swnn_args)
    #     pairs = swnn(self.X, self.Y, self.XF, self.YF, **dargs)[0]
    #     return pairs

    # def constrained_pairs(self):
    #     if self.use_swnn:
    #         pairs = self.swnn_pairs()
    #     if self.pairs is None:
    #         if self.use_swnn:
    #             self.pairs = pairs
    #     else:
    #         if self.use_swnn:
    #             self.pairs = self.xp.stack([self.pairs, pairs], 0)

    def init_points(self, X, Y, X_feat, Y_feat, pairs=None):
        X = self.to_tensor(X, dtype=self.floatxx, device='cpu')
        Y = self.to_tensor(Y, dtype=self.floatxx, device='cpu')

        if pairs is None:
            assert X.shape == Y.shape, "X and Y must have the same points number"
            # self.uX, self.X_idx = self.xp.unique(X, return_inverse=True) not consider features(pass)
            # self.uY, self.Y_idx = self.xp.unique(Y, return_inverse=True)
            xidx = self.xp.arange(X.shape[0])
            self.pairs = self.xp.vstack((xidx, xidx)).T
        else:
            self.pairs = pairs
        self.pairs = self.to_tensor(self.pairs, dtype=self.xp.int64, device=self.device)

        self.Xr = self.to_tensor(X, dtype=self.floatxx, device='cpu')
        self.Yr = self.to_tensor(Y, dtype=self.floatxx, device='cpu')

        self.N = self.M = self.pairs.shape[0]
        self.D = self.Xr.shape[1]

        self.fexist = not (X_feat is None or Y_feat is None)
        if self.fexist:
            self.XFr = self.init_feature(X_feat, self.Xr.shape[0])
            self.YFr = self.init_feature(Y_feat, self.Yr.shape[0])
            assert len(self.XFr) == len(self.YFr), "Features must have the same length"
            self.FL = len(self.XFr)
            self.FDs = []
            for il in range(self.FL):
                assert self.XFr[il].shape[1] == self.YFr[il].shape[1], "Features must have the same dimension"
                self.FDs.append(self.XFr[il].shape[1])
        else:
            self.FL = 0

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

    def init_outlier(self, w, wa, w_clip, wso, wto, theta, st_ratio):
        if w is None:
            w = self.rigid_outlier(self.Xr, self.Yr)
        self.w = self.to_tensor(w, dtype=self.floatx, device=self.device)
        if w_clip is not None:
            self.w = self.xp.clip(self.w, *w_clip)
        self.w_clip = w_clip
        self.wa = wa
        self.wso = wso
        self.wto = self.scalar2vetor(wto, L=self.FL)
        self.theta = theta
    
        self.st_ratio = [st_ratio/(st_ratio + 1.0)* 2.0, 2.0/(st_ratio + 1.0)]
    
    def normal_XY(self):
        Xa = [ self.Xr, self.Yr ]
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
        self.X, self.Y = Xn[0][self.pairs[:,0]], Xn[1][self.pairs[:,1]]

        self.Hx, self.Hy = Tf
        self.Mx, self.My = M.to(self.device, dtype=self.floatxx)
        self.Sx, self.Sy = S.to(self.device, dtype=self.floatxx)

        self.Vx = 1.0/self.xp.prod((self.xp.quantile(self.X, 0.98, dim=0) - self.xp.quantile(self.X, 0.02, dim=0)))

    def normal_features(self):
        if self.fexist:
            Fr, Fs = [self.XFr, self.YFr], []
            for n in range(len(Fr)):
                nFs = []
                for iL in range(self.FL):
                    inorm = self.feat_normal[iL]
                    iF = self.to_tensor(Fr[n][iL], dtype=self.floatx, device=self.device_pre)
                    if inorm in ['cosine', 'cos'] :
                        iF = self.normalize(iF)[0]
                    elif inorm == 'zc':
                        iF = self.centerlize(iF)[0]
                    elif inorm in ['ccosine', 'ccos']:     
                        iF = self.center_normalize(iF)[0]
                    elif inorm == 'pcc':    
                        iF = self.scaling(iF, anis_var=True)[0]
                    elif inorm == 'label':
                        iF = iF
                    elif inorm in ['pass', 'restc', 'resta']:    
                        pass
                    else:
                        logger.warning(f"Unknown feature normalization method: {inorm}")
                    nFs.append(iF)
                Fs.append(nFs)
            self.XF, self.YF = Fs
            self.Vf = [1.0/self.xp.prod((self.xp.quantile(xf, 0.98, dim=0) - 
                                     self.xp.quantile(xf, 0.02, dim=0))) for xf in self.XF]

    def decay_curve(self, decayto, decaystop, maxiter):
        assert 0 <= decaystop <= 1.0
        decaystop = int(decaystop * maxiter)
        decayrate = self.xp.ones((maxiter), device=self.device, dtype=self.floatx)
        decayrate[:decaystop] = self.xp.linspace(1.0, decayto, decaystop)
        decayrate[decaystop:] = decayto
        return decayrate

    def init_sigma2(self, sigma2=None, sigma2_clip=[1, None]):
        if sigma2 is None or sigma2 <= 0:
            sigma2 = self.sigma_square(self.X, self.Y)

        if sigma2_clip is not None:
            sigma2= self.xp.clip(sigma2, *sigma2_clip)

        self.sigma2 = self.to_tensor(sigma2, dtype=self.floatx, device=self.device)

        if self.transformer in 'P':
            self.logsigma2 = self.xp.nn.Parameter(
                self.xp.log(self.to_tensor(self.sigma2 or 1.0, dtype=self.floatx, device=self.device)).clone()
            )
    
    def init_tau2(self, tau2, tau2_clip=None):
        self.DK = self.scalar2vetor(2, L=self.FL)
        self.tau2_auto = [ bool(i) for i in self.tau2_auto ]
        self.tau2_grow = [ 1.0/self.decay_curve(self.tau2_decayto[i], self.tau2_decaystop[i], self.maxiter) for i in range(self.FL) ]
        self.tau2 = tau2
        if self.FL > 0:
            self.tau2_delta = [ i/ sum(self.tau2_delta) for i in self.tau2_delta]

        for iL in range(self.FL):
            if (tau2[iL] is None) or (tau2[iL] <= 0):
                if self.feat_normal[iL] in ['cos', 'cosine', 'ccosine', 'ccos', 'restc']:
                    self.tau2[iL] = self.sigma_square_cos(self.XF[iL], self.YF[iL]) #TODO
                    self.DK[iL] = 2
                elif self.feat_normal[iL] in ['label']:
                    self.tau2[iL] = 1.0
                    self.DK[iL] = 2
                else:
                    self.tau2[iL] = self.sigma_square(self.XF[iL], self.YF[iL])
                    self.DK[iL] = self.XF[iL].shape[1]
            else:
                tau2[iL] = self.to_tensor(tau2[iL], dtype=self.floatx, device=self.device)

            if tau2_clip is not None:
                self.tau2[iL] = self.xp.clip(self.tau2[iL], *tau2_clip)

    def init_params(self):
        self.init_sigma2(self.sigma2)
        self.init_tau2(self.tau2, getattr(self, 'tau2_clip', None))
        self.TY = self.Y.clone()
        # self.Yh = self.xp.zeros_like(self.Y)

        if self.transformer in 'H':
             self.Yh = self.Y.clone()

        self.Q = self.N * self.D * 0.5 * (1 + self.xp.log(self.sigma2))
        self.diff = self.Q

    def register(self, callback= None, **kwargs):
        self.adjustable_paras(**kwargs)
        self.init_params()
        self.echo_paras(ncols=self.each_ncols)
        self.spcompute_feauture()
        self.spcompute_regularizer()

        pbar = tqdm(range(self.maxiter), total=self.maxiter, colour='red', 
                    desc=f'{self.transformer}', disable=(self.verbose <1))
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

        # if self.get_final_P:
        #    self.P = self.update_P_xp(self.X, self.TY,self.sigma2, self.gs, xp=self.xp,
        #                             d2f=self.d2f, tau2=self.tau2, 
        #                             tau2_auto=self.tau2_auto, eps=self.eps,
        #                             tau2_alpha=self.tau2_grow[self.iteration], DF=self.DK)
        #    self.P = self.P.detach().cpu()
        self.update_normalize()
        # self.del_cache_attributes()
        self.detach_to_cpu(to_numpy=True)

    def postfix(self, **kargs):
        iargs = {'tol': f'{self.diff.item() :.3e}', 
                 'Q': f'{self.Q.item() :.3e}', 
                 #  'np': f'{self.Np :.3e}',
                 '\u03C3^2':  f'{self.sigma2.item() :.3e}'
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
            paras = ['N',  'D',  'FD', 'FL', 'fexist',  'use_keops', 
                    'w','wa', 'normal', 'scale_factor',
                     'feat_normal', 
                     'sigma2', 'sigma2_min', 
                     'tau2', 'tau2_clip', 'tau2_decayto', 'tau2_decaystop', 'tau2_auto',  
                     'tau2_delta', 'st_ratio',
                      'wso', 'wto', 'theta',
                      'record',
                      'maxiter', 'transformer',  'tol', 'verbose',
                      'device', 'device_pre',  'floatx', 'floatxx', 
            ]
        paras = sorted(paras)

        drop_paras = [
            'use_fast_low_rank', 'use_low_rank',
        ]
        logpara = [] 
        for ipara in paras:
            if hasattr(self, ipara):
                ivalue = self.echo_value(getattr(self, ipara))
                logpara.append([ipara, ivalue])
        # logpara = sorted(logpara, key=lambda x: x[0])

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
        
        if self.verbose > 1:
            logger.info(logpara1)

    def del_cache_attributes(self, attributes=None):
        if attributes is None:
            attributes = [ 'Pf', 'P1', 'Pt1', 'cdff', 
                          'PX', 'MY', 'Av', 'AvUCv', 
                         'd2f', 'dpf', 'fds',
                          #Xr, Yr,
                           'XFr', 'YFr',
                          'XF', 'YF',  'I',
                          'VAv', 'Fv', 'F', 'MPG' , 
                        #   'DR',
                          'L', 'LV', 'LY',  'AV', 'J', 'QY', 'LG', 'Yh',
                          'AG', 'RG', 'QG', 'QY' ]                
        for attr in attributes:
            if hasattr(self, attr):
                delattr(self, attr)
        self.clean_cache()

    def detach_to_cpu(self, attributes=None, to_numpy=True):
        if attributes is None:
            attributes = ['R', 'A', 'B', 't', 'd', 's',
                          'Xm', 'Xs', 'Xf',
                        #    'X', 'Ym', 'Ys', 'Y', 'Yf', 
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