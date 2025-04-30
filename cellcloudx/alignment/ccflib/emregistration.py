import numpy as np
from tqdm import tqdm

from ._neighbors import Neighbors
from .th_operation import thopt
from ._swnn import swnn
from ...transform import homotransform_point, ccf_deformable_transform_point
from ...io._logger import logger

class EMRegistration(thopt):
    def __init__(self, X, Y, X_feat=None, Y_feat=None, 
                 sigma2=None, 
                 sigma2_step= None, 
                 sigma2_iters =3,
                 tau2=None, 
                 tau2_auto=False,
                 tau2_clip= [5e-3, 5.0],
                 ts_ratio=1.0,

                 pairs=None,
                 c_alpha=1,
                 c_threshold=0.65,
                 use_swnn = False, 
                 swnn_args = {},

                 maxiter=None, 
                 tol=None, 
                 normal=None, 
                 feat_normal='cos', 
                 feat_model = 'gmm',
                 smm_dfs = 5,
                 theta = None,
                 data_level = 1,
                 data_split = None,
  
                 get_final_P = False,
    
                 floatx = None,
                 device = None,
                 device_pre = 'cpu',
                 use_keops=None,
                 
                
                 w=None, c=None,
                 wa=0.99,
                 w_clip=[0, 1.0-5e-2],
                 K=None, KF=None, p_epoch=None, 
                 kd_method='sknn', 
                 kdf_method='annoy',

                 seed = 200504,
                 verbose = 1, 
                 **kargs
                ):
        super().__init__(device=device, device_pre=device_pre,
                          floatx=floatx, seed=seed)
        self.verbose = verbose
        self.reg_core = None
        self.normal_ = normal
        self.data_level = data_level
        self.use_keops = ( (True if X.shape[0]*Y.shape[0] > 8e8 else False)
                            if use_keops is None else  use_keops)   
        if self.use_keops:
            try:
                import pykeops
                pykeops.set_verbose(False)
                from pykeops.torch import LazyTensor
                self.LazyTensor = LazyTensor
            except:
                raise ImportError('pykeops is not installed, `pip install pykeops`')
        self.expectation =  self.expectation_ko if self.use_keops else self.expectation_full

        self.data_split = data_split
        self.feat_normal = feat_normal
        self.get_final_P = get_final_P

        self.feat_model = feat_model
        self.smm_dfs = smm_dfs
        
        self.init_XY(X, Y, X_feat, Y_feat)
        self.maxiter = maxiter
        self.iteration = 0
        self.tol = tol
        self.theta = 0.1 if theta is None else theta

        self.w_clip = w_clip
        self.wa = wa
        self.init_outlier(w, w_clip)
        
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
        self.sigma2 = sigma2
        self.sigma2_step = sigma2_step
        self.sigma2_iters = sigma2_iters
        self.tau2 = tau2 
        self.tau2_auto = tau2_auto
        self.tau2_clip = tau2_clip
        self.ts_ratio = ts_ratio
        self.homotransform_point = homotransform_point
        self.ccf_deformable_transform_point = ccf_deformable_transform_point
        self.dvinfo = [self.get_memory(idv) for idv in [self.device, self.device_pre]]

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
        self.Xr = self.to_tensor(X, dtype=self.floatx, device='cpu')
        self.Yr = self.to_tensor(Y, dtype=self.floatx, device='cpu')

        self.N, self.Dx = self.Xr.shape
        self.M, self.Dy = self.Yr.shape
        self.D = self.Dx

        self.fexist = not (X_feat is None or Y_feat is None)
        if self.fexist:
            self.XF = self.to_tensor(X_feat, dtype=self.floatx, device='cpu')
            self.YF = self.to_tensor(Y_feat, dtype=self.floatx, device='cpu')

            assert self.XF.shape[1] == self.YF.shape[1]
            assert self.XF.shape[0] == self.Xr.shape[0]
            assert self.YF.shape[0] == self.Yr.shape[0]
            self.DF = self.XF.shape[1]

    def init_outlier(self, w, w_clip):
        # chulls = [np.ptp(i, axis=0).prod() for i in [X, Y]]
        if w is None:
            try:
                from scipy.spatial import ConvexHull
                chulls = [ ConvexHull(i.detach().cpu().numpy()).volume 
                        for i in [self.Xr, self.Yr]]
                phulls = [ int(i.shape[0]) for i in [self.Xr, self.Yr]]
                w = 1 - min([ min([i[0]/i[1], i[1]/i[0]]) for i in [chulls, phulls]])
            except:
                w = 0.
        w = float(w)
        if w_clip is not None:
            w = np.clip(w, *w_clip)
        self.w = self.to_tensor(w, dtype=self.floatx, device=self.device)

    def normalXY(self):
        if self.normal in ['each', 'isoscale']:
            _, self.Xm, self.Xs, self.Xf = self.centerlize(self.Xr, Xm=None, Xs=None)
            _, self.Ym, self.Ys, self.Yf = self.centerlize(self.Yr, Xm=None, Xs=None)
            if self.normal in ['isoscale']:
                self.Xs = sum([self.Xs, self.Ys])/2.0
                self.Ys = self.Xs

        elif self.normal  in [True, 'global']:
            XY = self.xp.concat((self.Xr, self.Yr), 0)
            _, XYm, XYs, XYf = self.centerlize(XY, Xm=None, Xs=None)

            self.Xm, self.Ym = XYm, XYm
            self.Xs, self.Ys = XYs, XYs
            self.Xf, self.Yf = XYf, XYf

        elif self.normal == 'X':
            _, self.Xm, self.Xs, self.Xf = self.centerlize(self.Xr, Xm=None, Xs=None)
            self.Ym, self.Ys, self.Yf = self.Xm, self.Xs, self.Xf
        
        elif self.normal in [False, 'pass']:
            self.Xm, self.Xs, self.Xf = (self.xp.zeros(self.Dx,  dtype=self.floatx, device=self.device), 
                                        self.xp.asarray(1,    dtype=self.floatx, device=self.device), 
                                        self.xp.eye(self.Dx+1, dtype=self.floatx, device=self.device))
            self.Ym, self.Ys, self.Yf = (self.xp.zeros(self.Dy,  dtype=self.floatx, device=self.device), 
                                        self.xp.asarray(1,    dtype=self.floatx, device=self.device), 
                                        self.xp.eye(self.Dy+1, dtype=self.floatx, device=self.device))
        else:
            raise ValueError(
                "Unknown normalization method: {}".format(self.normal))

        self.X, self.Xm, self.Xs, self.Xf = self.centerlize(self.Xr, Xm=self.Xm, Xs=self.Xs, device=self.device)
        self.Y, self.Ym, self.Ys, self.Yf = self.centerlize(self.Yr, Xm=self.Ym, Xs=self.Ys, device=self.device)
        self.TY = self.Y.clone().to(self.device)

    def normalXYfeatures(self): # TODO 
        if self.fexist:
            if self.feat_normal in ['cosine', 'cos'] :
                self.XF = self.normalize(self.XF, device=self.device_pre)
                self.YF = self.normalize(self.YF, device=self.device_pre)
            elif self.feat_normal == 'zc':
                self.XF = self.centerlize(self.XF, device=self.device_pre)[0]
                self.YF = self.centerlize(self.YF, device=self.device_pre)[0]
            elif self.feat_normal == 'pcc':
                self.XF = self.scaling(self.XF, anis_var=True, device=self.device_pre)
                self.YF = self.scaling(self.YF, anis_var=True, device=self.device_pre)
            else:
                logger.warning(f"Unknown feature normalization method: {self.feat_normal}")
            self.XF = self.XF.to(self.device_pre)
            self.YF = self.YF.to(self.device_pre)

    def adjustable_paras(self, **kargs):
        for key, value in kargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)

    def echo_paras(self, paras=None, maxrows=10, ncols = 2):
        if paras is None:
            paras = ['N', 'M', 'D', 'DF', 'sigma2', 'tau2', 'K', 'KF', 'alpha', 'beta', 
                     'device', 'device_pre', 'feat_model', 'use_keops', 'floatx',
                     'gamma1', 'feat_normal', 'maxiter', 'reg_core', 'tol', 'df_version',
                      'gamma2', 'kw', 'kl', 'beta_fg', 'use_fg', 'normal', 'low_rank', 
                      'fast_low_rank',  'w', 'c', 'lr', 'lr_gamma', 'lr_stepsize','opt']
        logpara = [] 
        for ipara in paras:
            if hasattr(self, ipara):
                ivalue = getattr(self, ipara)
                try:
                    if ((type(ivalue) in [float]) or
                         (self.xp.is_floating_point(ivalue))
                        ):
                        logpara.append([ipara, f'{ivalue:.4e}'])   
                    else:
                        logpara.append([ipara, f'{ivalue}'])
                except:
                    logpara.append([ipara, f'{ivalue}'])
        logpara = sorted(logpara, key=lambda x: x[0])
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
        dvinfo = ' '.join(sorted(set(self.dvinfo)))
        dvinfo = dvinfo.center(headsize, '-')
        logpara1 = '\n' + '\n'.join([headinfo] + logpara1 + [dvinfo])
        logger.info(logpara1)

    def postfix(self, **kargs):
        iargs = {'tol': f'{self.diff :.3e}', 
                 'Q': f'{self.q :.3e}', 
                 'np': f'{self.Np :.3e}',
                #  'w': f'{self.w :.3e}',
                 '\u03C3^2': f'{self.sigma2 :.3e}'}
        if self.fexist and self.tau2_auto:
            iargs.update({'\u03C4^2': f'{self.tau2 :.3e}'})
        iargs.update(kargs)
        return iargs

    def init_params(self):
        self.sigma2 =  self.to_tensor(self.sigma2 or self.sigma_square(self.X, self.TY), 
                                      device=self.device)
        self.q = 1.0 + self.N * self.D * 0.5 * self.xp.log(self.sigma2)
        self.gs = (self.M/self.N)*self.w/(1-self.w)
        self.Np = self.N
        self.diff = self.q
        if self.fexist:
            if self.tau2 is None:
                tau2 = self.sigma_square_cos(self.XF, self.YF)
                if self.tau2_auto:
                    self.tau2 = tau2
                else:
                    self.tau2 = 0.5 if tau2 >=1.3 else 0.1
            self.tau2 = self.to_tensor(self.tau2, device=self.device)

    def pre_compute_paras(self):
        if self.fexist:
            if self.use_keops:
                self.d2f = self.kodist( self.YF.to(self.device),
                                        self.XF.to(self.device))
                if not self.tau2_auto:
                    # self.d2f = self.d2f - self.d2f.min() #TODO ERROR
                    self.d2f = self.d2f/(-2*self.tau2)
            else:
                try:
                    self.d2f = self.xp.cdist(self.YF, self.XF, p=2)
                    self.d2f.pow_(2)
                    self.d2f = self.d2f.to(self.device)
                    if not self.tau2_auto:
                        # self.d2f.sub_(self.d2f.min())
                        self.d2f.div_(-2*self.tau2)
                except:
                    logger.error('Error in computing d2f, use `use_keops=True` instead.')

            if not self.pairs is None:
                iXF = self.XF.to(self.device)[self.pairs[0]]
                iYF = self.YF.to(self.device)[self.pairs[1]]
                self.dpf = (iXF - iYF).pow(2).sum(1)
                self.dpf = self.dpf.to(self.device)
                if not self.tau2_auto:
                    self.dpf.div_(-2*self.tau2)

    def kodist(self, X, Y):
        x_i = self.LazyTensor(X[:, None, :])
        y_j = self.LazyTensor(Y[None, :, :])
        return ((x_i - y_j)**2).sum(dim=2)

    def register(self, callback= None, **kwargs):
        self.adjustable_paras(**kwargs)
        self.init_params()
        self.echo_paras()
        self.pre_compute_paras()

        pbar = tqdm(range(self.maxiter), total=self.maxiter, colour='red', desc=f'{self.reg_core}', disable=(self.verbose==0))
        for i in pbar:
            if ((not self.sigma2_step is None) and 
                (i % self.sigma2_step == 0) and 
                (i // self.sigma2_step <= self.sigma2_iters)):
                self.sigma2 = self.sigma_square(self.X, self.TY)
                self.q = 1.0 + self.N * self.D * 0.5 * self.xp.log(self.sigma2)

            self.optimization()
            pbar.set_postfix(self.postfix())

            if callable(callback):
                kwargs = {'iteration': self.iteration,
                            'error': self.q, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)

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
           self.P = self.update_P(self.X, self.TY)
           self.P = self.P.detach().cpu()
        self.update_normalize()
        self.del_cache_attributes()
        self.detach_to_cpu(to_numpy=True)

    def expectation_full(self):
        P = self.update_P(self.X, self.TY)
        self.Pt1 = self.xp.sum(P, 0).to_dense()
        self.P1 = self.xp.sum(P, 1).to_dense()
        self.Np = self.xp.sum(self.P1)
        self.PX = P @ self.X
    
        self.w = (1- self.Pt1).clip(*self.w_clip)
        # w = (1.0-self.Np/self.N).clip(*self.w_clip)
        # self.w = self.wa*self.w + (1-self.wa)*w
        self.gs = (self.M/self.N)*self.w/(1-self.w)

        if self.fexist and self.tau2_auto:
            # tau2 = self.xp.einsum('ij,ij->', P, self.d2f)/self.Np/self.DF
            PF = P @ self.XF.to(self.device)
            trPfg = self.xp.sum(self.YF.to(self.device) * PF)
            self.tau2 = (2* self.Np - 2 * trPfg) / (self.Np ) #(self.Np * self.DF)
            self.tau2 = self.tau2.clip(*self.tau2_clip)

        self.update_P_ko_pairs()
        # if not self.pairs is None:
        #     pp = P[self.pairs[1], self.pairs[0]]
        #     pairs = self.pairs[:, pp >= self.c_threshold]

        #     if pairs.shape[1] > 0:
        #         c_Pt1, c_P1, c_PX = self.constrained_expectation(pairs)
        #         Pt1 = self.Pt1 + self.c_alpha*self.sigma2*c_Pt1
        #         P1 = self.P1 + self.c_alpha*self.sigma2*c_P1
        #         PX = self.PX + self.c_alpha*self.sigma2*c_PX
        #         Np = self.xp.sum(Pt1)

    def update_P(self, X, Y):
        P = self.xp.cdist(Y, X, p=2)
        P.pow_(2)
        P.mul_(-1.0/ (2.0*self.sigma2) )
        if self.fexist:
            if self.tau2_auto:
                P.add_(self.d2f, alpha=-self.ts_ratio/(2.0*self.tau2))
                cs = 0.5*(
                    self.D*self.xp.log(2.0*self.xp.pi*self.sigma2) 
                    # + self.DF*self.ts_ratio*self.xp.log(2.0*self.xp.pi*self.tau2)
                )
            else:
                P.add_(self.d2f)
                cs = 0.5*(
                    self.D*self.xp.log(2.0*self.xp.pi*self.sigma2) 
                    # + self.DF*self.xp.log(2.0*self.xp.pi*self.tau2)
                )
        else:
            cs = 0.5*(
                self.D*self.xp.log(2.0*self.xp.pi*self.sigma2)
            )
        if True:
            P.exp_()
            cs = self.xp.exp(cs)*self.gs
            cdfs = self.xp.sum(P, 0).to_dense()
            cdfs.add_(cs)
            cdfs.masked_fill_(cdfs == 0, 1.0)
            P.mul_(1.0/cdfs)
        else:
            cs = cs + self.xp.log(self.gs+self.eps)
            log_cdfs = self.xp.logsumexp(P, axis=0)
            log_cdfs = self.xp.logaddexp(log_cdfs, cs)
            P.sub_(log_cdfs)
            P.exp_()
        return P
    
    def expectation_ko(self):
        P, c = self.update_P_ko(self.X, self.TY)
        ft1 = P.sum(dim=0).flatten()
        a = (ft1 + c)
        a.masked_fill_(a == 0, self.eps)
        a = 1.0/a

        self.Pt1 = ft1*a #1 - a*c
        self.P1 = P @ a
        self.PX = P @ (self.X*a.unsqueeze(1))
        self.Np = self.xp.sum(self.Pt1)

        # self.w = (1- self.Pt1).clip(*self.w_clip)
        w = (1-self.Np/self.N).clip(*self.w_clip)
        self.w = self.wa*self.w + (1-self.wa)*w
        self.gs = (self.M/self.N)*self.w/(1-self.w)

        if self.fexist and self.tau2_auto:
            # self.tau2 = self.xp.einsum('ij,ij->', P, self.d2f)/self.Np/self.DF
            PF = P @ (self.XF.to(self.device)*a.unsqueeze(1))
            trPfg = self.xp.sum(self.YF.to(self.device) * PF)
            self.tau2 = (2* self.Np - 2 * trPfg) /self.Np #(self.Np * self.DF)
            self.tau2 = self.tau2.clip(*self.tau2_clip)

        self.update_P_ko_pairs(a)

    def update_P_ko_pairs(self, a=None):
        if not self.pairs is None:
            xids, yidx = self.pairs[0], self.pairs[1]
            iX = self.X[xids]
            iY = self.TY[yidx]

            dists = (iX- iY).pow(2).sum(1)
            dists.mul_(-1.0/ (2.0*self.sigma2) )
            if self.fexist:
                if self.tau2_auto:
                    dists.add_(self.dpf, alpha=-self.ts_ratio/(2.0*self.tau2))
                else:
                    dists.add_(self.dpf)
            # dists.exp_().mul_(a[xids])
            dists.exp_()
            # dists.mul_(a[xids])
            # dists.mul(1./(2*self.xp.pi*self.sigma2)**(self.D/2))
            self.pairs_idx = dists >= self.c_threshold
            pairs = self.pairs[:, self.pairs_idx]
            if pairs.shape[1] > 0:
                c_Pt1, c_P1, c_PX = self.constrained_expectation(pairs)
                self.Pt1 = self.Pt1 + self.c_alpha*self.sigma2*c_Pt1
                self.P1 = self.P1 + self.c_alpha*self.sigma2*c_P1
                self.PX = self.PX + self.c_alpha*self.sigma2*c_PX
                self.Np = self.xp.sum(self.Pt1)

    def update_P_ko(self, X, Y):
        P = self.kodist(Y, X)
        P = P/(-2*self.sigma2)
        if self.fexist:
            if self.tau2_auto:
                P = P  + self.ts_ratio/(-2.0*self.tau2)*self.d2f
                cs = 0.5*(
                    self.D*self.xp.log(2.0*self.xp.pi*self.sigma2) 
                    # + self.DF*self.ts_ratio*self.xp.log(2.0*self.xp.pi*self.tau2)
                )
            else: 
                P = P + self.d2f
                cs = 0.5*(
                    self.D*self.xp.log(2.0*self.xp.pi*self.sigma2) 
                    # + self.DF*self.xp.log(2.0*self.xp.pi*self.tau2)
                )
        else:
            cs = 0.5*(
                self.D*self.xp.log(2.0*self.xp.pi*self.sigma2)
            )

        P = P.exp()
        cs = self.xp.exp(cs)*self.gs
        return P, cs

    def constrained_expectation(self, pairs, pp=None):
        if pp is None:
            pp = self.xp.ones(pairs.shape[1]).to(self.device, dtype=self.floatx)
        V = self.xp.sparse_coo_tensor( 
                    self.xp.vstack([pairs[1], pairs[0]]), pp, 
                    size=(self.M, self.N))
        c_Pt1 = self.xp.sum(V, 0).to_dense()
        c_P1 = self.xp.sum(V, 1).to_dense()
        c_PX = V @ self.X
        return c_Pt1, c_P1, c_PX

    def pre_compute_paras_df(self):
        self.tau2_auto = False
        if self.fexist:
            if self.use_keops:
                self.d2f = self.kodist( self.YF.to(self.device), self.XF.to(self.device))
                self.d2f = self.d2f/(-2*self.tau2)
                self.cdff = self.d2f.exp()
                self.cdff = self.cdff.sum(dim=0).flatten()
                if not self.tau2_auto: #TODO
                    pass
            else:
                try:
                    self.d2f = self.xp.cdist(self.YF.to(self.device), self.XF.to(self.device), p=2)
                    self.d2f.pow_(2).div_(-2*self.tau2).exp_()
                    cdff = self.xp.sum(self.d2f, 0).to_dense()
                    self.d2f.div_(cdff)

                    if not self.tau2_auto: #TODO
                        pass 
                except:
                    logger.error('Error in computing d2f, use `use_keops=True` instead.')

    def expectation_full_df(self):
        P = self.update_P_df(self.X, self.TY)
        self.Pt1 = self.xp.sum(P, 0).to_dense()
        self.P1 = self.xp.sum(P, 1).to_dense()
        self.Np = self.xp.sum(self.P1)
        self.PX = P @ self.X

        # self.w = (1- self.Pt1).clip(*self.w_clip)
        # w = (1.0-self.Np/self.N).clip(*self.w_clip)
        # self.w = self.wa*self.w + (1-self.wa)*w
        self.gs = (self.M/self.N)*self.w/(1-self.w)

        if self.fexist and self.tau2_auto:
            # tau2 = self.xp.einsum('ij,ij->', P, self.d2f)/self.Np/self.DF
            PF = P @ self.XF.to(self.device)
            trPfg = self.xp.sum(self.YF.to(self.device) * PF)
            self.tau2 = (2* self.Np - 2 * trPfg) / (self.Np ) #(self.Np * self.DF)
            self.tau2 = self.tau2.clip(*self.tau2_clip)

        self.update_P_ko_pairs()

    def expectation_ko_df(self):
        P, a = self.update_P_ko_df(self.X, self.TY)
        ft1 = P.sum(dim=0).flatten()

        a.masked_fill_(a == 0, self.eps)
        a = 1.0/a

        self.Pt1 = ft1*a #1 - a*c
        self.P1 = P @ a
        self.PX = P @ (self.X*a.unsqueeze(1))
        self.Np = self.xp.sum(self.Pt1)

        # self.w = (1- self.Pt1).clip(*self.w_clip)
        # w = (1-self.Np/self.N).clip(*self.w_clip)
        # self.w = self.wa*self.w + (1-self.wa)*w
        self.gs = (self.M/self.N)*self.w/(1-self.w)

        if self.fexist and self.tau2_auto:
            # self.tau2 = self.xp.einsum('ij,ij->', P, self.d2f)/self.Np/self.DF
            PF = P @ (self.XF.to(self.device)*a.unsqueeze(1))
            trPfg = self.xp.sum(self.YF.to(self.device) * PF)
            self.tau2 = (2* self.Np - 2 * trPfg) /self.Np #(self.Np * self.DF)
            self.tau2 = self.tau2.clip(*self.tau2_clip)

        self.update_P_ko_pairs(a)
    
    def update_P_df(self, X, Y):
        P = self.xp.cdist(Y, X, p=2)
        P.pow_(2).mul_(-1.0/ (2.0*self.sigma2) ).exp_()
        cdfs = self.xp.sum(P, 0).to_dense()

        if self.fexist:
                P.multiply_(self.d2f)

        cs = 0.5*(
            self.D*self.xp.log(2.0*self.xp.pi*self.sigma2)
        )
        cs = self.xp.exp(cs)*self.gs
        cdfs.add_(cs)
        cdfs.masked_fill_(cdfs == 0, 1.0)
        P.mul_(1.0/cdfs)
        return P

    def update_P_ko_df(self, X, Y):
        P = self.kodist(Y, X)
        P = P/(-2*self.sigma2)
        cdf = P.exp()
        cdf = cdf.sum(dim=0).flatten()
        
        if self.fexist:
            if self.tau2_auto:
                P = P  + self.ts_ratio/(-2.0*self.tau2)*self.d2f
                cs = 0.5*(
                    self.D*self.xp.log(2.0*self.xp.pi*self.sigma2) 
                    + self.DF*self.ts_ratio*self.xp.log(2.0*self.xp.pi*self.tau2)
                )
            else: 
                P = P + self.d2f
                cs = 0.5*(
                    self.D*self.xp.log(2.0*self.xp.pi*self.sigma2) 
                    # + self.DF*self.xp.log(2.0*self.xp.pi*self.tau2)
                )
        else:
            cs = 0.5*(
                self.D*self.xp.log(2.0*self.xp.pi*self.sigma2)
            )

        P = P.exp()
        cs = self.xp.exp(cs)*self.gs
        a = (cs + cdf)*self.cdff
        return P, a

    def update_PK(self, X, Y, K):
        D, N, M = X.shape[1], X.shape[0], Y.shape[0]
        src, dst, P, sigma2 = self.cdist_k(X, Y, 
                                          knn=K,
                                          method=self.kd_method,
                                          sigma2=sigma2 )
        P.mul_(-0.5/sigma2)
        P.exp_()
        if self.fexist:
            P = P + self.d2f[dst, src]
            P = P.exp()
            cs = 0.5*(
                self.D*self.xp.log(2*self.xp.pi*self.sigma2) 
                # + self.DF*self.xp.log(2*self.xp.pi*self.tau2)
            )
        else:
            P = P.exp()
            cs = 0.5*(
                self.D*self.xp.log(2*self.xp.pi*self.sigma2)
            )

        P = self.xp.sparse_coo_tensor( self.xp.vstack([dst, src]), P, 
                                        size=(M, N), 
                                        dtype=self.floatx)
        cdfs = self.xp.sum(P, 0, keepdim=True).to_dense()

        Nx = self.xp.sum(cdfs>0)
        gs = M/Nx*self.w/(1. - self.w)
        cs = self.xp.exp(cs+gs)
        cdfs.add_(cs) 
        cdfs.masked_fill_(cdfs == 0, 1.0) 
        P.mul_(1.0/cdfs)
        return P

    def cdist_k(self, X, Y, sigma2=None, method='cunn',
                    metric='euclidean', 
                    knn=200, n_jobs=-1, **kargs):
        (N, D) = X.shape
        M = Y.shape[0]
        snn = Neighbors(method=method, metric=metric, 
                        device_index=self.device_index,
                        n_jobs=n_jobs)
        snn.fit(X, **kargs)
        ckdout = snn.transform(Y, knn=knn)
        nnidx = ckdout[1]

        src = self.xp.LongTensor(nnidx.flatten('C')).to(X.device)
        dst = self.xp.repeat_interleave(self.xp.arange(M, dtype=self.xp.int64), knn, dim=0).to(X.device)
        dist2 = self.xp.tensor(ckdout[0].flatten('C'), dtype=X.dtype, device=X.device)
        dist2.pow_(2)

        if sigma2 is None:
            sigma2 = self.xp.mean(dist2)/D
        return src, dst, dist2, sigma2
    
    def del_cache_attributes(self, attributes=None):
        if attributes is None:
            attributes = [ 'Pf', 'P1', 'Pt1', 'cdff', 
                         'PX', 'MY', 'Av', 'AvUCv', 
                         'd2f', 'dpf',
                          #Xr, Yr,
                          #'XF', 'YF', 
                          'VAv', 'Fv', 'F', 'MPG' ]                
        for attr in attributes:
            if hasattr(self, attr):
                delattr(self, attr)

        self.clean_cache()

    def detach_to_cpu(self, attributes=None, to_numpy=True):
        if attributes is None:
            attributes = ['R', 'A', 'B', 't', 'd', 's',
                          'Xm', 'Xs', 'Xf', 'X', 'Ym', 'Ys', 'Y', 'Yf', 
                          'beta', 'G', 'Q', 'S', 'W', 'inv_S',
                          'tmat', 'tmatinv', 'tform', 'tforminv',
                          'TY', 'P', 'C']    
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
                else:
                    value = value.detach().cpu()
                    setattr(self, a, value)

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