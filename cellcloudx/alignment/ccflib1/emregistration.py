import numpy as np
from tqdm import tqdm

from .th_operation import thopt
from ...transform import homotransform_point, ccf_deformable_transform_point
from ...io._logger import logger

class EMRegistration(thopt):
    def __init__(self, X, Y, X_feat=None, Y_feat=None, 
                 sigma2=None, 
                 sigma2_step= None, 
                 sigma2_iters =3,
                 sigma2f=None, maxiter=None, 
                 tol=None, w=None, 
                 normal=None, 
                 feat_normal='l2', 
                 feat_model = 'gmm',
                 smm_dfs = 5,
                 theta = None,
                 data_level = 1,
                 data_split = None,
  
                 pre_cumpute_fp = True,
                 get_final_P = False,
    
                 floatx = None,
                 device = None,
                 device_pre = 'cpu',
                 w_clip=[0, 1-1e-8],
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
        self.data_split = data_split
        self.feat_normal = feat_normal
        self.pre_cumpute_fp = pre_cumpute_fp
        self.get_final_P = get_final_P

        self.feat_model = feat_model
        self.smm_dfs = smm_dfs
        
        self.init_XY(X, Y, X_feat, Y_feat)
        self.maxiter = maxiter or 100
        self.iteration = 0
        self.tol = tol
        self.theta = 0.1 if theta is None else theta

        self.w_clip = w_clip
        self.init_outlier(w, w_clip)

        self.diff = self.xp.inf
        self.q = self.xp.inf
        self.kd_method = kd_method
        self.kdf_method = kdf_method
        self.K = K
        self.KF = KF
        self.sigma2 = sigma2
        self.sigma2_step = sigma2_step
        self.sigma2_iters = sigma2_iters
        self.sigma2f = sigma2f
        self.homotransform_point = homotransform_point
        self.ccf_deformable_transform_point = ccf_deformable_transform_point
        self.dvinfo = [self.get_memory(idv) for idv in [self.device, self.device_pre]]

    def init_XY(self, X, Y, X_feat, Y_feat):
        self.Xr = self.to_tensor(X, dtype=self.floatx, device='cpu')
        self.Yr = self.to_tensor(Y, dtype=self.floatx, device='cpu')

        assert self.Xr.shape[1] == self.Yr.shape[1]
        self.N, self.D = self.Xr.shape
        self.M = self.Yr.shape[0]

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
            from scipy.spatial import ConvexHull
            chulls = [ ConvexHull(i.detach().cpu().numpy()).volume 
                       for i in [self.Xr, self.Yr]]
            phulls = [ int(i.shape[0]) for i in [self.Xr, self.Yr]]
            w = 1 - max([ min([i[0]/i[1], i[1]/i[0]]) for i in [chulls, phulls]])
        w = float(w)
        if w_clip is not None:
            w = np.clip(w, *w_clip)
        self.w = self.to_tensor(w, dtype=self.floatx, device=self.device)

    def normalXY(self):
        if self.normal in [True, 'each']:
            _, self.Xm, self.Xs, self.Xf = self.centerlize(self.Xr, Xm=None, Xs=None)
            _, self.Ym, self.Ys, self.Yf = self.centerlize(self.Yr, Xm=None, Xs=None)

        elif self.normal == 'global':
            XY = self.xp.concat((self.Xr, self.Yr), 0)
            _, XYm, XYs, XYf = self.centerlize(XY, Xm=None, Xs=None)

            self.Xm, self.Ym = XYm, XYm
            self.Xs, self.Ys = XYs, XYs
            self.Xf, self.Yf = XYf, XYf

        elif self.normal == 'X':
            _, self.Xm, self.Xs, self.Xf = self.centerlize(self.Xr, Xm=None, Xs=None)
            self.Ym, self.Ys, self.Yf = self.Xm, self.Xs, self.Xf
        
        elif self.normal in [False, 'pass']:
            self.Xm, self.Xs, self.Xf = (self.xp.zeros(self.D, dtype=self.floatx, device=self.device), 
                                            self.xp.asarray(1,    dtype=self.floatx, device=self.device), 
                                            self.xp.eye(self.D+1, dtype=self.floatx, device=self.device))
            self.Ym, self.Ys, self.Yf = self.Xm, self.Xs, self.Xf
        else:
            raise ValueError(
                "Unknown normalization method: {}".format(self.normal))

        self.X, self.Xm, self.Xs, self.Xf = self.centerlize(self.Xr, Xm=self.Xm, Xs=self.Xs, device=self.device)
        self.Y, self.Ym, self.Ys, self.Yf = self.centerlize(self.Yr, Xm=self.Ym, Xs=self.Ys, device=self.device)
        self.TY = self.Y.clone().to(self.device)

    def normalXYfeatures(self): # TODO
        if self.fexist:
            if self.feat_normal == 'l2':
                self.XF = self.normalize(self.XF, device='cpu')
                self.YF = self.normalize(self.YF, device='cpu')
            elif self.feat_normal == 'zc':
                self.XF = self.centerlize(self.XF, device='cpu')[0]
                self.YF = self.centerlize(self.YF, device='cpu')[0]
            elif self.feat_normal == 'pcc':
                self.XF = self.scaling(self.XF, anis_var=True, device='cpu')
                self.YF = self.scaling(self.YF, anis_var=True, device='cpu')
            else:
                logger.warning(f"Unknown feature normalization method: {self.feat_normal}")
            self.XF = self.XF.to(self.device_pre)
            self.YF = self.YF.to(self.device_pre)

    def pre_cumpute_feature_p(self):
        if self.fexist:
            if self.pre_cumpute_fp :
                if self.KF:
                    self.Pf, self.gf, self.sigma2f = self.kernel_xmm_k( 
                                                            self.XF, 
                                                            self.YF,
                                                            knn=self.KF, 
                                                            method=self.kdf_method,
                                                            dfs=self.smm_dfs, 
                                                            kernel=self.feat_model,
                                                            sigma2=self.sigma2f)
                    K = self.KF
                else:
                    if self.data_level <=3:
                        self.Pf, self.gf, self.sigma2f = self.kernel_xmm(
                                                self.XF, 
                                                self.YF,
                                                dfs=self.smm_dfs, 
                                                kernel=self.feat_model,
                                                sigma2=self.sigma2f)
                self.Pf = self.Pf.detach().cpu().to(self.device)
                self.gf = self.gf.detach().cpu().to(self.device)

                if self.reg_core in ['rigid', 'affine']:
                    self.cdff = self.xp.sum(self.Pf, 0).to_dense() #/ K
                else:
                    self.cdff = self.xp.sum(self.Pf, 0).to_dense() #/ K
                self.cf = 0
                self.Pf.div_(self.cdff)
                self.cdff = 1
                self.gf = self.M # TODO
            else:
                self.sigma2f = self.sigma_square(self.XF, self.YF)

    def adjustable_paras(self, **kargs):
        for key, value in kargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)

    def echo_paras(self, paras=None, maxrows=10, ncols = 2):
        if paras is None:
            paras = ['N', 'M', 'D', 'DF', 'sigma2', 'sigma2f', 'K', 'KF', 'alpha', 'beta', 
                     'device', 'device_pre', 'feat_model', 'smm_dfs', 'data_level',
                     'gamma1', 'feat_normal', 'maxiter', 'reg_core', 'tol',
                      'gamma2', 'kw', 'kl', 'beta_fg', 'use_fg', 'normal', 'low_rank', 
                      'low_rank_g',  'w']
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

    def register(self, callback= None, **kwargs):
        self.adjustable_paras(**kwargs)
        self.sigma2 = self.sigma_square(self.X, self.Y)
        self.q = 1.0 + self.N * self.D * 0.5 * self.xp.log(self.sigma_square(self.X, self.Y))
        self.echo_paras()

        self.pre_cumpute_feature_p()
        if self.verbose:
            pbar = tqdm(range(self.maxiter), total=self.maxiter, colour='red', desc=f'{self.reg_core}')
            for i in pbar:
                if ((not self.sigma2_step is None) and 
                    (i % self.sigma2_step == 0) and 
                    (i // self.sigma2_step <= self.sigma2_iters)):
                    self.sigma2 = self.sigma_square(self.X, self.TY)
                    self.q = 1.0 + self.N * self.D * 0.5 * self.xp.log(self.sigma2)

                self.optimization()
                ww = 1- self.Np/self.N

                self.Ind = (self.P1 > self.theta)
                self.ww = self.xp.clip(self.Ind.sum() / self.M, *self.w_clip)

                log_prt = {
                        'tol': f'{self.diff :.4e}', 
                        'Q': f'{self.q :.5e}', 
                        'w': f'{ww :.5e}',
                        'ww': f'{self.ww :.5e}',
                        'np': f'{self.Np :.3e}',
                        # 'sigma2f': f'{self.sigma2f :.4e}',
                        'sigma2': f'{self.sigma2 :.4e}'}

                # if not self.w_clip is None:
                #     log_prt['w'] = f'{self.w :.3e}'

                pbar.set_postfix(log_prt)
                # if (self.p_epoch):
                #     pbar.update(self.p_epoch)

                if callable(callback):
                    kwargs = {'iteration': self.iteration,
                             'error': self.q, 'X': self.X, 'Y': self.TY}
                    callback(**kwargs)


                if (self.diff <= self.tol): #or (self.xp.abs(self.sigma2) < self.eps):
                    pbar.close()
                    logger.info(f'Tolerance is lower than {self.tol}. Program ends early.')
                    break
            pbar.close()
        else:
            while self.diff > self.tol and (self.iteration < self.maxiter):
                self.optimization()

        if self.get_final_P:
           self.P, self.C = self.update_P(self.X, self.TY)
           self.P = self.P.detach().cpu()
           self.C = self.C.detach().cpu()
        self.update_normalize()
        self.del_cache_attributes()
        self.detach_to_cpu(to_numpy=True)

    def expectation(self):
        P, C = self.update_P(self.X, self.TY)
        P.multiply_(C)

        # P = (P * self.OM) / (C*(1-self.w))
        # P = P.tocsc()[:, self.inlier]
        # self.Pt1 = self.Pt1[self.inlier]

        self.Pt1 = self.xp.sum(P, 0).to_dense()
        self.P1 = self.xp.sum(P, 1).to_dense()
        self.Np = self.xp.sum(self.P1)
        self.PX = P @ self.X

        # print( self.Np, self.N, self.M, 2222)
        # aa = self.Pt1.detach().cpu().numpy().flatten()
        # bb = self.P1.detach().cpu().numpy().flatten()
        # cc = P.detach().cpu().numpy().flatten()
        # dd = self.Pf.detach().cpu().numpy().flatten() * self.gf #.detach().cpu().numpy()
        # ee = self.cdff.detach().cpu().numpy().flatten() * self.gf #.detach().cpu().numpy()
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(3, 2, figsize=(15, 10))
        # axs[0,0].hist(aa, bins=100)
        # axs[0,0].set_yscale('log')
        # axs[0,1].hist(bb, bins=100)
        # axs[0,1].set_yscale('log')
        # axs[1,0].hist(cc, bins=100)
        # axs[1,0].set_yscale('log')
        # axs[1,1].hist( dd, bins=100)
        # axs[1,1].set_yscale('log')
        # axs[2,0].hist(ee, bins=100)
        # axs[2,0].set_yscale('log')
        # plt.show()

        #     Q = -np.sum(np.log(C)) + self.D*self.N*np.log(self.sigma2)/2

    def update_P(self, X, Y):
        if (not self.KF is None) and (not self.K is None): # PASS
            P, gs, self.sigma2 = self.kernel_xmm_k(X, Y, knn=self.K, 
                                                    method=self.kd_method, 
                                                    sigma2=self.sigma2,
                                                    kernel='gmm',)
            if self.fexist:
                self.Pf, self.gf, self.sigma2f = self.kernel_xmm_p(self.YF, self.X_feat, 
                                        pairs=P.nonzero(), 
                                        dfs=self.smm_dfs, 
                                        kernel=self.feat_model,
                                        sigma2=None,)
                Pe = P.to(self.xp.bool)*self.Pf
                Pa = Pe.sum(0)
                self.OM = Pe.to(self.xp.bool).sum(0)

        elif not self.KF is None: # Fast
            P, gs, self.sigma2 = self.kernel_xmm_p(Y, X, 
                                                pairs=self.Pf.nonzero(), 
                                                sigma2=self.sigma2,
                                                kernel='gmm')
            self.OM = self.KF
            if self.fexist:
                Pa = self.cdff
        elif not self.K is None: # Local bettwer
            P, gs, self.sigma2 = self.kernel_xmm_k(X, Y,
                                                    knn=self.K, 
                                                    method=self.kd_method, 
                                                    sigma2=self.sigma2,
                                                    kernel='gmm',)
            self.OM = self.K
            if self.fexist:
                # self.Pf, self.gf, self.sigma2f = kernel_xmm_p(self.Y_feat, self.X_feat, 
                #                         pairs=P.nonzero(), 
                #                         dfs=self.smm_dfs, 
                #                         kernel=self.feat_model,
                #                         sigma2=None)
                # Pa = self.Pf.sum(0) # sigma2f change slightly
                Pe = P.to(self.xp.bool)*self.Pf
                Pa = Pe.sum(0)
                # Pa = self.cdff

        else: # Best slow
            P, gs, self.sigma2 = self.kernel_xmm(X, Y, sigma2=self.sigma2, kernel='gmm')
            self.OM = self.M
            if self.fexist:
                Pa = self.cdff
        
        gs = self.OM/(1. - self.w)/gs
        cs = self.w/self.N
        cdfs = self.xp.sum(P, 0).to_dense()

        # self.wn = int((1-self.w) * self.N)
        # self.inlier = np.argpartition(1.0 - np.divide(1, cdfs+gs*cs), -self.wn)[-self.wn:]
        # self.inlier = np.argpartition(cdfs, -self.wn)[-self.wn:]
        # if not self.w_clip is None:
        #     self.w = np.clip(1-self.Np/self.N, *self.w_clip)
        # inlier_prob = 1.0 - np.divide(1, cdfs+gs*cs)
        # P = P * inlier_prob

        if self.fexist:
            cdfs = cdfs * Pa
            gs = gs / self.gf*self.M
            cs = cs + self.cf*self.w
            P.multiply_(self.Pf) #* self.M

        cdfs.masked_fill_(cdfs == 0, self.eps)
        Cv = 1.0/(cdfs + gs * cs)
        return P, Cv

    def del_cache_attributes(self, attributes=None):
        if attributes is None:
            attributes = ['Pf', 'P1', 'Pt1', 'cdff', 'PX', 'MY', 'Av', 'AvUCv', 
                          #Xr, Yr,
                          'XF', 'YF', 'VAv', 'Fv', 'F', 'MPG' ]                
        for attr in attributes:
            if hasattr(self, attr):
                delattr(self, attr)

        self.clean_cache()

    def detach_to_cpu(self, attributes=None, to_numpy=True):
        if attributes is None:
            attributes = ['R', 'B', 't', 's',  
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