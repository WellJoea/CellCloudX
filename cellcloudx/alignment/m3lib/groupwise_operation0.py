import numpy as np
import torch as th
from tqdm import tqdm
import itertools


from .operation_expectation import (rigid_outlier, rigid_overlap, kodist2, thdist2, update_P_xp, 
    update_P_ko, expectation_ko, expectation_xp, update_pi, update_pi2,
    expectation_ko1, expectation_xp1, expectation_ko2, expectation_xp2,
    expectation_ko_df, expectation_xp_df,
    expectation_xp_reg, expectation_ko_reg, expectation_xp_reg1, expectation_ko_reg1,
     expectation_xp_reg2, expectation_ko_reg2,
    features_pdist2, features_pdist2_df, sigma_square, default_transparas,
    expectation_imbids_otm, expectation_imbids_otm_log, 
    expectation_exbids_otm, 
    init_tmat)

from .manifold_regularizers import pwdeformable_regularizer
from ...transform import (
    homotransform_points,
    homotransform_point,
    ccf_deformable_transform_point,
    ccf_hierarchical_transform_point,
    homotransform_mat,
)

class gwcompute_feauture(object):
    def __init__(self, keops_thr=None, xp=th, verbose=0):
        self.keops_thr = keops_thr
        self.verbose = verbose
        self.xp = xp
        self.features_pdist2 = features_pdist2

    def compute_pairs(self, Fa, tau2, tau2_prediv=None, mask=None, fexist=None, device=None, dtype=None):
        L = len(Fa)
        if tau2_prediv is None:
            tau2_prediv = [True] * L
        with tqdm(total=L*L, 
                    desc="feature fusion",
                    colour='#AAAAAA', 
                    disable=(self.verbose==0)) as pbar:
            for i, j in itertools.product(range(L), range(L)):
                pbar.set_postfix(dict(i=int(i), j=int(j)))
                pbar.update()
                if not fexist is None:
                    assert fexist[i] == (Fa[i] is not None), f'Fa[{i}] is None'
                    assert fexist[j] == (Fa[j] is not None), f'Fa[{j}] is None'

                if (Fa[i] is not None) and (Fa[j] is not None):
                    if (mask is not None) and (mask[i,j]==0):
                        # setattr(self, f'f{i}_{j}', None)
                        continue
                    try:
                        use_keops = bool(self.keops_thr[i,j])
                    except:
                        use_keops = False

                    if use_keops:
                        fd = features_pdist2(Fa[j], Fa[i],
                                            tau2[i,j], 
                                            use_keops=use_keops, 
                                            tau2_prediv=tau2_prediv[i],
                                            device=device, dtype=dtype)
                        setattr(self,  f'f{i}_{j}', fd)
                    else:
                        if (hasattr(self, f'f{j}_{i}')  
                            #(getattr(self, f'f{j}_{i}', None) is not None
                            and self.xp.all(tau2[i,j]== tau2[j,i]) 
                            and (tau2_prediv[i]== tau2_prediv[j])):
                            setattr(self, f'f{i}_{j}', getattr(self, f'f{j}_{i}').T)
                        else:
                            fd = features_pdist2(Fa[j], Fa[i], tau2[i,j], 
                                                use_keops=use_keops, 
                                                tau2_prediv=tau2_prediv[i], 
                                                device=device, dtype=dtype)
                            setattr(self,  f'f{i}_{j}', fd)
                else:
                    setattr(self,  f'f{i}_{j}', None)

class gwEM_core():
    def __init__(self):
        super().__init__()
        self.init_tmat = init_tmat
        self.homotransform_points = homotransform_points
        self.ccf_deformable_transform_point = ccf_deformable_transform_point
        self.ccf_hierarchical_transform_point = ccf_hierarchical_transform_point
        self.homotransform_mat = homotransform_mat
        self.homotransform_point = homotransform_point
    
    def init_transformer(self):
        self.tmats = {}
        for iL in range(self.L):
            itrans = self.transformer[iL]
            # itrans = 'E' if iL == self.root else self.transformer[iL]
            if self.roots is not None and iL in self.roots:
                itrans = 'N'
                self.transformer[iL] = itrans
            self.tmats[iL] =  self.init_tmat(itrans, self.D, self.Ns[iL], 
                                             xp =self.xp, device=self.device, dtype=self.floatx)

    def init_regularizer(self):
        self.DR = {}
        DL = sum([i in "DH" for i in self.transformer])
        if DL == 0: return

        try:
            pbar = tqdm(range(self.L), total=self.L, colour='#AAAAAA', desc='gwRegularizer')
            for iL in pbar:
                idr = pwdeformable_regularizer(xp = self.xp, verbose=False, **self.transparas[iL]) # deformable_regularizer v0
                if (self.transformer[iL] in "DH") and (self.omega[iL].sum() > 0):
                    idr.compute(self.Xa[iL], device=self.device, dtype=self.floatx)
                self.DR[iL] = idr
            pbar.close()
        except Exception as error:
            del self.DR
            self.clean_cache()
            raise ValueError(
                'Failed to compute the deformable regularizer matrix. Check the Memory.'
            ) from error

    def edge_scale(self, xid, n_sample):
        scale = self.Ns[xid] / float(n_sample)
        if self.omega_normal:
            scale = scale / self.nu[xid]
        return scale
    @th.no_grad()
    def expectation_streams(self, iL, Xs, TYs, Ys,): # TODO
        yid, xids = iL, self.OM[iL]

        xl, wl =len(xids), self.omega[iL][self.OM[iL]]
        
        # if self.omega_normal:
        #     wl = wl/self.xp.sum(wl)

        Pt1s, P1s, PXs, Nps, ols = [],[],[],[],[]
        iTY, iY, iXs = TYs[yid], Ys[yid], [] # TODO sample for iTY will increase the complexity of the deformable.

        paras = []
        for i, (wlj, xid) in enumerate(zip(wl, xids)):
            iX = Xs[xid]
            xFa, yFa= self.Fa[xid], self.Fa[yid]
            fexist = (self.fexist[yid] == True and self.fexist[xid] == True)
            d2f= getattr(self.fds, f'f{yid}_{xid}', None)

            if fexist:
                assert not d2f is None
            use_samplex = self.use_sample and self.iteration < self.sample_stopiter
    
            if use_samplex:
                xsidx = self.sample_index(iL, xid, self.iteration)
                iX = iX[xsidx]

                if fexist:
                    xFa = [ ifa[xsidx] for ifa in self.Fa[xid]]
                    if bool(self.keops_thr[yid, xid]):
                        d2f = self.fds.features_pdist2(
                                                xFa, self.Fa[yid],
                                                self.tau2[yid, xid], 
                                                use_keops=True, 
                                                tau2_prediv=self.tau2_prediv[yid],
                                                device=self.device, dtype=self.floatx)
                    else:
                        d2f = d2f[:,xsidx]
            #use_sampley: #TODO
            
            D, Ni, Mi = iX.shape[1], iX.shape[0], iTY.shape[0]
            gs = Mi/Ni*self.ws[yid, xid]/(1-self.ws[yid, xid])
            isigma2 = (self.sigma2_expect[yid] if self.sigma2_sync
                       else self.sigma2[yid, xid])
            iparas = [iX, isigma2, gs, d2f, self.tau2[yid, xid], self.tau2_auto[iL], 
                        self.tau2_grow[iL][self.iteration], self.DK, self.feat_normal, xFa, yFa, self.keops_thr[yid, xid] ]

            paras.append(iparas)

        tau2ss =  []
        n_streams = min(len(xids), 8)
        parent_stream = th.cuda.current_stream(device=self.device)
        streams = [th.cuda.Stream(device=self.device) for _ in range(n_streams)]
        for stream in streams:
            stream.wait_stream(parent_stream)
        for i in range(xl):
            stream = streams[i % n_streams]
            with th.cuda.stream(stream):
                iX, isigma2, gs, d2f, itau2, itau2_auto, itau2_alpha, DK, feat_normal,  xFa, yFa, ikeops = paras[i]
                iexpectation = expectation_ko if ikeops else expectation_xp
                iPt1, iP1, iPX, iNp, tau2s = iexpectation(
                        iX, iTY, isigma2, gs, #sigma2_expect[yid]
                        d2f=d2f, tau2=itau2, 
                        tau2_auto=itau2_auto, eps=self.eps, 
                        tau2_alpha=itau2_alpha, DF=DK,
                        feat_normal=feat_normal, 
                        XF=xFa, YF=yFa,
                        device=self.device,
                        xp = self.xp )

                Pt1s.append(iPt1)
                P1s.append(iP1)
                PXs.append(iPX)
                Nps.append(iNp)

                iXs.append(iX)
                tau2ss.append(tau2s)
        # th.cuda.synchronize()
        for stream in streams:
            stream.synchronize()
        
        for i, (wlj, xid) in enumerate(zip(wl, xids)):
            iX = iXs[i]
            iNp = Nps[i]
            D, Ni, Mi = iX.shape[1], iX.shape[0], iTY.shape[0]

            self.tau2[yid, xid] = tau2ss[i]
            w = (1- iNp/Ni).clip(*self.w_clip)
            self.ws[yid, xid] = self.was[yid, xid] *self.ws[yid, xid] + (1-self.was[yid, xid])*w
            isigma2 = (self.sigma2_expect[yid] if self.omega_sync  else self.sigma2[yid, xid])
            olj = wlj * self.edge_scale(xid, Ni) / isigma2
            ols.append(olj)
        return Pt1s, P1s, Nps, PXs, ols, iY, iXs

    @th.no_grad()
    def optimization_streams(self): # TODO
        if self.graph_strategy == 'async':
            return self.optimization_series()

        GWexpectation = self.gwexpectation_lgm_m1 if self.use_wm else self.gwexpectation_lgm_m0
        TY_prev = self.TYs
        self.TYs_tmp = [iTY.clone() for iTY in TY_prev]
        self.Qs.zero_()
        self.Cs.zero_()
        self.Ls.zero_()
        self.edge_valid.zero_()
        self.edge_overlap.zero_()
        n_streams = min(len(self.OL), self.max_stream)
        streams = [th.cuda.Stream(device=self.device) for _ in range(n_streams)]
        parent_stream = th.cuda.current_stream(device=self.device)
        for stream in streams:
            stream.wait_stream(parent_stream)


        for k, iL in enumerate(self.OL):
            stream = streams[k % n_streams]
            with th.cuda.stream(stream):
                if (self.transformer[iL] == 'N') or (len(self.OM[iL]) == 0): continue
                Pt1s, P1s, Nps, PXs, ols, iY, iXs = GWexpectation(
                    iL, self.TYs, self.TYs_tmp, self.Xa,
                )
                if sum(iNp * iol for iNp, iol in zip(Nps, ols)) <= self.eps:
                    # self.Qs[iL] = 0
                    continue
                self.maximization(iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs, transformer=self.transformer[iL])
        for stream in streams:
            stream.synchronize()

        if self.graph_strategy == 'async':
            self.TYs = [ ity for ity in self.TYs_tmp ]
            # self.omega = self.omega_tmp
        if (self.iteration + 1) % self.inneriter == 0:
            self.TYs = [ ity.clone() for ity in self.TYs_tmp ]
            # self.omega = self.omega_tmp.clone()
        self.update_omega()
        qprev = self.Q
        self.Q = self.xp.sum(self.Qs)
        self.diff = self.xp.abs(self.Q - qprev) / self.xp.abs(qprev).clamp_min(1.0)
        self.iteration += 1

    @th.no_grad()
    def gwexpectation_lgm_m0(self, iL, Xs, TYs, Ys,):
        yid, xids = iL, self.OM[iL]
        wl = self.omega[iL][self.OM[iL]]
        
        # if self.omega_normal:
        #     wl = wl/self.xp.sum(wl)

        Pt1s, P1s, PXs, Nps, ols = [],[],[],[],[]
        iTY, iY, iXs = TYs[yid], Ys[yid], [] # TODO sample for iTY will increase the complexity of the deformable.

        for wlj, xid in zip(wl, xids):
            iX = Xs[xid]
            xFa, yFa= self.Fa[xid], self.Fa[yid]
            fexist = (self.fexist[yid] == True and self.fexist[xid] == True)
            d2f= getattr(self.fds, f'f{yid}_{xid}', None)

            if fexist:
                assert not d2f is None

            lexist = (self.aexist[yid] == True and self.aexist[xid] == True)
            use_keops = self.keops_thr[yid, xid] 
            if lexist > 0:
                iexpectation = expectation_ko_reg if use_keops else expectation_xp_reg
                largs = dict(
                    A = self.Aa[xid],
                    B = self.Aa[yid],
                    delta_h = self.lpas['delta_l'][yid],
                    use_gating = self.lpas['use_gating'][yid],
                    use_lowm = self.lpas['lowm_l'][yid],
                )
            else:
                iexpectation = expectation_ko if use_keops else expectation_xp
                largs = dict()

            use_samplex = self.use_sample and self.iteration < self.sample_stopiter
    
            if use_samplex:
                xsidx = self.sample_index(iL, xid, self.iteration)
                iX = iX[xsidx]

                if fexist:
                    xFa = [ ifa[xsidx] for ifa in self.Fa[xid]]
                    if bool(self.keops_thr[yid, xid]):
                        d2f = self.fds.features_pdist2(
                                                xFa, self.Fa[yid],
                                                self.tau2[yid, xid], 
                                                use_keops=True, 
                                                tau2_prediv=self.tau2_prediv[yid],
                                                device=self.device, dtype=self.floatx)
                    else:
                        d2f = d2f[:,xsidx]
                if lexist: 
                    largs['A'] = largs['A'][xsidx]
            #use_sampley: #TODO
            
            D, Ni, Mi = iX.shape[1], iX.shape[0], iTY.shape[0]
            gs = Mi/Ni*self.ws[yid, xid]/(1-self.ws[yid, xid])

            isigma2 = (self.sigma2_expect[yid] if self.sigma2_sync else self.sigma2[yid, xid])
            iPt1, iP1, iPX, iNp, tau2s = iexpectation(
                    iX, iTY, isigma2, gs,
                    d2f=d2f, tau2=self.tau2[yid, xid], 
                    tau2_auto=self.tau2_auto[iL], eps=self.eps, 
                    tau2_alpha=self.tau2_grow[iL][self.iteration],
                    DF=self.DK,
                    feat_normal=self.feat_normal, 
                    XF=xFa, YF=yFa,
                    device=self.device,
                    use_stable=self.use_stable,
                    xp = self.xp,
                    **largs )
        
            self.tau2[yid, xid] = tau2s
            w = (1- iNp/Ni).clip(*self.w_clip)
            self.ws[yid, xid] = self.was[yid, xid] *self.ws[yid, xid] + (1-self.was[yid, xid])*w

            isigma2 = (self.sigma2_expect[yid] if self.omega_sync else self.sigma2[yid, xid])
            # isigma2 = self.sigma2_expect[yid]
            olj = wlj * self.edge_scale(xid, Ni) / isigma2

            Pt1s.append(iPt1)
            P1s.append(iP1)
            PXs.append(iPX)
            Nps.append(iNp)
            ols.append(olj)
            iXs.append(iX)
        return Pt1s, P1s, Nps, PXs, ols, iY, iXs

    @th.no_grad()
    def gwexpectation_lgm_m1(self, iL, Xs, TYs, Ys,):
        yid, xids = iL, self.OM[iL]
        wl = self.omega[iL][self.OM[iL]]
        # if self.omega_normal:
        #     wl = wl/self.xp.sum(wl)

        Pt1s, P1s, PXs, Nps, ols = [],[],[],[],[]
        iTY, iY, iXs = TYs[yid], Ys[yid], [] # TODO sample for iTY will increase the complexity of the deformable.

        for wlj, xid in zip(wl, xids):
            iX = Xs[xid]
            xFa, yFa= self.Fa[xid], self.Fa[yid]
            fexist = (self.fexist[yid] == True and self.fexist[xid] == True)
            d2f= getattr(self.fds, f'f{yid}_{xid}', None)
            pi0, piu, U = self.ws[yid, xid], self.wms[yid][xid], self.U[yid][xid]
            a0, a1, mrf = self.a0[yid, xid], self.a1[yid, xid], None #self.mrf[yid, xid]
            wa, g_dr, wma = self.was[yid, xid], self.g_dr[yid][xid], self.wmas[yid][xid]

            if fexist:
                assert not d2f is None

            lexist = (self.aexist[yid] == True and self.aexist[xid] == True)
            use_keops = self.keops_thr[yid, xid] 
            if lexist > 0:
                iexpectation = expectation_ko_reg1 if use_keops else expectation_xp_reg1
                largs = dict(
                    A = self.Aa[xid],
                    B = self.Aa[yid],
                    delta_h = self.lpas['delta_l'][yid],
                    use_gating = self.lpas['use_gating'][yid],
                    use_lowm = self.lpas['lowm_l'][yid],
                )
            else:
                iexpectation = expectation_ko1 if use_keops else expectation_xp1
                largs = dict()

            use_samplex = self.use_sample and self.iteration < self.sample_stopiter
    
            if use_samplex:
                xsidx = self.sample_index(iL, xid, self.iteration)
                iX = iX[xsidx]

                if fexist:
                    xFa = [ ifa[xsidx] for ifa in self.Fa[xid]]
                    if bool(self.keops_thr[yid, xid]):
                        d2f = self.fds.features_pdist2(
                                                xFa, self.Fa[yid],
                                                self.tau2[yid, xid], 
                                                use_keops=True, 
                                                tau2_prediv=self.tau2_prediv[yid],
                                                device=self.device, dtype=self.floatx)
                    else:
                        d2f = d2f[:,xsidx]
                if lexist: 
                    largs['A'] = largs['A'][xsidx]
            #use_sampley: #TODO
            D, Ni, Mi = iX.shape[1], iX.shape[0], iTY.shape[0]
            Ui  = float(U or Ni)
            piu = ((1.0 - pi0)*Ui/Mi).expand(Mi) if piu is None else piu

            isigma2 = (self.sigma2_expect[yid] if self.sigma2_sync  else self.sigma2[yid, xid])
            iPt1, iP1, iPX, iNp, tau2s = iexpectation( 
                    iX, iTY, isigma2, piu, pi0,
                    d2f=d2f, 
                    tau2=self.tau2[yid, xid], 
                    tau2_auto=self.tau2_auto[yid], 
                    tau2_alpha=self.tau2_grow[yid][self.iteration], 
                    DF= self.DK,
                    feat_normal=self.feat_normal,
                    XF=xFa, YF=yFa, 
                    device=self.device,
                    use_stable=self.use_stable,
                    xp = self.xp,
                    **largs )

            pi0_new, piu_new = update_pi(iP1, iNp, Ni, Mi, pi0, 
                                        alpha=a1, alpha0=a0, 
                                        U = Ui,
                                        mrf_solver=mrf,
                                        w_clip=self.w_clip,
                                        xp=self.xp)

            # self.ws[ iL] = pi0_new
            # self.wms[iL] = piu_new
            pi0_n =  wa *pi0 + (1-wa)*pi0_new
            kappa = Ui * (1-pi0_n)
            piu_n = (g_dr*kappa / piu.sum().clamp_min(self.eps))*piu  + ((1-g_dr)*kappa /piu_new.sum().clamp_min(self.eps)) * piu_new

            self.ws[yid, xid]  = pi0_n
            self.wms[yid][xid] = piu_n
            self.tau2[yid, xid] = tau2s

            isigma2 = (self.sigma2_expect[yid] if self.omega_sync else self.sigma2[yid, xid])
            # isigma2 = self.sigma2_expect[yid]
            olj = wlj * self.edge_scale(xid, Ni) / isigma2
                
            Pt1s.append(iPt1)
            P1s.append(iP1)
            PXs.append(iPX)
            Nps.append(iNp)
            ols.append(olj)
            iXs.append(iX)
        return Pt1s, P1s, Nps, PXs, ols, iY, iXs

    @th.no_grad()
    def optimization_series(self):
        GWexpectation = self.gwexpectation_lgm_m1 if self.use_wm else self.gwexpectation_lgm_m0
        for iL in self.OL:
            if (self.transformer[iL] == 'N') or (len(self.OM[iL]) == 0): continue
            Pt1s, P1s, Nps, PXs, ols, iY, iXs = GWexpectation(iL, self.TYs, self.TYs_tmp, self.Xa, )
            if sum(iNp * iol for iNp, iol in zip(Nps, ols)) <= self.eps:
                self.Qs[iL] = 0
                continue
            # Pt1s, P1s, Nps, PXs, ols, iY, iXs = self.expectation_streams(iL, self.TYs, self.TYs_tmp, self.Xa, )
            self.maximization(iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs, transformer=self.transformer[iL])
        if self.graph_strategy == 'async':
            self.TYs = [ ity for ity in self.TYs_tmp ]
            # self.omega = self.omega_tmp
        if (self.iteration + 1) % self.inneriter == 0:
            self.TYs = [ ity.clone() for ity in self.TYs_tmp ]
            # self.omega = self.omega_tmp.clone()
        # self.update_edge_quality(iL, iXs, Pt1s, P1s, PXs, Nps, iTY)
        self.update_omega()
        qprev = self.Q
        self.Q = self.xp.sum(self.Qs)
        self.diff = self.xp.abs(self.Q - qprev) / self.xp.abs(qprev).clamp_min(1.0)
        self.iteration += 1

    @th.no_grad()
    def maximization(self, *args, transformer='E', **kwargs):
        if transformer == 'E':
            self.gw_rigid_maximization(*args, **kwargs)
        elif transformer in 'SRTILO':
            self.gw_similarity_maximization(*args, **kwargs)
        elif transformer == 'A':
            self.gw_affine_maximization(*args, **kwargs)
        elif transformer == 'D':
            self.gw_deformable_maximization(*args, **kwargs)
        elif transformer == 'H':
            self.gw_hierarchical_maximization(*args, **kwargs)
        else:
            raise ValueError(f"Unknown transformer: {transformer}")

    @th.no_grad()
    def gw_rigid_maximization(self, iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs):
        itranspara = self.transparas[iL]
        p_version = self.p_version
        delta = self.delta[iL]
        if p_version == 'v1':
            Af = self.Af
            tf = self.tf
            zeta = self.zeta[iL]
            # m_l = self.sigma2_expect[iL] * sum( iNp * iol for iNp, iol in zip(Nps, ols) )
            m_l = sum( iNp * iol * self.sigma2[iL, xid] for iNp, iol, xid in zip(Nps, ols, self.OM[iL]) )
            delta = delta * m_l
            zeta = zeta * m_l
        elif p_version == 'v2':
            delta = delta / self.nu[iL]

        # sigma2 = self.sigma2[iL][self.OM[iL]]
        # ols = [ i/j for i, j in zip(ols,sigma2) ]

        PX = sum( iPX * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
        Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        if p_version == 'v2':
            Np += delta * iY.shape[0]
            my1 = delta* iY.sum(0)
            muX = self.xp.divide(self.xp.sum(PX, 0)+ my1, Np)
            muY = self.xp.divide(iY.T @ P1 + my1, Np)

            Y_hat =  iY - muY
            U_hat =  iY - muX
            UhtYh = delta * (U_hat.T @ Y_hat)
            B = PX.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat) + UhtYh
        elif self.p_version == 'v1':
            Np += zeta
            muX = self.xp.divide(self.xp.sum(PX, 0)+ zeta*tf, Np)
            muY = self.xp.divide(iY.T @ P1, Np)
            Y_hat =  iY - muY
            # B = PX.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat) + delta*Af
            B = (PX.T @ Y_hat
                 - self.xp.outer(muX, P1 @ Y_hat)
                 + delta*Af
                 + zeta * self.xp.outer(muX - tf, muY)
                ) # TOCHECK
        else:
            muX = self.xp.divide(self.xp.sum(PX, 0), Np)
            muY = self.xp.divide(iY.T @ P1, Np)
            Y_hat  =  iY - muY
            B = PX.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat)

        ## R
        U, S, V = self.xp.linalg.svd(B, full_matrices=True)
        S = self.xp.diag(S)
        C = self.xp.eye(self.D, dtype=self.floatx, device=self.device)
        C[-1, -1] = self.xp.linalg.det(U @ V)
        R = U @ C @ V
        if self.xp.linalg.det(R) < 0:
            U[:, -1] = -U[:, -1]
            R = U @ C @ V
        
        ## S
        if itranspara['fix_s'] is False:
            trBR = self.xp.trace(B.T @ R)
            H = self.xp.sum(P1 * self.xp.sum(Y_hat * Y_hat, 1))
            if p_version == 'v2':
                H += delta * (Y_hat**2).sum()
            elif p_version == 'v1':
                # H += delta
                H += (zeta * self.xp.sum(muY * muY) +
                      delta * self.D) # TOCHECK
            s = trBR/H 
        else:
            s = self.to_tensor(1.0, dtype=self.floatx, device=self.device)

        if itranspara['s_clip'] is not None:
            s = self.xp.clip(s, *itranspara['s_clip'])
        t = muX - s * (R @ muY)

        iTY = s *  ( iY @ R.T ) + t
        self.tmats[iL]['R'] = R
        self.tmats[iL]['t'] = t
        self.tmats[iL]['s'] = s
        self.TYs_tmp[iL] = s * ( self.Xa[iL] @ R.T ) + t 

        self.update_sigma2(iL, iXs, Pt1s, P1s, PXs, Nps, iTY)

    @th.no_grad()
    def gw_similarity_maximization(self, iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs):
        itranspara = self.transparas[iL]
        p_version = self.p_version
        delta = self.delta[iL]
        if p_version == 'v1':
            Af = self.Af
            tf = self.tf
            zeta = self.zeta[iL]
            # m_l = self.sigma2_expect[iL] * sum(iNp * iol for iNp, iol in zip(Nps, ols))
            m_l = sum( iNp * iol * self.sigma2[iL, xid] for iNp, iol, xid in zip(Nps, ols, self.OM[iL]) )
            delta = delta * m_l
            zeta = zeta * m_l
        elif p_version == 'v2':
            delta = delta / self.nu[iL]

        PX = sum( iPX * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
        Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )
        PX_center = PX

        if p_version == 'v2':
            if not itranspara['fix_t']:
                Np += delta * iY.shape[0]
                my1 = delta* iY.sum(0)
                muX = self.xp.divide(self.xp.sum(PX, 0)+ my1, Np)
                muY = self.xp.divide(iY.T @ P1 + my1, Np)
                Y_hat =  iY - muY
                U_hat =  iY - muX
            else:
                PX_center = PX - P1[:, None] * self.tmats[iL]['t']
                muX = self.xp.zeros(self.D, dtype=self.floatx, device=self.device)
                Y_hat =  iY
                U_hat =  iY - self.tmats[iL]['t']
            UhtYh = delta * (U_hat.T @ Y_hat)
            B = PX_center.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat) + UhtYh
            
        elif self.p_version == 'v1':
            if not itranspara['fix_t']:
                Np += zeta
                muX = self.xp.divide(self.xp.sum(PX, 0)+ zeta*tf, Np)
                muY = self.xp.divide(iY.T @ P1, Np)
                Y_hat =  iY - muY
                B = (PX.T @ Y_hat
                     - self.xp.outer(muX, P1 @ Y_hat)
                     + zeta * self.xp.outer(muX - tf, muY)
                     + delta*Af) # TOCHECK
            else:
                PX_center = PX - P1[:, None] * self.tmats[iL]['t']
                muX = self.xp.zeros(self.D, dtype=self.floatx, device=self.device)
                Y_hat =  iY
                B = (PX_center.T @ Y_hat
                     - self.xp.outer(muX, P1 @ Y_hat)
                     + delta*Af) # TOCHECK
            # B = PX_center.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat) + delta*Af
        else:
            if not itranspara['fix_t']:
                muX = self.xp.divide(self.xp.sum(PX, 0), Np)
                muY = self.xp.divide(iY.T @ P1, Np)
            else:
                PX_center = PX - P1[:, None] * self.tmats[iL]['t']
                muX = self.xp.zeros(self.D, dtype=self.floatx, device=self.device)
                muY = self.xp.zeros(self.D, dtype=self.floatx, device=self.device)
            Y_hat  =  iY - muY
            B = PX_center.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat)

        ## RS
        if itranspara['isoscale']:
            R = self.tmats[iL]['R']
            if not itranspara['fix_R']:
                U, S, V = self.xp.linalg.svd(B, full_matrices=True)
                S = self.xp.diag(S)
                C = self.xp.eye(self.D, dtype=self.floatx, device=self.device)
                C[-1, -1] = self.xp.linalg.det(U @ V)
                R = U @ C @ V

                if self.xp.linalg.det(R) < 0:
                    U[:, -1] = -U[:, -1]
                    R = U @ C @ V
                self.tmats[iL]['R'] = R

            if itranspara['fix_s'] is False:
                trBR = self.xp.trace(B.T @ R)
                H = self.xp.sum(P1 * self.xp.sum(Y_hat * Y_hat, 1))
                if p_version == 'v2':
                    H += delta * (Y_hat**2).sum()
                elif p_version == 'v1':
                    # H += delta
                    H += delta * self.D
                    if not itranspara['fix_t']:
                        H += zeta * self.xp.sum(muY * muY) # TOCHECK
                s = trBR/H
                if itranspara['s_clip']  is not None:
                    s = self.xp.clip(s, *itranspara['s_clip'] )
                self.tmats[iL]['s'] = s.expand(self.D)
        else:
            YPY = self.xp.sum( Y_hat * Y_hat * P1.unsqueeze(1), 0)
            YPY.masked_fill_(YPY == 0, self.eps)
            if p_version == 'v2':
                H = YPY + delta * (Y_hat**2).sum(0)
            elif p_version == 'v1':
                H = YPY + delta
            else:
                H = YPY

            if (not itranspara['fix_s']) and (not itranspara['fix_R']):
                max_iter = 70
                error = True
                iiter = 0
                C = self.xp.eye(self.D, dtype=self.floatx, device=self.device)
    
                s = self.tmats[iL]['s'].clone()
                R = self.tmats[iL]['R'].clone()
                while (error):
                    U, S, V = self.xp.linalg.svd(B * s, full_matrices=True)
                    C[-1, -1] = self.xp.linalg.det(U @ V)
                    R_pre = R.clone()
                    R = U @ C @ V
                    if self.xp.linalg.det(R) < 0:
                        U[:, -1] = -U[:, -1]
                        R = U @ C @ V

                    s = self.xp.diagonal(B.T @ R)/H
                    if itranspara['s_clip']  is not None:
                        s = self.xp.clip(s, *itranspara['s_clip'] )
                    # s = self.xp.diag(s)  
                    iiter += 1
                    error = (self.xp.dist(R, R_pre) > 1e-8) and (iiter < max_iter)

                self.tmats[iL]['R'] = R
                self.tmats[iL]['s'] = s

            elif (not itranspara['fix_R']) and itranspara['fix_s']:
                U, S, V = self.xp.linalg.svd(B * self.tmats[iL]['s'], full_matrices=True)
                C = self.xp.eye(self.D, dtype=self.floatx, device=self.device)
                C[-1, -1] = self.xp.linalg.det(U @ V)
                R = U @ C @ V
                if self.xp.linalg.det(R) < 0:
                    U[:, -1] = -U[:, -1]
                    R = U @ C @ V
                self.tmats[iL]['R'] = R

            elif (not itranspara['fix_s']) and itranspara['fix_R']:
                s = self.xp.diagonal(B.T @ self.tmats[iL]['R'])/H
                if itranspara['s_clip']  is not None:
                    s = self.xp.clip(s, *itranspara['s_clip'] )
                # s = self.xp.diag(s)
                self.tmats[iL]['s'] = s

        if not itranspara['fix_t']:
            t = muX - (self.tmats[iL]['R'] * self.tmats[iL]['s']) @ muY
            self.tmats[iL]['t'] = t

        iTY = iY @ (self.tmats[iL]['R'] * self.tmats[iL]['s']).T + self.tmats[iL]['t']
        self.TYs_tmp[iL] = self.Xa[iL] @ (self.tmats[iL]['R'] * self.tmats[iL]['s']).T + self.tmats[iL]['t']

        self.update_sigma2(iL, iXs, Pt1s, P1s, PXs, Nps, iTY)

    @th.no_grad()
    def gw_affine_maximization(self, iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs):
        itranspara = self.transparas[iL]
        fix_t = itranspara.get('fix_t', False)
        p_version = self.p_version
        delta = self.delta[iL]

        if p_version == 'v1':
            Af = self.Af
            tf = self.tf
            zeta = self.zeta[iL]
            # m_l = self.sigma2_expect[iL] * sum(iNp * iol for iNp, iol in zip(Nps, ols))
            m_l = sum( iNp * iol * self.sigma2[iL, xid] for iNp, iol, xid in zip(Nps, ols, self.OM[iL]))
            delta = delta * m_l
            zeta = zeta * m_l
        elif p_version == 'v2':
            delta = delta / self.nu[iL]

        PX = sum(iPX * iol for iPX, iol in zip(PXs, ols))
        P1 = sum(iP1 * iol for iP1, iol in zip(P1s, ols))
        Np = sum(iNp * iol for iNp, iol in zip(Nps, ols))
        PX_center = PX

        if p_version == 'v2':
            if not fix_t:
                Np += delta * iY.shape[0]
                my1 = delta * iY.sum(0)
                muX = self.xp.divide( self.xp.sum(PX, 0) + my1, Np)
                muY = self.xp.divide(iY.T @ P1 + my1, Np)

                Y_hat = iY - muY
                U_hat = iY - muX
            else:
                t = self.tmats[iL]['t']
                PX_center = PX - P1[:, None] * t

                muX = self.xp.zeros(self.D, dtype=self.floatx, device=self.device)
                muY = self.xp.zeros(self.D, dtype=self.floatx, device=self.device)

                Y_hat = iY
                U_hat = iY - t
            UhtYh = delta * (U_hat.T @ Y_hat)
            YhtYh = delta * (Y_hat.T @ Y_hat)

            B = PX_center.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat) + UhtYh
            H = (Y_hat.T * P1) @ Y_hat + YhtYh

        elif self.p_version == 'v1':
            if not fix_t:
                Np += zeta
                muX = self.xp.divide(self.xp.sum(PX, 0) + zeta * tf, Np)
                muY = self.xp.divide(iY.T @ P1, Np)
                Y_hat = iY - muY
            else:
                PX_center = PX - P1[:, None] * self.tmats[iL]['t']
                muX = self.xp.zeros(self.D, dtype=self.floatx, device=self.device)
                muY = self.xp.zeros(self.D, dtype=self.floatx, device=self.device)
                Y_hat = iY
            B = ( PX_center.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat) + delta * Af)
            H = ( Y_hat.transpose(1, 0) * P1 ) @ Y_hat
            if not fix_t:
                B += zeta * self.xp.outer( muX - tf, muY )
                H += zeta * self.xp.outer( muY, muY )
            H.diagonal().add_(delta)

        else:
            if not fix_t:
                muX = self.xp.divide(self.xp.sum(PX, 0), Np)
                muY = self.xp.divide(iY.T @ P1, Np)
                Y_hat = iY - muY
            else:
                t = self.tmats[iL]['t']
                PX_center = PX - P1[:, None] * t

                muX = self.xp.zeros(self.D, dtype=self.floatx, device=self.device)
                muY = self.xp.zeros(self.D, dtype=self.floatx, device=self.device)
                Y_hat = iY

            B = ( PX_center.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat))
            H = (Y_hat.T * P1) @ Y_hat

        try:
            A = self.xp.linalg.solve(H, B.T).T
        except RuntimeError:
            A_pre = self.tmats[iL]['A']
            A_reg = (itranspara['delta'] * self.sigma2_expect[iL])

            H.diagonal().add_(A_reg)
            B = B + A_reg * A_pre
            A = self.xp.linalg.solve(H, B.T).T

        if not fix_t:
            self.tmats[iL]['t'] = muX - A @ muY

        t = self.tmats[iL]['t']
        iTY = iY @ A.T + t

        self.tmats[iL]['A'] = A
        self.TYs_tmp[iL] = self.Xa[iL] @ A.T + t

        self.update_sigma2(iL, iXs, Pt1s, P1s, PXs, Nps, iTY)
    
    @th.no_grad()
    def gw_deformable_maximization(self, iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs):
        itranspara = self.transparas[iL]
        delta, omega, om = self.delta[iL] / self.nu[iL], self.omega[iL], self.OM[iL]

        mu = itranspara['alpha_mu']**(self.iteration)
        alpha = itranspara['alpha'] 
        nu1 = nu2 = itranspara['gamma_nu']**(self.iteration)
        gamma1 = itranspara['gamma1']
        gamma2 = itranspara['gamma2']
        p1_thred = itranspara['p1_thred']
        use_p1 = itranspara['use_p1']

        wsa = sum( omega[i] * float(alpha[i]) * mu for i in om)
        PX = sum( iPX * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )

        B = PX - P1[:,None] * iY
        if delta>0:
            P1_hat = P1 + delta
        else:
            P1_hat = P1

        if itranspara['fast_rank']:
            U = self.DR[iL].U
            S = self.DR[iL].S
            I = self.DR[iL].I
            V = (U @ S)
            Pg = P1_hat[:,None]* V

            if (use_p1):
                P1s_thred = [ self.xp.where(iP1 < p1_thred[i], 0.0, iP1) for iP1, i in zip(P1s, om) ]
                # P1s_scaled = [ iP1 * self.edge_scale(i, iX.shape[0]) for iP1, iX, i in zip(P1s, iXs, om)]
                # P1s_thred = [ self.xp.where(iP1 < p1_thred[i], 0.0, iP1) for iP1, i in zip(P1s_scaled, om)]
                if any(gamma1[om] > 0):
                    P2 = sum( ( omega[i] * float(gamma1[i]) * nu1) * iP1 for iP1, i in zip(P1s_thred, om) )
                    P2 = self.DR[iL].E.T * P2[None,:]
                    B  -= P2 @ self.DR[iL].J3
                    Pg += P2 @ self.DR[iL].J1
                if any(gamma2[om] > 0):
                    P3 = sum( (omega[i]* float(gamma2[i]) * nu2) * iP1 for iP1, i in zip( P1s_thred,  om) )
                    Pg += (self.DR[iL].F.T * P3[None,:]) @ self.DR[iL].J2
            else:
                if any(gamma1[om] > 0):
                    P2 = sum( (omega[i] * float(gamma1[i]) * nu1)  for i in om)
                    B  -= P2 * self.DR[iL].J3
                    Pg += P2 * self.DR[iL].J1
                if any(gamma2[om] > 0):
                    P3 = sum( (omega[i] * float(gamma2[i]) * nu2)  for i in om)
                    Pg += P3 * self.DR[iL].J2

            W  = (B - Pg @ self.xp.linalg.solve(I*wsa + U.T @ Pg, U.T @ B)) / wsa
            H  = V @ (U.T @ W)

        else:
            G = self.DR[iL].G
            Pg = P1_hat[:,None]* G
            Pg.diagonal().add_(wsa)
    
            if (use_p1): # same as the fast_rank
                P1s_thred = [ self.xp.where(iP1 < p1_thred[i], 0.0, iP1) for iP1, i in zip(P1s, om) ]
                # P1s_scaled = [ iP1 * self.edge_scale(i, iX.shape[0]) for iP1, iX, i in zip(P1s, iXs, om)]
                # P1s_thred = [ self.xp.where(iP1 < p1_thred[i], 0.0, iP1) for iP1, i in zip(P1s_scaled, om)]
                if any(gamma1[om] > 0):
                    P2 = sum( ( omega[i] * float(gamma1[i]) * nu1) * iP1 for iP1, i in zip(P1s_thred, om) )
                    P2 = self.DR[iL].E.T * P2[None,:]
                    B  -= P2 @ self.DR[iL].J3
                    Pg += P2 @ self.DR[iL].J1
                if any(gamma2[om] > 0):
                    P3 = sum( (omega[i]* float(gamma2[i]) * nu2) * iP1 for iP1, i in zip( P1s_thred,  om) )
                    Pg += (self.DR[iL].F.T * P3[None,:]) @ self.DR[iL].J2
            else:  # same as the fast_rank
                if any(gamma1[om] > 0):
                    P2 = sum( (omega[i] * float(gamma1[i]) * nu1)  for i in om)
                    B  -= P2 * self.DR[iL].J3
                    Pg += P2 * self.DR[iL].J1
                if any(gamma2[om] > 0):
                    P3 = sum( (omega[i] * float(gamma2[i]) * nu2)  for i in om)
                    Pg += P3 * self.DR[iL].J2

            W = self.xp.linalg.solve(Pg, B)
            H = G @ W

        iTY = iY + H
        self.TYs_tmp[iL] = iTY #self.Xa[iL] +  V @ (U.T @ W)
        self.tmats[iL]['W'] = W

        Lp = self.xp.trace(W.T @ H) # TODO + LLE + LP
        Qp = [ float(alpha[i]) * mu * Lp/2.0 for i in self.OM[iL] ]
        self.update_sigma2(iL, iXs, Pt1s, P1s, PXs, Nps, iTY, Qp = Qp)

    @th.no_grad()
    def gw_deformable_maximization1(self, iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs):
        itranspara = self.transparas[iL]
        delta = self.delta[iL]

        mu = itranspara['alpha_mu']**(self.iteration)
        alpha = itranspara['alpha'] 
        nu1 = nu2 = itranspara['gamma_nu']**(self.iteration)
        gamma1 = itranspara['gamma1']
        gamma2 = itranspara['gamma2']
        p1_thred = itranspara['p1_thred']
        use_p1 = itranspara['use_p1']

        wsa = sum( w * self.sigma2_expect[iL] * float(alpha[i]) * mu for w, i in zip(ols, self.OM[iL]) )
        PX = sum( iPX * iol/wsa for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol/wsa for iP1, iol in zip(P1s, ols) )

        B = PX - P1[:,None] * iY
        if delta>0:
            P1_hat = P1 + delta/wsa
        else:
            P1_hat = P1

        if itranspara['fast_rank']:
            U = self.DR[iL].U
            S = self.DR[iL].S
            I = self.DR[iL].I
            V = (U @ S)
            Pg = P1_hat[:,None]* V

            if (use_p1):
                P1s_thred = [ self.xp.where(iP1 < p1_thred[i], 0.0, iP1) for iP1, i in zip(P1s, self.OM[iL]) ]
                if any(gamma1[self.OM[iL]] > 0):
                    P2 = sum( (w * self.sigma2_expect[iL] * float(gamma1[i]) * nu1/ wsa) * iP1 for w, iP1, i in zip(ols,  P1s_thred, self.OM[iL]) )
                    P2 = self.DR[iL].E.T * P2[None,:]
                    B  -= P2 @ self.DR[iL].J3
                    Pg += P2 @ self.DR[iL].J1
                    # LPLY = P2 @ self.DR[iL].J3
                    # LPLV = P2 @ self.DR[iL].J1
                    # LPLY = (self.DR[iL].E.T @ (P2[:,None]*self.DR[iL].J3))
                    # LPLV = (self.DR[iL].E.T @ (P2[:,None]*self.DR[iL].J1))
                    # B  -= LPLY
                    # Pg += LPLV
                if any(gamma2[self.OM[iL]] > 0):
                    P3 = sum( (w * self.sigma2_expect[iL] * float(gamma2[i]) * nu2/ wsa) * iP1 for w, iP1, i in zip(ols, P1s_thred, self.OM[iL]) )
                    Pg += (self.DR[iL].F.T * P3[None,:]) @ self.DR[iL].J2
            else:
                if any(gamma1[self.OM[iL]] > 0):
                    P2 = sum( (w * self.sigma2_expect[iL] * float(gamma1[i]) * nu1/ wsa)  for w, i in zip(ols, self.OM[iL]) )
                    B  -= P2 * self.DR[iL].J3
                    Pg += P2 * self.DR[iL].J1
                if any(gamma2[self.OM[iL]] > 0):
                    P3 = sum( (w * self.sigma2_expect[iL] * float(gamma2[i]) * nu2/ wsa)  for w, i in zip(ols, self.OM[iL]) )
                    Pg += P3 * self.DR[iL].J2

            W  = B - Pg @ self.xp.linalg.solve(I + U.T @ Pg, U.T @ B)
            H  = V @ (U.T @ W)

        else:
            G = self.DR[iL].G
            Pg = P1_hat[:,None]* G
            Pg.diagonal().add_(1.0)
    
            if (use_p1): # same as the fast_rank
                P1s_thred = [ self.xp.where(iP1 < p1_thred[i], 0.0, iP1) for iP1, i in zip(P1s, self.OM[iL]) ]
                if any(gamma1[self.OM[iL]] > 0):
                    P2 = sum( (w * self.sigma2_expect[iL] * float(gamma1[i]) * nu1/ wsa) * iP1 for w, iP1, i in zip(ols,  P1s_thred, self.OM[iL]) )
                    P2 = self.DR[iL].E.T * P2[None,:]
                    B  -= P2 @ self.DR[iL].J3
                    Pg += P2 @ self.DR[iL].J1
                if any(gamma2[self.OM[iL]] > 0):
                    P3 = sum( (w * self.sigma2_expect[iL] * float(gamma2[i]) * nu2/ wsa) * iP1 for w, iP1, i in zip(ols, P1s_thred, self.OM[iL]) )
                    Pg += (self.DR[iL].F.T * P3[None,:]) @ self.DR[iL].J2
            else:  # same as the fast_rank
                if any(gamma1[self.OM[iL]] > 0):
                    P2 = sum( (w * self.sigma2_expect[iL] * float(gamma1[i]) * nu1/ wsa)  for w, i in zip(ols, self.OM[iL]) )
                    B  -= P2 * self.DR[iL].J3
                    Pg += P2 * self.DR[iL].J1
                if any(gamma2[self.OM[iL]] > 0):
                    P3 = sum( (w * self.sigma2_expect[iL] * float(gamma2[i]) * nu2/ wsa)  for w, i in zip(ols, self.OM[iL]) )
                    Pg += P3 * self.DR[iL].J2

            W = self.xp.linalg.solve(Pg, B)
            H = G @ W

        iTY = iY + H
        self.TYs_tmp[iL] = iTY #self.Xa[iL] +  V @ (U.T @ W)
        self.tmats[iL]['W'] = W

        Lp = self.xp.trace(W.T @ H) # TODO + LLE + LP
        Qp = [ float(alpha[i]) * mu * Lp/2.0 for i in self.OM[iL] ]
        self.update_sigma2(iL, iXs, Pt1s, P1s, PXs, Nps, iTY, Qp = Qp)

    @th.no_grad()
    def gw_deformable_maximization0(self, iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs):
        itranspara = self.transparas[iL]
        delta = self.delta[iL] / self.nu[iL]

        mu = itranspara['alpha_mu']**(self.iteration)
        alpha = itranspara['alpha'] 

        wsa = sum( w * self.sigma2_expect[iL] * float(alpha[i]) * mu for w, i in zip(ols, self.OM[iL]) )
        PX = sum( iPX * iol/wsa for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol/wsa for iP1, iol in zip(P1s, ols) )
        B = PX - P1[:,None] * iY
        P1_h = P1 + delta/wsa

        if itranspara['fast_rank']:
            U = self.DR[iL].U
            S = self.DR[iL].S
            I = self.DR[iL].I
            V = (U @ S)

            Pg = P1_h[:,None]* V
            W  = B - Pg @ self.xp.linalg.solve(I + U.T @ Pg, U.T @ B)
            M  = V @ (U.T @ W)
        else:
            G = self.DR[iL].G
            A = P1_h[:,None]* G
            A.diagonal().add_(1.0)
            W = self.xp.linalg.solve(A, B)
            M = G @ W

        iTY = iY + M
        self.TYs_tmp[iL] = iTY #self.Xa[iL] +  V @ (U.T @ W)
        self.tmats[iL]['W'] = W

        Lp = self.xp.trace(W.T @ M)
        Qp = [ float(alpha[i]) * mu * Lp/2.0 for i in self.OM[iL] ]
        self.update_sigma2(iL, iXs, Pt1s, P1s, PXs, Nps, iTY, Qp = Qp)

    @th.no_grad()
    def gw_hierarchical_maximization(
        self, iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs
    ):
        """Update ``s R [Y + v(Y)] + t`` from all neighbours of point set iL."""
        itranspara = self.transparas[iL]
        itmat = self.tmats[iL]
        om = self.OM[iL]
        omega = self.omega[iL]
        Yh = self.Yhs[iL]

        mu = itranspara["alpha_mu"] ** self.iteration
        alpha = itranspara["alpha"]
        nu = itranspara["gamma_nu"] ** self.iteration
        gamma1 = itranspara["gamma1"]
        gamma2 = itranspara["gamma2"]
        p1_thred = itranspara["p1_thred"]
        use_p1 = itranspara["use_p1"]
        delta = self.delta[iL] / self.nu[iL]

        PX = sum(iPX * iol for iPX, iol in zip(PXs, ols))
        P1 = sum(iP1 * iol for iP1, iol in zip(P1s, ols))
        Np = sum(iNp * iol for iNp, iol in zip(Nps, ols))

        muX = self.xp.divide(self.xp.sum(PX, 0), Np)
        muY = self.xp.divide(Yh.T @ P1, Np)
        Y_hat = Yh - muY
        B = PX.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat)

        U, _, V = self.xp.linalg.svd(B, full_matrices=True)
        C = self.xp.eye(self.D, dtype=B.dtype, device=B.device)
        C[-1, -1] = self.xp.linalg.det(U @ V)
        R = U @ C @ V
        if self.xp.linalg.det(R) < 0:
            U[:, -1] = -U[:, -1]
            R = U @ C @ V

        if itranspara["fix_s"] is False:
            trBR = self.xp.trace(B.T @ R)
            Hs = self.xp.sum(P1 * self.xp.sum(Y_hat * Y_hat, 1))
            s = trBR / Hs
        else:
            s = itmat["s"]
        if itranspara["s_clip"] is not None:
            s = self.xp.clip(s, *itranspara["s_clip"])

        t = muX - s * (R @ muY)

        wsa = sum(omega[j] * float(alpha[j]) * mu for j in om)
        B = PX @ (s * R) - (s**2) * P1[:, None] * iY
        B -= P1[:, None] * (s * t @ R)
        P1_hat = P1 * (s**2) + delta if delta > 0 else P1 * (s**2)

        if use_p1:
            # P1s_scaled = [ iP1 * self.edge_scale(i, iX.shape[0]) for iP1, iX, i in zip(P1s, iXs, om)]
            # P1s_thred = [ self.xp.where(iP1 < p1_thred[i], 0.0, iP1) for iP1, i in zip(P1s_scaled, om)]
            P1s_thred = [ self.xp.where(iP1 < p1_thred[j], 0.0, iP1) for iP1, j in zip(P1s, om) ]

        if itranspara["fast_rank"]:
            U = self.DR[iL].U
            S = self.DR[iL].S
            V = U @ S
            Pg = P1_hat[:, None] * V

            if use_p1:
                if any(gamma1[om] > 0):
                    P2 = sum(
                        (omega[j] * float(gamma1[j]) * nu * (s**2)) * iP1
                        for iP1, j in zip(P1s_thred, om)
                    )
                    P2 = self.DR[iL].E.T * P2[None, :]
                    B -= P2 @ self.DR[iL].J3
                    Pg += P2 @ self.DR[iL].J1
                if any(gamma2[om] > 0):
                    P3 = sum(
                        (omega[j] * float(gamma2[j]) * nu) * iP1
                        for iP1, j in zip(P1s_thred, om)
                    )
                    Pg += (self.DR[iL].F.T * P3[None, :]) @ self.DR[iL].J2
            else:
                if any(gamma1[om] > 0):
                    P2 = sum(
                        omega[j] * float(gamma1[j]) * nu * (s**2) for j in om
                    )
                    B -= P2 * self.DR[iL].J3
                    Pg += P2 * self.DR[iL].J1
                if any(gamma2[om] > 0):
                    P3 = sum(omega[j] * float(gamma2[j]) * nu for j in om)
                    Pg += P3 * self.DR[iL].J2

            I = self.DR[iL].I * wsa
            W = (B - Pg @ self.xp.linalg.solve(I + U.T @ Pg, U.T @ B)) / wsa
            H = V @ (U.T @ W)
        else:
            G = self.DR[iL].G
            Pg = P1_hat[:, None] * G
            Pg.diagonal().add_(wsa)

            if use_p1:
                if any(gamma1[om] > 0):
                    P2 = sum(
                        (omega[j] * float(gamma1[j]) * nu * (s**2)) * iP1
                        for iP1, j in zip(P1s_thred, om)
                    )
                    P2 = self.DR[iL].E.T * P2[None, :]
                    B -= P2 @ self.DR[iL].J3
                    Pg += P2 @ self.DR[iL].J1
                if any(gamma2[om] > 0):
                    P3 = sum(
                        (omega[j] * float(gamma2[j]) * nu) * iP1
                        for iP1, j in zip(P1s_thred, om)
                    )
                    Pg += (self.DR[iL].F.T * P3[None, :]) @ self.DR[iL].J2
            else:
                if any(gamma1[om] > 0):
                    P2 = sum(
                        omega[j] * float(gamma1[j]) * nu * (s**2) for j in om
                    )
                    B -= P2 * self.DR[iL].J3
                    Pg += P2 * self.DR[iL].J1
                if any(gamma2[om] > 0):
                    P3 = sum(omega[j] * float(gamma2[j]) * nu for j in om)
                    Pg += P3 * self.DR[iL].J2

            W = self.xp.linalg.solve(Pg, B)
            H = G @ W

        Yh = iY + H
        iTY = Yh @ (R * s).T + t
        self.Yhs[iL] = Yh
        self.TYs_tmp[iL] = iTY
        itmat["R"] = R
        itmat["s"] = s
        itmat["t"] = t
        itmat["W"] = W

        # Lp = self.xp.trace(W.T @ H)
        Lp = self.xp.trace(W.T @ H) # TODO + LLE + LP
        Qp = [float(alpha[j]) * mu * Lp / 2.0 for j in om]
        self.update_sigma2(iL, iXs, Pt1s, P1s, PXs, Nps, iTY, Qp=Qp)

    @th.no_grad()
    def update_sigma2(self, iL, iXs, Pt1s, P1s, PXs, Nps, iTY, Qp = None):
        ixids = self.OM[iL]
        wls = self.omega[iL, ixids]

        Flj = []
        for i, iX in enumerate(iXs):
            trxPx = self.xp.sum(Pt1s[i] * self.xp.sum(iX * iX, dim=1))
            tryPy = self.xp.sum(P1s[i] * self.xp.sum(iTY * iTY, dim=1))
            trPXY = self.xp.sum(iTY * PXs[i])
            Flj.append(trxPx - 2.0 * trPXY + tryPy)

        J = self.xp.stack(Nps)
        F = self.xp.stack(Flj).clamp_min(0)
        n_sample = self.xp.as_tensor( [iX.shape[0] for iX in iXs], dtype=self.floatx, device=self.device,)
        edge_scale = self.Ns[ixids] / n_sample
        if self.omega_normal:
            edge_scale = edge_scale / self.nu[ixids]

        valid = ( (J > self.eps) & self.xp.isfinite(J) & self.xp.isfinite(F))
        self.Ls[iL].zero_()
        self.edge_valid[iL].zero_()
        self.edge_overlap[iL].zero_()
        if not bool(valid.any()):
            self.Qs[iL] = 0
            return

        mass_weight = wls * edge_scale
        sigma2_expect = ( (mass_weight[valid] * F[valid]).sum() / 
                          (mass_weight[valid] * J[valid] * self.D).sum().clamp_min(self.eps)
        ).clamp_min(self.sigma2_min)
        self.sigma2_expect[iL] = sigma2_expect

        if self.sigma2_sync:
            self.sigma2[iL, ixids] = sigma2_expect
        else:
            sigma2_edge = (F / (J * self.D).clamp_min(self.eps)).clamp_min(self.sigma2_min)
            self.sigma2[iL, ixids] = self.xp.where( valid, sigma2_edge, self.sigma2[iL, ixids])

        sigma2 = self.sigma2[iL, ixids]
        qgeom = ( F / (2.0 * sigma2) + J * self.D / 2.0 * (self.xp.log(sigma2) + np.log(2.0 * np.pi)) )
        q = edge_scale * qgeom
        if Qp is not None:
            q = q + self.xp.stack(Qp)
        q = q / self.nu_scale
        valid = valid & self.xp.isfinite(q)

        overlap = (J / n_sample).clamp(0.0, 1.0)
        self.edge_valid[iL, ixids] = valid
        self.edge_overlap[iL, ixids] = self.xp.where(valid, overlap, 0.0)
        self.Ls[iL, ixids] = self.xp.where(valid, q, 0.0)


    @th.no_grad()
    def update_omega(self):
        support = self.omega0 > 0
        valid = support & self.edge_valid
        qraw = self.Ls
        overlap = self.edge_overlap.clamp_min(self.eps)

        beta = self.omega_backbone
        q = qraw.clone()
        if beta > 0:
            mutual = valid & valid.T
            q = self.xp.where(mutual, 0.5 * (q + q.T), q)
            overlap = self.xp.where( mutual, self.xp.sqrt(overlap * overlap.T), overlap )

        warmup = self.omega_warmup
        if 0 < warmup < 1:
            warmup = int(self.maxiter * warmup)
        else:
            warmup = int(warmup)

        eta = self.omega_overlap
        rate = self.omega_rate
        omega_new = self.xp.zeros_like(self.omega)
        self.Qs.zero_()

        for iL in range(self.L):
            if self.transformer[iL] == 'N':
                continue

            ids = self.xp.where(support[iL])[0]
            if ids.numel() == 0:
                continue

            w0 = self.omega0[iL, ids]
            w0 = w0 / w0.sum().clamp_min(self.eps)
            qrow = q[iL, ids]
            rrow = overlap[iL, ids]
            valid_row = valid[iL, ids]
            kappa = float(self.kappa[iL])
            adaptive = (self.iteration >= warmup) and (kappa > 0)

            if not adaptive:
                wl = w0.clone()
                qeff = qrow
            else:
                if bool(valid_row.any()):
                    keep = self.xp.where(valid_row)[0]
                    qvalid = qrow[keep]
                    qrow = (qrow - qvalid.mean()) / qvalid.std(unbiased=False).clamp_min(self.eps)
                    qeff = qrow - kappa * eta * self.xp.log(rrow)
                    score = qeff[keep] - qeff[keep].min()
                    logits = self.xp.log(w0[keep].clamp_min(self.eps)) - score / kappa
                    wl = self.xp.zeros_like(w0)
                    wl[keep] = self.xp.nn.functional.softmax(logits, dim=0)

                    if beta > 0:
                        backbone = self.xp.zeros_like(w0)
                        near = keep[self.xp.argmin(self.xp.abs(ids[keep] - iL))]
                        backbone[near] = 1.0
                        wl = beta * backbone + (1.0 - beta) * wl
                else:
                    wl = w0.clone()

                old = self.omega[iL, ids]
                if bool(old.sum() > self.eps):
                    old = old / old.sum()
                    wl = (1.0 - rate) * old + rate * wl
                wl = wl / wl.sum().clamp_min(self.eps)

            omega_new[iL, ids] = wl
            self.Qs[iL] = ( wl * qraw[iL, ids]).sum()
            # self.Qs[iL] = (wl * qeff).sum()
            # if adaptive:
            #     self.Qs[iL] += kappa * ( wl * ( self.xp.log(wl.clamp_min(self.eps)) - self.xp.log(w0.clamp_min(self.eps))) ).sum()
        self.omega = omega_new

    @th.no_grad()
    def update_normalize(self):
        device, dtype = 'cpu', self.floatxx
        if self.root is not None:
            Xm_out = self.Xm[self.root].to(device, dtype=dtype)
        elif self.normal in ['global', True]:
            Xm_out = self.Xm[0].to(device, dtype=dtype)
        else:
            Xm_out = 0
        for iL in range(self.L):
            itf = self.transformer[iL]
            iTM = self.tmats[iL]
            iTM = { k: v.to(device, dtype=dtype) for k,v in iTM.items() }

            if itf in 'ESARTILON':
                iTM['tmat'] = self.homotransform_mat(itf, self.D, xp=self.xp, 
                                device=device, dtype=dtype, **iTM )
                if itf in ['E']:
                    A = iTM['R'] * iTM['s']
                elif itf in 'SRTILON':
                    A = iTM['R'] * iTM['s']
                elif itf in ['A']:
                    A = iTM['A']
                else:
                    raise ValueError(f"Unknown transformer: {itf}")

                iTM['t'] = (
                    iTM['t'] * self.Xs[iL].to(device, dtype=dtype)
                    - self.Xm[iL].to(device, dtype=dtype) @ A.T
                    + Xm_out
                )

                iTM['tform'] = self.homotransform_mat(itf, self.D, xp=self.xp, device=device, dtype=dtype, **iTM )
                self.TYs[iL] = self.Xr[iL].to(device, dtype=dtype) @ A.T + iTM['t']

            elif itf in ['D', 'H']:
                for ia in ['G', 'U', 'S']:
                    iv =  getattr(self.DR[iL], ia, None)
                    if iv is not None:
                        iTM[ia] = iv.to(device, dtype=dtype)
                    else:
                        iTM[ia] = iv

                iTM['Y'] = self.Xa[iL].to(device, dtype=dtype)
                iTM['Ym'] = self.Xm[iL].to(device, dtype=dtype)
                iTM['Ys'] = self.Xs[iL].to(device, dtype=dtype)
                iTM['Xs'] = self.Xs[iL].to(device, dtype=dtype)
                iTM['beta'] = self.transparas[iL]['beta']
                
                iTM['Xm'] = Xm_out

                if itf == 'H':
                    iTM['A'] = iTM['R'] * iTM['s']
                    iTM['recenter'] = False

                self.TYs[iL] = self.TYs[iL].to(device, dtype=dtype) * iTM['Xs'] + iTM['Xm']

            iTM['transformer'] = itf
            self.tmats[iL] = iTM
        # TYs = self.transform_points(self.Xr)

    # def update_records(self, ):
    #     for ird in self.record:
    #         if ird == 'TYs':
    #             Sx, Sy  = self.Sx, self.Sy
    #             Tx, Ty  = self.Mx, self.My
    #             # self.records[ird] = [Tx + Sx/Sy*(ity -Ty) for ity in self.records[ird]]
    #             self.records[ird] = [self.Y, *self.records[ird]]
    #             self.records[ird] = [Tx + Sx*(ity) for ity in self.records[ird]]

    def transform_points(self, Ys=None, use_keops=True, **kargs):
        Ys = self.Xr if Ys is None else Ys
        TYs = []
        for iL in range(self.L):
            if self.tmats[iL]['transformer'] in 'ESARTILOPN':
                iTY = self.homotransform_point(Ys[iL], self.tmats[iL]['tform'], inverse=False, xp=self.xp)
            elif self.tmats[iL]['transformer']  == 'D':
                iTY =  self.ccf_deformable_transform_point( Ys[iL], **self.tmats[iL], 
                                                           use_keops=use_keops, **kargs)
            elif self.tmats[iL]['transformer'] == 'H':
                iTY = self.ccf_hierarchical_transform_point(
                    Ys[iL], **self.tmats[iL], use_keops=use_keops, **kargs
                )
            else:
                raise ValueError(f"Unknown transformer: {self.tmats[iL]['transformer']}")
            TYs.append(iTY)
        return TYs
