import torch as th
from .neighbors_ensemble import Neighbors
from .operation_expectation import (rigid_outlier, kodist2, thdist2, update_P_xp, 
    update_P_ko, expectation_ko, expectation_xp,  expectation_ko_df, expectation_xp_df,
    features_pdist2, features_pdist2_df, sigma_square, default_transparas,
    init_tmat)

from .manifold_regularizers import pwdeformable_regularizer
from ...transform import homotransform_points, homotransform_point, ccf_deformable_transform_point, homotransform_mat

class pwEM_core():
    def __init__(self):
        super().__init__()
        self.rigid_outlier = rigid_outlier
        self.kodist2 = kodist2
        self.thdist2 = thdist2
        self.update_P_xp = update_P_xp
        self.update_P_ko = update_P_ko
        self.expectation_ko = expectation_ko
        self.expectation_xp = expectation_xp
        self.expectation_ko_df = expectation_ko_df
        self.expectation_xp_df = expectation_xp_df
        self.init_tmat = init_tmat
        self.default_transparas= default_transparas
        self.pwdeformable_regularizer = pwdeformable_regularizer
        self.homotransform_points = homotransform_points
        self.ccf_deformable_transform_point = ccf_deformable_transform_point
        self.homotransform_mat = homotransform_mat
        self.homotransform_point = homotransform_point

    def pwcompute_feauture(self):
        self.d2f = []
        for iL, fexist in enumerate(self.fexists):
            if fexist:
                d2f = features_pdist2(self.XF[iL], self.YF[iL], tau2s=self.tau2[iL], 
                                        use_keops=self.use_keops[iL], 
                                        tau2_prediv= not self.tau2_auto[iL],
                                        device=self.device, dtype=self.floatx)
                # if not self.pairs is None: #TODO
                #     iXF = self.XF.to(self.device)[self.pairs[0]]
                #     iYF = self.YF.to(self.device)[self.pairs[1]]
                #     self.dpf = (iXF - iYF).pow(2).sum(1)
                #     self.dpf = self.dpf.to(self.device)
                #     if not self.tau2_auto:
                #         self.dpf.div_(-2*self.tau2)
            else:
                d2f = None
            self.d2f.append(d2f)

    def pwcompute_feauture_df(self):
        self.d2f, self.cdff = [], []
        for iL, fexist in enumerate(self.fexists):
            if fexist:
                dd =  features_pdist2_df(self.XF[iL], self.YF[iL], self.tau2[iL], 
                                        use_keops=self.use_keops[iL], 
                                        device=self.device, dtype=self.floatx)
                if self.use_keops[iL]:
                    d2f, cdff = dd
                else:
                    d2f = dd
                    cdff = None
            else:
                d2f = None
                cdff = None
            self.d2f.append(d2f)
            self.cdff.append(cdff)

    def sample_index(self, N, itera, sample):
        gr = self.xp.Generator() #device=self.device
        gr.manual_seed( int(f'{N}{itera}' ))

        sample = min(1.0, sample*(self.sample_grow**itera))
        perm = self.xp.randperm(N, generator=gr)
        n_samples = int(max(N*sample, min(self.sample_min, N)))
        return perm[:n_samples]
    
    @th.no_grad()
    def pwexpectation(self):
        Pt1s, P1s,  PXs, Nps = [], [], [], []
        for iL in range(self.L):
            iX, iTY, fexist = self.Xs[iL], self.TY, self.fexists[iL]
            xFa, yFa, d2f, wi = self.XF[iL], self.YF[iL], self.d2f[iL], self.ws[iL]
            if fexist: 
                assert not d2f is None

            use_samplex = bool(self.sample[iL]) and (self.iteration < self.sample_stopiter)
            if use_samplex:
                xsidx = self.sample_index(iX.shape[0], self.iteration, self.sample[iL])
                iX = iX[xsidx]
        
                if fexist:
                    xFa = [ ifa[xsidx] for ifa in self.XF[iL]]
                    if self.use_keops[iL]:
                        d2f = features_pdist2(xFa, yFa, tau2s=self.tau2[iL], 
                                                use_keops=True, 
                                                tau2_prediv= not self.tau2_auto[iL],
                                                device=self.device, dtype=self.floatx)
                    else:
                        d2f = self.d2f[iL][:,xsidx]
            #use_sampley: #TODO

            D, Ni, Mi = iX.shape[1], iX.shape[0], iTY.shape[0]
            gs = Mi/Ni*wi/(1-wi)

            iexpectation = expectation_ko if self.use_keops[iL] else expectation_xp
            iPt1, iP1, iPX, iNp, tau2s = iexpectation( 
                    iX, iTY, self.sigma2[iL], gs,  #sigma2_exp
                    d2f=d2f, #
                    tau2=self.tau2[iL], 
                    tau2_auto=self.tau2_auto[iL], eps=self.eps, 
                    tau2_alpha=self.tau2_grow[self.iteration], 
                    DF=self.DK[iL],
                    feat_normal=self.feat_normal[iL], 
                    XF=xFa, YF=yFa, #
                    device=self.device,
                    xp = self.xp)
    
            nwi = (1- iNp/Ni).clip(*self.w_clip)
            self.ws[iL] = self.wa *wi + (1-self.wa)*nwi
            self.tau2[iL] = tau2s

            #  self.update_P_ko_pairs() # TODO
            Pt1s.append(iPt1)
            P1s.append(iP1)
            PXs.append(iPX)
            Nps.append(iNp)
        return Pt1s, P1s, PXs, Nps

    @th.no_grad()
    def pwexpectation_df(self):
        Pt1s, P1s, PXs, Nps = [], [], [], []
        for iL in range(self.L):
            iX, iTY, fexist, wi = self.Xs[iL], self.TY, self.fexists[iL], self.ws[iL]
            D, Ni, Mi = iX.shape[1], iX.shape[0], iTY.shape[0]
            gs = Mi/Ni*wi/(1-wi)
        
            iexpectation_df = expectation_ko_df if self.use_keops[iL] else expectation_xp_df
            iPt1, iP1, iPX, iNp = iexpectation_df(
                                        iX, iTY, self.sigma2[iL], gs,
                                        d2f=self.d2f[iL], cdff=self.cdff[iL],
                                        xp=self.xp)

            #  self.update_P_ko_pairs() # TODO
            Pt1s.append(iPt1)
            P1s.append(iP1)
            PXs.append(iPX)
            Nps.append(iNp)
        return Pt1s, P1s, PXs, Nps

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

    def optimization(self):
        qprev = self.Q
        Pt1s, P1s, PXs, Nps = self.pwexpectation()
        self.maximization(Pt1s, P1s, Nps, PXs, self.Y, self.Xs, transformer=self.transformer)
        self.diff = self.xp.abs(self.Q - qprev) #/self.Q
        self.iteration += 1

    def maximization(self, *args, transformer='E', **kwargs):
        if transformer == 'E':
            self.pw_rigid_maximization(*args, **kwargs)
        elif transformer in 'SRTILO':
            self.pw_similarity_maximization(*args, **kwargs)
        elif transformer == 'A':
            self.pw_affine_maximization(*args, **kwargs)
        elif transformer == 'P':
            self.pw_projective_maximization(*args, **kwargs)
        elif transformer == 'D':
            self.pw_deformable_maximization(*args, **kwargs)
        else:
            raise(f"Unknown transformer: {transformer}")

    @th.no_grad()
    def pw_rigid_maximization(self, Pt1s, P1s, Nps, PXs, Y, Xs):
        xp, D, L = self.xp, self.D, self.L
        sigma2_min, sigma2_exp, omega, itranspara, itmat = \
            self.sigma2_min, self.sigma2_exp, self.omega, self.transparas, self.tmats

        # ols = omega/sigma2 # raw
        ols = omega/sigma2_exp # exp 
        # ols = omega # same with not common constraint  exp #TODO

        if L == 1: #check the stability of numerical calculation 
            PX, P1, Np, Pt1, X = PXs[0], P1s[0], Nps[0], Pt1s[0], Xs[0]
        else:
            PX = sum( iPX * iol for iPX, iol in zip(PXs, ols) )
            P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
            Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        muX = xp.divide(xp.sum(PX, 0), Np)
        muY = xp.divide(Y.T @ P1, Np)
        Y_hat = Y - muY
        B = PX.T @ Y_hat - xp.outer(muX, P1 @ Y_hat)

        # R
        U, S, V = xp.linalg.svd(B, full_matrices=True)
        S = xp.diag(S)
        C = xp.eye(D, dtype=B.dtype, device=B.device)
        C[-1, -1] = xp.linalg.det(U @ V)
        R = U @ C @ V
        # R = U @ V

        if xp.linalg.det(R) < 0:
            U[:, -1] = -U[:, -1]
            R = U @ C @ V

        # S
        if itranspara['fix_s'] is False:
            trBR = xp.trace(B.T @ R)
            H = xp.sum(P1 * xp.sum(Y_hat * Y_hat, 1))
            s = trBR/H
        else:
            s = itmat['s']

        if itranspara['s_clip'] is not None:
            s = xp.clip(s, *itranspara['s_clip'])

        # T
        t = muX - s * (R @ muY)

        #TY
        TY = Y @ (R*s).T +  t

        # sigma2, Q
        if L == 1:  #check the stability of numerical calculation 
            X_hat = X - muX
            trAR = xp.trace(B.transpose(1,0) @ R)
            trXPX = xp.sum(Pt1 * xp.sum(X_hat * X_hat, 1))
            trYPY = xp.sum(P1 * xp.sum(Y_hat * Y_hat, 1))
            if itranspara['fix_s'] is False:
                sigma2 = (trXPX  - s * trAR) / (Np * D)
            else:
                sigma2 = (trXPX + s* s* trYPY - 2* s* trAR) / (Np * D)
            if sigma2 < sigma2_min: 
                sigma2 = xp.clip(sigma_square(X, TY), min=sigma2_min)
            Q = (Np * D)/2*(1+xp.log(sigma2))
            sigma2_exp = sigma2
            sigma2 = sigma2.expand(L)
        else:
            sigma2_exp, sigma2, Q = self.update_sigma2(Pt1s, P1s, Nps, PXs, Xs, TY, Qp = None)
        
        self.tmats['R'] = R
        self.tmats['s'] = s
        self.tmats['t'] = t
        self.TY = TY
        self.sigma2 = sigma2
        self.sigma2_exp = sigma2_exp
        self.Q = Q
        # return [[R, s, t,], TY, sigma2_exp, sigma2, Q]

    @th.no_grad()
    def pw_similarity_maximization(self, Pt1s, P1s, Nps, PXs, Y, Xs):
        xp, D, L, eps = self.xp, self.D, self.L, self.eps
        sigma2_min, itranspara, itmat = self.sigma2_min, self.transparas, self.tmats

        # ols = self.omega/self.sigma2 # raw
        ols = self.omega/self.sigma2_exp # exp 
        # ols = self.omega # same with not common constraint  exp #TODO

        if L == 1: #check the stability of numerical calculation 
            PX, P1, Np, Pt1, X = PXs[0], P1s[0], Nps[0], Pt1s[0], Xs[0]
        else:
            PX = sum( iPX * iol for iPX, iol in zip(PXs, ols) )
            P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
            Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        if not itranspara['fix_t']:
            muX = xp.divide(xp.sum(PX, 0), Np)
            muY = xp.divide(Y.transpose(1,0) @ P1, Np)

            Y_hat = Y - muY
            B = PX.transpose(1,0) @ Y_hat - xp.outer(muX, P1 @ Y_hat)
        else:
            muX = 0
            muY = 0

            Y_hat = Y
            B = PX.transpose(1,0) @ Y_hat - xp.outer(itmat['t'], P1 @ Y_hat)

        ## RS
        if itranspara['isoscale']:
            if not itranspara['fix_R']:
                U, S, V = xp.linalg.svd(B, full_matrices=True)
                S = xp.diag(S)
                C = xp.eye(D, dtype=B.dtype, device=B.device)
                C[-1, -1] = xp.linalg.det(U @ V)
                R = U @ C @ V

                if xp.linalg.det(R) < 0:
                    U[:, -1] = -U[:, -1]
                    R = U @ C @ V
            else:
                R = itmat['R']

            H = xp.sum(P1 * xp.sum(Y_hat * Y_hat, 1))
            YPY = xp.sum( Y_hat * Y_hat * P1.unsqueeze(1), 0)
    
            if not itranspara['fix_s']:
                trBR = xp.trace(B.T @ R)
                s = trBR/H
            else:
                s = itmat['s']
            s = s.expand(D)    
        else:
            YPY = xp.sum( Y_hat * Y_hat * P1.unsqueeze(1), 0)
            YPY.masked_fill_(YPY == 0, eps)
            H = YPY
            s = itmat['s'].clone()
            R = itmat['R'].clone()
            C = xp.eye(D, dtype=B.dtype, device=B.device)

            if (not itranspara['fix_s']) and (not itranspara['fix_R']):
                max_iter = 100
                error = True
                iiter = 0
                while (error):
                    # U, S, V = xp.linalg.svd(B @ s.T, full_matrices=True)
                    U, S, V = xp.linalg.svd(B * s, full_matrices=True)
                    C[-1, -1] = xp.linalg.det(U @ V)
                    R_pre = R.clone()
                    R = U @ C @ V
                    if xp.linalg.det(R) < 0:
                        U[:, -1] = -U[:, -1]
                        R = U @ C @ V

                    s = xp.diagonal(B.T @ R)/H
                    if itranspara['s_clip']  is not None:
                        s = xp.clip(s, *itranspara['s_clip'] )

                    iiter += 1
                    error = (xp.dist(R, R_pre) > 1e-8) and (iiter < max_iter)

            elif (not itranspara['fix_R']) and itranspara['fix_s']:
                U, S, V = xp.linalg.svd(B * s, full_matrices=True)
                C[-1, -1] = xp.linalg.det(U @ V)
                R = U @ C @ V
                if xp.linalg.det(R) < 0:
                    U[:, -1] = -U[:, -1]
                    R = U @ C @ V
    
            elif (not itranspara['fix_s']) and itranspara['fix_R']:
                s = xp.diagonal(B.T @ R )/H
    
        if itranspara['s_clip'] is not None:
            s = xp.clip(s, *itranspara['s_clip'])
        A = R * s
        if not itranspara['fix_t']:
            t = muX - A @ muY
        else:
            t = itmat['t']
        TY = Y @ A.T +  t #.T

        # sigma2, Q
        if L == 1:  #check the stability of numerical calculation 
            X_hat = X - muX
            trXPX = xp.sum(Pt1 * xp.sum(X_hat * X_hat, 1))
            trARS = xp.trace(A @ B.T)
            trSSYPY = xp.sum((s **2) * H)
            sigma2 = (trXPX  - 2*trARS + trSSYPY) / (Np * D)

            if sigma2 < sigma2_min: 
                sigma2 = xp.clip(sigma_square(X, TY), min=sigma2_min)
            Q = (Np * D)/2*(1+xp.log(sigma2))
            sigma2_exp = sigma2
            sigma2 = sigma2.expand(L)
        else:
            sigma2_exp, sigma2, Q = self.update_sigma2(Pt1s, P1s, Nps, PXs, Xs, TY, Qp = None)

        self.tmats['R'] = R
        self.tmats['s'] = s
        self.tmats['t'] = t
        self.TY = TY
        self.sigma2 = sigma2
        self.sigma2_exp = sigma2_exp
        self.Q = Q
        # return [[R, s, t,], TY, sigma2_exp, sigma2, Q]

    @th.no_grad()
    def pw_affine_maximization(self, Pt1s, P1s, Nps, PXs, Y, Xs):
        xp, D, L, eps = self.xp, self.D, self.L, self.eps
        sigma2_min, itranspara, itmat =  self.sigma2_min, self.transparas, self.tmats
        sigma2 = self.sigma2

        # ols = self.omega/self.sigma2 # raw
        ols = self.omega/self.sigma2_exp # exp 
        # ols = self.omega # same with not common constraint  exp #TODO

        if L == 1: #check the stability of numerical calculation 
            PX, P1, Np, Pt1, X = PXs[0], P1s[0], Nps[0], Pt1s[0], Xs[0]
        else:
            PX = sum( iPX * iol for iPX, iol in zip(PXs, ols) )
            P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
            Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        muX = xp.divide(xp.sum(PX, axis=0), Np)
        muY = xp.divide(Y.transpose(1,0) @ P1, Np)
        Y_hat = Y - muY
        B = PX.transpose(1,0) @ Y_hat - xp.outer(muX,  P1 @ Y_hat)

        YPYh = (Y_hat.transpose(1,0) * P1) @ Y_hat
        # if xp.det(YPYh) == 0:
        #     YPY = (Y.T * P1) @ Y
        #     XPY = PX.T @ Y
        #     error = True
        #     max_iter = 100
        #     iiter = 0
        #     while (error):
        #         YPT = xp.outer(t, xp.sum(Y.T * P1, 1))
        #         B_pre = B
        #         B = (XPY - YPT) @ xp.linalg.inv(YPY)
        #         t = muX - B @ muY
        #         iiter += 1
        #         error = (xp.linalg.norm(B - B_pre) > 1e-9) and (iiter < max_iter)
        # else:
        if itranspara['gamma1'] > 0: # TODO (P1WY)
            YtZY = ( itranspara['WY'].transpose(1,0) * P1) @ itranspara['WY']
            YPYh.add_( (2 * itranspara['gamma1'] * sigma2) * YtZY )

        try:
            YPYhv = xp.linalg.inv(YPYh)
            A = B @ YPYhv
        except:
            YPYh.diagonal().add_( itranspara['delta'] *sigma2)
            A = (B + itranspara['delta']*sigma2*itmat['A']) @ xp.linalg.inv(YPYh)

        t = muX - A @ muY
        TY = Y @ A.T +  t

        if L == 1: #check the stability of numerical calculation 
            X_hat = X - muX
            trAB = xp.trace(B @ A.T)
            trXPX = xp.sum(Pt1 * xp.sum(X_hat * X_hat, 1))
            sigma2 = (trXPX - trAB) / (Np * D)
            if sigma2 < sigma2_min: 
                sigma2 = xp.clip(sigma_square(X, TY), min=sigma2_min)
            Q = (Np * D)/2*(1+xp.log(sigma2))
            sigma2_exp = sigma2
            sigma2 = sigma2.expand(L)
        else:
            sigma2_exp, sigma2, Q = self.update_sigma2(Pt1s, P1s, Nps, PXs, Xs, TY, Qp = None)

        self.tmats['A'] = A
        self.tmats['t'] = t
        self.TY = TY
        self.sigma2 = sigma2
        self.sigma2_exp = sigma2_exp
        self.Q = Q
        # return [[A, t,], TY, sigma2_exp, sigma2, Q]

    def pw_projective_maximization(self, Pt1s, P1s, Nps, PXs, Y, Xs):
        with self.xp.no_grad(): 
            trXPXs = [ self.xp.sum(Pt1 * self.xp.sum(X * X, 1)) for X, Pt1 in zip( Xs, Pt1s) ]

        def closure():
            self.optimizer.zero_grad()
            self.logsigma2.data = self.xp.clamp(self.logsigma2, min=self.xp.log(self.sigma2_min))
            TY = self.projective_transform()

            Q = 0
            for iL in range(self.L):
                trXPY = self.xp.sum(PXs[iL] * TY)
                trYPY = self.xp.sum(TY * TY, 1) @ P1s[iL]
                res1 = (Nps[iL] * self.D / 2.0) * self.logsigma2[iL]
                iQ = (trXPXs[iL] - 2*trXPY + trYPY)/(2*self.xp.exp(self.logsigma2[iL])+ self.eps) + res1
                Q += iQ
            Q.backward(retain_graph=True)
            # self.xp.nn.utils.clip_grad_norm_([self.A, self.B, self.t, self.logsigma2], 1e5)
            return Q
        
        loss = self.optimizer.step(closure)
        if self.transparas['lr_stepsize']:
            self.scheduler.step()
    
        with self.xp.no_grad():
            self.TY = self.projective_transform()
            self.sigma2 = self.xp.exp(self.logsigma2)
            self.Q = self.xp.tensor(loss.item())

            for iL in range(self.L):
                if self.sigma2[iL] < self.sigma2_min: 
                    self.sigma2[iL] = self.xp.clip(sigma_square(Xs[iL], self.TY), min=self.sigma2_min)
                    self.logsigma2.data[iL] = self.xp.log(self.sigma2[iL])

    @th.no_grad()
    def pw_deformable_maximization0(self, Pt1s, P1s, Nps, PXs, Y, Xs):
        xp, D, L, eps = self.xp, self.D, self.L, self.eps
        sigma2_min, sigma2_exp, itranspara, itmat = self.sigma2_min, self.sigma2_exp, self.transparas, self.tmats
        sigma2 = self.sigma2
        # ols = self.omega/self.sigma2 # raw
        ols = self.omega/self.sigma2_exp # exp 
        # ols = self.omega # same with not common constraint  exp #TODO

        mu = itranspara['alpha_mu']**(self.iteration)
        alpha = itranspara['alpha'] 
        nu1 = nu2 = itranspara['gamma_nu']**(self.iteration)
        gamma1 = itranspara['gamma1']
        gamma2 = itranspara['gamma2']
        p1_thred = itranspara['p1_thred']
        use_p1 = itranspara['use_p1']
        delta = itranspara['delta']
   
        # wsa = sum( ols[iL] * sigma2[iL] * float(alpha[i]) * mu for iL in  range(L) )
        wsa = sum( ols[iL] * sigma2_exp * float(alpha[iL]) * mu for iL in  range(L) )
        PX = sum( iPX * iol/wsa for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol/wsa for iP1, iol in zip(P1s, ols) )

        B = PX - P1[:,None] * Y
        if delta>0:
            P1_hat = P1 + delta/wsa
        else:
            P1_hat = P1

        if itranspara['fast_rank']:
            U = self.DR.U
            S = self.DR.S
            I = self.DR.I
            V = (U @ S)
            Pg = P1_hat[:,None]* V

            if (use_p1):
                P1s_thred = [ xp.where(iP1 < ipth, 0.0, iP1) for iP1, ipth in zip(P1s, p1_thred) ]
                if any(gamma1 > 0):
                    P2 = sum( (w * sigma2_exp * float(ig1) * nu1/ wsa) * iP1 for w, iP1, ig1 in zip(ols,  P1s_thred, gamma1) )
                    P2 = self.DR.E.T * P2[None,:]
                    B  -= P2 @ self.DR.J3
                    Pg += P2 @ self.DR.J1
                if any(gamma2 > 0):
                    P3 = sum( (w * sigma2_exp * float(ig2) * nu2/ wsa) * iP1 for w, iP1, ig2 in zip(ols, P1s_thred, gamma2) )
                    Pg += (self.DR.F.T * P3[None,:]) @ self.DR.J2
            else:
                if any(gamma1 > 0):
                    P2 = sum( (w * sigma2_exp * float(ig1) * nu1/ wsa) for w, ig1 in zip(ols, gamma1) )
                    B  -= P2 * self.DR.J3
                    Pg += P2 * self.DR.J1
                if any(gamma2 > 0):
                    P3 = sum( (w * sigma2_exp * float(ig2) * nu2/ wsa)  for w, ig2 in zip(ols, gamma2) )
                    Pg += P3 * self.DR.J2

            W  = B - Pg @ xp.linalg.solve(I + U.T @ Pg, U.T @ B)
            H  = V @ (U.T @ W)

        else:
            G = self.DR.G
            Pg = P1_hat[:,None]* G
            Pg.diagonal().add_(1.0)
    
            if (use_p1): # same as the fast_rank
                P1s_thred = [ xp.where(iP1 < ipth, 0.0, iP1) for iP1, ipth in zip(P1s, p1_thred) ]
                if any(gamma1 > 0):
                    P2 = sum( (w * sigma2_exp * float(ig1) * nu1/ wsa) * iP1 for w, iP1, ig1 in zip(ols,  P1s_thred, gamma1) )
                    P2 = self.DR.E.T * P2[None,:]
                    B  -= P2 @ self.DR.J3
                    Pg += P2 @ self.DR.J1
                if any(gamma2 > 0):
                    P3 = sum( (w * sigma2_exp * float(ig2) * nu2/ wsa) * iP1 for w, iP1, ig2 in zip(ols, P1s_thred, gamma2) )
                    Pg += (self.DR.F.T * P3[None,:]) @ self.DR.J2
            else:  # same as the fast_rank
                if any(gamma1 > 0):
                    P2 = sum( (w * sigma2_exp * float(ig1) * nu1/ wsa) for w, ig1 in zip(ols, gamma1) )
                    B  -= P2 * self.DR.J3
                    Pg += P2 * self.DR.J1
                if any(gamma2 > 0):
                    P3 = sum( (w * sigma2_exp * float(ig2) * nu2/ wsa)  for w, ig2 in zip(ols, gamma2) )
                    Pg += P3 * self.DR.J2

            W = xp.linalg.solve(Pg, B)
            H = G @ W

        TY = Y + H
    
        Lp = xp.trace(W.T @ H) # TODO + LLE + LP
        # Qp = sum([ (float(alpha[i]) * mu * Lp + ill + ilp )/2.0 for i in range(L) ])
        Qp = sum(alpha) * mu * Lp/2.0
        sigma2_exp, sigma2, Q = self.update_sigma2(Pt1s, P1s, Nps, PXs, Xs, TY, Qp = Qp)

        self.tmats['W'] = W
        self.TY = TY
        self.sigma2 = sigma2
        self.sigma2_exp = sigma2_exp
        self.Q = Q

    @th.no_grad()
    def pw_deformable_maximization(self, Pt1s, P1s, Nps, PXs, Y, Xs):
        xp, D, L = self.xp, self.D, self.L
        sigma2_exp, itranspara, itmat = self.sigma2_exp, self.transparas, self.tmats
        sigma2, omega = self.sigma2, self.omega

        mu = itranspara['alpha_mu']**(self.iteration)
        alpha = itranspara['alpha'] 
        nu1 = nu2 = itranspara['gamma_nu']**(self.iteration)
        gamma1 = itranspara['gamma1']
        gamma2 = itranspara['gamma2']
        p1_thred = itranspara['p1_thred']
        use_p1 = itranspara['use_p1']
        delta = itranspara['delta']

        # ols = omega/sigma2 # raw
        ols = omega/sigma2_exp # exp 
        # ols = omega # same with not common constraint  exp #TODO
        wsa = sum( omega[iL] * float(alpha[iL]) * mu for iL in range(L) )

        PX = sum( iPX * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )

        B = PX - P1[:,None] * Y
        if delta>0:
            P1_hat = P1 + delta
        else:
            P1_hat = P1

        if itranspara['fast_rank']:
            U = self.DR.U
            S = self.DR.S
            V = (U @ S)
            Pg = P1_hat[:,None]* V

            if (use_p1):
                P1s_thred = [ xp.where(iP1 < ipth, 0.0, iP1) for iP1, ipth in zip(P1s, p1_thred) ]
                if any(gamma1 > 0):
                    P2 = sum( (w * float(ig1) * nu1) * iP1 for w, iP1, ig1 in zip(omega,  P1s_thred, gamma1) )
                    P2 = self.DR.E.T * P2[None,:]
                    B  -= P2 @ self.DR.J3
                    Pg += P2 @ self.DR.J1
                if any(gamma2 > 0):
                    P3 = sum( (w * float(ig2) * nu2) * iP1 for w, iP1, ig2 in zip(omega, P1s_thred, gamma2) )
                    Pg += (self.DR.F.T * P3[None,:]) @ self.DR.J2
            else:
                if any(gamma1 > 0):
                    P2 = sum( (w * float(ig1) * nu1) for w, ig1 in zip(omega, gamma1) )
                    B  -= P2 * self.DR.J3
                    Pg += P2 * self.DR.J1
                if any(gamma2 > 0):
                    P3 = sum( (w * float(ig2) * nu2)  for w, ig2 in zip(omega, gamma2) )
                    Pg += P3 * self.DR.J2
            
            I = self.DR.I * wsa
            W  = (B - Pg @ xp.linalg.solve(I + U.T @ Pg, U.T @ B)) / wsa
            H  = V @ (U.T @ W)

        else:
            G = self.DR.G
            Pg = P1_hat[:,None]* G
            Pg.diagonal().add_(wsa)
    
            if (use_p1): # same as the fast_rank
                P1s_thred = [ xp.where(iP1 < ipth, 0.0, iP1) for iP1, ipth in zip(P1s, p1_thred) ]
                if any(gamma1 > 0):
                    P2 = sum( (w * float(ig1) * nu1) * iP1 for w, iP1, ig1 in zip(omega, P1s_thred, gamma1) )
                    P2 = self.DR.E.T * P2[None,:]
                    B  -= P2 @ self.DR.J3
                    Pg += P2 @ self.DR.J1
                if any(gamma2 > 0):
                    P3 = sum( (w * float(ig2) * nu2) * iP1 for w, iP1, ig2 in zip(omega, P1s_thred, gamma2) )
                    Pg += (self.DR.F.T * P3[None,:]) @ self.DR.J2
            else:  # same as the fast_rank
                if any(gamma1 > 0):
                    P2 = sum( (w * float(ig1) * nu1) for w, ig1 in zip(omega, gamma1) )
                    B  -= P2 * self.DR.J3
                    Pg += P2 * self.DR.J1
                if any(gamma2 > 0):
                    P3 = sum( (w * float(ig2) * nu2) for w, ig2 in zip(omega, gamma2) )
                    Pg += P3 * self.DR.J2

            W = xp.linalg.solve(Pg, B)
            H = G @ W

        TY = Y + H
    
        Lp = xp.trace(W.T @ H) # TODO + LLE + LP
        # Qp = sum([ (float(alpha[i]) * mu * Lp + ill + ilp )/2.0 for i in range(L) ])
        Qp = sum(alpha) * mu * Lp/2.0
        sigma2_exp, sigma2, Q = self.update_sigma2(Pt1s, P1s, Nps, PXs, Xs, TY, Qp = Qp)

        self.tmats['W'] = W
        self.TY = TY
        self.sigma2 = sigma2
        self.sigma2_exp = sigma2_exp
        self.Q = Q
    
    def deformable_maximization0(self, Pt1, P1, Np, PX, Y, X, lr={}):
        xp, D, floatx, device, eps, sigma2_min = xp, self.D, self.floatx, self.device, self.eps,  self.sigma2_min
        isoscale, fix_R, fix_s, fix_t, s_clip  = self.isoscale, self.fix_R, self.fix_s, self.fix_t, self.s_clip
        iteration, sigma2 = self.iteration, self.sigma2

        mu = lr['alpha_mu']**(iteration)
        nu = lr['gamma_nu']**(iteration)
        alpha = alpha * mu
        gamma1 = gamma1 * nu
        gamma2 = gamma2 * nu
        DR = lr['DR']

        if lr['fast_rank']:
            B  = PX - xp.multiply(Y.T, P1).T
            B = 1.0/(sigma2*alpha) * B  # W = 1.0/(sigma2*alpha) * W
            Pg = 1.0/sigma2*(P1[:,None]* (DR.U @ DR.S))

            if (lr['use_p1']):
                P1 = P1.clone()
                P1[P1 < lr['p1_thred']] = 0

                if (gamma1>0):
                    LPLY = (DR.L.T @ (P1[:,None]*DR.LY))
                    LPLV = (DR.L.T @ (P1[:,None]*DR.LV))
                    B -=  gamma1 * sigma2 * LPLY
                    Pg += gamma1 * LPLV

                if (gamma2>0):
                    APAV = (DR.A.T @ (P1[:,None]*DR.AV))
                    Pg += gamma2 * APAV
            else:
                if (gamma1>0):
                    B -=  gamma1 * sigma2 * DR.QY
                    Pg += gamma1 * DR.J
                elif (gamma2>0):
                    Pg += gamma2 * DR.J

            Pg = 1.0/alpha * Pg
            W = B - Pg @ xp.linalg.solve( DR.I + DR.U.T @ Pg, DR.U.T @ B)
            V = (DR.U @ DR.S) @ (DR.U.T @ W)

        else:
            B  = PX - xp.multiply(Y.T, P1).T
            A = P1[:,None]* DR.G 
            A.diagonal().add_(alpha * sigma2)

            if (lr['use_p1']):
                P1 = P1.clone()
                P1[P1 < lr['p1_thred']] = 0

                if (gamma1>0):
                    LPLY = (DR.L.T @ (P1[:,None]*DR.LY))
                    QG = (DR.L.T @ (P1[:,None]*DR.LG))
                    B -=  gamma1 * sigma2 * LPLY
                    A +=  gamma1 * sigma2 * QG
                if (gamma2>0):
                    RG = (DR.A.T @ (P1[:,None]*DR.AG))
                    A += gamma2 * sigma2 * RG
            else:
                if (gamma1>0):
                    B -=  gamma1 * sigma2 * DR.QY
                    A +=  gamma1 * sigma2 * DR.QG
                if (gamma2>0):
                    A += gamma2 * sigma2 * DR.RG
            W = xp.linalg.solve(A, B)
            V = DR.G @ W
        
        TY = Y + V
        trxPx = xp.sum( Pt1 * xp.sum(X  * X, axis=1) )
        tryPy = xp.sum( P1 * xp.sum( TY * TY, axis=1))
        trPXY = xp.sum(TY * PX)

        sigma2 = (trxPx - 2 * trPXY + tryPy) / (Np * D)
        if sigma2 < eps: 
            sigma2 = xp.clip(self.sigma_square(X, TY), min=sigma2_min)

        Q = D * Np/2 * (1+xp.log(sigma2))
        return [[W, V], TY, sigma2, Q]

    @th.no_grad()
    def update_sigma2(self, Pt1s, P1s, Nps, PXs, Xs, TY, Qp = None):
        L, D, xp =  len(Xs), float(TY.shape[1]), self.xp
        sigma2_min, sigma2_sync, omega = self.sigma2_min, self.sigma2_sync, self.omega

        Flj = [] 
        for iL in range(L):
            trxPx = xp.sum( Pt1s[iL] * xp.sum(Xs[iL]  * Xs[iL], axis=1) )
            tryPy = xp.sum( P1s[iL]  * xp.sum(TY * TY, axis=1))
            trPXY = xp.sum( TY * PXs[iL])
            Flj.append(trxPx - 2 * trPXY + tryPy)

        # sigma2
        sigma2_exp =(sum( wlj * flj for wlj, flj in zip(omega, Flj))/
                        sum( wlj * jlj * D for wlj, jlj in zip(omega, Nps)))
        sigma2_exp = xp.clip(sigma2_exp, min=sigma2_min)
        if sigma2_sync:
            sigma2 = sigma2_exp.expand(L)
        else:
            sigma2 = xp.zeros(L, dtype=sigma2_exp.dtype, device=sigma2_exp.device)
            for iL in range(L):
                sigma2[iL] = Flj[iL]/Nps[iL]/D        
                if sigma2[iL] < sigma2_min: 
                    sigma2[iL] = xp.clip(sigma_square(Xs[iL], TY), min=sigma2_min)

        # Q
        Qp = Qp or 0
        for iL in range(L):
            iQ  = Flj[iL]/2.0/sigma2[iL] + Nps[iL] * D/2.0 * xp.log(sigma2[iL])
            # iQ  = Nps[iL] * D/2.0 * (1 + xp.log(sigma2[iL])) #asyn, sigma2_sync = False
            Qp += omega[iL]*iQ #/self.Ns[iL]  #TODO whether need to divide by Ns[iL]
        return sigma2_exp, sigma2, Qp

    @th.no_grad()
    def update_normalize(self):
        device, dtype = 'cpu', self.floatxx
        Sx, Sy  = self.Sx.to(device, dtype=dtype), self.Sy.to(device, dtype=dtype)
        Tx, Ty  = self.Mx.to(device, dtype=dtype), self.My.to(device, dtype=dtype)
        Hx, Hy  = self.Hx.to(device, dtype=dtype), self.Hy.to(device, dtype=dtype)
        Sr = Sx/Sy

        itf, itp = self.transformer, self.transparas
        iTM = self.tmats
        iTM = { k: v.detach().to(device, dtype=dtype) for k,v in iTM.items() }

        if itf in 'ESARTILO':
            iTM['tmat'] = self.homotransform_mat(itf, self.D, xp=self.xp, 
                            device=device, dtype=dtype, **iTM )
            
            if itf in 'ESRTILO':
                A = iTM['R'] * iTM['s'] * Sr
                iTM['s'] *= Sr
            elif itf in ['A']:
                A = iTM['A']* Sr
                iTM['A'] = A
            else:
                raise(f"Unknown transformer: {itf}")

            if not itp.get('fix_t' , False):
                iTM['t'] = iTM['t']*Sx + Tx - Ty @ A.T
            else: #TODO
                iTM['t'] = iTM['t']*Sx + Tx - Ty @ A.T

            iTM['tform'] = self.homotransform_mat(itf, self.D, xp=self.xp, 
                            device=device, dtype=dtype, **iTM )
        elif itf in ['P']:
            iTM['tmat'] = self.homotransform_mat(itf, self.D, xp=self.xp, 
                            device=device, dtype=dtype, **iTM )
            
            tform = Hx @ iTM['tmat'] @ self.xp.linalg.inv(Hy)
            tform /= tform[-1,-1]*iTM['d']
            iTM['tform_v0'] = tform

            iTM['A']= (Sx *iTM['A'] + self.xp.outer(Tx, iTM['B']))/Sy
            iTM['t'] = (iTM['t'] * Sx + Tx *iTM['d']) - iTM['A'] @ Ty
            iTM['B'] = iTM['B'] / Sy
            iTM['d'] = (iTM['d'] - iTM['B'] @ Ty)/ iTM['d'] #TODO inverse
            iTM['tform'] = self.homotransform_mat(itf, self.D, xp=self.xp, 
                                device=device, dtype=dtype, **iTM )

        elif itf in ['D']:
            for ia in ['G', 'U', 'S']:
                iv =  getattr(self.DR, ia, None)
                if iv is not None:
                    iTM[ia] = iv.to(device, dtype=dtype)
                else:
                    iTM[ia] = iv

            iTM['Y'] = self.Y.to(device, dtype=dtype)
            iTM['Ym'] = Ty
            iTM['Ys'] = Sy
            iTM['Xm'] = Tx
            iTM['Xs'] = Sx
            iTM['beta'] =  itp['beta'] # if center Y, set scale 
        else:
            raise(f"Unknown transformer: {itf}")

        self.TY = self.TY.detach().to(device, dtype=dtype) * Sx + Tx
        # self.TY = self.transform_points(self.Yr)
        self.tmats = iTM
        self.tmats['transformer'] = itf

    def transform_point(self, Yr, use_keops=True, **kargs):
        if self.tmats['transformer'] in 'ESARTILOP':
            return self.homotransform_point(Yr, self.tmats['tform'], inverse=False, xp=self.xp)
        elif self.tmats['transformer']  == 'D':
            return self.ccf_deformable_transform_point(Yr, **self.tmats, 
                                                        use_keops=use_keops, **kargs)
        else:
            raise(f"Unknown transformer: {self.tmats['transformer']}")

    def get_transformer(self):
        return self.tmats

    def paired_rigid_maximization(self, P, P1, Pt1, Np, X, Y,):
        xp = self.xp
        PX = P.dot(X)
        PY = P.dot(Y)
        muX = xp.divide(xp.sum(PX, axis=0), Np)
        muY = xp.divide(xp.sum(PY, axis=0), Np)
        # muY = xp.divide(xp.dot(xp.transpose(Y), P1), Np)

        X_hat = X - muX
        Y_hat = Y - muY

        # A = xp.dot(xp.transpose(P.dot(X_hat)), Y_hat)
        # A = X.T @ PY - ( muX[:,None] @ muY[:,None].T ) * Np
        A = xp.dot(PX.T, Y_hat) - xp.outer(muX, xp.dot(P1.T, Y_hat))

        U, S, Vh = xp.linalg.svd(A, full_matrices=True)
        S = xp.diag(S)
        C = xp.eye(self.D)
        C[-1, -1] = xp.linalg.det(xp.dot(U, Vh))
        R = xp.dot(xp.dot(U, C), Vh)

        # trAR = xp.trace(xp.dot(A.T, R))
        # trYPY = xp.trace(Y_hat.T @ xp.diag(P1) @ Y_hat)
        # trYPY = xp.sum(xp.multiply(Y.T**2, P1)) - Np*(muY.T @ muY)
        trAR = xp.trace(S @ C)
        trXPX = xp.sum(Pt1.T * xp.sum(xp.multiply(X_hat, X_hat), axis=1))
        trYPY = xp.sum(P1.T * xp.sum(xp.multiply(Y_hat, Y_hat), axis=1))

        ## LLE
        if self.scale is True:
            Z = self.LW.transpose().dot(P.dot(self.LW))
            YtZY = Y.T @ Z.dot(Y)
            s = trAR / (trYPY + 2 * self.alpha * sigma2 * xp.trace(YtZY))
            QZ = self.alpha * xp.linalg.norm(s * xp.sqrt(P) @ self.llW @ Y)
        else:
            QZ = 0

        t = muX - s * xp.dot(R, muY)
        TY = s * R @ Y.T + t
    
        V = xp.square(X - TY)
        sigma2 = xp.sum(V * P1[:, None]) / (self.D * Np)

        Ind = (P.diagonal() > self.theta)
        gamma = xp.clip(Ind.sum() / self.N, * self.gamma_clip)

        Q = 11 #TODO
        diff = xp.abs(self.Q - Q)
        diff_r = xp.abs(diff / Q)
        self.Q = Q

    def paired_affine_maximization(self, P, P1, Pt1, Np, X, Y, lr={}):
        xp, D, floatx, device, eps = self.xp, self.D, self.floatx, self.device, self.eps
        sigma2_min = self.sigma2_min
        sigma2 = self.sigma2

        PX = P.dot(X)
        PY = P.dot(Y)
        muX = xp.divide(xp.sum(PX, axis=0), Np)
        muY = xp.divide(xp.sum(PY, axis=0), Np)
        # muY = xp.divide(xp.dot(xp.transpose(Y), P1), Np)

        X_hat = X - muX
        Y_hat = Y - muY

        # A = xp.dot(xp.transpose(P.dot(X_hat)), Y_hat)
        A = X.T @ PY - (muX[:, None] @ muY[:, None].T) * Np
        # A = xp.dot(PX.T, Y_hat) - xp.outer(muX, xp.dot(P1.T, Y_hat))
        YPY = xp.dot(xp.dot(Y_hat.T, xp.diag(P1)), Y_hat)

        DR = lr['DR']
        ## LLE
        Z = DR.LW.transpose().dot(P.dot(DR.LW))
        YtZY = Y.T @ Z.dot(Y)
        YtZY *= (2 * lr['alpha'] * sigma2)

        B = xp.dot(A, xp.linalg.inv(YPY + YtZY))
        t = muX - xp.dot(B, muY)
        TY = B @ Y.T + t

        V = xp.square(X - TY)
        sigma2 = xp.sum(V * P1[:, None]) / (D * Np)
        Q = Np * D * 0.5 * (1.0 + xp.log(sigma2))
        Q += lr['alpha']  * xp.linalg.norm(B @ Y.T @ xp.sqrt(P) @ DR.LW.T)

        Ind = (P.diagonal() > self.theta)
        self.gamma = xp.clip(self.Ind.sum() / self.N, *self.gamma_clip)