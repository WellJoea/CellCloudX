import torch as xp
from ...transform import homotransform_points, homotransform_point, ccf_deformable_transform_point, homotransform_mat
from .operation_maximization import init_tmat
from .operation_expectation import (neighbor_weight, rigid_outlier, gwcompute_feauture,
                                    expectation_ko, expectation_xp, similarity_paras)

class gwEM():
    def __init__(self, sample=0.1, sample_grow=1.1, sample_min=1, sample_stopiter=10, xp=None):
        super().__init__()
        self.sample = sample
        self.sample_grow = sample_grow
        self.sample_min = sample_min
        self.sample_stopiter = sample_stopiter

        self.init_tmat = init_tmat
        self.homotransform_points = homotransform_points
        self.ccf_deformable_transform_point = ccf_deformable_transform_point
        self.homotransform_mat = homotransform_mat
        self.homotransform_point = homotransform_point

    def sample_index(self, iL, ins, itera):
        gr = self.xp.Generator() #device=self.device
        gr.manual_seed( int(f'{iL+1}{ins}{itera}' ))
        N = int(self.Ns[ins])
        sample = min(1.0, self.sample*(self.sample_grow**itera))
        perm = self.xp.randperm(N, generator=gr)
        n_samples = int(max(N*sample, min(self.sample_min, N)))
        return perm[:n_samples]


    def expectation(yid, xids, omega,  Xs, TYs, Ys):
        wl = omega[xids]
        wl = wl/xp.sum(wl)

        Pt1s, P1s, PXs, Nps, ols = [],[],[],[],[]
        iTY, iY, iXs = TYs[yid], Ys[yid], [] # TODO sample for iTY will increase the complexity of the deformable.

        for wlj, xid in zip(wl, xids):
            iX = Xs[xid] 

            if self.use_sample and self.iteration < self.sample_stopiter:
                xsidx = self.sample_index(iL, xid, self.iteration)
                iX = iX[xsidx]

            D, Ni, Mi = iX.shape[1], iX.shape[0], iTY.shape[0]

            gs = Mi/Ni*self.ws[yid, xid]/(1-self.ws[yid, xid])
            d2f= getattr(self.fds, f'f{yid}_{xid}')

            if d2f is None: #check:
                assert (self.fexist[yid] != True or self.fexist[xid] != True)
            else:
                assert (self.fexist[yid] == True and self.fexist[xid] == True)
        
            iexpectation = expectation_ko if self.keops_thr[yid, xid] else expectation_xp
            if self.use_sample and self.iteration < self.sample_stopiter:
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
                iPt1, iP1, iPX, iNp, tau2s = iexpectation( 
                        iX, iTY, self.sigma2[yid, xid], gs,  #sigma2_expect[yid]
                        xp = xp, d2f=d2f, tau2=self.tau2[yid, xid], 
                        tau2_auto=self.tau2_auto[iL], eps=self.eps, 
                        tau2_alpha=self.tau2_decay[iL]**self.iteration, DF=self.DK,
                        feat_normal=self.feat_normal, 
                        XF=xFa, YF=self.Fa[yid], device=self.device,)
            else:
                iPt1, iP1, iPX, iNp, tau2s = iexpectation(
                        iX, iTY, self.sigma2[yid, xid], gs, #sigma2_expect[yid]
                        xp = xp, d2f=d2f, tau2=self.tau2[yid, xid], 
                        tau2_auto=self.tau2_auto[iL], eps=self.eps, 
                        tau2_alpha=self.tau2_decay[iL]**self.iteration, DF=self.DK,
                        feat_normal=self.feat_normal, XF=self.Fa[xid], YF=self.Fa[yid], device=self.device,)
        
            if self.tau2_auto[iL] and self.fexist[iL]:
                self.tau2[yid, xid] = tau2s

            w = (1- iNp/Ni).clip(*self.w_clip)
            self.ws[yid, xid] = self.wa*self.ws[yid, xid] + (1-self.wa)*w
            # olj = wlj/2.0/self.sigma2[yid, xid] # TODO v0
            olj = wlj/self.sigma2_expect[yid] #self.sigma2_expect

            Pt1s.append(iPt1)
            P1s.append(iP1)
            PXs.append(iPX)
            Nps.append(iNp)
            ols.append(olj)
            iXs.append(iX)
        return Pt1s, P1s, Nps, PXs, ols, iY, iXs


    def init_transformer(self):
        self.tmats = {}
        for iL in range(self.L):
            itrans = 'E' if iL == self.root else self.transformer[iL]
            self.tmats[iL] =  self.init_tmat(itrans, self.D, self.Ns[iL], 
                                             xp =self.xp, device=self.device, dtype=self.floatx)

    def init_regularizer(self):
        self.DR = {}
        DL = sum([ 'D' in i for i in self.transformer ])
        if DL == 0: return

        try:
            pbar = tqdm(range(self.L), total=self.L, colour='#AAAAAA', desc='gwRegularizer')
            for iL in pbar:
                idr = deformable_regularizer(xp = self.xp, verbose=False, **self.transparas[iL])
                if ('D' in self.transformer[iL]) and (self.omega[iL].sum() > 0):
                    idr.compute(self.Xa[iL], device=self.device, dtype=self.floatx)
                self.DR[iL] = idr
            pbar.close()
        except:
            del self.DR
            self.clean_cache()
            raise ValueError('Failed to compute the deformable regularizer matrix. Check the Memory.')

    def optimization(self):
        for iL in self.OL:
            if len(self.OM[iL]) == 0: continue
            Pt1s, P1s, Nps, PXs, ols, iY, iXs = self.expectation(iL, self.TYs, self.TYs_tmp, self.Xa, )
            self.maximization(iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs, transformer=self.transformer[iL])
        if self.graph_strategy == 'async':
            self.TYs = [ ity for ity in self.TYs_tmp ]
            # self.omega = self.omega_tmp
        if (self.iteration + 1) % self.inneriter == 0:
            self.TYs = [ ity.clone() for ity in self.TYs_tmp ]
            # self.omega = self.omega_tmp.clone()
        qprev = self.Q
        self.Q = self.xp.sum(self.Qs)
        self.diff = self.xp.abs(self.Q - qprev) #/self.Q
        self.iteration += 1

    def maximization(self, *args, transformer='E', **kwargs):
        if transformer == 'E':
            self.rigid_maximization(*args, **kwargs)
        elif transformer in 'SRTILO':
            self.similarity_maximization(*args, **kwargs)
        elif transformer == 'A':
            self.affine_maximization(*args, **kwargs)
        elif transformer == 'D':
            self.deformable_maximization(*args, **kwargs)
        else:
            raise(f"Unknown transformer: {transformer}")

    def rigid_maximization(self, iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs):
        itranspara = self.transparas[iL]

        # sigma2 = self.sigma2[iL][self.OM[iL]]
        # ols = [ i/j for i, j in zip(ols,sigma2) ]

        PX = sum( iPX * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
        Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        muX = self.xp.divide(self.xp.sum(PX, 0), Np)
        muY = self.xp.divide(iY.T @ P1, Np)

        Y_hat  =  iY - muY
        B = PX.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat)
    
        U, S, V = self.xp.linalg.svd(B, full_matrices=True)
        S = self.xp.diag(S)
        C = self.xp.eye(self.D, dtype=self.floatx, device=self.device)
        C[-1, -1] = self.xp.linalg.det(U @ V)
        R = U @ C @ V

        if self.xp.linalg.det(R) < 0:
            U[:, -1] = -U[:, -1]
            R = U @ C @ V
        
        if itranspara['fix_s'] is False:
            trBR = self.xp.trace(B.T @ R)
            trYPY = self.xp.sum(P1 * self.xp.sum(Y_hat * Y_hat, 1))
            s = trBR/trYPY
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

    def similarity_maximization(self, iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs):
        itranspara = self.transparas[iL]

        # sigma2 = self.sigma2[iL][self.OM[iL]] # v1
        # ols = [ i/(2.0*j) for i, j in zip(ols,sigma2) ] # v1

        PX = sum( iPX * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
        Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        if not itranspara['fix_t']:
            muX = self.xp.divide(self.xp.sum(PX, 0), Np)
            muY = self.xp.divide(iY.T @ P1, Np)
        else:
            muX = self.xp.zeros(self.D, dtype=self.floatx, device=self.device)
            muY = self.xp.zeros(self.D, dtype=self.floatx, device=self.device)
            
        Y_hat  =  iY - muY
        B = PX.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat)
    
        if itranspara['isoscale']:
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
                trYPY = self.xp.sum(P1 * self.xp.sum(Y_hat * Y_hat, 1))
                s = trBR/trYPY
                if itranspara['s_clip']  is not None:
                    s = self.xp.clip(s, *itranspara['s_clip'] )
                self.tmats[iL]['s'] = s*self.xp.eye(self.D, dtype=self.floatx, device=self.device)
        else:
            YPY = self.xp.sum( Y_hat * Y_hat * P1.unsqueeze(1), 0)
            YPY.masked_fill_(YPY == 0, self.eps)
        
            if (not itranspara['fix_s']) and (not itranspara['fix_R']):
                max_iter = 70
                error = True
                iiter = 0
                C = self.xp.eye(self.D, dtype=self.floatx, device=self.device)
    
                s = self.tmats[iL]['s'].clone()
                R = self.tmats[iL]['R'].clone()
                while (error):
                    U, S, V = self.xp.linalg.svd(B @ s.T, full_matrices=True)
                    C[-1, -1] = self.xp.linalg.det(U @ V)
                    R_pre = R
                    R = U @ C @ V
                    if self.xp.linalg.det(R) < 0:
                        U[:, -1] = -U[:, -1]
                        R = U @ C @ V
                
                    s = self.xp.diagonal(B.T @ R)/YPY
                    if itranspara['s_clip']  is not None:
                        s = self.xp.clip(s, *itranspara['s_clip'] )
                    s = self.xp.diag(s)  
                    iiter += 1
                    error = (self.xp.dist(R, R_pre) > 1e-8) and (iiter < max_iter)

                self.tmats[iL]['R'] = R
                self.tmats[iL]['s'] = s

            elif (not itranspara['fix_R']) and itranspara['fix_s']:
                U, S, V = self.xp.linalg.svd(B @ self.tmats[iL]['s'], full_matrices=True)
                C = self.xp.eye(self.D, dtype=self.floatx, device=self.device)
                C[-1, -1] = self.xp.linalg.det(U @ V)
                R = U @ C @ V
                if self.xp.linalg.det(R) < 0:
                    U[:, -1] = -U[:, -1]
                    R = U @ C @ V
                self.tmats[iL]['R'] = R

            elif (not itranspara['fix_s']) and itranspara['fix_R']:
                s = self.xp.diagonal(B.T @ self.tmats[iL]['R'])/YPY
                if itranspara['s_clip']  is not None:
                    s = self.xp.clip(s, *itranspara['s_clip'] )
                s = self.xp.diag(s)
                self.tmats[iL]['s'] = s

        if not itranspara['fix_t']:
            t = muX - (self.tmats[iL]['R'] @ self.tmats[iL]['s']) @ muY
            self.tmats[iL]['t'] = t

        iTY = iY @ (self.tmats[iL]['R'] @ self.tmats[iL]['s']).T + self.tmats[iL]['t']
        self.TYs_tmp[iL] = self.Xa[iL] @ (self.tmats[iL]['R'] @ self.tmats[iL]['s']).T + self.tmats[iL]['t']

        self.update_sigma2(iL, iXs, Pt1s, P1s, PXs, Nps, iTY)
    
    def affine_maximization(self, iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs):
        itranspara = self.transparas[iL]

        PX = sum( iPX * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
        Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        muX = self.xp.divide(self.xp.sum(PX, 0), Np)
        muY = self.xp.divide(iY.T @ P1, Np)

        Y_hat  =  iY - muY
        B = PX.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat)
        YPYh = (Y_hat.transpose(1,0) * P1) @ Y_hat

        try:
            YPYhv = self.xp.linalg.inv(YPYh)
            A = B @ YPYhv
        except:
            A_pre = self.tmats[iL].get('A', self.xp.zeros((self.D, self.D), dtype=self.floatx, device=self.device))
            YPYh.diagonal().add_(itranspara['delta']*self.sigma2_expect[iL])
            A = (B + itranspara['delta']*self.sigma2_expect[iL]*A_pre) @ self.xp.linalg.inv(YPYh)

        t = muX - A @ muY
        iTY = iY @ A.T + t
        self.tmats[iL]['A'] = A
        self.tmats[iL]['t'] = t
        self.TYs_tmp[iL] = self.Xa[iL] @ A.T + t

        self.update_sigma2(iL, iXs, Pt1s, P1s, PXs, Nps, iTY)
    
    def deformable_maximization(self, iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs):
        itranspara = self.transparas[iL]

        mu = itranspara['alpha_mu']**(self.iteration)
        alpha = itranspara['alpha'] 

        wsa = sum( w * self.sigma2_expect[iL] * float(alpha[i]) * mu for w, i in zip(ols, self.OM[iL]) )
        PX = sum( iPX * iol/wsa for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol/wsa for iP1, iol in zip(P1s, ols) )
        B = PX - P1[:,None] * iY
 
        if itranspara['fast_rank']:
            U = self.DR[iL].U
            S = self.DR[iL].S
            I = self.DR[iL].I
            V = (U @ S)

            Pg = P1[:,None]* V
            W  = B - Pg @ self.xp.linalg.solve(I + U.T @ Pg, U.T @ B)
            M  = V @ (U.T @ W)
        else:
            G = self.DR[iL].G
            A = P1[:,None]* G
            A.diagonal().add_(1.0)
            W = self.xp.linalg.solve(A, B)
            M = G @ W

        iTY = iY + M
        self.TYs_tmp[iL] = iTY #self.Xa[iL] +  V @ (U.T @ W)
        self.tmats[iL]['W'] = W

        Lp = self.xp.trace(W.T @ M)
        Qp = [ float(alpha[i]) * mu * Lp/2.0 for i in self.OM[iL] ]
        self.update_sigma2(iL, iXs, Pt1s, P1s, PXs, Nps, iTY, Qp = Qp)

    def update_sigma2(self, iL, iXs, Pt1s, P1s, PXs, Nps, iTY, Qp = None):
        ixids = self.OM[iL]
        wls = self.omega[iL][ixids]

        # Flj
        Flj = []
        for i, xid in enumerate(ixids):
            trxPx = self.xp.sum( Pt1s[i] * self.xp.sum(iXs[i]  * iXs[i], axis=1) )
            tryPy = self.xp.sum( P1s[i]  * self.xp.sum(iTY * iTY, axis=1))
            trPXY = self.xp.sum( iTY * PXs[i])
            Flj.append(trxPx - 2 * trPXY + tryPy)

        # sigma2
        sigma2_expect =(sum( wlj * flj for wlj, flj in zip(wls, Flj))/
                        sum( wlj * jlj * self.D for wlj, jlj in zip(wls, Nps)))
        self.sigma2_expect[iL] = self.xp.clip(sigma2_expect, min=self.sigma2_min)

        if self.sigma2_sync:
            self.sigma2[iL, ixids] = sigma2_expect
            self.sigma2[iL, ixids] = self.xp.clip(sigma2_expect, min=self.sigma2_min)
        else:
            for xid, flj, jlj in zip(ixids, Flj, Nps):
                sigma2 = flj/jlj/self.D 
                self.sigma2[iL, xid] = sigma2
                # if sigma2 < self.sigma2_min: 
                #     sigma2 = self.xp.clip(self.sigma_square(iX, iTY), min=self.sigma2_min)
                self.sigma2[iL, xid] = self.xp.clip(sigma2, min=self.sigma2_min)  ## most for iL<->iL

        # Ql
        for xid, flj, jlj in zip(ixids, Flj, Nps):
            self.Ls[iL][xid] = flj/2.0/self.sigma2[iL, xid] + jlj * self.D/2.0 * self.xp.log(self.sigma2[iL, xid])
            # self.Ls[iL][xid] = jlj * self.D/2.0 *(1+self.xp.log(self.sigma2[iL, xid]))
            if Qp is not None:
                self.Ls[iL][xid] += Qp[i]

        # omega
        if self.kappa[iL] >0:
            Ql = self.Ls[iL]
            # wl1 = self.xp.exp(-Ql/self.kappa[iL] )[ixids]
            # wl1 = wl1/sum(wl1)
            wl = ( Ql/self.Ns/self.kappa[iL] )[ixids]
            wl = self.xp.nn.functional.softmax(-wl, dim=0)
            # self.omega_tmp[iL, self.OM[iL]] = wl
            self.omega[iL, self.OM[iL]] = wl
            # self.Qs[iL] = (self.omega[iL]*Ql).sum() + (self.kappa[iL]*wl*self.xp.log(wl)).sum()
            self.Qs[iL] = (self.omega[iL]*Ql/self.Nf[iL]).sum() + (self.kappa[iL]*wl*self.xp.log(wl)).sum()
        else:
            # self.Qs[iL] = (self.omega[iL]*self.Ls[iL]).sum()
            self.Qs[iL] = (self.omega[iL]*self.Ls[iL]/self.Nf[iL]).sum()

    def update_normalize(self):
        device, dtype = 'cpu', self.floatxx
        for iL in range(self.L):
            itf = self.transformer[iL]
            iTM = self.tmats[iL]
            iTM = { k: v.to(device, dtype=dtype) for k,v in iTM.items() }

            if itf in 'ESARTILO':
                iTM['tmat'] = self.homotransform_mat(itf, self.D, xp=self.xp, 
                                device=device, dtype=dtype, **iTM )
                if itf in ['E']:
                    A = iTM['R'] * iTM['s']
                elif itf in 'SRTILO':
                    A = iTM['R'] @ iTM['s']
                elif itf in ['A']:
                    A = iTM['A']
                else:
                    raise(f"Unknown transformer: {itf}")

                if not self.transparas[iL].get('fix_t' , False):
                    iTM['t'] = (iTM['t']*self.Xs[iL].to(device, dtype=dtype) - self.Xm[iL].to(device, dtype=dtype) @ A.T)
                if self.root is not None:
                    iTM['t'] += self.Xm[self.root].to(device, dtype=dtype)

                iTM['tform'] = self.homotransform_mat(itf, self.D, xp=self.xp, device=device, dtype=dtype, **iTM )
                self.TYs[iL] = self.Xr[iL].to(device, dtype=dtype) @ A.T + iTM['t']

            elif itf in ['D']:
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
                
                if self.root is not None:
                    iTM['Xm'] = self.Xm[self.root].to(device, dtype=dtype)
                else:
                    iTM['Xm'] = 0

                self.TYs[iL] = self.TYs[iL].to(device, dtype=dtype) * iTM['Xs'] + iTM['Xm']

            iTM['transformer'] = itf
            self.tmats[iL] = iTM
        # TYs = self.transform_points(self.Xr)

    def transform_points(self, Ys=None, use_keops=True, **kargs):
        TYs = []
        for iL in range(self.L):
            if self.tmats[iL]['transformer'] in 'ESARTILO':
                iTY = self.homotransform_point(Ys[iL], self.tmats[iL]['tform'], inverse=False, xp=self.xp)
            elif self.tmats[iL]['transformer']  == 'D':
                iTY =  self.ccf_deformable_transform_point( Ys[iL], **self.tmats[iL], 
                                                           use_keops=use_keops, **kargs)
            else:
                raise(f"Unknown transformer: {self.tmats[iL]['transformer']}")
            TYs.append(iTY)
        return TYs
