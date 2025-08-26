import numpy as np
import torch as th
from tqdm import tqdm
import itertools

from .operation_expectation import ( features_pdist2, init_tmat, expectation_ko, expectation_xp,)
from .manifold_regularizers import pwdeformable_regularizer
from ...transform import homotransform_points, homotransform_point, ccf_deformable_transform_point, homotransform_mat

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
        self.homotransform_mat = homotransform_mat
        self.homotransform_point = homotransform_point

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
                idr = pwdeformable_regularizer(xp = self.xp, verbose=False, **self.transparas[iL]) # deformable_regularizer v0
                if ('D' in self.transformer[iL]) and (self.omega[iL].sum() > 0):
                    idr.compute(self.Xa[iL], device=self.device, dtype=self.floatx)
                self.DR[iL] = idr
            pbar.close()
        except:
            del self.DR
            self.clean_cache()
            raise ValueError('Failed to compute the deformable regularizer matrix. Check the Memory.')

    @th.no_grad()
    def expectation(self, iL, Xs, TYs, Ys,):
        yid, xids = iL, self.OM[iL]

        wl = self.omega[iL][self.OM[iL]]
        
        if self.omega_normal:
            wl = wl/self.xp.sum(wl)

        Pt1s, P1s, PXs, Nps, ols = [],[],[],[],[]
        iTY, iY, iXs = TYs[yid], Ys[yid], [] # TODO sample for iTY will increase the complexity of the deformable.

        for wlj, xid in zip(wl, xids):
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

            iexpectation = expectation_ko if self.keops_thr[yid, xid] else expectation_xp
            iPt1, iP1, iPX, iNp, tau2s = iexpectation(
                    iX, iTY, self.sigma2[yid, xid], gs, #sigma2_expect[yid]
                    d2f=d2f, tau2=self.tau2[yid, xid], 
                    tau2_auto=self.tau2_auto[iL], eps=self.eps, 
                    tau2_alpha=self.tau2_decay[iL]**self.iteration, DF=self.DK,
                    feat_normal=self.feat_normal, 
                    XF=xFa, YF=yFa,
                    device=self.device,
                    xp = self.xp )
        
            self.tau2[yid, xid] = tau2s
            w = (1- iNp/Ni).clip(*self.w_clip)
            self.ws[yid, xid] = self.was[yid, xid] *self.ws[yid, xid] + (1-self.was[yid, xid])*w
            # olj = wlj/2.0/self.sigma2[yid, xid] # TODO v0
            olj = wlj/self.sigma2_expect[yid] #self.sigma2_expect

            Pt1s.append(iPt1)
            P1s.append(iP1)
            PXs.append(iPX)
            Nps.append(iNp)
            ols.append(olj)
            iXs.append(iX)
        return Pt1s, P1s, Nps, PXs, ols, iY, iXs

    @th.no_grad()
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

    @th.no_grad()
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

    @th.no_grad()
    def rigid_maximization(self, iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs):
        itranspara = self.transparas[iL]
        p_version = self.p_version
        delta = self.delta[iL]
        if p_version == 'v1':
            Af = self.Af
            tf = self.tf
            zeta = self.zeta[iL]

        # sigma2 = self.sigma2[iL][self.OM[iL]]
        # ols = [ i/j for i, j in zip(ols,sigma2) ]

        PX = sum( iPX * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
        Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        if p_version == 'v2':
            Np += delta
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
            B = PX.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat) + delta*Af
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
                H += delta
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
    def similarity_maximization(self, iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs):
        itranspara = self.transparas[iL]
        p_version = self.p_version
        delta = self.delta[iL]
        if p_version == 'v1':
            Af = self.Af
            tf = self.tf
            zeta = self.zeta[iL]

        PX = sum( iPX * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
        Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        if p_version == 'v2':
            if not itranspara['fix_t']:
                Np += delta
                my1 = delta* iY.sum(0)
                muX = self.xp.divide(self.xp.sum(PX, 0)+ my1, Np)
                muY = self.xp.divide(iY.T @ P1 + my1, Np)
                Y_hat =  iY - muY
                U_hat =  iY - muX
            else:
                muX = self.xp.zeros(self.D, dtype=self.floatx, device=self.device)
                Y_hat =  iY
                U_hat =  iY
            UhtYh = delta * (U_hat.T @ Y_hat)
            B = PX.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat) + UhtYh
            
        elif self.p_version == 'v1':
            if not itranspara['fix_t']:
                Np += zeta
                muX = self.xp.divide(self.xp.sum(PX, 0)+ zeta*tf, Np)
                muY = self.xp.divide(iY.T @ P1, Np)
                Y_hat =  iY - muY
            else:
                muX = self.xp.zeros(self.D, dtype=self.floatx, device=self.device)
                Y_hat =  iY
            B = PX.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat) + delta*Af
        else:
            if not itranspara['fix_t']:
                muX = self.xp.divide(self.xp.sum(PX, 0), Np)
                muY = self.xp.divide(iY.T @ P1, Np)
            else:
                muX = self.xp.zeros(self.D, dtype=self.floatx, device=self.device)
                muY = self.xp.zeros(self.D, dtype=self.floatx, device=self.device)
            Y_hat  =  iY - muY
            B = PX.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat)

        ## RS
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
                H = self.xp.sum(P1 * self.xp.sum(Y_hat * Y_hat, 1))
                if p_version == 'v2':
                    H += delta * (Y_hat**2).sum()
                elif p_version == 'v1':
                    H += delta
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
    def affine_maximization(self, iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs):
        itranspara = self.transparas[iL]
        p_version = self.p_version
        delta = self.delta[iL]
        if p_version == 'v1':
            Af = self.Af
            tf = self.tf
            zeta = self.zeta[iL]

        PX = sum( iPX * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
        Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        if p_version == 'v2':
            Np += delta
            my1 = delta* iY.sum(0)
            muX = self.xp.divide(self.xp.sum(PX, 0)+ my1, Np)
            muY = self.xp.divide(iY.T @ P1 + my1, Np)

            Y_hat =  iY - muY
            U_hat =  iY - muX
            UhtYh = delta * (U_hat.T @ Y_hat)
            YhtYh = delta * (Y_hat.T @ Y_hat)

            B = PX.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat) + UhtYh
            H = (Y_hat.transpose(1,0) * P1) @ Y_hat + YhtYh
    
        elif self.p_version == 'v1':
            Np += zeta
            muX = self.xp.divide(self.xp.sum(PX, 0)+ zeta*tf, Np)
            muY = self.xp.divide(iY.T @ P1, Np)
            Y_hat =  iY - muY
            B = PX.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat) + delta*Af
            H = (Y_hat.transpose(1,0) * P1) @ Y_hat
            H.diagonal().add_(delta)
        else:
            muX = self.xp.divide(self.xp.sum(PX, 0), Np)
            muY = self.xp.divide(iY.T @ P1, Np)
            Y_hat  =  iY - muY
            B = PX.T @ Y_hat - self.xp.outer(muX, P1 @ Y_hat)
            H = (Y_hat.transpose(1,0) * P1) @ Y_hat

        try:
            Hv = self.xp.linalg.inv(H)
            A = B @ Hv
        except:
            A_pre = self.tmats[iL].get('A', self.xp.zeros((self.D, self.D), dtype=self.floatx, device=self.device))
            H.diagonal().add_(itranspara['delta']*self.sigma2_expect[iL])
            A = (B + itranspara['delta']*self.sigma2_expect[iL]*A_pre) @ self.xp.linalg.inv(H)

        t = muX - A @ muY
        iTY = iY @ A.T + t
        self.tmats[iL]['A'] = A
        self.tmats[iL]['t'] = t
        self.TYs_tmp[iL] = self.Xa[iL] @ A.T + t

        self.update_sigma2(iL, iXs, Pt1s, P1s, PXs, Nps, iTY)
    
    @th.no_grad()
    def deformable_maximization(self, iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs):
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
        P1_hat = P1 + delta/wsa

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
    def deformable_maximization0(self, iL, Pt1s, P1s, Nps, PXs, ols, iY, iXs):
        itranspara = self.transparas[iL]
        delta = self.delta[iL]

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
            self.Qs[iL] = (self.omega[iL]*Ql/self.Nf[iL]).sum() + (self.kappa[iL]*wl*self.xp.log(wl)).sum() #TODO check Nf
        else:
            # self.Qs[iL] = (self.omega[iL]*self.Ls[iL]/self.Nf[iL]).sum()
            self.Qs[iL] = (self.omega[iL]*self.Ls[iL]).sum() #TODO check Nf

    @th.no_grad()
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
                    A = iTM['R'] * iTM['s']
                elif itf in ['A']:
                    A = iTM['A']
                else:
                    raise(f"Unknown transformer: {itf}")

                if not self.transparas[iL].get('fix_t' , False): # TODO
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