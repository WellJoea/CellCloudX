import torch as th
import numpy as np
from tqdm import tqdm

from .operation_expectation import ( features_pdist2,
                                    expectation_ko, expectation_xp)

from .manifold_regularizers import rfdeformable_regularizer
from ...transform import homotransform_points, homotransform_point, ccf_deformable_transform_point, homotransform_mat


from torch.utils.data import Dataset
from ...utilis._clean_cache import clean_cache

class rfRegularizer_Dataset(): #Dataset
    def __init__(self, Ys, delta, transparas, transformer,
                  verbose=0,
                  device=None, dtype=th.float32):
        self.transformer= transformer
        self.L = len(Ys)
        self.delta = delta
        assert self.L == len(transparas) and self.L == len(delta)

        DL = sum([ 'D' in i for i in self.transformer ])
        if DL == 0:
            pass
        else:
            try: #TODO: parallel
                with tqdm(total=self.L*self.L, 
                            desc="rfRegularizer",
                            colour='#AAAAAA', 
                            disable=(verbose==0)) as pbar:
                    for iL in range(self.L):
                        if not 'D' in self.transformer[iL]: continue
                        idr = rfdeformable_regularizer(xp = th, verbose=False, **transparas[iL])
                        for j in range(self.L):
                            pbar.set_postfix(dict(i=int(iL), j=int(j)))
                            pbar.update()

                            if iL==j:
                                idr.compute(Ys[iL][:,:-1], device=device, dtype=dtype) # exclude the z-axis
                                for ia in ['G', 'U', 'S', 'I', 'J1', 'J2', 'J3', 'E', 'F', ]:
                                    setattr(self, f'{iL}_{ia}', getattr(idr, ia))
                            else:
                                if self.delta[iL,j] > 0:
                                    H1 = np.array(transparas[iL]['beta'], dtype=np.float64)
                                    H2 = np.array(transparas[j]['beta'], dtype=np.float64)
                                    if (hasattr(self, f'{j}_{iL}_US') 
                                        and np.all( H1 == H2 )):
                                        US = getattr(self, f'{j}_{iL}_Vh').T
                                        Vh = getattr(self, f'{j}_{iL}_US').T
                                    else:
                                        U, S, Vh = idr.compute_pair(Ys[iL][:,:-1], Ys[j][:,:-1], # exclude the z-axis
                                                                  device=device, dtype=dtype)
                                        US = U * S
                                else:
                                    US, Vh = None, None
                                setattr(self, f'{iL}_{j}_US', US)
                                setattr(self, f'{iL}_{j}_Vh', Vh)
            except:
                del idr
                self.__dict__.clear()
                clean_cache()
                raise ValueError('Failed to compute the deformable regularizer matrix. Check the Memory.')
    
    def __len__(self):
        return self.L

    def __getitem__(self, idx): #TODO
        pass 

class rfCompute_Feauture(object): #TODO
    def __init__(self, keops_thr=None, xp=th, verbose=0):
        self.keops_thr = keops_thr
        self.verbose = verbose
        self.xp = xp
        self.features_pdist2 = features_pdist2

    def compute_pairs(self, FX, FY, tau2, tau2_prediv=None, device=None, dtype=None):
        if FX is None or FY is None:
            return
        elif all([ ifx is None for ifx in FX]) or all([ ify is None for ify in FY]):
            return

        L = len(FX)
        if tau2_prediv is None:
            tau2_prediv = [True] * L

        try:
            with tqdm(total=L, 
                        desc="feature fusion",
                        colour='#AAAAAA', 
                        disable=(self.verbose==0)) as pbar:  #TODO: parallel
                for iL in range(L):
                    pbar.set_postfix(dict(iL=int(iL)))
                    pbar.update()

                    iFx, iFy = FX[iL], FY[iL]
                    if type(self.keops_thr) in [bool]:
                        use_keops = self.keops_thr
                    else:
                        if iFx.shape[0] * iFy.shape[0] > self.keops_thr:
                            use_keops = True
                        else:
                            use_keops = False
                    fd = features_pdist2(iFx, iFy,
                                        tau2[iL], 
                                        use_keops=use_keops, 
                                        tau2_prediv=tau2_prediv[iL],
                                        device=device, dtype=dtype)
                    setattr(self,  f'f{iL}', fd)

        except:
            self.__dict__.clear()
            clean_cache()
            raise ValueError('Failed to compute the feature distance matrix. Check the Memory.')
            
class rfOptimization(): #th.nn.Module  #TODO: parallel
    def __init__(self):
        super().__init__()
        self.homotransform_points = homotransform_points
        self.ccf_deformable_transform_point = ccf_deformable_transform_point
        self.homotransform_mat = homotransform_mat
        self.homotransform_point = homotransform_point

        # self.DR = DR
        # self.use_keops = use_keops
        # self.feat_normal = feat_normal
        # self.w_clip = w_clip
        # self.was = was
        # self.fexist = fexist
        # self.tau2_auto = tau2_auto
        # self.device = device
        # self.DK = DK
        # self.FDs = FDs
        # self.sigma2_min = sigma2_min

    def optimization(self):
        for iL in range(self.L):
            if self.transformer[iL] == 'D':
                W, t_c, TY, sigma2, iQ, tau2, w = \
                    self.forward(iL, self.iteration, tcs=self.tcs, Ws =self.Ws)
                self.Ws_tmp[iL] = W
                self.tcs_tmp[iL] = t_c
                self.tmats[iL]['W'] = W
                self.tmats[iL]['tc'] = t_c
            else:
                A_ab, s_ab, t_ab, t_c, TY, sigma2, iQ, tau2, w = \
                    self.forward(iL, self.iteration,  tcs=self.tcs, As=self.As, tabs=self.tabs)
                self.As_tmp[iL] = A_ab * s_ab
                self.tabs_tmp[iL] = t_ab
                self.tcs_tmp[iL] = t_c

                if self.transformer[iL] == 'A':
                    self.tmats[iL]['A'] = A_ab
                    self.tmats[iL]['t'] = t_ab
                    self.tmats[iL]['tc'] = t_c
                else:
                    self.tmats[iL]['R'] = A_ab
                    self.tmats[iL]['s'] = s_ab
                    self.tmats[iL]['t'] = t_ab
                    self.tmats[iL]['tc'] = t_c

            self.Qs[iL] = iQ
            self.tau2[iL] = tau2
            self.sigma2[iL] = sigma2
            self.ws[iL] = w
            self.TYs[iL] = TY

        if (self.iteration + 1) % self.inneriter == 0: 
            if ('D' in self.transformer):
                self.Ws = [ iW.clone() for iW  in self.Ws_tmp ]
            else:
                self.As = self.As_tmp.clone()
                self.tabs = self.tabs_tmp.clone()
            self.tcs = self.tcs_tmp.clone()

        qprev = self.Q
        self.Q = self.xp.sum(self.Qs)
        self.diff = self.xp.abs(self.Q - qprev) #/self.Q
        self.iteration += 1
    
    def forward(self, iL, iteration, Ws=None, tcs=None, As =None, tabs=None,  ):
        X, Y, TY, XF, YF = self.Xs[iL], self.Ys[iL], self.TYs[iL], self.XFs[iL], self.YFs[iL]
        sigma2, tau2, tau2_alpha, w, wa = self.sigma2[iL], self.tau2[iL], self.tau2_grow[iteration], self.ws[iL], self.was[iL] 
        d2f, itranspara, itransformer = getattr(self.FDs, f'f{iL}', None), self.transparas[iL], self.transformer[iL]
        
        if (d2f is None) and self.fexist:
            raise ValueError(f'Please Check feature fusion with {iL}.')
    
        delta, zeta, eta = self.delta[iL], self.zeta[iL], self.eta[iL]

        Pt1, P1, PX, Np, tau2, w = \
            self.expectation(self, X, TY, XF, YF, sigma2, tau2, tau2_alpha, d2f, w, wa)
        if itransformer == 'D':
            W, t_c, TY, sigma2, iQ = \
                self.maximization(iL, Pt1, P1, PX, Np, X, Y, sigma2, delta,  eta, Ws, tcs, 
                        iteration, itranspara, transformer=itransformer)
            return W, t_c, TY, sigma2, iQ, tau2, w
        else:
            A_ab, s_ab, t_ab, t_c, TY, sigma2, iQ = \
                self.maximization(iL, Pt1, P1, PX, Np, X, Y, sigma2, delta, zeta, eta, As, tabs, tcs,
                                   itranspara, transformer=itransformer)
            return A_ab, s_ab, t_ab, t_c, TY, sigma2, iQ, tau2, w

    @th.no_grad()
    def expectation(self, iL, X, TY, XF, YF, sigma2, tau2, tau2_alpha, d2f, w, wa):
        D, Ni, Mi = X.shape[1], X.shape[0], TY.shape[0]

        # isigma2, tau2, tau2_alpha  = self.sigma2[iL], self.tau2[iL], self.tau2_grow[self.iteration]
        # gs = Mi/Ni*self.ws[iL]/(1-self.ws[iL])
        # d2f= getattr(self.FDs, f'f{iL}')
        gs = Mi/Ni*w/(1-w)

        if  not type(self.use_keops) in [bool, np.bool]:
            use_keops = True if Ni * Mi > self.use_keops else False
        else:
            use_keops = self.use_keops
        iexpectation = expectation_ko if use_keops else expectation_xp

        iPt1, iP1, iPX, iNp, tau2 = \
            iexpectation(X, TY, sigma2, gs, 
                                    XF=XF, YF=YF,
                                    d2f=d2f, tau2=tau2, 
                                    tau2_alpha=tau2_alpha, 
                                    tau2_auto=self.tau2_auto, 
                                    DF=self.DK, 
                                    feat_normal=self.feat_normal,
                                    xp=self.xp,
                                    device=self.device)

        wn = (1- iNp/Ni).clip(*self.w_clip)
        w = wa*w + (1-wa)*wn
        return iPt1, iP1, iPX, iNp, tau2, w

    @th.no_grad()
    def maximization(self, *args, transformer='E', **kwargs):
        if transformer == 'E':
            return self.rigid_maximization(*args, **kwargs)
        elif transformer in 'SRTILO':
            return self.similarity_maximization(*args, **kwargs)
        elif transformer == 'A':
            return self.affine_maximization(*args, **kwargs)
        elif transformer == 'D':
            return self.deformable_maximization(*args, **kwargs)
        else:
            raise(f"Unknown transformer: {transformer}")

    @th.no_grad()
    def rigid_maximization(self, iL, Pt1, P1, PX, Np, X, Y, sigma2, delta, zeta, eta, As, tabs, tcs, itranspara):
        M, D = Y.shape[0], X.shape[1]
        iY = Y[:,:D-1]
    
        Jab = Np + sigma2 * zeta.sum()
        Jc  = Np + sigma2 * eta.sum()

        TabE = sigma2 * (zeta @ tabs)
        TcE  = sigma2 * (eta @ tcs)

        muXab = th.divide(th.sum(PX[:,:D-1], axis=0)+TabE, Jab)
        muXc = th.divide(th.sum(PX[:,D-1])+TcE, Jc)
        muY = th.divide(iY.T @ P1, Jab)

        # X_hat_ab = X[:,:D-1] - muXab
        # X_hat_c =  X[:,D-1] - muXc
        Y_hat = iY - muY

        B = PX[:,:D-1].T @ Y_hat - th.outer(muXab, P1 @ Y_hat)
        B += sigma2*th.einsum('l,lij->ij', delta, As)


        U, S, V = th.linalg.svd(B, full_matrices=True)
        C = th.eye(D-1, dtype=B.dtype, device=B.device)
        C[-1, -1] = th.linalg.det(U @ V)
        R = U @ C @ V
        if th.linalg.det(R) < 0:
            U[:, -1] = -U[:, -1]
            R = U @ C @ V

        if itranspara['fix_s'] is False:
            H = (Y_hat.T * P1) @ Y_hat
            H.diagonal().add_(sigma2 * delta.sum())
            trAR = th.trace(B.T @ R)
            trH = th.trace(H)
            s = trAR/trH
        else:
            s = th.ones_like(Np)
        if itranspara['s_clip'] is not None:
            s = th.clip(s, *itranspara['s_clip'])
        
        A_ab = R * s
        t_ab = muXab - A_ab @ muY
        t_c  = muXc
        TY   = th.hstack([iY @ A_ab.T + t_ab, t_c.expand(M,1)])

        trXPX = th.sum( Pt1 * th.sum(X*X, axis=1) )
        trYPY = th.sum( P1  * th.sum(TY * TY, axis=1))
        trXPY = th.sum( TY * PX)
        if self.use_projection: #TODO
            sigma2 = (trXPX - 2*trXPY + trYPY) / (Np * (D))
        else:
            sigma2 = (trXPX - 2*trXPY + trYPY) / (Np * (D-1))
        sigma2 = th.clip(sigma2, self.sigma2_min, None)

        iQ = D * Np/2 *(1+ th.log(sigma2))  \
                + th.einsum('l,lij->', delta, (A_ab - As)**2)/2 \
                + th.einsum('l,li->',  zeta, (t_ab - tabs)**2)/2 \
                + th.einsum('l,l->',   eta, (t_c - tcs)**2)/2

        return R, s, t_ab, t_c, TY, sigma2, iQ

    @th.no_grad()
    def similarity_maximization(self, iL, Pt1, P1, PX, Np, X, Y, sigma2, delta, zeta, eta, As, tabs, tcs, itranspara):
        M, D = Y.shape[0], X.shape[1]
        iY = Y[:,:D-1]
    
        Jab = Np + sigma2 * zeta.sum()
        Jc  = Np + sigma2 * eta.sum()

        TabE = sigma2 * (zeta @ tabs)
        TcE  = sigma2 * (eta @ tcs)

        muXab = th.divide(th.sum(PX[:,:D-1], axis=0)+TabE, Jab)
        muXc = th.divide(th.sum(PX[:,D-1])+TcE, Jc)
        muY = th.divide(iY.T @ P1, Jab)

        # X_hat_ab = X[:,:D-1] - muXab
        # X_hat_c =  X[:,D-1] - muXc
        Y_hat = iY - muY

        B = PX[:,:D-1].T @ Y_hat - th.outer(muXab, P1 @ Y_hat)
        B += sigma2*th.einsum('l,lij->ij', delta, As)

        if itranspara['isoscale'] :
            U, S, V = th.linalg.svd(B, full_matrices=True)
            C = th.eye(D-1, dtype=B.dtype, device=B.device)
            C[-1, -1] = th.linalg.det(U @ V)
            R = U @ C @ V
            if th.linalg.det(R) < 0:
                U[:, -1] = -U[:, -1]
                R = U @ C @ V

            if itranspara['fix_s'] is False:
                H = (Y_hat.T * P1) @ Y_hat
                H.diagonal().add_(sigma2 * delta.sum())
                trAR = th.trace(B.T @ R)
                trH = th.trace(H)
                s = trAR/trH
            else:
                s = th.ones_like(Np)
            if itranspara['s_clip'] is not None:
                s = th.clip(s, *itranspara['s_clip'])
        else:
            H = (Y_hat.T * P1) @ Y_hat
            H.diagonal().add_(sigma2 * delta.sum())
            max_iter = 70
            error = True
            iiter = 0
            s, R = th.ones(D-1, dtype=B.dtype, device=B.device), th.eye(D-1, dtype=B.dtype, device=B.device)
            while (error):
                U, S, V = th.linalg.svd(B * s, full_matrices=True)
                C = th.eye(D-1, dtype=B.dtype, device=B.device)
                C[-1, -1] = th.linalg.det(U @ V)
                R_pre = R.clone()
                R = U @ C @ V
                if th.linalg.det(R) < 0:
                    U[:, -1] = -U[:, -1]
                    R = U @ C @ V

                s = th.diagonal(B @ R.T)/th.diagonal(H)
                if itranspara['s_clip'] is not None:
                    s = th.clip(s, *itranspara['s_clip'])

                iiter += 1
                error = (th.dist(R, R_pre) > 1e-8) and (iiter < max_iter)

        A_ab = R * s
        t_ab = muXab - A_ab @ muY
        t_c  = muXc
        TY   = th.hstack([iY @ A_ab.T + t_ab, t_c.expand(M,1)])

        trXPX = th.sum( Pt1 * th.sum(X*X, axis=1) )
        trYPY = th.sum( P1  * th.sum(TY * TY, axis=1))
        trXPY = th.sum( TY * PX)
        if self.use_projection: #TODO
            sigma2 = (trXPX - 2*trXPY + trYPY) / (Np * (D))
        else:
            sigma2 = (trXPX - 2*trXPY + trYPY) / (Np * (D-1))
        sigma2 = th.clip(sigma2, self.sigma2_min, None)

        iQ = D * Np/2 *(1+ th.log(sigma2))  \
                + th.einsum('l,lij->', delta, (A_ab - As)**2)/2 \
                + th.einsum('l,li->',  zeta, (t_ab - tabs)**2)/2 \
                + th.einsum('l,l->',   eta, (t_c - tcs)**2)/2

        return R, s, t_ab, t_c, TY, sigma2, iQ

    @th.no_grad()
    def affine_maximization(self, iL, Pt1, P1, PX, Np, X, Y, sigma2, delta, zeta, eta, As, tabs, tcs, itranspara):
        M, D = Y.shape[0], X.shape[1]
        iY = Y[:,:D-1]
    
        Jab = Np + sigma2 * zeta.sum()
        Jc  = Np + sigma2 * eta.sum()

        TabE = sigma2 * (zeta @ tabs)
        TcE  = sigma2 * (eta @ tcs)

        muXab = th.divide(th.sum(PX[:,:D-1], axis=0)+TabE, Jab)
        muXc = th.divide(th.sum(PX[:,D-1])+TcE, Jc)
        muY = th.divide(iY.T @ P1, Jab)

        # X_hat_ab = X[:,:D-1] - muXab
        # X_hat_c =  X[:,D-1] - muXc
        Y_hat = iY - muY

        B = PX[:,:D-1].T @ Y_hat - th.outer(muXab, P1 @ Y_hat)
        H = (Y_hat.T * P1) @ Y_hat

        B += sigma2*th.einsum('l,lij->ij', delta, As)
        H.diagonal().add_(sigma2 * delta.sum())

        try:
            A_ab = B @ th.linalg.inv(H)
        except:
            B.diagonal().add_(0.001*sigma2)
            H.diagonal().add_(0.001*sigma2)
            A_ab = B @ th.linalg.inv(H)
        t_ab = muXab - A_ab @ muY
        t_c  = muXc
        TY   = th.hstack([iY @ A_ab.T + t_ab, t_c.expand(M,1)])

        trXPX = th.sum( Pt1 * th.sum(X*X, axis=1) )
        trYPY = th.sum( P1  * th.sum(TY * TY, axis=1))
        trXPY = th.sum( TY * PX)
        if self.use_projection: #TODO
            sigma2 = (trXPX - 2*trXPY + trYPY) / (Np * (D))
        else:
            sigma2 = (trXPX - 2*trXPY + trYPY) / (Np * (D-1))
        sigma2 = th.clip(sigma2, self.sigma2_min, None)

        iQ = D * Np/2 *(1+ th.log(sigma2))  \
                + th.einsum('l,lij->', delta, (A_ab - As)**2)/2 \
                + th.einsum('l,li->',  zeta, (t_ab - tabs)**2)/2 \
                + th.einsum('l,l->',   eta, (t_c - tcs)**2)/2

        return A_ab, 1, t_ab, t_c, TY, sigma2, iQ

    @th.no_grad()
    def deformable_maximization(self, iL, Pt1, P1, PX, Np, X, Y, sigma2, delta,  eta, Ws, tcs, iteration, itranspara):
        mu = itranspara['alpha_mu']**(iteration)
        nu1 = nu2 = itranspara['gamma_nu']**(iteration)
        alpha = itranspara['alpha'] * mu
        gamma1 = itranspara['gamma1'] * nu1
        gamma2 = itranspara['gamma2'] * nu2
        p1_thred = itranspara['p1_thred']
        use_p1 = itranspara['use_p1']
   
        M, D = Y.shape[0], X.shape[1]

        Jc  = Np + sigma2 * eta.sum()
        TcE  = sigma2 * (eta @ tcs)
        muXc = th.divide(th.sum(PX[:,D-1])+TcE, Jc)
        t_c = muXc

        J1 = getattr(self.DR, f'{iL}_J1')
        J2 = getattr(self.DR, f'{iL}_J2')
        J3 = getattr(self.DR, f'{iL}_J3')
        E = getattr(self.DR, f'{iL}_E')
        F  = getattr(self.DR, f'{iL}_F')
        U = getattr(self.DR, f'{iL}_U')
        S = getattr(self.DR, f'{iL}_S')
        I = getattr(self.DR, f'{iL}_I')
        G = getattr(self.DR, f'{iL}_G')
        
        iY = Y[:,:D-1]
        B = PX[:,:D-1] - P1[:,None] * iY
        if itranspara['fast_rank']:
            V = (U @ S)
            H_del = 0
            for i, idel in enumerate(delta):
                if idel >0:
                    if (iL == i):
                        H_del += idel* (V @ (U.T @ Ws[iL]))
                    elif (iL != i):
                        Us = getattr(self.DR, f'{iL}_{i}_US')
                        Vh = getattr(self.DR, f'{iL}_{i}_Vh')
                        H_del += idel*(Us @ (Vh @ Ws[i]))

            B += sigma2*H_del
            Pg = (P1 + sigma2*delta.sum())[:,None] * V
            if (use_p1):
                P1_thred = th.where(P1 < p1_thred, 0.0, P1)
                if gamma1 > 0 :
                    P2 = (sigma2* gamma1) * (E.T * P1_thred[None,:])
                    B  -= P2 @ J3
                    Pg += P2 @ J1
                if gamma2 > 0 :
                    P3 =  (sigma2* gamma2) * (F.T * P1_thred[None,:])
                    Pg += P3 @ J2
            else:
                if gamma1 > 0 :
                    P2 = (sigma2* gamma1)
                    B  -= P2 * J3
                    Pg += P2 * J1
                if gamma2 > 0 :
                    P3 =  (sigma2* gamma2)
                    Pg += P3 * J2
            
            B  *= 1.0/(sigma2*alpha)
            Pg *= 1.0/(sigma2*alpha)
            W  = B - Pg @ th.linalg.solve(I + U.T @ Pg, U.T @ B)
            H  = V @ (U.T @ W)
        else:
            H_del = 0
            for i, idel in enumerate(delta):
                if idel >0:
                    if (iL == i):
                        H_del += idel*(G @ Ws[iL])
                    elif (iL != i):
                        Us = getattr(self.DR, f'{iL}_{i}_US')
                        Vh = getattr(self.DR, f'{iL}_{i}_Vh')
                        H_del += idel*(Us @ (Vh @ Ws[i]))

            B += sigma2 * H_del
            Pg = (P1 + sigma2*delta.sum())[:,None]*G
            if (use_p1):
                P1_thred = th.where(P1 < p1_thred, 0.0, P1)
                if gamma1 > 0 :
                    P2 = (sigma2* gamma1) * (E.T * P1_thred[None,:])
                    B  -= P2 @ J3
                    Pg += P2 @ J1
                if gamma2 > 0 :
                    P3 =  (sigma2* gamma2) * (F.T * P1_thred[None,:])
                    Pg += P3 @ J2
            else:
                if gamma1 > 0 :
                    P2 = (sigma2* gamma1)
                    B  -= P2 * J3
                    Pg += P2 * J1
                if gamma2 > 0 :
                    P3 =  (sigma2* gamma2)
                    Pg += P3 * J2
            Pg.diagonal().add_(sigma2 *alpha)
            W = th.linalg.solve(Pg, B)
            H = G @ W
        
        TY   = th.hstack([ iY + H, t_c.expand(M,1) ])

        trXPX = th.sum( Pt1 * th.sum(X*X, axis=1) )
        trYPY = th.sum( P1  * th.sum(TY * TY, axis=1))
        trXPY = th.sum( TY * PX)

        if self.use_projection: #TODO
            sigma2 = (trXPX - 2*trXPY + trYPY) / (Np * (D))
        else:
            sigma2 = (trXPX - 2*trXPY + trYPY) / (Np * (D-1))
        sigma2 = th.clip(sigma2, self.sigma2_min, None)
    
        # if self.use_projection:
        #     trxPx = th.sum( Pt1 * th.sum(iX **2, axis=1) )
        #     tryPy = th.sum( P1 * th.sum( TY **2, axis=1))
        #     trPXY = th.sum( TY * PX)
        #     sigma2 = (trxPx - 2 * trPXY + tryPy) / (Np * self.D)
        # else:
        #     trxPx = th.sum( Pt1 * th.sum(iX[:,:self.D-1] **2, axis=1) )
        #     tryPy = th.sum( P1 * th.sum( TY[:,:self.D-1] **2, axis=1))
        #     trPXY = th.sum( TY[:,:self.D-1] * PX[:,:self.D-1])
        #     sigma2 = (trxPx - 2 * trPXY + tryPy) / (Np * (self.D-1))

        if False:
            # PASS:TODO
            LV = self.delta[iL, iL] * th.sum( ( (Q * S) @ (Q.T @ (W -self.W[iL])))**2 )
            for i in range(self.L):
                if hasattr(self.Gs, f'US{iL}{i}'):
                    Us = getattr(self.Gs, f'US{iL}{i}')
                    Vh = getattr(self.Gs, f'Vh{iL}{i}')
                    LV += self.delta[iL, i] * th.sum( (V - Us @ (Vh @ self.W[i]))**2 )
            PV = alpha/2 * xp.trace(W.T @ H) 
        else:
            LV = 0
            PV = 0
        iQ = D * Np/2 *(1+ th.log(sigma2)) + PV + LV

        return W, t_c, TY, sigma2, iQ