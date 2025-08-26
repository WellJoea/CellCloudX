import torch as th
import numpy as np
from tqdm import tqdm

from .operation_expectation import ( features_pdist2, sigma_square, expectation_ko, expectation_xp)

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

    def is_null(self, X):
        if X is None:
            return True
        elif len(X) == 0:
            return True
        elif isinstance(X, (list, tuple)):
            return all([ self.is_null(i) for i in X])
        else:
            return False
            
    def compute_pairs(self, FX, FY, tau2, tau2_prediv=None, device=None, dtype=None):
        if FX is None or FY is None:
            return
        elif self.is_null(FX) or self.is_null(FY):
            return

        L = len(FX)
        if tau2_prediv is None:
            tau2_prediv = [True] * L

        try:
            with tqdm(total=L, 
                        desc="Featurefusion",
                        colour='#AAAAAA', 
                        disable=(self.verbose==0)) as pbar:  #TODO: parallel
                for iL in range(L):
                    pbar.set_postfix(dict(iL=int(iL)))
                    pbar.update()

                    iFx, iFy = FX[iL], FY[iL]
                    if self.is_null(iFx) or self.is_null(iFy) :
                        continue

                    use_keops = bool(self.keops_thr[iL])
                    FL = len(iFx)
                    assert FL == len(iFy)

                    d2f = []
                    for ifl in range(FL):
                        if self.is_null(iFx[ifl]) or self.is_null(iFy[ifl]) :
                            fd = None
                        else:
                            fd = features_pdist2(iFx[ifl], iFy[ifl],
                                                tau2[iL][ifl], 
                                                use_keops=use_keops, 
                                                tau2_prediv=tau2_prediv[iL],
                                                device=device, dtype=dtype)
                        d2f.append(fd)
                    setattr(self,  f'f{iL}', d2f)

        except:
            self.__dict__.clear()
            clean_cache()
            raise ValueError('Failed to compute the feature distance matrix. Check the Memory.')
            
class rfEM_core(): #th.nn.Module  #TODO: parallel
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
                W, t_c, TY, sigma2_exp, sigma2, iQ = \
                    self.forward(iL, tcs=self.tcs, Ws =self.Ws)
                self.Ws_tmp[iL] = W
                self.tcs_tmp[iL] = t_c
                self.tmats[iL]['W'] = W
                self.tmats[iL]['tc'] = t_c
            else:
                A_ab, s_ab, t_ab, t_c, TY, sigma2_exp, sigma2, iQ = \
                    self.forward(iL, tcs=self.tcs, As=self.As, tabs=self.tabs)
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
            self.sigma2_exp[iL] = sigma2_exp
            self.sigma2[iL] = sigma2
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
    
    def forward(self, iL,  Ws=None, tcs=None, As =None, tabs=None,  ): # 
        Xs, Y, TY, XF, YF = self.Xa[iL], self.Ya[iL], self.TYs[iL], self.XFa[iL], self.YFa[iL]
        itransformer = self.transformer[iL]

        Pt1s, P1s, PXs, Nps = self.expectation(iL, Xs, TY, XF, YF)
        if itransformer == 'D':
            return self.maximization(iL, Pt1s, P1s, PXs, Nps, Xs, Y,  Ws, tcs,  transformer=itransformer)
        else:
            return self.maximization(iL, Pt1s, P1s, PXs, Nps, Xs, Y, As, tabs, tcs, transformer=itransformer)

    @th.no_grad()
    def expectation(self, iL, Xs, TY, XF, YF):
        min_points = 10
        XL, DF, feat_normal = self.XL, self.DK, self.feat_normal
        sigma2, tau2, tau2_auto = self.sigma2[iL], self.tau2[iL], self.tau2_auto[iL]
        w, wa = self.ws[iL], self.was[iL] 
        use_keops = self.keops_thr[iL]
        tau2_alpha = self.tau2_grow[self.iteration]
        d2f = getattr(self.FDs, f'f{iL}', [None]*XL)

        Pt1s, P1s, PXs, Nps = [], [], [], []
        for xl in range(XL):
            iX, iTY = Xs[xl], TY
            D, Ni, Mi = iX.shape[1], iX.shape[0], iTY.shape[0]
            if (Ni <= min_points) or (Mi <= min_points):
                iPt1, iP1, iNp, iPX = 0, 0, 0, self.xp.zeros( (1, D), device=self.device, dtype=self.floatx)
            else:
                gs = Mi/Ni*w[xl]/(1-w[xl])
                iexpectation = expectation_ko if use_keops else expectation_xp
                
                # TODO
                #use_samplex = bool(self.sample[xl]) and (self.iteration < self.sample_stopiter)
                #if use_samplex: 

                iPt1, iP1, iPX, iNp, tau2s = iexpectation( 
                        iX, iTY, sigma2[xl], gs,  #sigma2_exp
                        d2f=d2f[xl], 
                        tau2=tau2[xl], 
                        tau2_auto=tau2_auto, 
                        tau2_alpha=tau2_alpha, 
                        DF= DF[xl],
                        feat_normal=feat_normal[xl], 
                        XF=XF[xl], YF=YF[xl], 
                        device=self.device,
                        xp = self.xp, )

                nwi = (1- iNp/Ni).clip(*self.w_clip)
                self.ws[iL][xl] = wa[xl] *w[xl] + (1-wa[xl])*nwi
                self.tau2[iL][xl] = tau2s
            Pt1s.append(iPt1)
            P1s.append(iP1)
            PXs.append(iPX)
            Nps.append(iNp)
        return Pt1s, P1s, PXs, Nps

    @th.no_grad()
    def maximization(self, *args, transformer='E', **kwargs):
        if transformer == 'E':
            return self.rf_rigid_maximization(*args, **kwargs)
        elif transformer in 'SRTILO':
            return self.rf_similarity_maximization(*args, **kwargs)
        elif transformer == 'A':
            return self.rf_affine_maximization(*args, **kwargs)
        elif transformer == 'D':
            return self.rf_deformable_maximization(*args, **kwargs)
        else:
            raise(f"Unknown transformer: {transformer}")

    @th.no_grad()
    def rf_rigid_maximization(self, iL, Pt1s, P1s, PXs, Nps, Xs, Y, As, tabs, tcs):
        omega, delta, zeta, eta, tc_walk = self.omega[iL], self.delta[iL], self.zeta[iL], self.eta[iL], self.tc_walk
        sigma2, sigma2_exp, sigma2_min = self.sigma2[iL], self.sigma2_exp[iL], self.sigma2_min
        itranspara, itmat = self.transparas[iL], self.tmats[iL]

        XL, D, M = len(Nps), Y.shape[1], Y.shape[0]
        # ols = omega/sigma2 # raw
        ols = omega/sigma2_exp # exp 
        # ols = omega # same with not common constraint  exp #TODO

        PX_c = sum( iPX[:,D-1].sum() * iol for iPX, iol, iwalk in zip(PXs, ols, tc_walk) if iwalk ) 
        J_c = sum( iNp * iol for iNp, iol, iwalk in zip(Nps, ols, tc_walk) if iwalk ) 
        t_c  = th.divide(PX_c + (eta @ tcs), J_c + eta.sum())

        PX = sum( iPX[:,:D-1] * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
        Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        iY = Y[:,:D-1]
        Jab = Np + zeta.sum()
        TabE = (zeta @ tabs) #check
        muXab = th.divide(th.sum(PX, axis=0) +TabE, Jab)
        muY   = th.divide(iY.T @ P1, Jab)
        Y_hat = iY - muY

        B = PX.T @ Y_hat - th.outer(muXab, P1 @ Y_hat)
        B += th.einsum('l,lij->ij', delta, As)

        U, S, V = th.linalg.svd(B, full_matrices=True)
        C = th.eye(D-1, dtype=B.dtype, device=B.device)
        C[-1, -1] = th.linalg.det(U @ V)
        R = U @ C @ V
        if th.linalg.det(R) < 0:
            U[:, -1] = -U[:, -1]
            R = U @ C @ V

        if itranspara['fix_s'] is False:
            H = (Y_hat.T * P1) @ Y_hat
            H.diagonal().add_(delta.sum())
            trAR = th.trace(B.T @ R)
            trH = th.trace(H)
            s = trAR/trH
        else:
            s = itmat['s']
        if itranspara['s_clip'] is not None:
            s = th.clip(s, *itranspara['s_clip'])
        
        A_ab = R * s
        t_ab = muXab - A_ab @ muY
        TY   = th.hstack([iY @ A_ab.T + t_ab, t_c.expand(M,1)])

        Qp = th.einsum('l,lij->', delta, (A_ab - As)**2)/2 \
                + th.einsum('l,li->',  zeta, (t_ab - tabs)**2)/2 \
                + th.einsum('l,l->',   eta, (t_c - tcs)**2)/2
        sigma2_exp, sigma2, iQ = self.update_sigma2(iL, Pt1s, P1s, Nps, PXs, Xs, TY, Qp = Qp)
        return R, s, t_ab, t_c, TY, sigma2_exp, sigma2, iQ

    @th.no_grad()
    def rf_similarity_maximization(self, iL, Pt1s, P1s, PXs, Nps, Xs, Y, As, tabs, tcs):
        omega, delta, zeta, eta, tc_walk = self.omega[iL], self.delta[iL], self.zeta[iL], self.eta[iL], self.tc_walk
        sigma2, sigma2_exp, sigma2_min = self.sigma2[iL], self.sigma2_exp[iL], self.sigma2_min
        itranspara, itmat = self.transparas[iL], self.tmats[iL]

        XL, D, M = len(Nps), Y.shape[1], Y.shape[0]

        # ols = omega/sigma2 # raw
        ols = omega/sigma2_exp # exp 
        # ols = omega # same with not common constraint  exp #TODO

        PX_c = sum( iPX[:,D-1].sum() * iol for iPX, iol, iwalk in zip(PXs, ols, tc_walk) if iwalk ) 
        J_c = sum( iNp * iol for iNp, iol, iwalk in zip(Nps, ols, tc_walk) if iwalk ) 
        t_c  = th.divide(PX_c + (eta @ tcs), J_c + eta.sum())
        # t_c = itmat['tc']

        PX = sum( iPX[:,:D-1] * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
        Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        iY = Y[:,:D-1]
        Jab = Np + zeta.sum()
        TabE = (zeta @ tabs) #check

        if not itranspara['fix_t']:
            muXab = th.divide(th.sum(PX, axis=0) +TabE, Jab)
            muY   = th.divide(iY.T @ P1, Jab)

            Y_hat = iY - muY
            B = PX.T @ Y_hat - th.outer(muXab, P1 @ Y_hat)
            B += th.einsum('l,lij->ij', delta, As)
        else:
            muXab = 0
            muY = 0

            Y_hat = Y
            B = PX.T @ Y_hat - th.outer(itmat['t'], P1 @ Y_hat)

        if itranspara['isoscale'] :
            if not itranspara['fix_R']:
                U, S, V = th.linalg.svd(B, full_matrices=True)
                C = th.eye(D-1, dtype=B.dtype, device=B.device)
                C[-1, -1] = th.linalg.det(U @ V)
                R = U @ C @ V
                if th.linalg.det(R) < 0:
                    U[:, -1] = -U[:, -1]
                    R = U @ C @ V
            else:
                R = itmat['R']

            if itranspara['fix_s'] is False:
                H = (Y_hat.T * P1) @ Y_hat
                H.diagonal().add_(delta.sum())
                trAR = th.trace(B.T @ R)
                trH = th.trace(H)
                s = trAR/trH
            else:
                s = itmat['s']
            s = s.expand(D-1)
        else:
            H = (Y_hat.T * P1) @ Y_hat
            H.diagonal().add_( delta.sum())

            s = itmat['s'].clone()
            R = itmat['R'].clone()
            C = th.eye(D-1, dtype=B.dtype, device=B.device)

            if (not itranspara['fix_s']) and (not itranspara['fix_R']):
                max_iter = 80
                error = True
                iiter = 0

                while (error):
                    U, S, V = th.linalg.svd(B * s, full_matrices=True)
                    C[-1, -1] = th.linalg.det(U @ V)
                    R_pre = R.clone()
                    R = U @ C @ V
                    if th.linalg.det(R) < 0:
                        U[:, -1] = -U[:, -1]
                        R = U @ C @ V

                    s = th.diagonal(B.T @ R)/H
                    if itranspara['s_clip']  is not None:
                        s = th.clip(s, *itranspara['s_clip'] )

                    iiter += 1
                    error = (th.dist(R, R_pre) > 1e-8) and (iiter < max_iter)

            elif (not itranspara['fix_R']) and itranspara['fix_s']:
                U, S, V = th.linalg.svd(B * s, full_matrices=True)
                C[-1, -1] = th.linalg.det(U @ V)
                R = U @ C @ V
                if th.linalg.det(R) < 0:
                    U[:, -1] = -U[:, -1]
                    R = U @ C @ V
    
            elif (not itranspara['fix_s']) and itranspara['fix_R']:
                s = th.diagonal(B.T @ R )/H

        if itranspara['s_clip'] is not None:
            s = th.clip(s, *itranspara['s_clip'])

        A_ab = R * s
        if not itranspara['fix_t']:
            t_ab = muXab - A_ab @ muY
        else:
            t_ab = itmat['t']

        TY   = th.hstack([iY @ A_ab.T + t_ab, t_c.expand(M,1)])

        Qp = th.einsum('l,lij->', delta, (A_ab - As)**2)/2 \
                + th.einsum('l,li->',  zeta, (t_ab - tabs)**2)/2 \
                + th.einsum('l,l->',   eta, (t_c - tcs)**2)/2
        sigma2_exp, sigma2, iQ = self.update_sigma2(iL, Pt1s, P1s, Nps, PXs, Xs, TY, Qp = Qp)
        return R, s, t_ab, t_c, TY, sigma2_exp, sigma2, iQ

    @th.no_grad()
    def rf_affine_maximization(self, iL, Pt1s, P1s, PXs, Nps, Xs, Y,  As, tabs, tcs):
        omega, delta, zeta, eta, tc_walk = self.omega[iL], self.delta[iL], self.zeta[iL], self.eta[iL], self.tc_walk
        sigma2, sigma2_exp, sigma2_min = self.sigma2[iL], self.sigma2_exp[iL], self.sigma2_min
        itranspara, itmat = self.transparas[iL], self.tmats[iL]

        XL, D, M = len(Nps), Y.shape[1], Y.shape[0]
        # ols = omega/sigma2 # raw
        ols = omega/sigma2_exp # exp 
        # ols = omega # same with not common constraint  exp #TODO

        PX_c = sum( iPX[:,D-1].sum() * iol for iPX, iol, iwalk in zip(PXs, ols, tc_walk) if iwalk ) 
        J_c = sum( iNp * iol for iNp, iol, iwalk in zip(Nps, ols, tc_walk) if iwalk ) 
        t_c  = th.divide(PX_c + (eta @ tcs), J_c + eta.sum())

        PX = sum( iPX[:,:D-1] * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
        Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        iY = Y[:,:D-1]
        Jab = Np + zeta.sum()
        TabE = (zeta @ tabs) #check

        muXab = th.divide(th.sum(PX, axis=0) +TabE, Jab)
        muY   = th.divide(iY.T @ P1, Jab)
        Y_hat = iY - muY

        B = PX.T @ Y_hat - th.outer(muXab, P1 @ Y_hat)
        H = (Y_hat.T * P1) @ Y_hat

        B += th.einsum('l,lij->ij', delta, As)
        H.diagonal().add_(delta.sum())

        try:
            A_ab = B @ th.linalg.inv(H)
        except:
            B.diagonal().add_(0.001*sigma2)
            H.diagonal().add_(0.001*sigma2)
            A_ab = B @ th.linalg.inv(H)

        t_ab = muXab - A_ab @ muY
        TY   = th.hstack([iY @ A_ab.T + t_ab, t_c.expand(M,1)])
        
        Qp = th.einsum('l,lij->', delta, (A_ab - As)**2)/2 \
                + th.einsum('l,li->',  zeta, (t_ab - tabs)**2)/2 \
                + th.einsum('l,l->',   eta, (t_c - tcs)**2)/2
        sigma2_exp, sigma2, iQ = self.update_sigma2(iL, Pt1s, P1s, Nps, PXs, Xs, TY, Qp = Qp)
        return A_ab, 1.0, t_ab, t_c, TY, sigma2_exp, sigma2, iQ

    @th.no_grad()
    def rf_deformable_maximization(self, iL, Pt1s, P1s, PXs, Nps, Xs, Y, Ws, tcs):
        omega, delta, eta, tc_walk = self.omega[iL], self.delta[iL], self.eta[iL], self.tc_walk
        sigma2, sigma2_exp, sigma2_min = self.sigma2[iL], self.sigma2_exp[iL], self.sigma2_min
        itranspara, itmat = self.transparas[iL], self.tmats[iL]
        iteration = self.iteration

        mu = itranspara['alpha_mu']**(iteration)
        nu1 = nu2 = itranspara['gamma_nu']**(iteration)
        alpha = itranspara['alpha']
        gamma1 = itranspara['gamma1']
        gamma2 = itranspara['gamma2']
        p1_thred = itranspara['p1_thred']
        use_p1 = itranspara['use_p1']
   
        XL, D, M = len(Nps), Y.shape[1], Y.shape[0]
        # ols = omega/sigma2 # raw
        ols = omega/sigma2_exp # exp 
        # ols = omega # same with not common constraint  exp #TODO

        PX_c = sum( iPX[:,D-1].sum() * iol for iPX, iol, iwalk in zip(PXs, ols, tc_walk) if iwalk ) 
        J_c = sum( iNp * iol for iNp, iol, iwalk in zip(Nps, ols, tc_walk) if iwalk ) 
        t_c  = th.divide(PX_c + (eta @ tcs), J_c + eta.sum())

        iY = Y[:,:D-1]
        wsa = sum( omega[xl] * float(alpha[xl]) * mu for xl in range(XL) )
        PX = sum( iPX[:,:D-1] * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
        Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        J1 = getattr(self.DR, f'{iL}_J1')
        J2 = getattr(self.DR, f'{iL}_J2')
        J3 = getattr(self.DR, f'{iL}_J3')
        E = getattr(self.DR, f'{iL}_E')
        F = getattr(self.DR, f'{iL}_F')
        U = getattr(self.DR, f'{iL}_U')
        S = getattr(self.DR, f'{iL}_S')
        I = getattr(self.DR, f'{iL}_I')
        G = getattr(self.DR, f'{iL}_G')

        B = PX - P1[:,None] * iY
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
            B += H_del
            Pg = (P1 + delta.sum())[:,None] * V
            if (use_p1):
                P1s_thred = [ th.where(iP1 < ipth, 0.0, iP1) for iP1, ipth in zip(P1s, p1_thred) ]
                if any(gamma1 > 0):
                    P2 = sum( (w * float(ig1) * nu1) * iP1 for w, iP1, ig1 in zip(omega,  P1s_thred, gamma1) )
                    P2 =  E.T *  P2[None,:]
                    B  -= P2 @ J3
                    Pg += P2 @ J1
                if any(gamma2 > 0):
                    P3 = sum( (w * float(ig2) * nu2) * iP1 for w, iP1, ig2 in zip(omega, P1s_thred, gamma2) )
                    Pg += (F.T * P3[None,:]) @ J2
            else:
                if any(gamma1 > 0):
                    P2 = sum( (w * float(ig1) * nu1) for w, ig1 in zip(omega, gamma1) )
                    B  -= P2 * J3
                    Pg += P2 * J1
                if any(gamma2 > 0):
                    P3 = sum( (w * float(ig2) * nu2)  for w, ig2 in zip(omega, gamma2) )
                    Pg += P3 * J2
    
            W  = (B - Pg @ th.linalg.solve(I * wsa + U.T @ Pg, U.T @ B)) / wsa
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

            B +=  H_del
            Pg = (P1 + delta.sum())[:,None]*G
            Pg.diagonal().add_(wsa)

            if (use_p1): # same as the fast_rank
                P1s_thred = [ th.where(iP1 < ipth, 0.0, iP1) for iP1, ipth in zip(P1s, p1_thred) ]
                if any(gamma1 > 0):
                    P2 = sum( (w * float(ig1) * nu1) * iP1 for w, iP1, ig1 in zip(omega, P1s_thred, gamma1) )
                    P2 = E.T * P2[None,:]
                    B  -= P2 @ J3
                    Pg += P2 @ J1
                if any(gamma2 > 0):
                    P3 = sum( (w * float(ig2) * nu2) * iP1 for w, iP1, ig2 in zip(omega, P1s_thred, gamma2) )
                    Pg += (F.T * P3[None,:]) @ J2
            else:  # same as the fast_rank
                if any(gamma1 > 0):
                    P2 = sum( (w * float(ig1) * nu1) for w, ig1 in zip(omega, gamma1) )
                    B  -= P2 * J3
                    Pg += P2 * J1
                if any(gamma2 > 0):
                    P3 = sum( (w * float(ig2) * nu2) for w, ig2 in zip(omega, gamma2) )
                    Pg += P3 * J2
            W = th.linalg.solve(Pg, B)
            H = G @ W
        
        TY   = th.hstack([ iY + H, t_c.expand(M,1) ])

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
            Lp = th.trace(W.T @ H)
        Qp = Lp #+ (PV + LV)
        sigma2_exp, sigma2, iQ = self.update_sigma2(iL, Pt1s, P1s, Nps, PXs, Xs, TY, Qp = Qp)
        return W, t_c, TY, sigma2_exp, sigma2, iQ

    @th.no_grad()
    def update_sigma2(self, iL, Pt1s, P1s, Nps, PXs, Xs, TY, Qp = None):
        XL, D, xp =  len(Xs), float(TY.shape[1]), self.xp
        sigma2_min, sigma2_sync, omega = self.sigma2_min, self.sigma2_sync, self.omega[iL]

        Flj = [] 
        for xl in range(XL):
            if Xs[xl].shape[0] == 0:
                Flj.append(0)
            else:
                trxPx = xp.sum( Pt1s[xl] * xp.sum(Xs[xl]  * Xs[xl], axis=1) )
                tryPy = xp.sum( P1s[xl]  * xp.sum(TY * TY, axis=1))
                trPXY = xp.sum( TY * PXs[xl])
                Flj.append(trxPx - 2 * trPXY + tryPy)

        # sigma2
        sigma2_exp =(sum( wlj * flj for wlj, flj in zip(omega, Flj))/
                        sum( wlj * jlj * D for wlj, jlj in zip(omega, Nps)))
        sigma2_exp = xp.clip(sigma2_exp, min=sigma2_min)
        if sigma2_sync:
            sigma2 = sigma2_exp.expand(XL)
        else:
            sigma2 = xp.ones(XL, dtype=sigma2_exp.dtype, device=sigma2_exp.device)
            for xl in range(XL):
                if Xs[xl].shape[0] > 0:
                    if self.use_projection: #TODO
                        sigma2[xl] = Flj[xl]/Nps[xl]/D        
                    else:
                        sigma2[xl] = Flj[xl]/Nps[xl]/(D-1)
                    if sigma2[xl] < sigma2_min: 
                        sigma2[xl] = xp.clip(sigma_square(Xs[xl], TY), min=sigma2_min)

        # Q
        Qp = Qp or 0
        for xl in range(XL):
            iQ  = Flj[xl]/2.0/sigma2[xl] + Nps[xl] * D/2.0 * xp.log(sigma2[xl])
            Qp += omega[xl]*iQ
        return sigma2_exp, sigma2, Qp

    @th.no_grad()
    def update_normalize(self):
        device, dtype = 'cpu', self.floatxx
        Sx, Sy  = self.Sx.to(device, dtype=dtype), self.Sy.to(device, dtype=dtype)
        Tx, Ty  = self.Mx.to(device, dtype=dtype), self.My.to(device, dtype=dtype)
        Sr = Sx/Sy
        D, self.TY = self.D, self.xp.zeros((self.M, self.D), device=device, dtype=dtype)
        for iL in range(self.L):
            itf, itp = self.transformer[iL], self.transparas[iL]
            iTM = self.tmats[iL]
            iTM = { k: v.to(device, dtype=dtype) for k,v in iTM.items() }

            if itf in 'ESARTILO':
                if itf in ['E']:
                    A = iTM['R'] * iTM['s'] * Sr
                    iTM['s'] *= Sr
                elif itf in 'SRTILO':
                    A = iTM['R'] * iTM['s']* Sr
                    iTM['s'] *= Sr
                elif itf in ['A']:
                    A = iTM['A']* Sr
                    iTM['A'] = A
                else:
                    raise(f"Unknown transformer: {itf}")

                if not itp.get('fix_t' , False):
                    t = iTM['t']*Sx + Tx[:-1] - Ty[:-1] @ A.T
                    tc = iTM['tc']*Sx + Tx[-1]
                else: #TODO
                    t = iTM['t']*Sx + Tx[:-1] - Ty[:-1] @ A.T
                    tc = iTM['tc']*Sx + Tx[-1]
                
                H = self.xp.eye(D+1, D+1, device=device, dtype=dtype)
                H[:D-1, :D-1] = A
                H[:D-1,  D] = t
                H[D-1, D] = tc
                H[D-1, D-1] = 0 # drop Yz

                TY = self.TYs[iL].to(device, dtype=dtype) * Sx + Tx
                # TY1 = self.Yrs[iL].to(device, dtype=dtype) @ H[:-1,:-1].T + self.xp.hstack([t, tc]) #TODO check
                # TY2 = self.homotransform_point(self.Yrs[iL], H)
                # print(iL, H, TY - TY1, TY1-TY2)               

                iTM['t'] = t
                iTM['tc'] = tc
                iTM['tform'] = H

            elif itf in ['D']:
                for ia in ['G', 'U', 'S']:
                    iv =  getattr(self.DR, f'{iL}_{ia}', None)
                    if iv is not None:
                        iTM[ia] = iv.to(device, dtype=dtype)
                    else:
                        iTM[ia] = iv

                iTM['Y'] = self.Ya[iL][:, :-1].to(device, dtype=dtype)
                iTM['Ym'] = Ty[:-1]
                iTM['Ys'] = Sy
                iTM['Xm'] = Tx[:-1]
                iTM['Xs'] = Sx
                iTM['beta'] = itp['beta']
                iTM['tc'] = iTM['tc']*Sx + Tx[-1]

                if not iTM['G'] is None:
                    H = iTM['G']  @  iTM['W']
                else:
                    H = iTM['U'] @ iTM['S'] @ (iTM['U'].T @  iTM['W'])
        
                TY = self.TYs[iL].to(device, dtype=dtype) * Sx + Tx
                # TY1 = self.xp.hstack(
                #         [ ((self.Yrs[iL] - Ty)[:,:-1]/Sy + H)* iTM['Xs'] + iTM['Xm'],
                #          iTM['tc'].expand(self.Yrs[iL].shape[0], 1) ]
                # )
                # print(iL, self.xp.dist(TY, TY1), TY, TY1  )

            iTM['transformer'] = itf
            self.tmats[iL] = iTM
            self.TY[self.Yins[iL]] = TY
            # self.TYs[iL] = TY
        # TY1 = self.transform_point(self.Yr).to(device, dtype=dtype)
        # print(self.xp.dist(self.TY, TY1), self.TY - TY1)

    def transform_point(self, Yr, tmats = None, device=None, dtype=None,  **kargs):
        device = self.device if device is None else device
        dtype = self.floatxx if dtype is None else dtype

        tmats = self.tmats if tmats is None else tmats
        Yins = [ Yr[:,self.D-1] == i for i in self.Zid ]
        Ys =   [ Yr[i] for i in Yins ]
        TYS = self.transform_points(Ys, tmats, device=device, dtype=dtype, **kargs)
        T = self.xp.zeros(Yr.shape, device=device, dtype=dtype)
        for ity, idx in zip(TYS, Yins):
            T[idx] = ity
        return T

    def transform_points(self, Ys, tmats, device=None, dtype=None,  use_keops=True,):
        assert len(Ys )== len(tmats)
        L = len(Ys)
        device = self.device if device is None else device
        dtype = self.floatxx if dtype is None else dtype
        Ys = [ self.xp.asarray(iY, device=device, dtype=dtype) for iY in Ys]

        TYs = []
        for iL in range(L):
            iTM = tmats[iL]
            itf = iTM['transformer']

            if itf == 'D':
                iTY = self.ccf_deformable_transform_point(Ys[iL][:,:-1],
                         dtype=dtype, device=device, use_keops=use_keops, **iTM)
                iTYc = self.xp.ones((Ys[iL].shape[0], 1), device=device, dtype=dtype) * float(iTM['tc'])
                iTY = self.xp.hstack([iTY, iTYc])
            else:
                iTY = self.homotransform_point(Ys[iL], iTM['tform'],
                        xp=self.xp, dtype=dtype, device=device,)
            TYs.append(iTY)
        return TYs

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