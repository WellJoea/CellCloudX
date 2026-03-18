import torch as th
from .reference_emregistration import rfEMRegistration

class rfCombinedRegistration(rfEMRegistration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'rf-combined'

    def optimization(self):
        for iL in range(self.L):
            if self.transformer[iL] == 'D':
                W, t_c, TY, sigma2_exp, sigma2, iQ = self.forward(iL, H_upd=self.H_upd)
                self.H_tmp[iL]['W'] = W
                self.H_tmp[iL]['tc'] = t_c
                self.tmats[iL]['W'] = W
                self.tmats[iL]['tc'] = t_c
            else:
                A_ab, s_ab, t_ab, t_c, TY, sigma2_exp, sigma2, iQ = self.forward(iL, H_upd=self.H_upd,)
                
                self.H_tmp[iL]['A'] = A_ab * s_ab
                self.H_tmp[iL]['t'] = t_ab
                self.H_tmp[iL]['tc'] = t_c

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
            self.H_upd.update(self.H_tmp)

        qprev = self.Q
        self.Q = self.xp.sum(self.Qs)
        self.diff = self.xp.abs(self.Q - qprev) #/self.Q
        self.iteration += 1

    def forward(self, iL,  H_upd=None ): # 
        Xs, Y, TY, XF, YF = self.Xa[iL], self.Ya[iL], self.TYs[iL], self.XFa[iL], self.YFa[iL]
        itransformer = self.transformer[iL]
        Pt1s, P1s, PXs, Nps = self.expectation(iL, Xs, TY, XF, YF)
        return self.maximization(iL, Pt1s, P1s, PXs, Nps, Xs, Y, H_upd, transformer=itransformer)

    @th.no_grad()
    def maximization(self, *args, transformer='E', **kwargs):
        if transformer == 'E':
            return self.rfc_rigid_maximization(*args, **kwargs)
        elif transformer in 'SRTILO':
            return self.rfc_similarity_maximization(*args, **kwargs)
        elif transformer == 'A':
            return self.rfc_affine_maximization(*args, **kwargs)
        elif transformer == 'D':
            return self.rfc_deformable_maximization(*args, **kwargs)
        else:
            raise(f"Unknown transformer: {transformer}")

    @th.no_grad()
    def rfc_predictTYs(self, iL, delta, H_upd, Y):
        D = self.D
        TYp, TYq = 0, 0
        for il, idel in enumerate(delta):
            itf = self.transformer[il]
            if idel > 0:
                if il == iL:
                    iTY = self.TYs[iL][:,:D-1]
                else:
                    if itf in 'D':
                        Us = getattr(self.DR, f'S_{iL}_{il}_US')
                        Vh = getattr(self.DR, f'S_{iL}_{il}_Vh')
                        W  =  H_upd[il]['W']
                        H  = Us @ (Vh @ W)
                        iTY = Y[:,:D-1] + H
                    else:
                        iTY = Y[:,:D-1] @ H_upd[il]['A'].T + H_upd[il]['t']
                TYp += iTY * idel
                # TYq += (iTY**2).sum() * idel # use for Qp pass
            # else:
            #     iTY = th.zeros(1, D-1, device=Y.device, dtype=Y.dtype)
        return TYp
            
    @th.no_grad()
    def rfc_rigid_maximization(self, iL, Pt1s, P1s, PXs, Nps, Xs, Y, H_upd):
        omega, delta, eta, tc_walk = self.omega[iL], self.delta[iL], self.eta[iL], self.tc_walk
        sigma2, sigma2_exp, sigma2_min = self.sigma2[iL], self.sigma2_exp[iL], self.sigma2_min
        itranspara, itmat = self.transparas[iL], self.tmats[iL]

        XL, D, M = len(Nps), Y.shape[1], Y.shape[0]
        # ols = omega/sigma2 # raw
        ols = omega/sigma2_exp # exp 
        # ols = omega # same with not common constraint  exp #TODO

        PX_c = sum( iPX[:,D-1].sum() * iol for iPX, iol, iwalk in zip(PXs, ols, tc_walk) if iwalk ) 
        J_c  = sum( iNp * iol for iNp, iol, iwalk in zip(Nps, ols, tc_walk) if iwalk ) 
        TcE  = sum( H_upd[il]['tc'] for il, iieta in enumerate(eta) if iieta >0)
        t_c  = th.divide(PX_c + TcE, J_c + eta.sum())

        PX = sum( iPX[:,:D-1] * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
        Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        iY = Y[:,:D-1]
        S_delta = delta.sum()
        if S_delta>0:
            TYp = self.rfc_predictTYs(iL, delta, H_upd, Y)
            Jab = Np + S_delta
            TabE = TYp.sum(0)
            my1  = S_delta* iY.sum(0)
            muXab = th.divide(th.sum(PX, axis=0) +TabE, Jab)
            muY   = th.divide(iY.T @ P1 + my1, Jab)
            Y_hat = iY - muY
            T_hat = TYp - S_delta* muXab

            B = PX.T @ Y_hat - th.outer(muXab, P1 @ Y_hat)
            # H = (Y_hat.T * P1) @ Y_hat
            B += T_hat.T @ Y_hat
            # H += S_delta * (Y_hat.T @ Y_hat)
        else:
            Jab = Np 
            muXab = th.divide(th.sum(PX, axis=0), Jab)
            muY   = th.divide(iY.T @ P1, Jab)
            Y_hat = iY - muY
            B = PX.T @ Y_hat - th.outer(muXab, P1 @ Y_hat)
            # H = (Y_hat.T * P1) @ Y_hat

        U, S, V = th.linalg.svd(B, full_matrices=True)
        C = th.eye(D-1, dtype=B.dtype, device=B.device)
        C[-1, -1] = th.linalg.det(U @ V)
        R = U @ C @ V
        if th.linalg.det(R) < 0:
            U[:, -1] = -U[:, -1]
            R = U @ C @ V

        if itranspara['fix_s'] is False:
            H = (Y_hat.T * P1) @ Y_hat +  S_delta * (Y_hat.T @ Y_hat)
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

        Qp = 0 # \| TY - TYp \|_2^2 pass
        sigma2_exp, sigma2, iQ = self.update_sigma2(iL, Pt1s, P1s, Nps, PXs, Xs, TY, Qp = Qp)
        return R, s, t_ab, t_c, TY, sigma2_exp, sigma2, iQ

    @th.no_grad()
    def rfc_similarity_maximization(self, iL, Pt1s, P1s, PXs, Nps, Xs, Y, H_upd):
        omega, delta, eta, tc_walk = self.omega[iL], self.delta[iL], self.eta[iL], self.tc_walk
        sigma2, sigma2_exp, sigma2_min = self.sigma2[iL], self.sigma2_exp[iL], self.sigma2_min
        itranspara, itmat = self.transparas[iL], self.tmats[iL]

        XL, D, M = len(Nps), Y.shape[1], Y.shape[0]

        # ols = omega/sigma2 # raw
        ols = omega/sigma2_exp # exp 
        # ols = omega # same with not common constraint  exp #TODO

        PX_c = sum( iPX[:,D-1].sum() * iol for iPX, iol, iwalk in zip(PXs, ols, tc_walk) if iwalk ) 
        J_c  = sum( iNp * iol for iNp, iol, iwalk in zip(Nps, ols, tc_walk) if iwalk ) 
        TcE  = sum( H_upd[il]['tc'] for il, iieta in enumerate(eta) if iieta >0)
        t_c  = th.divide(PX_c + TcE, J_c + eta.sum())

        PX = sum( iPX[:,:D-1] * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
        Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        S_delta, TabE  = delta.sum(), 0
        if S_delta>0:
            TYp = self.rfc_predictTYs(iL, delta, H_upd, Y)
            TabE = TYp.sum(0)

        iY = Y[:,:D-1]
        Jab = Np + S_delta

        if not itranspara['fix_t']:
            my1  = S_delta* iY.sum(0)
            muXab = th.divide(th.sum(PX, axis=0) +TabE, Jab)
            muY   = th.divide(iY.T @ P1 + my1, Jab)
            Y_hat = iY - muY
            B = PX.T @ Y_hat - th.outer(muXab, P1 @ Y_hat)
        else:
            muXab = 0
            muY = 0
            Y_hat = Y
            B = PX.T @ Y_hat - th.outer(itmat['t'], P1 @ Y_hat)

        if S_delta>0:
            T_hat = TYp - S_delta* muXab
            B += T_hat.T @ Y_hat

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
                H = (Y_hat.T * P1) @ Y_hat + S_delta* (Y_hat.T @ Y_hat)
                # H = (Y_hat.T * (P1+S_delta)) @ Y_hat
                trAR = th.trace(B.T @ R)
                trH = th.trace(H)
                s = trAR/trH
            else:
                s = itmat['s']
            s = s.expand(D-1)
        else:
            H = (Y_hat.T * P1) @ Y_hat + S_delta * (Y_hat.T @ Y_hat)
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
        Qp = 0 # \| TY - TYp \|_2^2 pass
        sigma2_exp, sigma2, iQ = self.update_sigma2(iL, Pt1s, P1s, Nps, PXs, Xs, TY, Qp = Qp)
        return R, s, t_ab, t_c, TY, sigma2_exp, sigma2, iQ

    @th.no_grad()
    def rfc_affine_maximization(self, iL, Pt1s, P1s, PXs, Nps, Xs, Y, H_upd):
        omega, delta, eta, tc_walk = self.omega[iL], self.delta[iL], self.eta[iL], self.tc_walk
        sigma2, sigma2_exp, sigma2_min = self.sigma2[iL], self.sigma2_exp[iL], self.sigma2_min
        itranspara, itmat = self.transparas[iL], self.tmats[iL]

        XL, D, M = len(Nps), Y.shape[1], Y.shape[0]
        # ols = omega/sigma2 # raw
        ols = omega/sigma2_exp # exp 
        # ols = omega # same with not common constraint  exp #TODO

        PX_c = sum( iPX[:,D-1].sum() * iol for iPX, iol, iwalk in zip(PXs, ols, tc_walk) if iwalk ) 
        J_c  = sum( iNp * iol for iNp, iol, iwalk in zip(Nps, ols, tc_walk) if iwalk ) 
        TcE  = sum( H_upd[il]['tc'] for il, iieta in enumerate(eta) if iieta >0)
        t_c  = th.divide(PX_c + TcE, J_c + eta.sum())

        PX = sum( iPX[:,:D-1] * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
        Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        iY = Y[:,:D-1]
        S_delta = delta.sum()
        if S_delta>0:
            TYp = self.rfc_predictTYs(iL, delta, H_upd, Y)
            Jab = Np + S_delta
            TabE = TYp.sum(0)
            my1  = S_delta* iY.sum(0)
            muXab = th.divide(th.sum(PX, axis=0) +TabE, Jab)
            muY   = th.divide(iY.T @ P1 + my1, Jab)
            Y_hat = iY - muY
            T_hat = TYp - S_delta* muXab

            B = PX.T @ Y_hat - th.outer(muXab, P1 @ Y_hat)
            H = (Y_hat.T * P1) @ Y_hat

            B += T_hat.T @ Y_hat
            H += S_delta * (Y_hat.T @ Y_hat)

        else:
            Jab = Np 
            muXab = th.divide(th.sum(PX, axis=0), Jab)
            muY   = th.divide(iY.T @ P1, Jab)
            Y_hat = iY - muY
            B = PX.T @ Y_hat - th.outer(muXab, P1 @ Y_hat)
            H = (Y_hat.T * P1) @ Y_hat

        try:
            A_ab = B @ th.linalg.inv(H)
        except:
            B.diagonal().add_(0.001*sigma2)
            H.diagonal().add_(0.001*sigma2)
            A_ab = B @ th.linalg.inv(H)

        t_ab = muXab - A_ab @ muY
        TY   = th.hstack([iY @ A_ab.T + t_ab, t_c.expand(M,1)])
        
        Qp = 0 # \| TY - TYp \|_2^2 pass
        sigma2_exp, sigma2, iQ = self.update_sigma2(iL, Pt1s, P1s, Nps, PXs, Xs, TY, Qp = Qp)
        return A_ab, 1.0, t_ab, t_c, TY, sigma2_exp, sigma2, iQ

    @th.no_grad()
    def rfc_deformable_maximization(self, iL, Pt1s, P1s, PXs, Nps, Xs, Y, H_upd):
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
        J_c  = sum( iNp * iol for iNp, iol, iwalk in zip(Nps, ols, tc_walk) if iwalk ) 
        TcE  = sum( H_upd[il]['tc'] for il, iieta in enumerate(eta) if iieta >0)
        t_c  = th.divide(PX_c + TcE, J_c + eta.sum())

        wsa = sum( omega[xl] * float(alpha[xl]) * mu for xl in range(XL) )
        PX = sum( iPX[:,:D-1] * iol for iPX, iol in zip(PXs, ols) )
        P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
        # Np = sum( iNp * iol for iNp, iol in zip(Nps, ols) )

        J1 = getattr(self.DR, f'E_{iL}_J1')
        J2 = getattr(self.DR, f'E_{iL}_J2')
        J3 = getattr(self.DR, f'E_{iL}_J3')
        E = getattr(self.DR, f'E_{iL}_E')
        F = getattr(self.DR, f'E_{iL}_F')
        U = getattr(self.DR, f'E_{iL}_U')
        S = getattr(self.DR, f'E_{iL}_S')
        I = getattr(self.DR, f'E_{iL}_I')
        G = getattr(self.DR, f'E_{iL}_G')

        iY = Y[:,:D-1]
        B = PX - P1[:,None] * iY

        S_delta = delta.sum()
        if S_delta>0:
            TYp = self.rfc_predictTYs(iL, delta, H_upd, Y)
            H_del = TYp - S_delta* iY
            B += H_del

        if itranspara['fast_rank']:
            V = (U @ S)
            if S_delta>0:
                Pg = (P1 + S_delta)[:,None] * V
            else:
                Pg = P1[:,None] * V

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
            if S_delta>0:
                Pg = (P1 + S_delta)[:,None] * G
            else:
                Pg = P1[:,None] * G
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