import torch as th
import math
from .neighbors_ensemble import Neighbors
from .operation_expectation import (rigid_outlier, sigma_square, default_transparas, init_tmat)

from .manifold_regularizers import spdeformable_regularizer
from ...transform import homotransform_points, homotransform_point, ccf_deformable_transform_point, homotransform_mat, ccf_hierarchical_transform_point

class spEM_core():
    def __init__(self):
        super().__init__()
        self.init_tmat = init_tmat
        self.default_transparas= default_transparas
        self.rigid_outlier = rigid_outlier

        self.homotransform_points = homotransform_points
        self.ccf_deformable_transform_point = ccf_deformable_transform_point
        self.ccf_hierarchical_transform_point = ccf_hierarchical_transform_point
        self.homotransform_mat = homotransform_mat
        self.homotransform_point = homotransform_point
        self.spdeformable_regularizer = spdeformable_regularizer

    def spcompute_feauture(self):
        self.d2f = []
        for iL in range(self.FL):
            xidx, yidx = self.pairs[:,0], self.pairs[:,1]
            d2f = (self.XF[iL][xidx] - self.YF[iL][yidx]).pow(2).sum(1)
            self.d2f.append(d2f.to(self.device, dtype=self.floatx))

    def spcompute_regularizer(self):
        self.DR = self.spdeformable_regularizer(**self.transparas, xp = self.xp, verbose=self.verbose>1)
        self.DR.compute(self.Y, device=self.device, dtype=self.floatx)

    @th.no_grad()
    def spexpectation(self):
        xp , eps =self.xp, self.eps
        X, TY = self.X, self.TY
        sigma2, tau2 = self.sigma2, self.tau2

        N, D, DK, FL = float(X.shape[0]), float(X.shape[1]), self.DK, self.FL
        wso, wto, w, wa, w_clip, theta =self.wso, self.wto, self.w, self.wa, self.w_clip, self.theta
        tau2_auto = self.tau2_auto 
        tau2_grow = self.tau2_grow
        tau2_delta = self.tau2_delta
        st_ratio = self.st_ratio

        D2i  = ((X - TY)**2).sum(dim=1)
        D2f = self.d2f

        ## outlier I  ##rbf Feature
        sigma2_n = -0.5 * st_ratio[0]/sigma2
        P_in =  D2i * sigma2_n
        P_ou =  D2i * (sigma2_n/wso)
        c_oi = (1-w)/w * (wso)**(-0.5 *st_ratio[0]* D) #  (wso)**(-D) too small

        tau2_n = [ -0.5 * st_ratio[1] * tau2_grow[iL][self.iteration] * tau2_delta[iL]/tau2[iL] for iL in range(FL) ]
        for iL in range(FL):
            P_in += D2f[iL]*  tau2_n[iL]
            P_ou += D2f[iL]* (tau2_n[iL] *wto[iL])

            # if feature term is treated as Gaussian density, keep this;
            # if treated as kernel similarity, this term can be omitted intentionally.
            # c_oi *= wto[iL] ** (-0.5 * st_ratio[1] * DK[iL])
            # P_ou += D2f[iL]* (tau2_n[iL]/wto[iL])

        # P_in = xp.exp(P_in)
        # P_ou = xp.exp(P_ou) * c_oi
        # P  = P_in/(P_in+P_ou + eps)

        P_ou = P_ou + xp.log(c_oi + eps)
        P = 1.0 / (1.0 + xp.exp(P_ou - P_in))
        
        # Np0 = P.sum()
        # # simgma_o^2 = k * sigma2, tau2_o^2 = k * tau2
        # P = P + (1-P)/wso
        # Pf =  [ P + (1-P)/iwto for iwto in wto ]

        P1 = P
        Pt1= P
        PX = P[:, None] * X
        Np = P.sum()

        '''
        # update st_ratio: #TODO
        if FL >0:
            E_g = (P * D2i/ (2.0* sigma2)).sum()
            E_f = sum([ D2f[iL]* (0.5 * tau2_grow[iL] * tau2_delta[iL]/tau2[iL]) for iL in range(FL) ])
            E_f = (P * E_f).sum()
            E_r = E_g/E_f
            self.st_ratio = [E_r/(E_r + 1.0)* 2.0, 2.0/(E_r + 1.0)]

        # update tau2_delta: #TODO
        if FL >1:
            for iL in range(FL):
                tau2_delta[iL] = (P * xp.exp(D2f[iL]*  tau2_n[iL])).sum()/Np
            self.tau2_delta = [ i/ sum(tau2_delta) for i in tau2_delta]
        '''

        for iL in range(FL):
            if  tau2_auto[iL]:
                self.tau2[iL] = (P*D2f[iL]).sum()/ (Np*DK[iL])
                # self.tau2[iL] = (P*D2f[iL]).sum()/ (Np)

        # soft update is more EM-consistent
        self.w = wa *w + (1-wa)* Np/float(N)
        # self.w = wa *w + (1-wa)* (P > theta).sum()/float(N)
        self.w = self.w.clip(*w_clip)
        self.P = P
        return Pt1, P1, PX, Np

    def optimization(self):
        qprev = self.Q
        Pt1s, P1s, PXs, Nps = self.spexpectation()
        self.maximization(Pt1s, P1s, Nps, PXs, self.Y, self.X, transformer=self.transformer)
        self.diff = self.xp.abs(self.Q - qprev) #/self.Q
        self.iteration += 1

    def maximization(self, *args, transformer='E', **kwargs):
        if transformer == 'E':
            self.sp_rigid_maximization(*args, **kwargs)
        elif transformer in 'SRTILO':
            self.sp_similarity_maximization(*args, **kwargs)
        elif transformer == 'A':
            self.sp_affine_maximization(*args, **kwargs)
        elif transformer == 'P':
            self.sp_projective_maximization(*args, **kwargs)
        elif transformer == 'D':
            self.sp_deformable_maximization(*args, **kwargs)
        elif transformer == 'H':
            self.sp_hierarchical_maximization(*args, **kwargs)
        else:
            raise(f"Unknown transformer: {transformer}")

    @th.no_grad()
    def sp_rigid_maximization(self, Pt1, P1, Np, PX, Y, X):
        xp, D = self.xp, self.D
        sigma2, sigma2_min, itranspara, itmat = self.sigma2, self.sigma2_min, self.transparas, self.tmats
        p1_thred = itranspara['p1_thred']
        use_p1 = itranspara['use_p1']
        gamma1 = itranspara['gamma1']
        gamma3 = itranspara['gamma3']

        muX = xp.divide(xp.sum(PX, 0), Np)
        muY = xp.divide(Y.T @ P1, Np)
        # muY = xp.divide(xp.sum(P1[:, None] * Y, 0), Np)
        Y_hat = Y - muY
        X_hat = X - muX
        B = (X_hat* P1[:, None]).T @ Y_hat

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
            if (gamma1>0):
                if (use_p1):
                    P1_t = xp.where(P1 < p1_thred, 0.0, P1)
                    trYQY = ((self.DR.J3 **2).sum(1) * P1_t).sum()
                else:
                    trYQY = xp.trace(Y.T @ self.DR.J3)
                H += gamma1* sigma2 * trYQY
            if gamma3 > 0:
                trYOY = xp.trace(self.DR.C)
                H += gamma3 * sigma2 * trYOY
            s = trBR/H
        else:
            s = itmat['s']

        if itranspara['s_clip'] is not None:
            s = xp.clip(s, *itranspara['s_clip'])

        # T
        t = muX - s * (R @ muY)
        #TY
        TY = Y @ (R*s).T +  t

        sigma2, Q = self.update_sigma2(P1, Np, X, TY, Qp = None)
        self.tmats['R'] = R
        self.tmats['s'] = s
        self.tmats['t'] = t
        self.TY = TY
        self.sigma2 = sigma2
        self.Q = Q

    @th.no_grad()
    def sp_similarity_maximization(self, Pt1, P1, Np, PX, Y, X):
        xp, D, eps = self.xp, self.D, self.eps
        sigma2, sigma2_min, itranspara, itmat = self.sigma2, self.sigma2_min, self.transparas, self.tmats
        p1_thred = itranspara['p1_thred']
        use_p1 = itranspara['use_p1']
        gamma1 = itranspara['gamma1']
        gamma3 = itranspara['gamma3']

        if not itranspara['fix_t']:
            muX = xp.divide(xp.sum(PX, 0), Np)
            # muY = xp.divide(Y.T @ P1, Np)
            muY = xp.divide(xp.sum(P1[:, None] * Y, 0), Np)
            Y_hat = Y - muY
            X_hat = X - muX
            # B = PX.T @ Y_hat - xp.outer(muX, P1 @ Y_hat)
            B = (X_hat* P1[:, None]).T @ Y_hat
            ## check B- B1
        else:
            muX = 0
            muY = 0

            Y_hat = Y
            B = PX.transpose(1,0) @ Y_hat - xp.outer(itmat['t'], P1[:, None] * Y_hat)
            ## TODO:check
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
            # YPY = xp.sum( Y_hat * Y_hat * P1.unsqueeze(1), 0)

            if not itranspara['fix_s']:
                trBR = xp.trace(B.T @ R)
                if (gamma1>0):
                    if (use_p1):
                        P1_t = xp.where(P1 < p1_thred, 0.0, P1)
                        trYQY = ((self.DR.J3 **2).sum(1) * P1_t).sum()
                    else:
                        trYQY = xp.trace(Y.T @ self.DR.J3)
                    H += gamma1* sigma2 * trYQY
                if gamma3 > 0:
                    trYOY = xp.trace(self.DR.C)
                    H += gamma3 * sigma2 * trYOY
                s = trBR/H
            else:
                s = itmat['s']
            s = s.expand(D)    
        else:
            YPY = xp.sum( Y_hat * Y_hat * P1.unsqueeze(1), 0)
            YPY.masked_fill_(YPY == 0, eps)
            H = YPY

            if (gamma1>0):
                if (use_p1):
                    P1_t = xp.where(P1 < p1_thred, 0.0, P1)
                    dYQY = ((self.DR.J3 **2) * P1_t).sum(0)
                else:
                    dYQY = xp.diag(Y.T @ self.DR.J3)
                H += gamma1* sigma2 * dYQY
            if gamma3 > 0:
                dYOY = xp.diag(self.DR.C)
                H += gamma3 * sigma2 * dYOY

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
        sigma2, Q = self.update_sigma2(P1, Np, X, TY, Qp = None)

        self.tmats['R'] = R
        self.tmats['s'] = s
        self.tmats['t'] = t
        self.TY = TY
        self.sigma2 = sigma2
        self.Q = Q
        # return [[R, s, t,], TY, sigma2_exp, sigma2, Q]

    @th.no_grad()
    def sp_affine_maximization(self, Pt1, P1, Np, PX, Y, X):
        xp, D = self.xp, self.D
        sigma2, sigma2_min, itranspara, itmat = self.sigma2, self.sigma2_min, self.transparas, self.tmats
        p1_thred = itranspara['p1_thred']
        use_p1 = itranspara['use_p1']
        gamma1 = itranspara['gamma1']
        gamma3 = itranspara['gamma3']

        # Np = P1.sum()
        muX = xp.divide(xp.sum(PX, 0), Np)
        muY = xp.divide(Y.T @ P1, Np)
        # muY = xp.divide(xp.sum(P1[:, None] * Y, 0), Np)
        Y_hat = Y - muY
        X_hat = X - muX
        # B1 = PX.T @ Y_hat - xp.outer(muX, P1 @ Y_hat)
        B = (X_hat* P1[:, None]).T @ Y_hat

        YPYh = (Y_hat.transpose(1,0) * P1) @ Y_hat

        if (gamma1>0):
            if (use_p1):
                P1_t = xp.where(P1 < p1_thred, 0.0, P1)
                YQY = self.DR.J3.T @ (P1_t[:, None] * self.DR.J3)
            else:
                YQY = xp.diag(Y.T @ self.DR.J3)
            YPYh += gamma1* sigma2 * YQY
        if gamma3 > 0:
            YOY = self.DR.C
            YPYh += gamma3 * sigma2 * YOY

        if itranspara['delta'] > 0:
            YPYh.diagonal().add_( itranspara['delta'] *sigma2)
            A = (B + itranspara['delta']*sigma2*itmat['A']) @ xp.linalg.inv(YPYh)
        else:
            try:
                YPYhv = xp.linalg.inv(YPYh)
                A = B @ YPYhv
            except:
                delta = 1e-2
                YPYh.diagonal().add_(delta*sigma2)
                A = (B + delta*sigma2*itmat['A']) @ xp.linalg.inv(YPYh)

        t = muX - A @ muY
        TY = Y @ A.T +  t

        sigma2, Q = self.update_sigma2(P1, Np, X, TY, Qp = None)
        self.tmats['A'] = A
        self.tmats['t'] = t
        self.TY = TY
        self.sigma2 = sigma2
        self.Q = Q

    def sp_projective_maximization(self, Pt1, P1, Np, PX, Y, X):
        xp, D = self.xp, float(self.D)
        sigma2, sigma2_min, itranspara, itmat = self.sigma2, self.sigma2_min, self.transparas, self.tmats
        p1_thred = itranspara['p1_thred']
        use_p1 = itranspara['use_p1']
        gamma1 = itranspara['gamma1']
        gamma3 = itranspara['gamma3']

        def closure():
            self.optimizer.zero_grad()
            # self.logsigma2.data = self.xp.nan_to_num(self.logsigma2, nan=0.0, posinf=12.0, neginf=-12)
            self.logsigma2.data = self.xp.clamp(self.logsigma2, min=self.xp.log(self.sigma2_min))
            TY = self.projective_transform()

            d1 = (X - TY)
            trXPT = ((d1**2).sum(1) * P1).sum()
            Q = trXPT /(2*self.xp.exp(self.logsigma2)+ self.eps) + D * Np / 2 * self.logsigma2

            if (gamma1>0):
                q1 = ((self.DR.E @ TY)**2).sum(1)
                if (use_p1):
                    P1_t = xp.where(P1 < p1_thred, 0.0, P1)
                    q1 = q1 * P1_t
                Q += 0.5* gamma1 * q1.sum()
            if gamma3 > 0:
                q3 = xp.trace(TY.T @ self.DR.F @ TY)
                Q += 0.5 * gamma3 * q3

            Q.backward(retain_graph=True)
            # self.xp.nn.utils.clip_grad_norm_([self.A, self.B], max_norm=10.0)
            # self.xp.nn.utils.clip_grad_norm_([self.t, self.logsigma2],  max_norm=100.0)
            return Q
        
        loss = self.optimizer.step(closure)
        if self.transparas['lr_stepsize']:
            self.scheduler.step()
    
        with self.xp.no_grad():
            self.TY = self.projective_transform()
            self.sigma2 = self.xp.exp(self.logsigma2)
            self.Q = self.xp.tensor(loss.item())

            if self.sigma2 < self.sigma2_min:
                self.sigma2 = self.xp.clip(sigma_square(X, self.TY), min=self.sigma2_min)
                self.logsigma2.data = self.xp.log(self.sigma2)

    @th.no_grad()
    def sp_deformable_maximization(self, Pt1, P1, Np, PX, Y, X):
        xp, D = self.xp, self.D
        sigma2, sigma2_min, itranspara, itmat = self.sigma2, self.sigma2_min, self.transparas, self.tmats
        
        p1_thred = itranspara['p1_thred']
        use_p1 = itranspara['use_p1']
        mu = itranspara['alpha_mu']**(self.iteration)
        alpha = itranspara['alpha'] 
        nu1 = nu2 = itranspara['gamma_nu']**(self.iteration)
        gamma1 = itranspara['gamma1']
        gamma2 = itranspara['gamma2']
        delta = itranspara['delta']

        wsa = alpha * mu
        B = (PX - P1[:,None] * Y)/sigma2
        if delta>0:
            P1_hat = P1/sigma2 + delta
        else:
            P1_hat = P1/sigma2

        if itranspara['fast_rank']:
            U = self.DR.U
            S = self.DR.S
            V = (U @ S)
            Pg = P1_hat[:,None]* V

            if (use_p1):
                P1_t = xp.where(P1 < p1_thred, 0.0, P1)
                if (gamma1 > 0):
                    P2 = P1_t * (gamma1 * nu1)
                    P2 = self.DR.E.T * P2[None,:]
                    B  -= P2 @ self.DR.J3
                    Pg += P2 @ self.DR.J1
                if (gamma2 > 0):
                    P3 = P1_t * (gamma2 * nu2)
                    Pg += (self.DR.F.T * P3[None,:]) @ self.DR.J2
            else:
                if (gamma1 > 0):
                    P2 = (gamma1 * nu1)
                    B  -= P2 * self.DR.J3
                    Pg += P2 * self.DR.J1
                if (gamma2 > 0):
                    P3 = (gamma2 * nu2)
                    Pg += P3 * self.DR.J2
            
            I = self.DR.I * wsa
            W  = (B - Pg @ xp.linalg.solve(I + U.T @ Pg, U.T @ B)) / wsa
            H  = V @ (U.T @ W)

        else:
            G = self.DR.G
            Pg = P1_hat[:,None]* G
            Pg.diagonal().add_(wsa)
    
            if (use_p1): # same as the fast_rank

                P1_t = xp.where(P1 < p1_thred, 0.0, P1)
                if (gamma1 > 0):
                    P2 = P1_t * (gamma1 * nu1)
                    P2 = self.DR.E.T * P2[None,:]
                    B  -= P2 @ self.DR.J3
                    Pg += P2 @ self.DR.J1
                if (gamma2 > 0):
                    P3 = P1_t * (gamma2 * nu2)
                    Pg += (self.DR.F.T * P3[None,:]) @ self.DR.J2
            else:
                if (gamma1 > 0):
                    P2 = (gamma1 * nu1)
                    B  -= P2 * self.DR.J3
                    Pg += P2 * self.DR.J1
                if (gamma2 > 0):
                    P3 = (gamma2 * nu2)
                    Pg += P3 * self.DR.J2

            W = xp.linalg.solve(Pg, B)
            H = G @ W

        TY = Y + H
    
        Lp = xp.trace(W.T @ H) # TODO + LLE + LP
        # Qp = sum([ (float(alpha[i]) * mu * Lp + ill + ilp )/2.0 for i in range(L) ])
        Qp = alpha * mu * Lp/2.0
        sigma2, Q = self.update_sigma2(P1, Np, X, TY, Qp = Qp)

        self.tmats['W'] = W
        self.TY = TY
        self.sigma2 = sigma2
        self.Q = Q

    @th.no_grad()
    def sp_hierarchical_maximization(self, Pt1, P1, Np, PX, Y, X):
        xp, D = self.xp, self.D
        sigma2, sigma2_min, itranspara, itmat = self.sigma2, self.sigma2_min, self.transparas, self.tmats

        p1_thred = itranspara['p1_thred']
        use_p1 = itranspara['use_p1']
        mu = itranspara['alpha_mu']**(self.iteration)
        alpha = itranspara['alpha'] 
        nu1 = nu2 = itranspara['gamma_nu']**(self.iteration)
        gamma1 = itranspara['gamma1']
        gamma2 = itranspara['gamma2']
        gamma3 = itranspara['gamma3']
        delta = itranspara['delta']

        Yh = self.Yh
        muX = xp.divide(xp.sum(PX, 0), Np)
        # muY = xp.divide( Yh.T @ P1, Np)
        muY = xp.divide(xp.sum(P1[:, None] * Yh, 0), Np)
        Y_hat = Yh - muY
        X_hat = X - muX
 
        # B = PX.T @ Y_hat - xp.outer(muX, P1 @ Y_hat)
        B = (X_hat* P1[:, None]).T @ Y_hat

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
            if (gamma1>0):
                if (use_p1):
                    P1_t = xp.where(P1 < p1_thred, 0.0, P1)
                    trHQH = (((self.DR.E @ Yh)**2).sum(1) * P1_t).sum() 
                else:
                    trHQH = ((self.DR.E @ Yh)**2).sum()
                H += gamma1* sigma2 * trHQH
            if gamma3 > 0:
                trYOY = xp.trace(Yh.T @ self.DR.F @ Yh)
                H += gamma3 * sigma2 * trYOY
            s = trBR/H
        else:
            s = itmat['s']

        if itranspara['s_clip'] is not None:
            s = xp.clip(s, *itranspara['s_clip'])

        # T
        t = muX - s * (R @ muY)

        # W
        ols = 1.0/sigma2
        wsa = alpha * mu
        B = (PX @ (s*R) - (s**2) * P1[:,None] * Y - (P1[:, None] * (s* t @ R))) * ols
        if delta>0:
            P1_hat = P1*(ols* (s**2)) + delta #  delta * \bar{P_1}
        else:
            P1_hat = P1*(ols* (s**2))

        if itranspara['fast_rank']:
            U = self.DR.U
            S = self.DR.S
            V = (U @ S)
            Pg = P1_hat[:,None]* V

            if (use_p1):
                P1_t = xp.where(P1 < p1_thred, 0.0, P1)
                if (gamma1 > 0):
                    P2 = P1_t * (gamma1 * nu1 * (s**2))
                    P2 = self.DR.E.T * P2[None,:]
                    B  -= P2 @ self.DR.J3
                    Pg += P2 @ self.DR.J1
                if (gamma2 > 0):
                    P3 = P1_t * (gamma2 * nu2)
                    Pg += (self.DR.F.T * P3[None,:]) @ self.DR.J2
            else:
                if (gamma1 > 0):
                    P2 = (gamma1 * nu1 * (s**2))
                    B  -= P2 * self.DR.J3
                    Pg += P2 * self.DR.J1
                if (gamma2 > 0):
                    P3 = (gamma2 * nu2)
                    Pg += P3 * self.DR.J2

            I = self.DR.I * wsa
            W  = (B - Pg @ xp.linalg.solve(I + U.T @ Pg, U.T @ B)) / wsa
            H  = V @ (U.T @ W)

        else:
            G = self.DR.G
            Pg = P1_hat[:,None]* G
            Pg.diagonal().add_(wsa)

            if (use_p1): # same as the fast_rank

                P1_t = xp.where(P1 < p1_thred, 0.0, P1)
                if (gamma1 > 0):
                    P2 = P1_t * (gamma1 * nu1 * (s**2))
                    P2 = self.DR.E.T * P2[None,:]
                    B  -= P2 @ self.DR.J3
                    Pg += P2 @ self.DR.J1
                if (gamma2 > 0):
                    P3 = P1_t * (gamma2 * nu2)
                    Pg += (self.DR.F.T * P3[None,:]) @ self.DR.J2
            else:
                if (gamma1 > 0):
                    P2 = (gamma1 * nu1 * (s**2))
                    B  -= P2 * self.DR.J3
                    Pg += P2 * self.DR.J1
                if (gamma2 > 0):
                    P3 = (gamma2 * nu2)
                    Pg += P3 * self.DR.J2

            W = xp.linalg.solve(Pg, B)
            H = G @ W

        Yh = Y + H
        TY = Yh @ (R*s).T +  t

        Lp = xp.trace(W.T @ H) # TODO + LLE + LP
        # Qp = sum([ (float(alpha[i]) * mu * Lp + ill + ilp )/2.0 for i in range(L) ])
        Qp = alpha * mu * Lp/2.0
        sigma2, Q = self.update_sigma2(P1, Np, X, TY, Qp = Qp)

        self.tmats['R'] = R
        self.tmats['s'] = s
        self.tmats['t'] = t
        self.tmats['A'] = R * s
        self.tmats['W'] = W
        self.Yh = Yh
        self.TY = TY
        self.sigma2 = sigma2
        self.Q = Q

    @th.no_grad()
    def update_sigma2(self, P, Np, X, TY, Qp = None):
        D, N, xp = float(TY.shape[1]), float(TY.shape[0]), self.xp
        d1 = (X - TY)
        sigma2 = ((d1**2).sum(1) * P).sum()/Np/D
        # sigma2 = ((d1**2).sum(1) * P).sum()/P.sum()/D
        # sigma2 = ((d1**2).sum(1) * P).sum()/N/D
        Q = D * Np/2 * (1+xp.log(sigma2)) + (Qp or 0)

        if sigma2 < self.sigma2_min: 
            sigma2 = xp.clip(sigma_square(X, TY), min= self.sigma2_min)
        return sigma2, Q

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

        # T =  \frac{X_s}{Y_s} Y sR^T +  X_s V sR^T + 1_m (X_m^T + (X_s t)^T- \frac{X_s}{Y_s} Y_m^T sR^T )  
        # T =  \frac{X_s}{Y_s} Y A^T +  X_s V A^T + 1_m (X_m^T + (X_s t)^T- \frac{X_s}{Y_s} Y_m^T A^T )  
        if not itf in 'ESRTILOAPDNH':
            raise ValueError(f'Invalid transformation type {itf}, should be in "ESRTILOAPDNH"')
    
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
        if itf in ['P']:
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

        if itf in ['D']:
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
        
        if itf in ['H']: 
            recenter = False
            if recenter:
                iTM['recenter'] = True      
                A = iTM['R'] * iTM['s'] * Sr
                iTM['s'] *= Sr
                iTM['A'] = A

                if not itp.get('fix_t' , False):
                    iTM['t'] = iTM['t']*Sx + Tx - Ty @ A.T
                else: #TODO
                    iTM['t'] = iTM['t']*Sx + Tx - Ty @ A.T


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
                iTM['recenter'] = False   
                iTM['A'] = iTM['R'] * iTM['s']
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

        self.TY = self.TY.detach().to(device, dtype=dtype) * Sx + Tx
        # self.TY = self.transform_points(self.Yr)
        iTM['Ym'] = Ty
        iTM['Ys'] = Sy
        iTM['Xm'] = Tx
        iTM['Xs'] = Sx
        self.tmats = iTM
        self.tmats['transformer'] = itf

    def transform_point(self, Yr, use_keops=True, **kargs):
        if self.tmats['transformer'] in 'ESARTILOP':
            return self.homotransform_point(Yr, self.tmats['tform'], inverse=False, xp=self.xp)
        elif self.tmats['transformer']  == 'D':
            return self.ccf_deformable_transform_point(Yr, **self.tmats, 
                                                        use_keops=use_keops, **kargs)
        elif self.tmats['transformer']  == 'H':
            return self.ccf_hierarchical_transform_point(Yr, **self.tmats, 
                                                        use_keops=use_keops, **kargs)
        else:
            raise(f"Unknown transformer: {self.tmats['transformer']}")

    def get_transformer(self):
        return self.tmats