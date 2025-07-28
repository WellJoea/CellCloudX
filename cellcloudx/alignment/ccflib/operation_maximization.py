import numpy as np
import torch as th
from .manifold_regularizers import deformable_regularizer, gwdeformable_regularizer
from tqdm import tqdm

from ...transform import homotransform_points, homotransform_point, ccf_deformable_transform_point, homotransform_mat

class pwMaximization():
    def __init__(self):
        super().__init__()
        self.init_tmat = init_tmat
        self.homotransform_points = homotransform_points
        self.ccf_deformable_transform_point = ccf_deformable_transform_point
        self.homotransform_mat = homotransform_mat
        self.homotransform_point = homotransform_point
    
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

    def pair_rigid_maximization(self, P, P1, Pt1, Np, X, Y,):
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

    def pair_affine_maximization(self, P, P1, Pt1, Np, X, Y, lr={}):
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

    def rigid_maximization(self, Pt1, P1, Np, PX, Y, X):
        xp, D, floatx, device, eps = self.xp, self.D, self.floatx, self.device, self.eps
        fix_s, s_clip, sigma2_min = self.fix_s, self.s_clip, self.sigma2_min

        muX = xp.divide(xp.sum(PX, 0), Np)
        muY = xp.divide(Y.transpose(1,0) @ P1, Np)

        X_hat = X - muX
        Y_hat = Y - muY

        A = PX.transpose(1,0) @ Y_hat - xp.outer(muX, P1 @ Y_hat)
        U, S, V = xp.linalg.svd(A, full_matrices=True)
        S = xp.diag(S)
        C = xp.eye(D, dtype=floatx, device=device)
        C[-1, -1] = xp.linalg.det(U @ V)
        R = U @ C @ V
        # R = U @ V

        if xp.linalg.det(R) < 0:
            U[:, -1] = -U[:, -1]
            R = U @ C @ V

        trAR = xp.trace(A.transpose(1,0) @ R)
        trXPX = xp.sum(Pt1 * xp.sum(X_hat * X_hat, 1))
        trYPY = xp.sum(P1 * xp.sum(Y_hat * Y_hat, 1))

        if fix_s is False:
            s = trAR/trYPY
        if s_clip is not None:
            s = xp.clip(s, *s_clip)

        t = muX - s * (R @ muY)
        TY = s * (Y @ R.transpose(1,0)) +  t #.T

        if fix_s is False:
            sigma2 = (trXPX  - s * trAR) / (Np * D)
        else:
            sigma2 = (trXPX + s* s* trYPY - 2* s* trAR) / (Np * D)

        if sigma2 < eps: 
            sigma2 = xp.clip(self.sigma_square(X, TY), min=sigma2_min)
    
        Q = (Np * D)/2*(1+xp.log(sigma2))

        return [[R, s, t,], TY, sigma2, Q]

    def similarity_maximization(self, Pt1, P1, Np, PX, Y, X):
        xp, D, floatx, device, eps, sigma2_min = self.xp, self.D, self.floatx, self.device, self.eps,  self.sigma2_min
        isoscale, fix_R, fix_s, fix_t, s_clip  = self.isoscale, self.fix_R, self.fix_s, self.fix_t, self.s_clip
    
        if not fix_t:
            muX = xp.divide(xp.sum(PX, 0), Np)
            muY = xp.divide(Y.transpose(1,0) @ P1, Np)

            X_hat = X - muX
            Y_hat = Y - muY
            A = PX.transpose(1,0) @ Y_hat - xp.outer(muX, P1 @ Y_hat)
        else:
            X_hat = X
            Y_hat = Y
            A = PX.transpose(1,0) @ Y_hat

        trXPX = xp.sum(Pt1 * xp.sum(X_hat * X_hat, 1))

        if isoscale:
            if not fix_R:
                U, S, V = xp.linalg.svd(A, full_matrices=True)
                C = xp.eye(D, dtype=floatx, device=device)
                C[-1, -1] = xp.linalg.det(U @ V)
                R = U @ C @ V
                if xp.linalg.det(R) < 0:
                    U[:, -1] = -U[:, -1]
                    R = U @ C @ V

            trAR = xp.trace(A.transpose(1,0) @ R )
            trYPY = xp.sum(P1 * xp.sum(Y_hat * Y_hat, 1))

            if not fix_s:
                s = trAR/trYPY
                s = xp.eye(D).to(device) * s
            
            if s_clip is not None:
                s = xp.clip( s.diagonal(), *s_clip).diag()
                
            if not fix_t:
                t = muX - R @ s @ muY

            TY = Y @ (R @ s).T +  t #.T
            sigma2 = (trXPX + s[0,0]* s[0,0]* trYPY - 2* s[0,0] * trAR) / (Np * D)
        else:
            YPY = xp.sum( Y_hat * Y_hat *P1.unsqueeze(1), 0)
            YPY.masked_fill_(YPY == 0, eps)
            if (not fix_s) and (not fix_R):
                max_iter = 70
                error = True
                iiter = 0
                C = xp.eye(D, dtype=floatx, device=device)
                while (error):
                    U, S, V = xp.linalg.svd(A @ s, full_matrices=True) #A @ s.T
                    C[-1, -1] = xp.linalg.det(U @ V)
                    R = U @ C @ V
                    if xp.linalg.det(R) < 0:
                        U[:, -1] = -U[:, -1]
                        R = U @ C @ V
                    s_pre = s
                    s = xp.diagonal(A.T @ R)/YPY
                    if s_clip is not None:
                        s = xp.clip(s, *s_clip)
                    s = xp.diag(s) 
                    iiter += 1
                    error = (xp.linalg.norm(s - s_pre) > 1e-8) and (iiter < max_iter)

            elif (not fix_R) and fix_s:
                U, S, V = xp.linalg.svd(A @ s, full_matrices=True)
                C = xp.eye(D, dtype=floatx, device=device)
                C[-1, -1] = xp.linalg.det(U @ V)
                R = U @ C @ V
                if xp.linalg.det(R) < 0:
                    U[:, -1] = -U[:, -1]
                    R = U @ C @ V
    
            elif (not fix_s) and fix_R:
                s = xp.diagonal(A.T @ R)/YPY
                if s_clip is not None:
                    s = xp.clip(s, *s_clip)
                s = xp.diag(s) 

            if not fix_t:
                t = muX -  (R @ s) @ muY

            TY = Y @ (R @ s).transpose(1,0) +  t
            trARS = xp.trace(R @ s @ A.transpose(1,0))
            trSSYPY = xp.sum((s **2) * YPY)
            sigma2 = (trXPX  - 2*trARS + trSSYPY) / (Np * D)

        if sigma2 < eps: 
            sigma2 = xp.clip(self.sigma_square(X, TY), min=sigma2_min)

        Q = D * Np/2 * (1+xp.log(sigma2))
        return [[R, s, t,], TY, sigma2, Q]

    def affine_maximization(self, Pt1, P1, Np, PX, Y, X, lr={}):
        xp, D, floatx, device, eps = self.xp, self.D, self.floatx, self.device, self.eps
        sigma2_min = self.sigma2_min

        muX = xp.divide(xp.sum(PX, axis=0), Np)
        muY = xp.divide(Y.transpose(1,0) @ P1, Np)

        X_hat = X - muX
        Y_hat = Y - muY

        A = PX.transpose(1,0) @ Y_hat - \
                 xp.outer(muX,  P1 @ Y_hat)

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
        if lr['gamma1'] > 0: # TODO (P1WY)
            YtZY = ( lr['WY'].transpose(1,0) * P1) @ lr['WY']
            YPYh.add_( (2 * lr['gamma1'] * sigma2) * YtZY )

        try:
            YPYhv = xp.linalg.inv(YPYh)
            B = A @ YPYhv
        except:
            B_pre = B
            YPYh.diagonal().add_( lr['delta'] *sigma2)
            B = (A+ lr['delta']*sigma2*B_pre) @ xp.linalg.inv(YPYh)

        t = muX - B @ muY
        TY = Y @ B.transpose(1,0) +  t
        trAB = xp.trace(A @ B.transpose(1,0))
        trXPX = xp.sum(Pt1 * xp.sum(X_hat * X_hat, 1))

        sigma2 = (trXPX - trAB) / (Np * D)
        if sigma2 < eps: 
            sigma2 = xp.clip(self.sigma_square(X, TY), min=sigma2_min)

        Q = D * Np/2 * (1+xp.log(sigma2))
        return [[A, t,], TY, sigma2, Q]

    def deformable_maximization(self, Pt1, P1, Np, PX, Y, X, lr={}):
        xp, D, floatx, device, eps, sigma2_min = self.xp, self.D, self.floatx, self.device, self.eps,  self.sigma2_min
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

class gwMaximization():
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
                idr = gwdeformable_regularizer(xp = self.xp, verbose=False, **self.transparas[iL]) # deformable_regularizer v0
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
                self.tmats[iL]['s'] = s*self.xp.eye(self.D, dtype=self.floatx, device=self.device)
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
                    U, S, V = self.xp.linalg.svd(B @ s.T, full_matrices=True)
                    C[-1, -1] = self.xp.linalg.det(U @ V)
                    R_pre = R.clone()
                    R = U @ C @ V
                    if self.xp.linalg.det(R) < 0:
                        U[:, -1] = -U[:, -1]
                        R = U @ C @ V

                    s = self.xp.diagonal(B.T @ R)/H
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
                s = self.xp.diagonal(B.T @ self.tmats[iL]['R'])/H
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

def init_tmat(T, D, N= None, s_clip=None, xp =th, device=None, dtype=th.float32):
    itam = {}
    if T in ['E']:
        itam['R'] = xp.eye(D, device=device, dtype=dtype).clone()
        itam['t'] = xp.zeros(D, device=device, dtype=dtype).clone()
        itam['s'] = xp.tensor(1.0, device=device, dtype=dtype).clone()
        if s_clip is not None:
            itam['s'] = xp.clip(itam['s'], *s_clip)
    elif T in 'SRTILO':
        itam['R'] = xp.eye(D, device=device, dtype=dtype).clone()
        itam['t'] = xp.zeros(D, device=device, dtype=dtype).clone()
        itam['s'] = xp.eye(D, device=device, dtype=dtype).clone()
        if s_clip is not None:
            itam['s'] = xp.clip(itam['s'], *s_clip).diag().diag()
    elif T in ['A']:
        itam['A'] = xp.eye(D, device=device, dtype=dtype).clone()
        itam['t'] = xp.zeros(D, device=device, dtype=dtype).clone()
    elif T in ['D']:
        itam['W'] = xp.zeros((N,D), device=device, dtype=dtype).clone()
    return itam