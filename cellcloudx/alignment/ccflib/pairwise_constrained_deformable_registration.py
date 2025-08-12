from builtins import super
import numpy as np
import numbers
from .pairwise_deformable_registration import pwDeformableRegistration


class pwConstrainedDeformableRegistration(pwDeformableRegistration):
    """
    Constrained deformable registration.

    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.

    beta: float(positive)
        Width of the Gaussian kernel.

    e_alpha: float (positive)
        Reliability of correspondence priors. Between 1e-8 (very reliable) and 1 (very unreliable)
    
    source_id: numpy.ndarray (int) 
        Indices for the points to be used as correspondences in the source array

    target_id: numpy.ndarray (int) 
        Indices for the points to be used as correspondences in the target array

    """

    def __init__(self, *args, e_alpha = None, Y_src = None, X_dst= None,
                 X_idx = None,
                 e_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
 
        self.e_alpha = 1e2 if e_alpha is None else e_alpha
        self.Y_src = Y_src
        self.X_dst = X_dst
        self.X_idx = X_idx
        self.e_weight = e_weight
        self.e_pwexpectation()

    def e_pwexpectation(self):
        assert self.Y_src is not None, "Source points not provided"
        assert self.X_dst is not None, "Target points not provided"
        assert self.X_idx is not None, "Indices for target points not provided"
        
        self.Y_src = self.to_tensor(self.Y_src, device=self.device, dtype  = self.xp.int64)
        self.X_dst = self.to_tensor(self.X_dst, device=self.device, dtype  = self.xp.int64)
        self.X_idx = int(self.X_idx)

        if self.e_weight is None:
            self.e_weight = self.xp.ones(self.Y_src.shape[0], dtype=self.floatx)
        else:
            self.e_weight = self.to_tensor(self.e_weight, device=self.device, dtype=self.floatx)

        M, N = self.Y, self.Xs[self.X_idx]
        P = self.xp.sparse_coo_tensor( (self.Y_src, self.X_ds), self.e_weight, shape=(M, N), 
                                        dtype=self.floatx)

        self.P1_tilde = self.xp.sum(P, axis=1)
        self.PX_tilde = self.xp.sparse.mm(P, self.Xs[self.X_idx])

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

        #ADD CONSTRAINT, check
        PX += (self.e_alpha*sigma2_exp)*self.PX_tilde
        P1 += (self.e_alpha*sigma2_exp)*self.P1_tilde

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

