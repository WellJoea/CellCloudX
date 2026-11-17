from builtins import super
import numpy as np
import numbers
from .pairwise_deformable_registration import pwDeformableRegistration


class pwConstrainedDeformableRegistration(pwDeformableRegistration):
    """
    Constrained deformable registration.

    Attributes
    ----------
    """

    def __init__(self, *args, e_alpha = None, pairs=None,
                 e_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
 
        self.e_alpha = 1e4 if e_alpha is None else e_alpha
        self.pairs = pairs
        self.e_weight = e_weight
        self.e_pwexpectation()

    def e_pwexpectation(self):
        assert self.pairs is not None, "pairs indices not provided"
        if isinstance(self.pairs, (list, tuple)):
            assert len(self.pairs) == self.L
        else:
            self.pairs = [self.pairs]
        LP = len(self.pairs)

        if isinstance(self.e_alpha, (float, int)):
            self.e_alpha = [ self.e_alpha for i in range(LP) ]

        self.pair_num, self.P1_tilde, self.PX_tilde = [], 0, 0
        for ilp in range(LP):
            ipair = self.pairs[ilp]
            if ipair is None or len(ipair) == 0:
                self.pair_num.append(0)         
                continue
            self.pair_num.append(len(ipair))

            ipair = self.to_tensor(ipair, device=self.device, dtype  = self.xp.int64)
            M, N = self.Y.shape[0], self.Xs[ilp].shape[0]
            if self.e_weight is None:
                e_weight = self.xp.ones( (ipair.shape[0],), dtype=self.floatx, device=self.device)
            else:
                e_weight = self.to_tensor(self.e_weight[ilp], device=self.device, dtype  = self.floatx)
                assert e_weight.shape[0] == ipair.shape[0]

            P = self.xp.sparse_coo_tensor( ipair[:,[1,0]].T, e_weight, size=(M, N), 
                                            dtype=self.floatx, device=self.device)
            P1_tilde = self.xp.sum(P, axis=1).to_dense()
            PX_tilde = self.xp.sparse.mm(P, self.Xs[ilp]).to_dense()

            self.P1_tilde += P1_tilde*self.e_alpha[ilp]
            self.PX_tilde += PX_tilde*self.e_alpha[ilp]

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
        # PX += (self.e_alpha*sigma2_exp)*self.PX_tilde
        # P1 += (self.e_alpha*sigma2_exp)*self.P1_tilde
        PX += self.PX_tilde
        P1 += self.P1_tilde

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

