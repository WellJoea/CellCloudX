from builtins import super
import numpy as np
import numbers
from .emregistration import EMRegistration

class DeformableRegistration(EMRegistration):
    """
    Deformable registration.

    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.

    beta: float(positive)
        Width of the Gaussian kernel.
    
    low_rank: bool
        Whether to use low rank approximation.
    
    num_eig: int
        Number of eigenvectors to use in lowrank calculation.
    """

    def __init__(self, *args, alpha=None, beta=None,
                 agg_fg=False, beta_fg=None, low_rank=True, kw=None, num_eig=100,
                   **kwargs):
        super().__init__(*args, **kwargs)
        if alpha is not None and (not isinstance(alpha, numbers.Number) or alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter alpha(lambda). Instead got: {}".format(alpha))

        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError(
                "Expected a positive value for the width of the coherent Gaussian kerenl. Instead got: {}".format(beta))

        self.alpha = 1 if alpha is None else alpha
        self.beta_ = 2 if beta is None else beta
        self.kw = kw

        self.low_rank_ = low_rank
        self.num_eig_ = num_eig
        self.agg_fg = agg_fg
        self.beta_fg = beta_fg
        self.W = np.zeros((self.M, self.D))
        
        self.init_G()
        self.update_transformer()

    @property
    def beta(self):
        return self.beta_
    @property
    def num_eig(self):
        return self.num_eig_
    @property
    def low_rank(self):
        return self.low_rank_

    def init_G(self):
        G, self.beta_  = self.kernal_gmm(self.Y, self.Y, sigma2=self.beta_, temp=1)
        if self.agg_fg:
            G1, beta_fg = self.kernal_gmm(self.Y_feat, self.Y_feat, sigma2=self.beta_fg)

        if self.low_rank is True:
            if self.agg_fg:
                G = G * G1
            self.G = G
            self.Q, self.S = self.low_rank_eigen(G, self.num_eig)
            self.inv_S = np.diag(1./self.S)
            self.S = np.diag(self.S)
            self.E = 0.
        else:
            if self.agg_fg:
                G = G * G1
            self.G = G

    def optimization(self):
        self.expectation()
        self.update_transform()
        self.update_transformer()
        self.transform_point()
        self.update_variance()
        self.iteration += 1

    def update_transform(self):
        if self.low_rank is False:
            # A = self.xp.dot(self.xp.diag(self.P1), self.G) + \
            #     self.alpha * self.sigma2 * self.xp.eye(self.M)
            # B = self.PX - self.xp.dot(self.xp.diag(self.P1), self.Y)
            A = self.xp.multiply(self.P1, self.G).T + \
                self.alpha * self.sigma2 * self.xp.eye(self.M)
            B = self.PX - self.xp.multiply(self.Y.T, self.P1).T

            self.W = self.xp.linalg.solve(A, B)

        elif self.low_rank is True:
            # dP = np.diag(self.P1)
            # dPQ = np.matmul(dP, self.Q)
            #F = self.PX - np.matmul(dP, self.Y)
            dPQ = np.multiply(self.Q.T, self.P1).T
            F = self.PX - np.multiply(self.Y.T, self.P1).T

            self.W = 1 / (self.alpha * self.sigma2) * (F - np.matmul(dPQ, 
                (np.linalg.solve((self.alpha * self.sigma2 * self.inv_S + np.matmul(self.Q.T, dPQ)),
                                (np.matmul(self.Q.T, F))))))
            QtW = np.matmul(self.Q.T, self.W)
            # self.E = self.E + self.alpha / 2 * np.trace(np.matmul(QtW.T, np.matmul(self.S, QtW)))

    def update_transformer(self):
        if self.low_rank is False:
            self.tmat = self.xp.dot(self.G, self.W)

        elif self.low_rank is True:
            self.tmat = np.matmul(self.Q, np.matmul(self.S, np.matmul(self.Q.T, self.W)))
        self.tform = self.tmat

    def transform_point(self, Y=None, reset_G=False ):
        if Y is None:
            self.TY = self.Y + self.tmat
        else:
            if reset_G or np.array_equal(self.Y, Y.astype(self.Y.dtype)):
                G = self.kernal_gmm(Y, self.Y, sigma2=self.beta)[0]
                tmat = np.dot(G, self.W)
            else:
                tmat = self.tmat

            if self.normal:
                Y_n = (Y -self.Ym)/self.Ys + tmat
                Y_n = Y_n * self.Xs + self.Xm
            else:
                Y_n = Y + tmat
            return Y_n

    def update_variance(self):
        qprev = self.sigma2

        # trxPx = np.dot(np.transpose(self.Pt1), 
        #              np.sum(np.multiply(self.X, self.X), axis=1))
        # tryPy = np.dot(np.transpose(self.P1),  
        #              np.sum(np.multiply(self.TY, self.TY), axis=1))
        
        trxPx = np.sum( self.Pt1.T * np.sum(np.multiply(self.X, self.X), axis=1) )
        tryPy = np.sum( self.P1.T * np.sum(np.multiply(self.TY, self.TY), axis=1))
        trPXY = np.sum(np.multiply(self.TY, self.PX))

        self.sigma2 = (trxPx - 2 * trPXY + tryPy) / (self.Np * self.D)
        if self.sigma2 <= 0:
            # self.sigma2 = self.tolerance / 10
            self.sigma2 = 0.1

        self.diff = np.abs(self.sigma2 - qprev)

    def update_normalize(self):
        # self.tmat = self.tmat * self.Xs + self.Xm
        self.TY = self.TY * self.Xs + self.Xm
        self.tform = self.TY - (self.Y * self.Ys + self.Ym)

    def get_registration_parameters(self):
        """
        Return the current estimate of the deformable transformation parameters.


        Returns
        -------
        self.G: numpy array
            Gaussian kernel matrix.

        self.W: numpy array
            Deformable transformation matrix.
        """
    
        return self.H , self.G, self.W
