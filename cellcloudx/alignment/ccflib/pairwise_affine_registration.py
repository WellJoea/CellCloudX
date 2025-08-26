from .pairwise_emregistration import pwEMRegistration
from .manifold_regularizers import lle_w, gl_w
from ...io._logger import logger

class pwAffineRegistration(pwEMRegistration):
    def __init__(self, *args, 
                 A=None, t = None,
                 gamma1=None, delta=0.01, 
                 kw=15, kd_method='sknn', **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = 'A'

        self.tol =  1e-9 if self.tol is None else self.tol
        self.normal = 'isoscale' if self.normal is None else self.normal #or global
        self.maxiter = self.maxiter or 150

        self.normal_XY()
        self.normal_features()
        # self.constrained_pairs()
        self.init_transparas(delta=delta, gamma1=gamma1, kw=kw, kd_method=kd_method)
        self.init_transformer( A=A, t=t)
        self.init_L()

    def init_transparas(self, delta=0.01, gamma1 = 0, kw=15, kd_method='sknn'):
        self.transparas = self.default_transparas()[self.transformer]
        self.transparas['delta'] = delta or 0.01
        self.transparas['gamma1'] = gamma1 or 0
        self.transparas['kw'] = kw
        self.transparas['kd_method'] = kd_method


    def init_transformer(self, A=None, t=None):
        self.tmats = self.init_tmat(self.transformer, self.D, A=A, t=t, 
                        xp=self.xp, device=self.device, dtype=self.floatx)
        #TODO check
        if not A is None:
            self.tmats['A'] = self.tmats['A']/self.Sy
        if not t is None:
            self.tmats['t'] = (self.tmats['t'] - self.My)/self.Sy

    def init_L(self,  use_unique=False,):
        if self.transparas['gamma1'] > 0: #TODO
            logger.info(f'compute Y lle...')
            W = lle_w(self.Y, use_unique = False, 
                      kw=self.transparas['kw'] , 
                      method=self.transparas['kd_method'])
            # W = gl_w(self.Y, kw=kw, method=method)
            self.WY = (W @ self.Y).to(self.device)
            