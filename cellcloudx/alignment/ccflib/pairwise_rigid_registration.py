from .pairwise_emregistration import pwEMRegistration
from ...io._logger import logger

class pwRigidRegistration(pwEMRegistration):
    def __init__(self, *args, 
                   R = None, s = None, t=None,
                   fix_s=True, s_clip=None,
                   **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = 'E'
        self.tol =  1e-9 if self.tol is None else self.tol
        self.normal = 'isoscale' if self.normal is None else self.normal #or global
        self.maxiter = self.maxiter or 150

        self.normal_XY()
        self.normal_features()
        self.init_transparas(fix_s=fix_s, s_clip=s_clip)
        self.init_transformer( R=R, s = s, t=t)

    def init_transparas(self, **kargs):
        self.transparas = self.default_transparas()[self.transformer]
        self.transparas.update(kargs)

    def init_transformer(self, **kargs):
        self.tmats = self.init_tmat(self.transformer, self.D, 
                        xp=self.xp, device=self.device, dtype=self.floatx, **kargs)
        #TODO check
        if not kargs.get('s', None) is None:
            self.tmats['s'] = self.tmats['s']/self.Sy
        if not kargs.get('t', None) is None:
            self.tmats['t'] = (self.tmats['t'] - self.My)/self.Sy