from .pairwise_emregistration import pwEMRegistration
from ...io._logger import logger

class pwSimilarityRegistration(pwEMRegistration):
    def __init__(self, *args, 
                   R = None, s = None, t=None,
                   transformer='S',
                   isoscale=False, fix_R=False, fix_t=False, 
                   fix_s=False, s_clip=None, transparas={},
                   **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = transformer
        self.tol =  1e-9 if self.tol is None else self.tol
        self.normal = 'isoscale' if self.normal is None else self.normal #or global
        self.maxiter = self.maxiter or 150

        self.tau2_decayto = self.scalar2vetor( 0.15 if self.tau2_decayto is None else self.tau2_decayto , L=self.L)   
        self.tau2_decaystop = self.scalar2vetor(0.8 if self.tau2_decaystop is None else self.tau2_decaystop, L=self.L)

        self.normal_XY()
        self.normal_features()
        self.init_transparas(fix_s=fix_s, s_clip=s_clip, fix_R=fix_R,
                              fix_t=fix_t, isoscale=isoscale, transparas=transparas)
        self.init_transformer( R=R, s = s, t=t)

    def init_transparas(self, transparas={}, **kargs):
        self.transparas = self.default_transparas()[self.transformer]
        self.transparas.update(kargs)
        self.transparas.update(transparas)

    def init_transformer(self, **kargs):
        self.tmats = self.init_tmat(self.transformer, self.D, 
                        xp=self.xp, device=self.device, dtype=self.floatx, **kargs)
        #TODO check
        if not kargs.get('s', None) is None:
            self.tmats['s'] = self.tmats['s']/self.Sy
        if not kargs.get('t', None) is None:
            self.tmats['t'] = (self.tmats['t'] - self.My)/self.Sy