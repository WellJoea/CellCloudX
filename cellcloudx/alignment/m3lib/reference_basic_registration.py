import torch as th
import numpy as np

from .reference_emregistration import rfEMRegistration

class rfBasicRegistration(rfEMRegistration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'rf-basic'

    def mask_paras(self,):
        self.delta = self.to_tensor(self.delta, dtype=self.floatx, device=self.device)
        self.zeta = self.to_tensor(self.zeta, dtype=self.floatx, device=self.device)
        self.eta = self.to_tensor(self.eta, dtype=self.floatx, device=self.device)

        tfs =  np.array(self.transformer)
        tfs[tfs!='D'] = 'L'
        tfs = th.asarray(tfs[:, None] == tfs[None,:]).to(self.device)
        self.mask = (self.delta >0 ) & tfs

        self.delta *= self.mask
        self.zeta *= self.mask
        self.eta *= self.mask