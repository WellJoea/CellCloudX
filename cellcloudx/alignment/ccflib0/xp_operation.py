import numpy as np
import random
import os

class xpopt():
    def __init__(self, device='cpu', core='numpy', floatx=None,
                 seed=None):
        self.device = device
        self.core = core
        
        if core == 'numpy':
            self.device = device
            self.xp = np
            _floatx = 'float64'
        elif core == 'cupy':
            import cupy
            self.xp = cupy
            _floatx = 'float32'
        elif core == 'torch':
            import torch
            self.xp = torch
            _floatx = 'float32'

        floatx = _floatx if floatx is None else floatx
        self.floatx = eval(f'self.xp.{floatx}')
        self.seed(seed)
        self.eps = self.xp.finfo(self.xp.float64).eps



    def from_numpy(self, x, **kargs):
        if self.core == 'numpy':
            return x
        elif self.core == 'cupy':
            return self.xp.asarray(x, **kargs)
        elif self.core == 'torch':
            return self.xp.from_numpy(x, **kargs)

    def to(self, x, device=None):
        device = self.device if device is None else device

        if self.core == 'numpy':
            return self.xp
        elif self.core == 'cupy':
            return self.xp.asarray(device)
        elif self.core == 'torch':
            return self.xp.to(device)
    
    def to_numpy(self, x):
        if self.core == 'numpy':
            return self.xp
        elif self.core == 'cupy':
            return self.xp.asnumpy()
        elif self.core == 'torch':
            return self.xp.cpu().numpy()

    def seed(self, seed=None):
        if not seed is None:
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            if self.core in ['cupy', 'numpy']:
                self.xp.random.seed(seed)

            if self.core == 'torch':
                self.xp.manual_seed(seed)
                self.xp.cuda.manual_seed(seed)
                self.xp.cuda.manual_seed_all(seed)
                self.xp.mps.manual_seed(seed)
                self.xp.backends.cudnn.deterministic = True
                self.xp.backends.cudnn.benchmark = False
                # torch.backends.cudnn.enabled = False
                # try:
                #     torch.use_deterministic_algorithms(True)
                # except:
                #     pass

class torch_opt(xpopt):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
    
    def array(self, x):
        return self.xp.from_numpy(x)
    