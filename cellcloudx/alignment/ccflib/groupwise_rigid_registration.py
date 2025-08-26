from .groupwise_emregistration import gwEMRegistration

class gwRigidRegistration(gwEMRegistration):
    def __init__(self, *args, 
                   fix_s=True, s_clip=None,  transparas=None,
                   **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'gw-Euclidean'
        self.normal = 'isoscale' if self.normal is None else self.normal

        self.maxiter = self.maxiter or 400
        self.inneriter = self.inneriter or 1
        assert self.maxiter % self.inneriter == 0, "maxiter must be multiple of inneriter"
        self.tol =  1e-9 if self.tol is None else self.tol

        self.transformer =  list('E' * self.L)
    
        self.init_transparas(fix_s, s_clip, transparas)
        self.init_transformer()
        self.normal_Xs()
        self.normal_Fs()

    def init_transparas(self, fix_s=True, s_clip=None, transparas=None):
        self.transparas = {}
        fix_s = self.scalar2vetor(fix_s, self.L)
        s_clip = self.init_sclip(s_clip)
        for iL in range(self.L):
            self.transparas[iL] = {'s_clip': s_clip[iL], 'fix_s': fix_s[iL]}
            if transparas is not None:
                self.transparas[iL] = {
                    **self.transparas[iL]
                    **transparas.get('E', {})
                    **transparas.get(iL, {}).get('E',{}),
                    **transparas.get(iL, {})
                }

    def init_sclip(self, s_clip):
        if s_clip is None:
            s_clip = self.scalar2vetor(None, self.L)
        else:
            assert self.xp.tensor(s_clip).ndim >0
            if (self.xp.tensor(s_clip).ndim == 1) \
                and (self.xp.tensor(s_clip).shape[0] == 2):
                s_clip = self.scalar2vetor(s_clip, self.L, force=True)
            elif (self.xp.tensor(s_clip).ndim == 2) \
                and (tuple(self.xp.tensor(s_clip).shape) == (self.L, 2)):
                s_clip = s_clip
            else:
                raise ValueError('s_clip should be a scalar or a 1D vector of length 2 or a 2D matrix of shape (L, 2)')
        return s_clip
        
    def transform_point0(self, Ys=None):
        if Ys is None:
            self.TYs = self.homotransform_points(self.Xa, self.tmat, inverse=False, xp=self.xp)
            return
        else:
            return self.homotransform_points(Ys, self.tform, inverse=False, xp=self.xp)

    def update_normalize0(self):
        '''
        tmat: not scaled transform matrix
        tform: scaled transform matrix
        '''
        self.R = self.R.to(self.floatxx)
        self.t = self.t.to(self.floatxx)
        self.s = float(self.s)

        self.tmat = self.update_transformer()

        self.s *= self.Xs/self.Ys 
        self.t = (self.t * self.Xs + self.Xm) - self.s * self.R @ self.Ym
        
        self.tform =self.update_transformer()
        # self.tform = self.Xf @ self.tmat @ self.xp.linalg.inv(self.Yf)
        # TY = self.TY * self.Xs + self.Xm
        # self.TY  = self.s * (self.Yr.to(self.device) @ self.R.T) +  self.t
        self.TY = self.transform_point(self.Yr.to(self.device))