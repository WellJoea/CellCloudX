from .groupwise_emregistration import gwEMRegistration

class gwAffineRegistration(gwEMRegistration):
    def __init__(self, *args, gamma1=None, delta=0.1, 
                 theta=0.1, kw=15,  transparas= None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'gw-Affine'
        self.normal = 'isoscale' if self.normal is None else self.normal

        self.maxiter = self.maxiter or 400
        self.inneriter = self.inneriter or 10
        assert self.maxiter % self.inneriter == 0, "maxiter must be multiple of inneriter"
        self.tol =  1e-9 if self.tol is None else self.tol

        self.transformer = self.scalar2vetor('A', self.L)
        if self.root is not None: self.transformer[self.root] = 'E'
    
        self.init_transparas(delta, transparas)
        self.init_transformer()
        self.normal_Xs()
        self.normal_Fs()
    
    def init_transparas(self, delta=0.1, transparas=None):
        self.transparas = {}
        delta = self.scalar2vetor(delta, self.L)
        for iL in range(self.L):
            self.transparas[iL] = {'delta': delta[iL]}
            if transparas is not None:
                self.transparas[iL] = {
                    **self.transparas[iL]
                    **transparas.get('A', {})
                    **transparas.get(iL, {}).get('A',{}),
                    **transparas.get(iL, {})
                }

    def transform_point0(self, Ys=None):
        if Ys is None:
            self.TYs = self.homotransform_points(self.Xa, self.tmat, inverse=False, xp=self.xp)
            return
        else:
            return self.homotransform_points(Ys, self.tform, inverse=False, xp=self.xp)

    def update_transformer0(self):
        tmat = self.xp.eye(self.D+1, dtype=self.floatxx, device=self.device).expand(self.L,-1,-1).clone()
        tmat[:,:self.D, :self.D] = self.A
        tmat[:,:self.D,  self.D] = self.t
        return tmat

    def update_normalize0(self):
        self.A = self.A.to(self.floatxx)
        self.t = self.t.to(self.floatxx)
        self.tmat = self.update_transformer()
    
        t = self.xp.vstack([ self.t[il]*self.Xs[il].to(self.device, dtype=self.floatxx) - 
                            self.Xm[il].to(self.device, dtype=self.floatxx) @ self.A[il].T
                            for il in range(self.L) ])
        self.t = self.t* self.Xs[:, None].to(self.device, dtype=self.floatxx) - \
                    self.xp.einsum('li,lji->lj', self.Xm.to(self.device, dtype=self.floatxx), self.A)

        if self.root is not None:
            self.t += self.Xm[self.root].to(self.device, dtype=self.floatxx)
    
        self.tform = self.update_transformer()
        # TYs = [self.TYs[il] * self.Xs[il] + self.Xm[self.root].to(self.device, dtype=self.floatxx)
        #         for il in range(self.L)]
        self.TYs = self.transform_point(self.Xr)
