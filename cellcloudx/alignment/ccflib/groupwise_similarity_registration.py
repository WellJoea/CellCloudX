from .groupwise_emregistration import gwEMRegistration

class gwSimilarityRegistration(gwEMRegistration):
    def __init__(self, *args, isoscale=False,
                 fix_R=False, fix_t=False, fix_s=False,
                 s_clip=None, transparas=None,
                 #R=None, t=None, s=None,  TODO
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_core = 'gw-Similarity'
        self.normal = 'isoscale' if self.normal is None else self.normal

        self.maxiter = self.maxiter or 400
        self.inneriter = self.inneriter or 10
        assert self.maxiter % self.inneriter == 0, "maxiter must be multiple of inneriter"
        self.tol =  1e-9 if self.tol is None else self.tol

        self.transformer =  self.scalar2vetor('S', self.L)
        if self.root is not None: self.transformer[self.root] = 'E'

        self.init_transparas( isoscale=isoscale, fix_R=fix_R, fix_t=fix_t, 
                             fix_s=fix_s, s_clip=s_clip, transparas=transparas)
        self.init_transformer()
        self.normal_Xs()
        self.normal_Fs()

    def init_transparas(self, isoscale=False, fix_R=False, fix_t=False, fix_s=False, s_clip=None, transparas=None):
        self.transparas = {}
        isoscale = self.scalar2vetor(isoscale, self.L) 
        fix_R =  self.scalar2vetor(fix_R, self.L) 
        fix_t =  self.scalar2vetor(fix_t, self.L) 
        fix_s =  self.scalar2vetor(fix_s, self.L) 
        s_clip = self.init_sclip(s_clip)

        for iL in range(self.L):
            self.transparas[iL] = {'isoscale': isoscale[iL], 'fix_R': fix_R[iL], 'fix_t': fix_t[iL], 
                                   'fix_s': fix_s[iL], 's_clip': s_clip[iL]}
            if transparas is not None:
                self.transparas[iL] = {
                    **self.transparas[iL]
                    **transparas.get('S', {})
                    **transparas.get(iL, {}).get('S',{}),
                    **transparas.get(iL, {})
                }

    def init_sclip(self, s_clip):
        if s_clip is None:
            s_clip = self.scalar2vetor(None, self.L, force=True)
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

    def update_transformer0(self):
        tmat = self.xp.eye(self.D+1, dtype=self.floatxx, device=self.device).expand(self.L,-1,-1).clone()
        tmat[:,:self.D, :self.D] = self.R @ self.s
        tmat[:,:self.D,  self.D] = self.t
        return tmat

    def update_normalize0(self):
        self.R = self.R.to(self.floatxx)
        self.t = self.t.to(self.floatxx)
        self.s = self.s.to(self.floatxx)

        self.tmat = self.update_transformer()

        # t = self.xp.vstack([ self.t[il]*self.Xs[il].to(self.device, dtype=self.floatxx) - 
        #                     self.Xm[il].to(self.device, dtype=self.floatxx) @ ( self.R[il] @ self.s[il] ).T
        #                     for il in range(self.L) ])
        self.t = self.t* self.Xs[:, None].to(self.device, dtype=self.floatxx) - \
                    self.xp.einsum('li,lji,lkj->lk', self.Xm.to(self.device, dtype=self.floatxx), self.s, self.R)
        
        if self.root is not None:
            self.t += self.Xm[self.root].to(self.device, dtype=self.floatxx)

        self.tform = self.update_transformer()
        # TYs = [self.TYs[il] * self.Xs[il] for il in range(self.L)]
        self.TYs = self.transform_point(self.Xr)