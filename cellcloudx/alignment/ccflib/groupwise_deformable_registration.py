from .groupwise_emregistration import gwEMRegistration


class gwDeformableRegistration(gwEMRegistration):
    def __init__(self, *args, 
                 beta=4.0, alpha=5e2, 
                  low_rank= 3000,
                  low_rank_type = 'keops',
                  fast_low_rank = 10000, num_eig=100, 
                    gamma1=0, gamma2=0,  kw= 15, kl=15,
                    alpha_decayto = 1,  use_p1=False, p1_thred = 0,
                    gamma_growto = 1, kd_method='sknn',  transparas=None,
                  **kwargs):
    
        super().__init__(*args, **kwargs)
        self.reg_core = 'gw-Deformable'
        self.normal = 'isoscale' if self.normal is None else self.normal

        self.maxiter = self.maxiter or 400
        self.inneriter = self.inneriter or 10
        assert self.maxiter % self.inneriter == 0, "maxiter must be multiple of inneriter"
        self.tol =  1e-9 if self.tol is None else self.tol

        self.transformer = self.scalar2vetor('D', self.L)
        if self.root is not None: self.transformer[self.root] = 'E'
    
        self.init_transparas(beta=beta, alpha=alpha,
                    low_rank=low_rank,
                    fast_low_rank=fast_low_rank, num_eig=num_eig,
                    gamma1=gamma1, gamma2=gamma2, kw=kw, kl=kl,
                    low_rank_type=low_rank_type,
                    alpha_decayto=alpha_decayto, use_p1=use_p1, p1_thred=p1_thred,
                    gamma_growto=gamma_growto, kd_method=kd_method, transparas=transparas)
        self.init_transformer()
        self.normal_Xs()
        self.normal_Fs()
    
    def init_transparas(self, **kwargs):
        transparas =  kwargs.get('transparas', None) 
        kwargs = { k: self.scalar2vetor(v, self.L) for k,v in kwargs.items() }
        self.transparas = {}
        for iL in range(self.L):
            ipara = { k: v[iL] for k,v in kwargs.items() if k != 'transparas' }
            ipara['alpha_mu'] =  ipara['alpha_decayto'] **(1.0/ float(self.maxiter-1)) 
            ipara['gamma_nu'] =  ipara['gamma_growto'] **(1.0/ float(self.maxiter-1)) 
            ipara['use_low_rank'] = ( ipara['low_rank'] if type(ipara['low_rank']) == bool  
                                            else bool(self.Ns[iL] >= ipara['low_rank']) )
            ipara['use_fast_low_rank'] = ( ipara['fast_low_rank'] if type(ipara['fast_low_rank']) == bool  
                                            else bool(self.Ns[iL] >= ipara['fast_low_rank']) )
            ipara['fast_rank'] = ipara['use_low_rank'] or ipara['use_fast_low_rank']
            self.transparas[iL] = ipara
            if transparas is not None:
                self.transparas[iL] = {
                    **self.transparas[iL]
                    **transparas.get('D', {})
                    **transparas.get(iL, {}).get('D',{}),
                    **transparas.get(iL, {})
                }
            for iarr in ['alpha', 'gamma1', 'gamma2', 'p1_thred']:
                self.transparas[iL][iarr] = self.xp.tensor(self.scalar2vetor(self.transparas[iL][iarr], self.L), dtype=self.floatx)
    
    def transform_point0(self, Y=None, root = None ): #TODO
        TYs = []
        root = self.root if root is None else root
        Xm = 0 if root is None else self.Xm[root]
        for iL in range(self.L):
            if root is not None and iL == root:
                TYs.append(Y[iL])
            else:
                iTY =  self.ccf_deformable_transform_point(
                            Y[iL], Y=self.Xa[iL], Ym=self.Xm[iL], Ys=self.Xs[iL], 
                            Xm=Xm, Xs=self.Xs[iL], beta=self.beta[iL], 
                            W=self.W[iL], G=self.G[iL], U=self.U[iL], S=self.S[iL])
                TYs.append(iTY)
        return TYs

    def update_normalize0(self):
        for ia in ['G', 'U', 'S']:
            setattr(self, ia, {})
            for iL in range(self.L):
                iv = getattr(self.DR[iL], ia, None)
                if iv is not None:
                    iv = iv.clone().to(self.floatxx)
                getattr(self, ia)[iL] = iv
                if iv is not None:
                    delattr(self.DR[iL], ia)
        self.clean_cache()

        for ia in ['W', 'Xa', 'Xm', 'Xs', 'Xr']:
            if type(getattr(self, ia)) == list:
                for iL in range(self.L):
                    getattr(self, ia)[iL] = getattr(self, ia)[iL].to(device=self.device, dtype=self.floatxx)
            else:
                setattr(self, ia, getattr(self, ia).to(device=self.device, dtype=self.floatxx))
        
        if self.root is not None:
            Xm = self.Xm[self.root].to(self.device, dtype=self.floatxx)
        else:
            Xm = 0

        self.TYs = [self.TYs[il] * self.Xs[il] + Xm for il in range(self.L)]
        # self.TYs = self.transform_point(self.Xr)