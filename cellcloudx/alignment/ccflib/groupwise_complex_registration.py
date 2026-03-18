import re
from .groupwise_emregistration import gwEMRegistration

class gwComplexRegistration(gwEMRegistration):
    def __init__(self, *args, transformer='E',
                 transparas= None,
                  **kwargs):
    
        super().__init__(*args, **kwargs)
        self.reg_core = 'gw-complex'
        self.normal = 'global' if self.normal is None else self.normal

        self.maxiter = self.maxiter or 300
        self.inneriter = self.inneriter or 30
        assert self.maxiter % self.inneriter == 0, "maxiter must be multiple of inneriter"

        self.tau2_decayto = 0.2 if self.tau2_decayto is None else self.tau2_decayto 

        self.tol =  -1 if self.tol is None else self.tol
        self.alltype = 'ESADRTILOPN'
        if (type(transformer) == str):
            self.transformer =  list( re.sub(r"\s+", "", transformer))
            if len(self.transformer) == 1:
                self.transformer = self.transformer*self.L
            elif len(self.transformer) == self.L:
                pass
            else:
                raise ValueError('the length of transformer should be 1 or L')
        else:
            self.transformer = self.scalar2vetor(transformer, self.L)
            
        for i in self.transformer:
            if i not in self.alltype:
                raise ValueError(f'transformer {i} is not supported, should be one of {self.alltype}')
    
        # if self.root is not None: self.transformer[self.root] = 'E'

        self.init_transformer()
        self.init_transparas(transparas)
        self.normal_Xs()
        self.normal_Fs()

    def init_transparas(self, nparas, alltype='ESADRTILOPN'): 
        # self.default_transparas()
        nparas = {} if nparas is None else nparas
        self.transparas = {}
        for iL in range(self.L):
            itrans = self.transformer[iL]

            ipara = {**self.dparas[itrans], 
                     **nparas.get(itrans,{}),
                     **nparas.get(iL, {}).get(itrans,{}),
                     **nparas.get(iL, {}) #TODO
                     } 
            if itrans == 'D':
                ipara['alpha_mu'] = ipara['alpha_decayto']**(1.0/ float(self.maxiter-1)) 
                ipara['gamma_nu'] = ipara['gamma_growto']**(1.0/ float(self.maxiter-1))
            
                ipara['use_low_rank'] = ( ipara['low_rank'] if type(ipara['low_rank']) == bool  
                                                else bool(self.Ns[iL] >= ipara['low_rank']) )
                ipara['use_fast_low_rank'] = ( ipara['fast_low_rank'] if type(ipara['fast_low_rank']) == bool  
                                                else bool(self.Ns[iL] >= ipara['fast_low_rank']) )
                ipara['fast_rank'] = ipara['use_low_rank'] or ipara['use_fast_low_rank']
                ipara['alpha'] = self.scalar2vetor(ipara['alpha'], self.L)
                for iarr in ['alpha', 'gamma1', 'gamma2', 'p1_thred']:
                    ipara[iarr] = self.xp.tensor(self.scalar2vetor(ipara[iarr], self.L), dtype =self.floatx)

            self.transparas[iL] = ipara