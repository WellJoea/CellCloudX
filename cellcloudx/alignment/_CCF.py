import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse

from ..transform import ccf_transform_point
from ..utilis._arrays import isidentity, list_iter
from ..utilis._clean_cache import clean_cache

from .ccflib.pairwise_ansac_registration import pwAnsac
from .ccflib.ccd_registration import pwccd
from ..io._logger import logger

class ccf:
    def __init__(self, X, Y, X_feat=None, Y_feat=None, 
                 floatx='float64', *args, **kwargs):
        self.xp = np
        self.floatx = eval(f'self.xp.{ floatx or "float64"}')
        self.X = self.xp.asarray(X, dtype=self.floatx ).copy()
        self.Y = self.xp.asarray(Y, dtype=self.floatx ).copy()
        self.D = self.X.shape[1]

        self.X_feat = X_feat if not X_feat is None else None
        self.Y_feat = Y_feat if not Y_feat is None else None
        self.TY = self.xp.asarray(Y, dtype=self.floatx).copy()

    def regist( self, 
                    # method=['ansac', 'ccd'], 
                    # transformer=['rigid', 'rigid'],
                    method=None,
                    transformer=None,
                   
                    c_maxiter=None,
                    kd_method='keops',
                    mnn=10,
                    fnn =50, 
                    snn =30,
                    max_pairs=5e4,
                    min_pairs=500,
                    lower = 0.01, 
                    upper=0.995,
                    min_score= 0.30,
                    min_samples=100,
                    stop_pairs=300,
                    stop_merror=1,
                    stop_derror=1,
                    ansac_kargs={},

                    maxiter=None,
                    tol=None,
                    normal=None, 
                    tau2 = None,
                    alpha= None,
                    gamma1= None,
                    gamma2=None,
                    beta= None,
                    w = None,
                    record=None,
                    mulks= {},

                    show_reg=True,
                    figsize=(6,6),
                    pointsize = 0.1, 

                    record_attr =[],
                    verbose = 2,
                    **kargs):

        self.TYs = []
        self.tmats = []
        self.tforms = []
        self.paras = []
        self.attrs = {}
        self.records = {}
    
        if transformer is None:
            transformer = 'E'
    
        rMethod = []
        for i, j in enumerate(transformer):
            if method is None:
                rMethod.append('F')
            else:
                rMethod.append(method[i])

        L = min(len(rMethod), len(transformer))
        Fidx = [ i for i in range(L) if rMethod[i] in ['F', 'ccd'] ]
        FL = len(Fidx)

        mks = { 'w':w, 'normal':normal, 'alpha':alpha, 'maxiter':maxiter, 'record':record,
               'gamma1':gamma1, 'gamma2':gamma2, 'tau2':tau2, 'beta':beta, 'tol':tol, }
        mks.update(mulks)
        for k, v in mks.items():
            mks[k] = self.scalar2vetor(k, v, FL)

        J = 0
        for I, (imth, itrans) in enumerate(zip(rMethod, transformer)):
            if verbose > 0: logger.info(f'pw-registration {I+1}: {imth} -> {itrans}')
            pre_TY = self.TY.copy()
            if imth in ['ansac', 'C']:
                model = pwAnsac(self.X, self.TY, X_feat=self.X_feat, Y_feat=self.Y_feat,
                                device=kargs.get('device', None),transformer=itrans, 
                                verbose=verbose,)
                model.register(
                                kd_method=kd_method,
                                mnn = mnn,
                                fnn = fnn,
                                snn = snn,
                                max_pairs=max_pairs,
                                min_pairs=min_pairs,
                                lower = lower,
                                upper = upper,
                                min_score=min_score,
                                min_samples=min_samples,
                                stop_pairs=stop_pairs,
                                stop_merror=stop_merror,
                                stop_derror=stop_derror,
                                maxiter=c_maxiter,
                                **ansac_kargs)
                self.tmats.append(model.tform)
                # if len(self.tforms) and self.tforms[-1].shape == (self.D+1, self.D+1):
                #     self.tforms[-1] = np.dot(self.tforms[-1], tform)
                # else:
                #     self.tforms.append(tform)
                self.TY = np.array(model.TY, dtype=np.float64)

            elif imth in ['ccd', 'F']:
                iargs = {k: v[J] for k, v in mks.items()}
                iargs.update(kargs)
                model = pwccd(self.X, self.TY, X_feat=self.X_feat, Y_feat=self.Y_feat,
                            transformer=itrans, 
                                verbose=verbose,
                                **iargs)
                # model.register( callback=callback )
                self.records[I] = model.records
                self.tmats.append(model.tform)
                self.TY = np.array(model.TY, dtype=np.float64)
                J += 1
            else:
                raise ValueError(f'Unknown method: {imth}')

            self.TYs.append(self.TY.copy())
            self.paras.append(model.get_transformer())
    
            for iattr in record_attr:
                ival = getattr(model, iattr, None)
                if iattr not in self.attrs:
                    self.attrs[iattr] = [ival]
                else:
                    self.attrs[iattr].append(ival)

            if show_reg:
                fig, ax = plt.subplots(1,1, figsize=figsize)
                ax.scatter(pre_TY[:,0], pre_TY[:,1], s=pointsize, c='#888888')
                ax.scatter(self.X[:,0], self.X[:,1], s=pointsize, c='r')
                ax.scatter(self.TY[:,0], self.TY[:,1], s=pointsize, c='b')
                ax.set_aspect('equal', 'box')
                plt.show()
        clean_cache('model')
        self.merge_tmats()
        return self

    def transform_point(self, Y=None, paras=None, **kargs):
        if Y is None:
            Y = self.Y
        if paras is None:
            paras = self.paras
        TY = ccf_transform_point(Y, paras, **kargs)
        if hasattr(TY, 'detach'):
            TY = TY.detach().cpu().numpy()
        return TY

    def is_linear_mat(self, tmat):
        return (tmat.shape == (self.D+1, self.D+1) and
                (tmat[-1,:-1] ==0).all())

    def merge_tmats(self):
        self.tforms = []
        for itm in self.tmats:
            is_linear = (self.is_linear_mat(itm) and 
                         len(self.tforms) and
                         self.is_linear_mat(self.tforms[-1]))
            if is_linear:
                self.tforms[-1] = np.dot(itm, self.tforms[-1])
            else:
                self.tforms.append(itm)

    def scalar2vetor(self, n, x, L, default = None, force=False):
        if force:
            xs = [ x for i in range(L) ]
        else:
            if ((type(x) in [str, float, int, bool]) 
                or isinstance(x, (str, bytes))
                or np.isscalar(x)
                or (x is None)):
                xs = [ x for i in range(L) ]
            else:
                assert len(x) == L, f'len({n})={len(x)} != L={L}'
                xs = x
        if default is not None:
            xs = [ default if x is None else x for x in xs ]
        return xs

    @staticmethod
    def trancer(P, C=None, top_ratio=0.75, state='posterior'): #deprecated
        import scipy.sparse as sp
        if C is None:
            C = 1
        else:
            assert C.shape[0] == P.shape[1]

        if state == 'posterior':
            P = P.copy()/C
        elif state == 'prior':
            P = P
        else:
            raise ValueError(f'Unknown state: {state}')

        if sp.issparse(P):
            q_idx = sp.csr_matrix.argmax(P.tocsr(), axis=0)
            p_score = P.max(0).toarray().flatten()
        else:
            q_idx = np.argmax(P, axis=0)
            p_score = P.max(0)

        top_n = int(top_ratio * P.shape[0])
        top_n = min(top_n, np.sum(p_score > 0))

        r_idx = np.argpartition(p_score, -top_n)[-top_n:]
        p_score = p_score[r_idx]
        q_idx = q_idx[r_idx]

        return [r_idx, q_idx, p_score]

def ccf_registration(X, Y, X_feat=None, Y_feat=None, 
                       method=['ansac', 'ccd'], 
                       transformer='EE',
                       **kargs):
    model = ccf(X, Y, X_feat=X_feat, Y_feat=Y_feat)
    model.regist(method=method, transformer=transformer, **kargs)
    return model