import numpy as np
import matplotlib.pyplot as plt

from ..transform import ccf_transform_point
from ..utilis._clean_cache import clean_cache

from .ccflib.pairwise_ansac_registration import pwAnsac
from .ccflib.ccd_registration import pwccd
from ..io._logger import logger
from ..plotting._colors import random_colors, color_palette

class ccf:
    def __init__(self, X, Y, X_feat=None, Y_feat=None, 
                 floatx='float64', *args, **kwargs):
        self.xp = np
        self.floatx = eval(f'self.xp.{ floatx or "float64"}')
        self.X = X
        self.Y = self.xp.asarray(Y, dtype=self.floatx ).copy()
        self.D = self.Y.shape[1]

        self.X_feat = X_feat if not X_feat is None else None
        self.Y_feat = Y_feat if not Y_feat is None else None
        self.TY = self.xp.asarray(Y, dtype=self.floatx).copy()

    def regist( self, 
                    transformer=None,
                    tf=None,
                    c_maxiter=None,
                    kd_method='keops',
                    mnn=15,
                    fnn =50, 
                    snn =30,
                    max_pairs=5e4,
                    min_pairs=500,
                    lower = 0.01, 
                    upper=0.995,
                    min_score= 0.30,
                    min_samples=100,
                    stop_pairs=300,
                    stop_merror=0.5, 
                    stop_derror=0.5,
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
                    pointsize = None, 

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
            transformer = tf
            
        if transformer is None:
            transformer = 'E'
    
        FTs = [ i for i in transformer if i not in 'esap']
        FL = len(FTs)
        mks = { 'w':w, 'normal':normal, 'alpha':alpha, 'maxiter':maxiter, 'record':record,
               'gamma1':gamma1, 'gamma2':gamma2, 'tau2':tau2, 'beta':beta, 'tol':tol, }
        mks.update(mulks)

        for k, v in mks.items():
            mks[k] = self.scalar2vetor(k, v, FL)

        add_mks = ['tau2_auto', 'tau2_clip', 'tau2_decayto', 'tau2_decaystop', 'feat_normal',
                   'use_stable', 'use_keops', 'wa', 'wma', 'use_wm', 'use_mrf',  
                    'a0', 'a1', 'w_clip', 
                    'sample', 'sample_growto', 'sample_min',  'sample_stop']
        for k in add_mks:
            if k in kargs:
                mks[k] = self.scalar2vetor(k, kargs[k], FL)
                del kargs[k]

        J = 0
        for I, (itrans) in enumerate(transformer):
            if verbose > 0: logger.info(f'pw-registration {I+1}: {itrans}')
            pre_TY = self.TY.copy()
            if itrans in 'esap':
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

            else:
                iargs = {k: v[J] for k, v in mks.items()}
                iargs.update(kargs)
                model = pwccd(self.X, self.TY, X_feat=self.X_feat, Y_feat=self.Y_feat,
                            transformer=itrans, 
                                verbose=verbose,
                                **iargs)
                # model.register( callback=callback )
                self.records[I] = model.records
                self.tmats.append(model.tmats.get('tform', None))
                self.TY = np.array(model.TY, dtype=np.float64)
                J += 1

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
                ax.scatter(pre_TY[:,0], pre_TY[:,1], s= (pointsize or 2000/pre_TY.shape[0]), c='#888888')
                
                ncol =  len(self.X) + 1 if isinstance(self.X, (list, tuple)) else 2
                cols = color_palette(ncol)
                if isinstance(self.X, (list, tuple)):
                    for icol, X in zip(cols, self.X):
                        ax.scatter(X[:,0], X[:,1], s=(pointsize or 2000/X.shape[0]), c=icol)
                else:
                    ax.scatter(self.X[:,0], self.X[:,1], s=(pointsize or 2000/self.X.shape[0]),  c=cols[0])
                ax.scatter(self.TY[:,0], self.TY[:,1], s=(pointsize or 2000/self.TY.shape[0]), c=cols[-1])
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
        if tmat is None:
            return False
        else:
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

def ccf_pwr(X, Y, X_feat=None, Y_feat=None, 
                        transformer=None,
                        method=None, 
                       **kargs):
    model = ccf(X, Y, X_feat=X_feat, Y_feat=Y_feat)
    model.regist(method=method, transformer=transformer, **kargs)
    return model