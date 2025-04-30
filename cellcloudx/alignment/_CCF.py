import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse

from ..transform import ccf_transform_point
from ..utilis._arrays import isidentity, list_iter
from ..utilis._clean_cache import clean_cache

from .ccflib.ansac_registration import ansac
from .ccflib.ccd_registration import ccd
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

    def regist( self, method=['ansac', 'ccd'], 
                    transformer=['rigid', 'rigid'],
                    maxiter=None,
                    kda_method = 'annoy',
                    kd_method='sknn',

                    use_dpca=False,
                    m_neighbor=10,
                    e_neighbor =50, 
                    s_neighbor =30,
                    o_neighbor =None,
                    max_pairs=10000,
                    min_pairs=500,
                    upper=0.995,
                    min_samples=100,

                    CI = 0.95,
                    score_threshold= 0.3,
                    stop_merror=1,
                    stop_derror=1,

                    drawmatch=False,
                    verbose = 1,
                    ansac_kargs={},

                    tol=None,
                    get_larst_P = False,
                    normal=None, 
                    alpha=0.5,
                    gamma1=0,
                    gamma2=0,
                    beta=1,
                    w = None,
                    w_clip=[0, 1-1e-6],

                    K=None, 
                    KF=None,
                    callback=None,
            
                    show_reg=True,
                    figsize=(6,6),
                    pointsize = 0.1, 
                    mulks= {},
                    **kargs):
        if not isinstance(method, list):
            method = [method]
        if not isinstance(transformer, list):
            transformer = [transformer]
        self.TYs = []
        self.tmats = []
        self.tforms = []
        self.paras = []
        self.P = None
        self.C = None

        mks = {'K': K, 'KF': KF, 'w':w, 'normal':normal, 'alpha':alpha, 
               'gamma1':gamma1, 'gamma2':gamma2,
               'kd_method':kd_method, 'beta':beta, 'tol':tol,}
        mks.update(mulks)
        for k, v in mks.items():
            mks[k] = list_iter(v)
        maxiter = list_iter(maxiter)
        kda_method = list_iter(kda_method)

        K, J = 0,0
        A = min(len(method), len(transformer))

        if 'ccd' in method[:A]:
            L = np.where(np.array(method[:A]) == 'ccd')[0][-1]
        else:
            L = -1

        for I, (imth, itrans) in enumerate(zip(method, transformer)):
            logger.info(f'registration method {I+1}: {imth} -> {itrans}')
            pre_TY = self.TY.copy()
            if imth == 'ansac':
                model = ansac(self.X, self.TY, X_feat=self.X_feat, Y_feat=self.Y_feat)
                model.register(transformer=itrans,
                                use_dpca=use_dpca,
                                m_neighbor=m_neighbor,
                                e_neighbor = e_neighbor, 
                                s_neighbor = s_neighbor,
                                o_neighbor = o_neighbor,
                                min_samples = min_samples,
                                min_pairs=min_pairs,

                                upper = upper,
                                CI = CI,
                                stop_merror=stop_merror,
                                max_pairs=max_pairs,
                                score_threshold = score_threshold,
                                maxiter=maxiter[I],
                                kd_method=kda_method[K],
                                stop_derror=stop_derror,
                                drawmatch=drawmatch,
                                verbose=verbose,
                                **ansac_kargs)
                self.tmats.append(model.tform)
                # if len(self.tforms) and self.tforms[-1].shape == (self.D+1, self.D+1):
                #     self.tforms[-1] = np.dot(self.tforms[-1], tform)
                # else:
                #     self.tforms.append(tform)
                self.TY = np.array(model.TY, dtype=np.float64)
                K +=1

            elif imth == 'ccd':
                iargs = {k: v[J] for k, v in mks.items()}
                iargs.update(kargs)
                if I == L:
                    iargs['get_final_P'] = get_larst_P
                model = ccd(self.X, self.TY, X_feat=self.X_feat, Y_feat=self.Y_feat,
                            transformer=itrans, 
                                maxiter=maxiter[I],
                                w_clip = w_clip,
                                verbose=verbose,
                                **iargs)
                # model.register( callback=callback )
                self.tmats.append(model.tform)

                self.TY = np.array(model.TY, dtype=np.float64)
                J += 1
            else:
                raise ValueError(f'Unknown method: {imth}')

            self.TYs.append(self.TY.copy())
            self.paras.append(model.get_transformer())
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

    @staticmethod
    def trancer(P, C=None, top_ratio=0.75, state='posterior'):
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
                       transformer=['rigid', 'rigid'],
                       **kargs):
    model = ccf(X, Y, X_feat=X_feat, Y_feat=Y_feat)
    model.regist(method=method, transformer=transformer, **kargs)
    return model