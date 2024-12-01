import numpy as np
from scipy.sparse import issparse, csr_array
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import PCA
from typing import List, Optional
from typing import Optional, Union
from anndata import AnnData
from scipy.sparse import issparse, spmatrix
import scanpy as sc

from ..preprocessing._normalize import scale_array
from ..tools._glmpca import glmpca

EPS = np.finfo(float).eps
def dualPCA(X:np.ndarray, Y:np.ndarray, 
            n_comps:Optional[int]=50,
            scale:Optional[bool]=True,
            seed:Optional[int]=200504,
            axis:Optional[int]=0,
            zero_center:Optional[bool]=True,
            **kargs
    ) -> List:
    assert X.shape[1] == Y.shape[1]
    if scale:
        X = scale_array(X, axis=axis, zero_center=zero_center, **kargs)
        Y = scale_array(Y, axis=axis, zero_center=zero_center, **kargs)
    cor_var = X @ Y.T
    U, S, V = randomized_svd(cor_var, n_components=n_comps, random_state=seed)
    S = np.diag(S)
    Xh = U @ np.sqrt(S)
    Yh = V.T @ np.sqrt(S)
    return Xh, Yh

def glmPCA(data:  Union[AnnData, np.ndarray, spmatrix],
           n_comps=None, 
           use_highly_variable = True,
           inplace = True,
           use_approximate = True,
           doscale = True, 
           model='deviance',
           svd_solver='arpack',
           fam="poi", 
           ctl={"maxIter": 1000, "eps": 1e-4, "optimizeTheta": True}, penalty=1,
           verbose=False, init={"factors": None, "loadings": None},
           nb_theta=100, X=None, Z=None, sz=None, **kargs):

    data_is_AnnData = isinstance(data, AnnData)
    if data_is_AnnData:
        adata = data if inplace else data.copy()
    else:
        adata = AnnData(data, dtype=data.dtype)

    if not data_is_AnnData:
        use_highly_variable = False
    if use_highly_variable is True and 'highly_variable' not in adata.var.keys():
        raise ValueError(
            'Did not find adata.var[\'highly_variable\']. '
            'Either your data already only consists of highly-variable genes '
            'or consider running `pp.highly_variable_genes` first.'
        )
    adata_comp = adata[:, adata.var['highly_variable']] if use_highly_variable else adata
    n_comps = n_comps or min([50, adata_comp.shape[1]-1])

    DY = adata_comp.X
    if issparse(DY):
        DY = DY.toarray()

    if use_approximate:
        g_pca, g_pcs = nuResPCA(DY, n_comps = n_comps, doscale=doscale, svd_solver=svd_solver,
                                model=model, fam=fam, **kargs)
    else:
        res = glmpca(DY.T, n_comps, fam=fam, ctl=ctl, penalty=penalty, verbose=verbose, init=init,
                    nb_theta=nb_theta, X=X, Z=Z, sz=sz)
        g_pca = res["factors"]
        g_pcs = res["loadings"]
        # {"factors": factors, "loadings": loadings, "coefX": A, "coefZ": G}
        # res["dev"] = dev[range(t + 1)]
        # res["glmpca_family"] = gf
    if data_is_AnnData:
        adata.obsm['X_gpca'] = g_pca
        adata.uns['X_gpcs'] = g_pcs
        if not inplace:
            return adata
    else:
        return g_pca, g_pcs

def nuResPCA(mtx, n_comps=None, doscale=True, model='deviance', fam='binomial',  
                  svd_solver='arpack',size_factors=None, seed = 200504, **kargs):
    mtx = nullResiduals(mtx, model=model, fam=fam)
    if doscale:
        mtx = scale_array(mtx, axis=0)

    n_comps = n_comps or 50
    # X_pca, X_PCs = sc.pp.pca(mtx, n_comps=n_comps, svd_solver=svd_solver, 
    #                          random_state=seed, 
    #                          return_info=True,**kargs )[:2]
    if svd_solver=='arpack':
        n_comps = min([n_comps, mtx.shape[0], mtx.shape[1]-1])
    pca_ = PCA(n_components=n_comps, svd_solver=svd_solver, random_state=seed, **kargs)
    pca_.fit(mtx)
    X_pca = pca_.transform(mtx)
    X_PCs = pca_.components_.T

    return X_pca, X_PCs

def nullResiduals(mtx, model='deviance', fam='binomial', size_factors=None):
    '''
    # from https://bioconductor.org/packages/release/bioc/vignettes/scry/inst/doc/scry.html
    Args:
        mtx: obs*var
        model: "deviance", "pearson"
        fam: "binomial","bn", "poisson", "poi"
    Returns:
        mtx
    '''

    mtx = mtx.toarray() if issparse(mtx) else mtx.copy()
    sz = mtx.sum(1) if size_factors is None else size_factors

    if (fam in ["binomial", 'bn']):
        phat = mtx.sum(0)/mtx.sum()
        mhat = np.outer(sz, phat) + EPS
        if (model == "deviance"):
            nx = sz[:, None] - mtx
            nmhat = np.outer(sz, 1 - phat) + EPS
            term2 = nx * np.log(nx/nmhat + EPS)
            term1 = mtx * np.log(mtx/mhat + EPS)
            term1[ np.isnan(term1) ] =  0
            res = np.sign(mtx - mhat) * np.sqrt(2*(term1 + term2))
            res[ np.isnan(res) ] = 0
            return res
        elif (model == "pearson"):
            res = (mtx - mhat) /np.sqrt(mhat * (1 - phat))
            res[np.isnan(res)] = 0
            return res
    elif (fam in ["poi", "poisson"]):
        lsz = np.log(sz)
        sz = np.exp(lsz - np.mean(lsz))
        lamb = mtx.sum(0)/np.sum(sz)
        mhat = np.outer(sz, lamb) + EPS
        if (model == "deviance"):
            term1 = mtx * np.log(mtx/mhat + EPS)
            term1[ np.isnan(term1) ] =  0
            term2 = mhat - mtx
            res = np.sign(mtx - mhat) * np.sqrt(np.abs( 2*(term1 + term2) ))
            res[ np.isnan(res) ] = 0
            return res
        elif (model == "pearson"):
            res = (mtx - mhat)/np.sqrt(mhat)
            res[np.isnan(res)] = 0
            return res