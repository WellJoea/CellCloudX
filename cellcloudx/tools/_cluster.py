import os
import numpy as np
import pandas as pd

import numpy as np
from anndata import AnnData

def cutsne():
    import cudf
    import cuml
    import cudf, requests
    import cupy

def ftsne(data,
            use_rep='X_pca', 
            add_key='X_tsne',
            inplace=True, 
            perplexity = 30, 
            n_components=2,
            n_jobs=5,
            metric="euclidean", #cosine
            initialization='pca', #random
            learning_rate= 'auto', #'auto',
            early_exaggeration_iter=250,
            early_exaggeration='auto',
            n_iter=500,
            exaggeration=None,
            initial_momentum=0.8,
            final_momentum=0.8,
            verbose=True,
            random_state=19491001,
            **kargs):

    data_is_AnnData = isinstance(data, AnnData)
    X = data.obsm[use_rep].copy() if data_is_AnnData else data.copy()

    if type(perplexity) in [int, float]:
        import openTSNE
        tsne = openTSNE.TSNE(
            n_components=n_components,
            n_jobs=n_jobs,
            perplexity=perplexity,
            metric=metric,
            initialization=initialization,
            learning_rate=learning_rate,
            early_exaggeration_iter=early_exaggeration_iter,
            early_exaggeration=early_exaggeration,
            n_iter=n_iter,
            exaggeration=exaggeration,
            initial_momentum=initial_momentum,
            final_momentum=final_momentum,
            random_state=random_state,
            verbose=verbose,
            **kargs
        )
        embedding_train = tsne.fit(X)
    else:
        import openTSNE
        affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
            X,
            perplexities=perplexity,
            metric=metric,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        if initialization=='pca':
            init = openTSNE.initialization.pca(X, n_components=n_components, random_state=random_state)
        else:
            init = openTSNE.initialization.random(X, n_components=n_components,  random_state=random_state)

        tsne = openTSNE.TSNE(n_jobs=n_jobs,
                            n_components=n_components,
                            metric=metric,
                            learning_rate=learning_rate,
                            early_exaggeration_iter=early_exaggeration_iter,
                            early_exaggeration=early_exaggeration,
                            n_iter=n_iter,
                            exaggeration=exaggeration,
                            initial_momentum=initial_momentum,
                            final_momentum=final_momentum,
                            random_state=random_state,
                            verbose=verbose,
                            **kargs
            )
        embedding_train = tsne.fit(
            affinities=affinities_multiscale_mixture,
            initialization=init,
        )

    if data_is_AnnData:
        data = data if inplace else data.copy()
        data.obsm[add_key] = np.array(embedding_train)
        if not inplace:
            return data
    else:
        return embedding_train

def rmclust(adata, nclust=None, use_rep='pca', X=None,  modelNames='EEE',
             add_key='mclust', copy=False,
             R_HOME = None, R_USER=None, verbose=False,
            random_seed=491001):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    https://mclust-org.github.io/mclust/reference/mclustModelNames.html
    modelNames: A vector of character strings indicating the models to be
          fitted in the EM phase of clustering. The default is:

            • for univariate data (d = 1): ‘c("E", "V")’

            • for multivariate data (n > d): all the models available
              in ‘mclust.options("emModelNames")’
                "EII" "VII" "EEI" "VEI" "EVI" "VVI" "EEE" "VEE" "EVE" 
                "VVE" "EEV" "VEV" "EVV" "VVV"

            • for multivariate data (n <= d): the spherical and
              diagonal models, i.e. ‘c("EII", "VII", "EEI", "EVI",
              "VEI", "VVI")’
    """
    if R_HOME:
        os.environ['R_HOME'] = R_HOME
    if R_USER:
        os.environ['R_USER'] = R_USER
    if copy:
        adata = adata.copy()
    if X is None:
        X = adata.obsm[use_rep]

    np.random.seed(random_seed)
    from rpy2 import robjects as robj
    robj.r.library("mclust")

    import rpy2.robjects.numpy2ri as npr
    npr.activate()
    r_random_seed = robj.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robj.r['Mclust']
    summary = robj.r['summary']
    nclust = robj.rinterface.NULL if nclust is None else nclust #robj.r.seq(1, nclust)

    #mclust::adjustedRandIndex(kowalczyk.integrated$seurat_clusters, cell_info$cell_type_label)
    res = rmclust(npr.numpy2rpy(X), nclust, modelNames=modelNames)
    if verbose:
        print(summary(res))
    try:
        clust = np.int32(res[-2])-1
        adata.obs[add_key] = pd.Categorical(clust.astype(str),
                                    categories=np.unique(clust).astype(str))
    except:
        adata.obs[add_key] = '0'

    if copy:
        return adata
