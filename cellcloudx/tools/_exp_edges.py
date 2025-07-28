import numpy as np
import scanpy as sc
import anndata as ad
import copy
from scipy.sparse import issparse, issparse, csr_array
import matplotlib.pyplot as plt

from ..utilis._arrays import list_iter,checksymmetric
from ..tools._neighbors import Neighbors, mtx_similarity
from ..tools._decomposition import glmPCA, nullResiduals
from ..preprocessing._normalize import NormScale

def exp_edges(adata, groupby = None, 
                  merge_edges = False,
                  add_key=None, **kargs):
    add_key = add_key or 'exp_edges'

    if isinstance(adata, ad.AnnData) and (groupby is None):
        exp_edge(adata, inplace = True, **kargs)

    elif isinstance(adata, list) and isinstance(adata[0], ad.AnnData):
        for i in range(len(adata)):
            iargs = copy.deepcopy(kargs)
            for iar in ['n_genes', 'n_pca', 'n_neighbors', 'knn', 'simi_genes']:
                if iar in kargs:
                    iargs[iar] = list_iter(kargs[iar])[i]
            exp_edge(adata[i], inplace = True, **iargs)

    elif isinstance(adata, ad.AnnData) and (not groupby is None):
        try:
            groups = adata.obs[groupby].cat.remove_unused_categories().cat.categories
        except:
            groups = adata.obs[groupby].unique()

        verbose = kargs.get('verbose', 1)
        kargs['verbose']  = verbose

        cellidx = np.arange(adata.shape[0]).astype(np.int64)
        edges_info = []
        for i, igrp in enumerate(groups):
            idx = (adata.obs[groupby]==igrp)
            iadata = adata[idx]
            icellid = cellidx[idx]

            iargs = copy.deepcopy(kargs)
            for iar in ['n_genes', 'n_pca', 'n_neighbors', 'knn']:
                if iar in kargs:
                    iargs[iar] = list_iter(kargs[iar])[i]

            edge_info = exp_edge(iadata, inplace = True, title=igrp,
                                      return_edges = True, **iargs)

            if merge_edges:
                edge_info['edges'] = icellid[edge_info['edges']]
            edges_info.append(edge_info)

        if merge_edges:
            adata.uns[add_key] = { k: np.concatenate([info[k] for info in edges_info], axis=1) 
                                  for k in edges_info[0].keys()}
        else:
            adata.uns[add_key] = dict(zip(groups, edges_info))

        if verbose:
            print('computing expression edges...\n'
                f"finished: added to `.uns['{add_key}']`")

def exp_edge(adata, adj = None, add_key = None, n_genes=3500, n_pca=50, 
             n_neighbors=15, knn = 10, min_dist=0.5, res=1, color=None, 
             remove_loop = False,
             title = None, inplace=True,
             weight_thred = 0.1, show_plot=True,
             use_esimi=False, normal_type='normal_hgvs',
             simi_pcs=100, cor_method = 'cosine', simi_thred = 0.1,
             simi_genes = 8000, remove_lowesimi=True, verbose=2, 
             return_edges=False, **kargs):
    assert not (adata is None and adj is None), 'adata and adj cannot be None'
    add_key = add_key or 'exp_edge'
    n_neighbors = max(n_neighbors, knn)

    if not adata is None:
        adata = adata if inplace else adata.copy()
    if adj is None:
        adatai = sc_dimreduction(adata.copy(), n_top_genes=n_genes, n_pca=n_pca, 
                                    n_neighbors = n_neighbors, min_dist=min_dist,
                                    res=res, color=color)
        A = adatai.obsp['connectivities']
        # pca_emb = adatai.obsm['X_pca']
        # snn = Neighbors(method='annoy', n_jobs=-1)
        # snn.fit(pca_emb)
        # cdkout = snn.transform(pca_emb, knn=n_neighbors)
        # iscr, idest, idist = snn.translabel(cdkout, return_type='lists')

        # A1 = fuzzy_connectivities(None, knn_indices=cdkout[1], knn_dists=cdkout[0],
        #                         n_obs=adatai.shape[0],
        #                         random_state=None,
        #                          n_neighbors=n_neighbors).toarray()

        # np.fill_diagonal(A1, 1)
        # src1, dst1, value1 = gettopn_min(-A1, knn)
        # set3 = set(zip(src1, dst1))
    else:
        A = adj

    if issparse(A):
        A = A.toarray()
    if  remove_loop:
        np.fill_diagonal(A, 0)
    else:
        np.fill_diagonal(A, 1)

    src, dst, value = gettopn_min(-A, knn)
    value = -value
    idx = value> weight_thred

    if show_plot:
        fig, ax = plt.subplots(1,1, figsize=(3.5,3.5))
        ax.hist(value, bins=50)
        ax.axvline(weight_thred, color='black', label=f'neighbor_weight_thred: {weight_thred :.3f}')
        if not title is None:
            ax.set_title(f'{title} connectivities weight distribution')
        ax.legend()
        plt.show()

    src = src[idx]
    dst = dst[idx]
    edge = np.vstack([src, dst])
    value = value[idx]

    _, counts = np.unique(dst, return_counts=True)
    mean_neig = np.mean(counts)
    verbose and print(f'total edges: {len(idx)}, drop edges: {len(idx) - len(value)}, mean edges: {mean_neig}')
    edges_info =  {'edges': edge, 'edges_attr': value[None,:], 'edges_weight': value[None,:]}

    if use_esimi:
        exp_simi, exp_idx = exp_similarity(adata, edges=edge, 
                                            normal_type=normal_type,
                                            n_pcs=simi_pcs,
                                            method = cor_method, simi_thred = simi_thred,
                                            n_top_genes = simi_genes,
                                            title = title, show_plot=show_plot, **kargs)
        # exp_simi = np.sqrt(exp_simi)
        edges_info['edges_weight'] = np.vstack([exp_simi, exp_idx])
        verbose and print(f'low_threshold edges: {exp_idx.shape[0] - exp_idx.sum()}')

        if remove_lowesimi:
            for k,v in edges_info.items():
                edges_info[k] = v[:, exp_idx.astype(np.bool_)]

    if return_edges:
        return edges_info

    if not adata is None:
        adata.uns[add_key] = edges_info
        (verbose >=2) and  print(f"    : added to `.uns['{add_key}']`")
        if not inplace:
            return adata
    else:
        return edges_info

def normal_adata(adatalist,  normal_type='counts_hgvs',
                   donormal=False, doscale=False, n_pcs=100, 
                   model='deviance', fam="poi",show_plot=True,
                   n_top_genes = 10000,):
    verbose = sc.settings.verbosity.real
    sc.settings.verbosity = 0
    if isinstance(adatalist, ad.AnnData):
        adatalist = [adatalist]
    else:
        assert isinstance(adatalist, list)

    Order = range(len(adatalist))
    adatasi = ad.concat(adatalist, label="batch", 
                       keys=Order,
                       index_unique="-") 

    if normal_type  == 'counts':
            similar = adatasi.X.toarray() if issparse(adatasi.X) else adatasi.X
    elif normal_type == 'glmPCA':
        sc.pp.highly_variable_genes(adatasi, n_top_genes=n_top_genes, flavor='seurat_v3')
        glmPCA(adatasi, n_comps=n_pcs, svd_solver='arpack',model=model, fam=fam)
        similar = adatasi.obsm['X_gpca']

    elif normal_type == 'nullResiduals':
        sc.pp.highly_variable_genes(adatasi, n_top_genes=n_top_genes, flavor='seurat_v3')
        similar = nullResiduals( adatasi[:, adatasi.var['highly_variable']].X, 
                                model=model, fam=fam)

    elif normal_type  == 'counts_hgvs':
        sc.pp.highly_variable_genes(adatasi, n_top_genes=n_top_genes, flavor='seurat_v3')
        if donormal:
            sc.pp.normalize_total( adatasi, target_sum=1e4)
            sc.pp.log1p(adatasi)
        adatasi = adatasi[:, adatasi.var.highly_variable]
        similar = adatasi.X.toarray() if issparse(adatasi.X) else adatasi.X
    elif normal_type  == 'normal_hgvs':
        adatasi = NormScale(adatasi, batch_key='batch', 
                                donormal=True, doscale=doscale,
                                saveraw=False, dropMT=True,
                                n_top_genes = n_top_genes,
                                show_plot=show_plot, 
                                usehvgs=True, verbose=0)
        similar = adatasi.X.toarray() if issparse(adatasi.X) else adatasi.X
    elif normal_type  == 'pca':
        adatasi = NormScale(adatasi, batch_key='batch', 
                                donormal=True, doscale=True,
                                saveraw=False, dropMT=True,
                                n_top_genes = n_top_genes,
                                show_plot=show_plot,
                                usehvgs=True, verbose=0)
        sc.tl.pca(adatasi, svd_solver='arpack', n_comps=n_pcs)
        similar = adatasi.obsm['X_pca']

    sc.settings.verbosity = verbose
    similars = [  similar[(adatasi.obs['batch'] == i).values] for i in Order ]
    return similars
    
def exp_similarity(adata, adatay=None,  edges=None, normal_type='counts_hgvs',
                   n_pcs=100, n_top_genes = 10000, method='cosine',
                   title = None, simi_thred = None, show_hist=False):

    if adatay is None:
        exp_matrx = normal_adata(adata, normal_type=normal_type,n_pcs=n_pcs, n_top_genes = n_top_genes, show_plot=show_hist)
        similar = mtx_similarity(exp_matrx[0], exp_matrx[0], method = method, pairidx = edges)
    else:
        exp_matrx = normal_adata([adata, adatay], normal_type=normal_type,n_pcs=n_pcs, n_top_genes = n_top_genes, show_plot=show_hist)
        similar = mtx_similarity(exp_matrx[0], exp_matrx[1], method = method, pairidx = edges)

    if (method == 'fuzzy'):
        similar = similar[0].toarray()
        if not edges is None:
            similar = np.array( similar[edges[1], edges[0]] ).flatten()

    if simi_thred is None:
        return similar
    else:
        simi_idx =  np.abs(similar) >= simi_thred
        if (not edges is None) and (show_hist):
            fig, ax = plt.subplots(1,2, figsize=(7.5,3))
            ax[0].hist(np.abs(similar).flatten(), bins=100)
            if simi_thred >0 :
                ax[0].axvline(simi_thred, color='black', label=f'{normal_type}\nexp_simi_thred: {simi_thred :.3f}')
                if similar.min() < 0:
                    ax[0].axvline(-simi_thred, color='gray')
            ax[0].legend()
            _, counts = np.unique(edges[1][simi_idx], return_counts=True)
            mean_neig = np.mean(counts)
            bins = np.max(counts)
            ax[1].hist(counts, bins=bins, facecolor='b', 
                    label=f'mean_neighbors:{mean_neig :.3f}\nmean exp_similarity(abs):{np.mean(np.abs(similar)) :.3f}')
            ax[1].legend()

            if not title is None:
                ax[0].set_title(f'{title} expression similarity distribution')
                ax[1].set_title(f'{title} mean neighbor distribution')
            plt.tight_layout()
            plt.show()
        return similar, simi_idx

def sc_dimreduction(adata, n_top_genes=3500, n_pca=50,
             n_neighbors = 15, run_umap=False,
             min_dist=0.5, res=None, color=None ):
    adata = NormScale(adata.copy(), batch_key=None, 
                             donormal=True, doscale=True,
                             saveraw=False, dropMT=True,
                             n_top_genes = n_top_genes,
                             minnbat=0, min_mean=0.1, min_disp=0.2, max_mean=6,
                             usehvgs=True)
    # import scanpy as sc
    # sc._settings.ScanpyConfig(verbosity=0)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pca)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pca)

    if run_umap:
        if not res is None:
            sc.tl.leiden(adata, resolution=res)

        if color is None:
            color = ['leiden']
        elif isinstance(color, str):
            color = ['leiden', color]
        elif isinstance(color, list):
            color = ['leiden'] + color

        if adata.obs.columns.isin(color).any() or adata.var.index.isin(color).any():
            sc.tl.umap(adata, min_dist=min_dist)
            sc.pl.umap(adata, color=color, use_raw=False)

    return adata

def gettopn_min(mtx, kth):
    if issparse(mtx):
        mtx = mtx.toarray()
    indices = np.argpartition(mtx, kth=kth, axis=1)
    indices = indices[:,:kth]
    src = indices.flatten('C')
    dst = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
    value = mtx[dst, src]
    return [src, dst, value]