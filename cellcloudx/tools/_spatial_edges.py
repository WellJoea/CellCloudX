import anndata as ad
import matplotlib.pyplot as plt
from skimage import filters
import collections
import numpy as np
import pandas as pd
import copy

import scipy as sci

from ..tools._neighbors import Neighbors
from ..utilis._arrays import list_iter
from ..tools._exp_edges import exp_similarity
from ..tools._outlier import Invervals
from ..tools._search import searchidx
from ..plotting._imageview import drawMatches

def spatial_edges(adata, groupby = None, 
                  basis='spatial',

                  add_key=None,
                  listarg = [],
                  root=None, 
                  regist_pair=None,
                  full_pair=False,
                  cross_group=False,
                  
                  normal_type='normal_hgvs',
                  self_pair=True,
                  cross_simi_thred=None,
                  cross_simi_genes=5000,
                  cross_simi_pcs=100,
                  cross_knn =None,

                  show_tree = True,
                  figsize=(8,6),
                  layout="spectral", 
                 **kargs):
    add_key = add_key or f'{basis}_edges'
    listargs =  set(['knn', 'radius', 'CI', 'simi_thred', 'simi_genes', ] + listarg)
    crosslist = ['cross_simi_thred', 'cross_simi_genes', 'cross_simi_pcs', 'cross_knn']
    kargs['basis'] = basis

    if isinstance(adata, ad.AnnData) and (groupby is None):
        spatial_edge(adata, inplace = True, **kargs)

    elif isinstance(adata, list) and isinstance(adata[0], ad.AnnData):
        for i, iadata in enumerate(adata):
            iargs = copy.deepcopy(kargs)
            for iar in listargs:
                if iar in kargs:
                    iargs[iar] = list_iter(kargs[iar])[i]
            spatial_edge(iadata, inplace = True, **iargs)

    elif isinstance(adata, ad.AnnData) and (not groupby is None):
        try:
            groups = adata.obs[groupby].cat.remove_unused_categories().cat.categories
        except:
            groups = adata.obs[groupby].unique()

        verbose = kargs.get('verbose', 1)
        kargs['verbose']  = verbose

        if cross_group:
            pairs, _ = searchidx(len(groups),
                                labels=groups,
                                step=1, 
                                self_pair=self_pair,
                                root=root, 
                                regist_pair=regist_pair,
                                search_type='bfs',
                                layout=layout, 
                                figsize=figsize,
                                show_tree=show_tree,
                                full_pair=full_pair)

        else:
            pairs = list(zip(groups, groups))
    
        edges_info = []
        for i, (iscr, idst) in enumerate(pairs):
            datascr = adata[(adata.obs[groupby]==iscr)]
            datadst = adata[(adata.obs[groupby]==idst)] if idst != iscr else None

            iargs = copy.deepcopy(kargs)
            for iar in listargs:
                if iar in kargs:
                    iargs[iar] = list_iter(kargs[iar])[i]
            
            for icross in crosslist:
                if (eval(icross) is not None) and (iscr != idst):
                    iargs[icross.replace('cross_', '')] = list_iter(eval(icross))[i] 

            iedge_info = spatial_edge(datascr, adatay=datadst,
                                      inplace = False,
                                      title = (iscr, idst), 
                                      return_edges = True, **iargs)
            iedge_info['paires'] = f'{iscr}<->{idst}'
            edges_info.append(iedge_info)

        adata.uns[add_key] = pd.concat(edges_info, axis=0).reset_index(drop=True)
        if verbose:
            print('computing spatial edges...\n'
                 f"finished: added to `.uns['{add_key}']`")

def spatial_edge(adata, adatay = None,
                basis='spatial',
                add_key=None,
                title = None,

                knn=10,
                radius = None,

                show_hist = True,
                show_match = False,

                method='sknn',
                self_loop=False,
                CI=0.985, 

                use_esimi=True, 
                normal_type='normal_hgvs',
                simi_pcs=100, 
                cor_method = 'cosine', 
                simi_thred = None,
                simi_genes = 5000,

                verbose=2,
                inplace = True,
                return_edges = False,

                line_width=0.75,
                point_size=5,
                line_sample=1000,
                line_alpha=0.8,
                n_jobs=-1):

    if title is None:
        titles = [f'src', f'dst']
    else:
        if isinstance(title, (str, int, float)):
            titles = [f'{title}_src', f'{title}_dst']
        else:
            assert isinstance(title, (list, tuple))
            titles = title[:2]

    adata = adata if inplace else adata.copy()
    add_key = add_key or f'{basis}_edges'
    if adatay is None:
        coordx = adata.obsm[basis]
        labelx = adata.obs_names.values
        nnode = adata.shape[0]

        src, dst, dist = coord_edges(coordx, coordy=None, knn=knn, radius=radius, 
                                     method=method, n_jobs = n_jobs)

        edges_pair = [src, dst]
        edges_info = pd.DataFrame({'src': labelx[src], 'dst': labelx[dst], 'edges_dist': dist})

        edges_info['src_name'] = titles[0]
        edges_info['dst_name'] = titles[1]

    else:
        pairs = [(0,1), (1,0)]
        pdata = [adata, adatay]
        nnode = adata.shape[0] + adatay.shape[0]
        edges_info = []
        edges_pair = []
        for i, ipair in enumerate(pairs):
            sidx, didx = ipair
            idatax = pdata[sidx]
            idatay = pdata[didx]
    
            coordx = idatax.obsm[basis]
            coordy = idatay.obsm[basis]
    
            labelx = idatax.obs_names.values
            labely = idatay.obs_names.values

            src, dst, dist = coord_edges(coordx, coordy=coordy, knn=knn, radius=radius, 
                                            method=method, n_jobs = n_jobs)
            i_info = pd.DataFrame({'src': labelx[src], 'dst': labely[dst], 'edges_dist': dist})
            i_info['src_name'] = titles[sidx]
            i_info['dst_name'] = titles[didx]
            edges_info.append(i_info)

            if i == 0:
                edges_pair = [src, dst]
            elif i == 1:
                edges_pair[0] = np.concatenate([edges_pair[0], dst]) 
                edges_pair[1] = np.concatenate([edges_pair[1], src])            
        edges_info = pd.concat(edges_info, axis=0)

    if use_esimi:
        exp_simi = exp_similarity(adata, adatay=adatay, 
                                    edges=edges_pair, 
                                    normal_type=normal_type,
                                    n_pcs=simi_pcs,
                                    method = cor_method,
                                    n_top_genes = simi_genes,
                                    simi_thred = None,
                                    title = None, show_hist=False)
        edges_info['edges_weight'] = exp_simi

    keep_idx = state_edges(edges_info,
                                nnode=nnode,
                                CI = CI,
                                self_loop = self_loop,
                                simi_thred = simi_thred,
                                show_hist= show_hist,
                                title = ' <-> '.join(titles),
                                verbose = verbose,)
    edges_info = edges_info.loc[keep_idx].reset_index(drop=True)

    if show_match:
        adatay = adata if adatay is None else adatay
        coordx = adata.obsm[basis][:,:2]
        coordy = adatay.obsm[basis][:,:2]

        scr_coord = coordx[edges_pair[0][keep_idx]]
        dst_coord = coordy[edges_pair[1][keep_idx]]

        drawMatches((scr_coord, dst_coord), bgs=(coordx, coordy),
                       line_width=line_width,
                        titles= titles,
                        size=point_size,
                        line_sample=line_sample,
                        line_alpha=line_alpha,
                        fsize=4)

    if verbose >= 2:
        print('computing spatial edges...\n'
             f"finished: added to `.uns['{add_key}']`")
    if return_edges:
        return edges_info
    else:
        adata.uns[add_key] = edges_info
        if not inplace:
            return adata

def coord_edges(coordx, coordy=None,
                knn=50,
                radius=None,
                
                max_neighbor = int(1e4),
                method='sknn' ,
                n_jobs = -1):

    if (not coordy is None):
        assert coordx.shape[1] == coordy.shape[1], 'coordx and coordy must have the same number of dimentions'
        
        cknn = Neighbors( method=method ,metric='euclidean', n_jobs=n_jobs)
        cknn.fit(coordx, radius_max= None,max_neighbor=max_neighbor)
        distances, indices = cknn.transform(coordy, knn=knn, radius = radius)

        # cknn = Neighbors( method=method ,metric='euclidean', n_jobs=n_jobs)
        # cknn.fit(coordy, radius_max= None,max_neighbor=max_neighbor)
        # distances2, indices2 = cknn.transform(coordx, knn=knn, radius = radius)
        # dst2 = np.concatenate(indices2, axis=0)
        # src2 = np.repeat(np.arange(len(indices2)), list(map(len, indices2)))
        # dist2 = np.concatenate(distances2, axis=0)

        # src = np.concatenate([src1, src2], axis=0)
        # dst = np.concatenate([dst1, dst2], axis=0)
        # dist = np.concatenate([dist1, dist2], axis=0)
        # psets = list(zip(src, dst))
        # pairs, idx = set(), []
        # for i in range(len(src)):
        #     ipair= psets[i]
        #     if not ipair in pairs:
        #         pairs.add(ipair)
        #         idx.append(i)
        # assert len(idx) == len(set(zip(src, dst)))
        # src = src[idx]
        # dst = dst[idx]
        # dist = dist[idx]

        # idx = np.lexsort((dst,src))
        # src = src[idx]
        # dst = dst[idx]
        # dist = dist[idx]

        # coord_idx = np.concatenate([np.arange(coordx.shape[0]), np.arange(coordy.shape[0])], axis=0)
        # coordxy = np.concatenate([coordx, coordy], axis=0)
        # cknn = Neighbors( method=method ,metric='euclidean', n_jobs=n_jobs)
        # cknn.fit(coordxy, radius_max= None, max_neighbor=max_neighbor)
        # distances, indices = cknn.transform(coordxy, knn=knn, radius = radius)
        # src = np.concatenate(indices, axis=0)
        # dst = np.repeat(np.arange(len(indices)), list(map(len, indices)))
        # dist = np.concatenate(distances, axis=0)
    else:
        coordy = coordx
        cknn = Neighbors( method=method ,metric='euclidean', n_jobs=n_jobs)
        cknn.fit(coordx, radius_max= None,max_neighbor=max_neighbor)
        distances, indices = cknn.transform(coordy, knn=knn, radius = radius)

    # distance neighbors
    src = np.concatenate(indices, axis=0).astype(np.int64)
    dst = np.repeat(np.arange(len(indices)), list(map(len, indices))).astype(np.int64)
    dist = np.concatenate(distances, axis=0)
    # src = indices.flatten('C')
    # dst = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
    # dist = distances.flatten('C')
    return [src, dst, dist]

def state_edges(edges_info, nnode,
                CI = 1,
                simi_thred = None,
                kernel='poi',
                self_loop = False,
                show_hist= True,
                title = None,
                verbose = True,):

    src = edges_info['src']
    dst = edges_info['dst']
    dist = edges_info['edges_dist'].values
    if 'edges_weight' in edges_info.columns:
        eweig = np.abs(edges_info['edges_weight'].values)
    else:
        eweig = None
    assert nnode >= len( (set(dst) | set(src)) )

    # filter edges by distance
    keep_idx = np.ones(dist.shape[0], dtype=bool)
    if not self_loop:
        keep_idx &= (src != dst)

    if (not CI is None) and (CI < 1):
        radiu_trim = Invervals(dist[dist>0], CI=CI,  kernel=kernel, tailed ='two')[2]
        keep_idx &= (dist<=radiu_trim)
    else:
        radiu_trim = None

    if (not eweig is None) and (not simi_thred is None):
        if simi_thred == 'ci':
            eweig_k = eweig[ (keep_idx & (eweig<1)) ]
            simi_thred = Invervals(eweig_k, CI=CI,  kernel=kernel, tailed ='two')[1]
        assert (0<=simi_thred<1)
        keep_idx &= (eweig>=simi_thred)
    else:
        simi_thred = None

    # state and plot
    dst_k = dst[keep_idx]
    dist_k = dist[keep_idx]
    k_nodes, counts = np.unique(dst_k, return_counts=True)
    mean_neig = np.mean(counts)
    nedge = len(dist_k)
    mean_radiu = np.mean(dist_k)
    if simi_thred is None:
        mean_simi = None
    else:
        mean_simi = np.mean(eweig[keep_idx])

    if verbose:
        if title:
            print('*'*10 + f' {title} ' + '*'*10)
        else:
            print('*'*20)
        print(f'nodes: {nnode}, edges: {len(dst)}\n'
              f'keep nodes:{k_nodes.shape[0]}, keep edges: {nedge}, drop edges:{len(dst)-nedge}\n'
              f'mean edges: {mean_neig :.6f}.\n'
              f'mean distance: {mean_radiu :.6f}.')
        if not simi_thred is None:
            print(f'mean similarity: {mean_simi :.6f}.')

    if show_hist:
        ncols = 3 if simi_thred else 2
        fig, ax = plt.subplots(1, ncols, figsize=((ncols+0.1)*3, 3))
        fig.suptitle(f"{title} edge statistics")#, fontsize=10)

        ax[0].bar( *np.unique(counts, return_counts=True), facecolor='b',
                   label=f'nodes: {nnode}\nedges: {nedge}\nmean edges:{mean_neig :.2f}\nmean distance:{np.mean(dist_k) :.2f}' )
        ax[0].legend()
        ax[0].set_title(f'edges distribution')

        ax[1].hist(dist, histtype='barstacked', bins=50, facecolor='red', alpha=1)
        if not radiu_trim is None:
            ax[1].axvline(radiu_trim, color='black', 
                          label=f'distance thred: {radiu_trim :.2f}\ndrop edges: {(dist>radiu_trim).sum()}')
            ax[1].set_title(f'distance distribution')
            ax[1].legend()
        if simi_thred:
            ax[2].hist(eweig, histtype='barstacked', bins=50, facecolor='blue', alpha=1)
            ax[2].axvline(simi_thred, color='black', 
                          label=f'similarity thred: {simi_thred :.2f}\ndrop edges: {(eweig<simi_thred).sum()}')
            ax[2].legend()
            ax[2].set_title(f'similarity distribution')

        plt.tight_layout()
        plt.show()

    return keep_idx

def min_dist(coord, algorithm='auto', metric = 'minkowski', quantiles = [0.05, 0.95], n_jobs=-1):
    from sklearn.neighbors import NearestNeighbors as sknn
    nbrs = sknn(n_neighbors=2,
                p=2,
                n_jobs=n_jobs,
                algorithm=algorithm, metric=metric)
    nbrs.fit(coord)
    distances, indices = nbrs.kneighbors(coord, 2, return_distance=True)
    distances = distances[:, 1].flatten()
    dmin, dmax = np.quantile(distances, quantiles)
    mean_dist = np.mean(distances[( (distances>=dmin) & (distances<=dmax) )])
    return mean_dist

def dist_similarity(distances, coords=None, method='exp', scale=None ):
    # distance to similarity
    if (not coords is None)  and (scale is None):
        scale = min_dist(coords) 

    if(scale is None):
        scale = 1
    elif isinstance(scale, str):
        if scale == 'max':
            scale = np.max(distances)
        elif scale == 'mean':
            scale = np.mean(distances)
        elif scale == 'median':
            md = np.ma.masked_where(distances == 0, distances)
            scale = np.ma.median(md)
        elif scale == 'l2':
            scale = np.linalg.norm(distances, axis=None, ord=2)**0.5
        # elif scale == 'fuzzy':
        #     compute_connectivities_umap
        else:
            scale = 1

    distances = distances/scale
    if method == 'linear':
        simi = 1- distances
    elif method == 'negsigmid':
        simi = (2*np.exp(-distances))/(1+np.exp(-distances))
    elif method == 'exp':
        simi = np.exp(-distances)
    elif method == 'exp1':
        nonz_min = np.min(distances[distances>0])
        distances = np.clip(distances, nonz_min, None)
        simi = np.exp(1-distances)
    elif method == 'log':
        nonz_min = np.min(distances[distances>0])
        distances = np.clip(distances-nonz_min, 0, None)
        simi = 1/(1+np.log(1+distances))
    return simi