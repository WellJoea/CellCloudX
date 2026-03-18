import os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import seaborn as sns
import sys
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def pca_embedding(adatas, donormal=True, n_top_genes=5000, min_cells=0, n_comps=50, **kargs):
    adatas = [ i.copy() for i in adatas ]
    L =len(adatas)
    if L == 1:
        return adatas[0]

    cgene = list(set(adatas[0].var_names))
    for i in range(L):
        adata = adatas[i]
        if donormal:
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
        cgene = list(set(cgene) & set(adata.var_names))
        print(f'common gene: {len(cgene)}')
    print(f'final common gene: {len(cgene)}')

    adatac = {}
    for i in range(L):
        adata = adatas[i]
        adatac[i] = adata[:, list(cgene)]
        print(f'{i}:', adata.X.max())

    adatac = ad.concat(adatac, label='Batch')
    adatac.obs['Batch'] = pd.Categorical(adatac.obs['Batch'])
    groupby = 'Batch'

    sc.pp.highly_variable_genes(adatac, n_top_genes=n_top_genes, batch_key='Batch')
    adatac = adatac[:, adatac.var.highly_variable]
    sc.tl.pca(adatac, n_comps=min(n_comps, adatac.shape[1]-1))
    return adatac

def hgvs(adata, n_top_genes=5000, batch_key=None):
    hgvdf = sc.pp.highly_variable_genes(adata.copy(), n_top_genes=n_top_genes,
                                        batch_key=batch_key, flavor='seurat_v3', inplace=False)
    return hgvdf.index[hgvdf['highly_variable']]

def paste_regist(adata1, adata2, alpha=0.1, filter_gene= 10000, numItermax=300, use_gpu =False, **kargs):
    import paste as pst
    adata1 = adata1.copy()
    adata2 = adata2.copy()
    sc.pp.filter_genes(adata1, min_cells=8)
    sc.pp.filter_genes(adata2, min_cells=8)

    hgv_g1 = hgvs(adata1, n_top_genes=filter_gene)
    hgv_g2 = hgvs(adata2, n_top_genes=filter_gene)
    cgenes = list(set(hgv_g1) & set(hgv_g2))

    slices = [adata1[:, cgenes], adata2[:, cgenes]]
    pi12 = pst.pairwise_align(*slices, alpha=alpha, use_gpu =use_gpu, numItermax=numItermax, **kargs)
    new_slice0 = pst.stack_slices_pairwise(slices, [pi12])
    TX = new_slice0[0].obsm['spatial']
    TY = new_slice0[1].obsm['spatial']
    Xidx = new_slice0[0].obs_names
    Yidx = new_slice0[1].obs_names
    columns = list('XYZ')[:TX.shape[1]]

    posreg = pd.DataFrame(np.r_[ TX, TY], 
                          index=np.r_[Xidx, Yidx],
                          columns=columns)
    posreg['group'] = np.repeat(['f', 'm'], (TX.shape[0], TY.shape[0]))
    posreg[['X0', 'Y0']] = (np.r_[ slices[0].obsm['spatial'], slices[1].obsm['spatial']]) 
    return posreg

def paste2_regist(adata1, adata2, s=0.5, filter_gene= 10000, use_rep =None, dissimilarity='glmpca', **kargs):
    from paste2 import PASTE2, projection
    adata1 = adata1.copy()
    adata2 = adata2.copy()
    sc.pp.filter_genes(adata1, min_cells=1)
    sc.pp.filter_genes(adata2, min_cells=1)

    hgv_g1 = hgvs(adata1, n_top_genes=filter_gene)
    hgv_g2 = hgvs(adata2, n_top_genes=filter_gene)
    cgenes = list(set(hgv_g1) & set(hgv_g2))

    slices = [adata1[:, cgenes], adata2[:, cgenes]]

    pi_AB = PASTE2.partial_pairwise_align(*slices, s=s, use_rep=use_rep, dissimilarity=dissimilarity, **kargs)
    new_slices = projection.partial_stack_slices_pairwise(slices, [pi_AB])
    TX = new_slices[0].obsm['spatial']
    TY = new_slices[1].obsm['spatial']
    Xidx = new_slices[0].obs_names
    Yidx = new_slices[1].obs_names
    # columns = list('XYZ')[:TX.shape[0]]

    posreg = pd.DataFrame(np.r_[ TX, TY], 
                          index=np.r_[Xidx, Yidx],
                          columns=['X', 'Y'])
    posreg['group'] = np.repeat(['f', 'm'], (TX.shape[0], TY.shape[0]))
    posreg[['X0', 'Y0']] = (np.r_[ slices[0].obsm['spatial'], slices[1].obsm['spatial']]) 
    return posreg

def santo_regist(adata1, adata2, filter_gene= 10000, epochs=30, mode = 'align',
                version=1, normal=True,
                 lr=0.01, k=5, alpha=0.1, diff_omics=False,  device = 'cpu'):
    import sys

    if version==1:
        sys.path.append('/gpfs/home/user19/JupyterCode/')
        import SANTO
        from SANTO.SANTO_utils import santo, simulate_stitching, evaluation
        from SANTO.SANTO_data import intersect
    else:
        sys.path.append('/gpfs/home/user19/WorkSpace/00sortware/SANTO/build/lib/')
        import santo
        from santo.utils import santo, simulate_stitching, evaluation
        from santo.data import intersect

    import easydict

    train_ad1 = adata1.copy()
    train_ad2 = adata2.copy()

    sc.pp.filter_genes(train_ad1, min_cells=1)
    sc.pp.filter_genes(train_ad2, min_cells=1)

    if normal:
        sc.pp.normalize_total(train_ad1)
        sc.pp.normalize_total(train_ad2)

    hgv_g1 = hgvs(train_ad1, n_top_genes=filter_gene)
    hgv_g2 = hgvs(train_ad2, n_top_genes=filter_gene)
    cgenes = list(set(hgv_g1) & set(hgv_g2))
    print('common gene', len(cgenes))
    train_ad1 = train_ad1[:, cgenes]
    train_ad2 = train_ad2[:, cgenes]

    args = easydict.EasyDict({})
    args.epochs = epochs
    args.lr = lr
    args.k = k
    args.alpha = alpha # weight of transcriptional loss
    args.diff_omics = diff_omics # whether to use different omics data
    args.mode = mode # Choose the mode among 'align', 'stitch' and None
    args.dimension =  train_ad1.obsm['spatial'].shape[1]  # choose the dimension of coordinates (2 or 3)
    args.device = device #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # choose the device
    align_source_cor, trans_dict = santo(train_ad2, train_ad1, args)

    TX = train_ad1.obsm['spatial']
    TY = align_source_cor
    Xidx = train_ad1.obs_names
    Yidx = train_ad2.obs_names
    columns = list('XYZ')[:TX.shape[1]]
    columns0= ['X0', 'Y0', 'Z0'][:TX.shape[1]]
    posreg = pd.DataFrame(np.r_[ TX, TY], 
                          index=np.r_[Xidx, Yidx],
                          columns=columns)
    posreg['group'] = np.repeat(['f', 'm'], (TX.shape[0], TY.shape[0]))
    posreg[columns0] = (np.r_[ train_ad1.obsm['spatial'], train_ad2.obsm['spatial']]) 
    return posreg

def spacel_regist(adata1, adata2, min_cells=8, n_neighbors=15, filter_gene= 10000):
    import SPACEL
    from SPACEL import Scube
    adata1 = adata1.copy()
    adata2 = adata2.copy()
    sc.pp.filter_genes(adata1, min_cells=min_cells)
    sc.pp.filter_genes(adata2, min_cells=min_cells)

    hgv_g1 = hgvs(adata1, n_top_genes=filter_gene)
    hgv_g2 = hgvs(adata2, n_top_genes=filter_gene)
    cgenes = list(set(hgv_g1) & set(hgv_g2))

    adata1 = adata1[:, cgenes]
    adata2 = adata2[:, cgenes]

    adata1.obs['slice'] = 0
    adata1.obs['slice'] = adata1.obs['slice'].astype('category')
    adata2.obs['slice'] = 1
    adata2.obs['slice'] = adata2.obs['slice'].astype('category')

    Scube.align([adata1, adata2],
        cluster_key='slice', 
        n_neighbors = 15, 
        n_threads=10,
        p=1,
        write_loc_path=None,
        )
    
    TX = adata1.obsm['spatial_aligned']
    TY = adata2.obsm['spatial_aligned']
    Xidx = adata1.obs_names
    Yidx = adata2.obs_names
    columns = list('XYZ')[:TX.shape[1]]

    posreg = pd.DataFrame(np.r_[ TX, TY], 
                          index=np.r_[Xidx, Yidx],
                          columns=columns)
    posreg['group'] = np.repeat(['f', 'm'], (TX.shape[0], TY.shape[0]))
    posreg[['X0', 'Y0']] = (np.r_[ adata1.obsm['spatial'], adata2.obsm['spatial']]) 
    return posreg

def spateo_regist(adata1, adata2, min_cells=8, filter_gene= 3000,
                  normal=True,  transtype='rigid', use_hvg=True,
                  spatial_key = 'spatial', mode = 'SN-S', device=None, max_iter=200,
                  key_added = 'align_spatial', **kargs):
    import sys
    import spateo
    import spateo as st
    import torch
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    slice1 = adata1.copy()
    slice2 = adata2.copy()
    del adata1, adata2

    if 'batch' in slice1.obs.columns:
        slice1.obs.drop('batch', axis=1, inplace=True)
    if 'batch' in slice2.obs.columns:
        slice2.obs.drop('batch', axis=1, inplace=True)

    slice1.layers["counts"] = slice1.X.copy()
    # sc.pp.filter_cells(slice1, min_genes=min_cells)
    sc.pp.filter_genes(slice1, min_cells=min_cells)
    if use_hvg:
        if normal:
            sc.pp.normalize_total(slice1)
            sc.pp.log1p(slice1)
        sc.pp.highly_variable_genes(slice1, n_top_genes=filter_gene)
        slice1 = slice1[:, slice1.var.highly_variable] 

    slice2.layers["counts"] = slice2.X.copy()
    # sc.pp.filter_cells(slice2, min_genes=10)
    if use_hvg:
        sc.pp.filter_genes(slice2, min_cells=min_cells)
        if normal:
            sc.pp.normalize_total(slice2)
            sc.pp.log1p(slice2)
        sc.pp.highly_variable_genes(slice2, n_top_genes=filter_gene)
        slice2 = slice2[:, slice2.var.highly_variable]
    cgene = set(slice1.var_names) & set(slice2.var_names)
    print(f'common gene: {len(cgene)}')
    slice1 = slice1[:, list(cgene)]
    slice2 = slice2[:, list(cgene)]

    st.align.group_pca([slice1,slice2], pca_key='X_pca')
    cluster_key = 'seurat_clusters'
    print("Running this notebook on: ", device)
    # spateo return aligned slices as well as the mapping matrix
    aligned_slices, pis = st.align.morpho_align(
        models=[slice1, slice2],
        verbose=False,
        mode = mode,
        spatial_key=spatial_key,
        key_added=key_added,
        max_iter=max_iter, 
        device=device, **kargs
    )
    # st.pl.overlay_slices_2d(slices = aligned_slices, spatial_key = key_added, height=5, overlay_type='backward')
    # st.pl.overlay_slices_2d(slices = aligned_slices, spatial_key = key_added+'_nonrigid', height=5, overlay_type='backward')
    del slice1, slice2
    del st
    if transtype == 'rigid':
        TX = aligned_slices[0].obsm[key_added]
        TY = aligned_slices[1].obsm[key_added]
    else:
        TX = aligned_slices[0].obsm[key_added+'_nonrigid']
        TY = aligned_slices[1].obsm[key_added+'_nonrigid']
    Xidx = aligned_slices[0].obs_names
    Yidx = aligned_slices[1].obs_names
    columns = list('XYZ')[:TX.shape[1]]
    columns0= ['X0', 'Y0', 'Z0'][:TX.shape[1]]

    posreg = pd.DataFrame(np.r_[ TX, TY], 
                          index=np.r_[Xidx, Yidx],
                          columns=columns)
    posreg['group'] = np.repeat(['f', 'm'], (TX.shape[0], TY.shape[0]))
    posreg[columns0] = (np.r_[ aligned_slices[0].obsm[spatial_key], 
                                   aligned_slices[1].obsm[spatial_key]]) 
    return aligned_slices, posreg

def spateo_regist_paraing(adata1, adata2, min_cells=8, filter_gene= 3000,
                        normal=True,  transtype='rigid', use_hvg=True,
                        beta_list = [1e-4, 1e-3, 1e-2, 1e-1, 1],
                        lambdaVF_list = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],
                        K_list=[50],
                        spatial_key = 'spatial', mode = 'SN-S', 
                        device=None, max_iter=500,
                        figscale=5, size=0.5, save=None,
                        key_added = 'align_spatial', **kargs):
    import sys
    import spateo
    import spateo as st
    import torch
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    slice1 = adata1.copy()
    slice2 = adata2.copy()
    del adata1, adata2

    import gc
    gc.collect()
    import gc
    gc.collect()

    slice1.layers["counts"] = slice1.X.copy()
    # sc.pp.filter_cells(slice1, min_genes=min_cells)
    sc.pp.filter_genes(slice1, min_cells=min_cells)
    if use_hvg:
        if normal:
            sc.pp.normalize_total(slice1)
            sc.pp.log1p(slice1)
        sc.pp.highly_variable_genes(slice1, n_top_genes=filter_gene)
        slice1 = slice1[:, slice1.var.highly_variable] 

    slice2.layers["counts"] = slice2.X.copy()
    # sc.pp.filter_cells(slice2, min_genes=10)
    if use_hvg:
        sc.pp.filter_genes(slice2, min_cells=min_cells)
        if normal:
            sc.pp.normalize_total(slice2)
            sc.pp.log1p(slice2)
        sc.pp.highly_variable_genes(slice2, n_top_genes=filter_gene)
        slice2 = slice2[:, slice2.var.highly_variable]
    cgene = set(slice1.var_names) & set(slice2.var_names)
    slice1 = slice1[:, list(cgene)]
    slice2 = slice2[:, list(cgene)]

    st.align.group_pca([slice1,slice2], pca_key='X_pca')
    cluster_key = 'seurat_clusters'
    print("Running this notebook on: ", device)
    
    nrow = len(beta_list)
    ncol = len(lambdaVF_list)
    fig, axs = plt.subplots(nrow, ncol, figsize=(figscale*ncol, figscale*nrow))
    for i, beta in enumerate(beta_list):
        for j, lambdaVF in enumerate(lambdaVF_list):
            print(beta, lambdaVF)
            import spateo as st
            aligned_slices, pis = st.align.morpho_align(
                models=[slice1, slice2],
                verbose=False,
                K=K_list[0],
                beta=beta,
                lambdaVF=lambdaVF,
                mode = mode,
                spatial_key=spatial_key,
                key_added=key_added,
                max_iter=max_iter, 
                device=device, **kargs
            )
            # st.pl.overlay_slices_2d(slices = aligned_slices, spatial_key = key_added, height=5, overlay_type='backward')
            # st.pl.overlay_slices_2d(slices = aligned_slices, spatial_key = key_added+'_nonrigid', height=5, overlay_type='backward')

            TX1 = aligned_slices[0].obsm[key_added]
            TY1 = aligned_slices[1].obsm[key_added]

            TX2 = aligned_slices[0].obsm[key_added+'_nonrigid']
            TY2 = aligned_slices[1].obsm[key_added+'_nonrigid']

            iax = axs[i,j]
            iax.scatter(TX2[:,0], TX2[:,1], s=size, edgecolors='none', c='#b39ddf')
            iax.scatter(TY2[:,0], TY2[:,1], s=size, edgecolors='none', c='c')
            iax.grid(False)
            iax.set_aspect('equal', adjustable='box')

            del aligned_slices
            del st
            import gc
            gc.collect()
            import gc
            gc.collect()

    fig.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()
    del slice1, slice2

    # if transtype == 'rigid':
    #     TX = aligned_slices[0].obsm[key_added]
    #     TY = aligned_slices[1].obsm[key_added]
    # else:
    #     TX = aligned_slices[0].obsm[key_added+'_nonrigid']
    #     TY = aligned_slices[1].obsm[key_added+'_nonrigid']
    # Xidx = aligned_slices[0].obs_names
    # Yidx = aligned_slices[1].obs_names
    # columns = list('XYZ')[:TX.shape[1]]

    # posreg = pd.DataFrame(np.r_[ TX, TY], 
    #                       index=np.r_[Xidx, Yidx],
    #                       columns=columns)
    # posreg['group'] = np.repeat(['f', 'm'], (TX.shape[0], TY.shape[0]))
    # posreg[['X0', 'Y0']] = (np.r_[ aligned_slices[0].obsm['spatial'], 
    #                                aligned_slices[1].obsm['spatial']]) 
    # return aligned_slices, posreg

def cast_clust(adata, groupby, output_path, basis='spatial', k=10, 
               resolution=0.5, filter_gene=15000,
                minibatch=True,gpu_t=None, **kargs):
    import CAST
    from CAST import CAST_MARK
    from CAST import CAST_STACK
    from CAST.CAST_Stack import reg_params
    import torch
    import scipy as sci
    adatas = adata.copy()
    adatas.layers['norm_1e4'] = sc.pp.normalize_total(adatas, target_sum=1e4, inplace=False)['X']

    if sci.sparse.issparse(adatas.layers['norm_1e4']):
        adatas.layers['norm_1e4'] = adatas.layers['norm_1e4'].toarray()
    samples = np.unique(adata.obs[groupby])

    from collections import OrderedDict
    coords_raw = OrderedDict()
    exp_dict = {}
    cellidx = {}
    for sample_t in samples:
        idx = adatas.obs_names[(adatas.obs[groupby] == sample_t)]
        coords_raw[sample_t] = torch.DoubleTensor(adatas[idx].obsm[basis])
        exp_dict[sample_t] =  adatas[idx].layers['norm_1e4']
        cellidx[sample_t] = idx
    embed_dict = CAST_MARK(coords_raw,exp_dict,output_path, gpu_t=gpu_t, **kargs)
    embed_arr = pd.concat([pd.DataFrame(embed_dict[sample_t].cpu().numpy(),index=cellidx[sample_t]) 
                            for sample_t in samples ], axis=0)
    adatas.obsm['cast_emb'] = embed_arr.loc[adatas.obs_names,].values

    sc.pp.neighbors(adatas, use_rep='cast_emb')
    sc.tl.leiden(adatas,resolution=resolution)
    sc.tl.umap(adatas)

    # cc.pl.splitplot(adatas, groupby='SID', splitby="kmeans", basis='spatial', size=5)

    from sklearn.cluster import KMeans, MiniBatchKMeans
    embed_stack = adatas.obsm['cast_emb']
    kmeans = KMeans(n_clusters=k,random_state=0).fit(embed_stack) if minibatch == False else MiniBatchKMeans(n_clusters=k,random_state=0).fit(embed_stack)
    adatas.obs['kmeans'] = kmeans.labels_.astype(str)
    sc.pl.umap(adatas, color=["leiden", 'kmeans'])
    return adatas

def cast_regist(adata1, adata2, output_path, filter_gene=15000, gpu_t=None,resolution=0.5, k=20,
                minibatch=True, dist_penalty1=0, min_cells=5, diff_step=5,
                                bleeding=500, dimens=2,  mesh_weight = [None],
                iterations=500, iterations_bs=[400],
                  basis='spatial',  normal=True, gpu=0, transform_type='affine', **kargs):
    import CAST
    from CAST import CAST_MARK
    from CAST import CAST_STACK
    from CAST.CAST_Stack import reg_params
    import torch
    from scipy.sparse import issparse, csr_matrix
    adata1 = adata1.copy()
    adata2 = adata2.copy()

    idx1 = sc.pp.filter_genes(adata1, min_cells=min_cells, inplace=False)[0]
    idx2 = sc.pp.filter_genes(adata2, min_cells=min_cells, inplace=False)[0]
    adata1 = adata1[:, idx1]
    adata2 = adata2[:, idx2]

    # if normal:
    #     adata1.layers['norm_1e4'] = sc.pp.normalize_total(adata1, target_sum=1e4, inplace=False)['X']
    #     adata2.layers['norm_1e4'] = sc.pp.normalize_total(adata2, target_sum=1e4, inplace=False)['X']
    # else:
    #     adata1.layers['norm_1e4'] = adata1.X.copy()
    #     adata2.layers['norm_1e4'] = adata2.X.copy()

    cgenes = list(set(adata1.var_names[idx1]) & set(adata1.var_names[idx2]))
    adataxy = ad.concat({'X': adata1[:, cgenes], 'Y':adata2[:, cgenes]}, label='Batch')
    #adataxy.obs.index = adataxy.obs.index + adataxy.obs['Batch'].astype(str)
    del adata1, adata2

    if normal:
        adataxy.layers['norm_1e4'] = sc.pp.normalize_total(adataxy, target_sum=1e4, inplace=False)['X']
    else:
        adataxy.layers['norm_1e4'] = adataxy.X.copy()


    # cgenes = hgvs(adataxy, n_top_genes=filter_gene, batch_key='Batch')
    print(f'common gene: {len(cgenes)}')
    # adataxy = adataxy[:, cgenes]

    if issparse(adataxy.X):
        adataxy.X = adataxy.X.toarray()
    if issparse(adataxy.layers['norm_1e4']):
        adataxy.layers['norm_1e4'] = adataxy.layers['norm_1e4'].toarray()

    samples = [ 'Y', 'X']
    from collections import OrderedDict
    coords_raw = OrderedDict()
    exp_dict = {}
    cellidx = {}
    for sample_t in samples:
        idx = adataxy.obs_names[(adataxy.obs['Batch'] == sample_t)]
        coords_raw[sample_t] = torch.DoubleTensor(adataxy[idx].obsm[basis][:,:dimens])
        exp_dict[sample_t] =  adataxy[idx].layers['norm_1e4']
        cellidx[sample_t] = idx
    embed_dict = CAST_MARK(coords_raw,exp_dict,output_path,  gpu_t=gpu_t,)
    embed_arr = pd.concat([pd.DataFrame(embed_dict[sample_t].cpu().numpy(),index=cellidx[sample_t]) 
                            for sample_t in samples ], axis=0)

    adataxy.obsm['cast_emb'] = embed_arr.loc[adataxy.obs_names,].values
    # sc.pp.neighbors(adataxy, use_rep='cast_emb')
    # sc.tl.leiden(adataxy,resolution=resolution)

    from sklearn.cluster import KMeans, MiniBatchKMeans
    embed_stack = adataxy.obsm['cast_emb']
    kmeans = KMeans(n_clusters=k,random_state=0).fit(embed_stack) if minibatch == False else MiniBatchKMeans(n_clusters=k,random_state=0).fit(embed_stack)
    adataxy.obs['kmeans'] = kmeans.labels_.astype(str)
    # sc.pl.embedding(adataxy, color=["leiden", 'kmeans'],basis=basis)

    if  transform_type == 'affine':
        iterations_bs = [0]
    elif transform_type == 'deformable':
         iterations_bs = iterations_bs
    params_dist = CAST.reg_params(dataname = samples[0], # S2 is the query sample
                                gpu = gpu,
                                diff_step =diff_step ,
                                #### Affine parameters
                                iterations=iterations,
                                dist_penalty1=dist_penalty1,
                                bleeding=bleeding,
 
                                #### FFD parameters
                                mesh_weight =mesh_weight ,
                                iterations_bs = iterations_bs,
                                 **kargs)
    params_dist.alpha_basis = torch.Tensor([1/1000,1/1000,1/50,5,5]).reshape(5,1).to(params_dist.device)
    # Run CAST Stack
    coords_reg = CAST.CAST_STACK(coords_raw,embed_dict,output_path,samples,params_dist)

    TX = coords_reg['X']
    TY = coords_reg['Y']
    Xidx = cellidx['X']
    Yidx = cellidx['Y']

    posreg = pd.DataFrame(np.r_[ TX, TY], 
                          index=np.r_[Xidx, Yidx],
                          columns=['X', 'Y'])
    posreg['group'] = np.repeat(['f', 'm'], (TX.shape[0], TY.shape[0]))
    posreg[['X0', 'Y0']] = (np.r_[ coords_raw['X'], coords_raw['Y'],]) 

    return adataxy, posreg

def ccf_embedding(adata1, adata2,  hidden_dims=[512, 40], n_epochs=1000, donormal=True,
               knn=8, n_top_genes=5000,  n_comps=50, min_cells=8, 
               use_gate = False, device='cuda:1',  gconvs='gatv3',**kargs):
    sys.path.append('/gpfs/home/user19/JupyterCode/CellCloudX_v126/')
    import cellcloudx
    import cellcloudx as cc
    from cellcloudx.nn._GATE import GATE


    slice1 = adata1.copy()
    slice2 = adata2.copy()
    del adata1, adata2

    slice1.raw = slice1.copy()
    # sc.pp.filter_cells(slice1, min_genes=min_cells)
    sc.pp.filter_genes(slice1, min_cells=min_cells)

    if donormal:
        sc.pp.normalize_total(slice1)
        sc.pp.log1p(slice1)
        sc.pp.highly_variable_genes(slice1, n_top_genes=n_top_genes)
        slice1 = slice1[:, slice1.var.highly_variable] 

    slice2.raw = slice2.copy()
    # sc.pp.filter_cells(slice2, min_genes=10)
    if donormal:
        sc.pp.normalize_total(slice2)
        sc.pp.log1p(slice2)
        sc.pp.highly_variable_genes(slice2, n_top_genes=n_top_genes)
        slice2 = slice2[:, slice2.var.highly_variable]
    cgene = list(set(slice1.var_names) & set(slice2.var_names))
    slice1 = slice1[:, list(cgene)]
    slice2 = slice2[:, list(cgene)]

    print(f'common gene: {len(cgene)}')
    adataxy = ad.concat({'X': slice1, 'Y': slice2}, label='Batch')
    adataxy.obs.index = adataxy.obs.index + adataxy.obs['Batch'].astype(str)
    adataxy.obs['Batch'] = pd.Categorical(adataxy.obs['Batch'], categories=['X', 'Y'])
    groupby = 'Batch'

    if use_gate:
        cc.tl.spatial_edges(adataxy, groupby=groupby, 
                            basis = 'spatial',
                            knn= knn,
                            radius=None,

                            simi_thred = 0.1,
                            show_hist=False)

        ccal = GATE( Lambda=0)
        ccal.train(adataxy, groupby='Batch',
                            basis='spatial', hidden_dims=hidden_dims, n_epochs=n_epochs, lr=1e-3, 
                        device=device, gconvs=gconvs,
                        step_size =500, step_gamma=0.5, use_scheduler=False)
    
    sc.tl.pca(adataxy, n_comps=n_comps)
    return adataxy
    # ccf_reg = cc.ag.ccf_wrap(adataxy.obsm['spatial'], adataxy.obsm['GATE'], adataxy.obs['Batch'], levels=['X', 'Y'])
    # ccf_reg.regists(root='X', method=method, transformer=transformer, tol=tol, **kargs)
    # return ccf_reg.TY

def ccf_embedding0(adata1, adata2,  hidden_dims=[512, 40], n_epochs=1000, donormal=True,
               knn=8, n_top_genes=5000,  n_comps=50,
               use_gate = False, device='cuda:1',  gconvs='gatv3',**kargs):
    sys.path.append('/gpfs/home/user19/JupyterCode/CellCloudX_v126/')
    import cellcloudx
    import cellcloudx as cc
    from cellcloudx.nn._GATE import GATE
    
    idx1 = sc.pp.filter_genes(adata1, min_cells=5, inplace=False)[0]
    idx2 = sc.pp.filter_genes(adata2, min_cells=5, inplace=False)[0]

    cgenes = list(set(adata1.var_names[idx1]) & set(adata2.var_names[idx2]))
    adataxy = ad.concat({'X': adata1[:, cgenes], 'Y':adata2[:, cgenes]}, label='Batch')
    adataxy.obs.index = adataxy.obs.index + adataxy.obs['Batch'].astype(str)
    adataxy.obs['Batch'] = pd.Categorical(adataxy.obs['Batch'], categories=['X', 'Y'])
    groupby = 'Batch'

    adataxy.raw = adataxy.copy()

    if use_gate:
        cc.tl.spatial_edges(adataxy, groupby=groupby, 
                            basis = 'spatial',
                            knn= knn,
                            radius=None,

                            simi_thred = 0.1,
                            show_hist=False)

    adataxy = cc.pp.NormScale(adataxy, batch_key='Batch', donormal=donormal, doscale=False,
                            n_top_genes = n_top_genes,
                            minnbat=2,
                            min_mean=0.1, min_disp=0.15, max_mean=7)

    if use_gate:
        
        ccal = GATE( Lambda=0)
        ccal.train(adataxy, groupby='Batch',
                            basis='spatial', hidden_dims=hidden_dims, n_epochs=n_epochs, lr=1e-3, 
                        device=device, gconvs=gconvs,
                        step_size =500, step_gamma=0.5, use_scheduler=False)
    sc.tl.pca(adataxy, n_comps=n_comps)
    return adataxy
    # ccf_reg = cc.ag.ccf_wrap(adataxy.obsm['spatial'], adataxy.obsm['GATE'], adataxy.obs['Batch'], levels=['X', 'Y'])
    # ccf_reg.regists(root='X', method=method, transformer=transformer, tol=tol, **kargs)
    # return ccf_reg.TY

def ccf_regist(adata1, adata2, use_gate=False, basis='spatial',
                hidden_dims=[512, 40], n_epochs=500, donormal=True,
               knn=8, n_top_genes=5000,  n_comps=40,
                device='cuda:1', 
                use_rep=None,

                method=['ansac', 'ccd'], transformer=['rigid', 'rigid'],
                    feat_normal='l2',
                    K = 100,
                    KF = None,
                    kd_method='sknn',
                    w=None,
                    # w_clip=[1e-5, 1-1e-5],
                    maxiter=[100,300],**kargs
             ):
    sys.path.append('/gpfs/home/user19/JupyterCode/CellCloudX_v126/')
    import cellcloudx as cc
    from scipy.sparse import issparse
    adataS = ccf_embedding(adata1, adata2, use_gate=use_gate, n_top_genes=n_top_genes,
                            donormal=donormal, device=device,knn=knn,
                           hidden_dims=hidden_dims, n_epochs=n_epochs, n_comps=n_comps )
    if use_rep is None:
        if use_gate:
            Features = adataS.obsm['GATE']
        else:
            Features = adataS.obsm['X_pca']
    else:
        if use_rep in adataS.obsm.keys():
            Features = adataS.obsm[use_rep]
        elif use_rep == 'X':
            Features = adataS.X
    
    if issparse(Features):
        Features = Features.toarray()
    
    if use_gate:
        ccf_reg = cc.ag.ccf_wrap(adataS.obsm[basis], Features, 
                                 adataS.obs['Batch'], levels=['X', 'Y'])
    else:
        ccf_reg = cc.ag.ccf_wrap(adataS.obsm[basis], Features,
                                  adataS.obs['Batch'], levels=['X', 'Y'])
    ccf_reg.regists(root='X', method=method, transformer=transformer,
                    feat_normal=feat_normal,
                    K = K,
                    KF = KF,
                    kd_method=kd_method,
                    w=w,
                    # w_clip=[1e-5, 1-1e-5],
                    maxiter=maxiter,
                     **kargs)
    adataS.obsm['ccf'] = ccf_reg.TY

    columns = list('XYZ')[:adataS.obsm['ccf'].shape[1]]
    columns = columns + [ f'{i}0'for i in columns]
    posreg = pd.DataFrame(np.c_[adataS.obsm['ccf'], adataS.obsm[basis]],
                          index=adataS.obs_names,
                          columns=columns)
    posreg['group'] = adataS.obs['Batch'].cat.rename_categories({'X':'f', 'Y':'m'})
    return adataS, ccf_reg, posreg

def P_value(df, groupy, stats):
    from scipy.stats import mannwhitneyu, ttest_ind
    pars = df[groupy].unique()
    p_vals = []
    for i in range(len(pars)-1):
        xdf = df[(df[groupy] == pars[i])]
        ydf = df[(df[groupy] == pars[i+1])]
        for ist in stats:
            #p1 = mannwhitneyu(xdf[ist], ydf[ist], method="exact")[1]
            ix = xdf[ist].values
            iy = ydf[ist].values
            ix = ix[~pd.isna(ix)]
            iy = iy[~pd.isna(iy)]

            p2 = ttest_ind(ix, iy)[1]
            ps = [pars[i], pars[i+1], ist, p2]
            p_vals.append(ps)
    return p_vals