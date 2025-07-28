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
    sc.pp.filter_genes(adata1, min_cells=8)
    sc.pp.filter_genes(adata2, min_cells=8)

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
        sys.path.append('/home/zhouw/WorkSpace/00sortware/')
        import SANTO
        from SANTO.SANTO_utils import santo, simulate_stitching, evaluation
        from SANTO.SANTO_data import intersect
    else:
        sys.path.append('/home/zhouw/WorkSpace/00sortware/SANTO/build/lib/')
        import santo
        from santo.utils import santo, simulate_stitching, evaluation
        from santo.data import intersect

    import easydict

    train_ad1 = adata1.copy()
    train_ad2 = adata2.copy()

    if normal:
        sc.pp.normalize_total(train_ad1)
        sc.pp.normalize_total(train_ad2)

    sc.pp.filter_genes(train_ad1, min_cells=8)
    sc.pp.filter_genes(train_ad2, min_cells=8)

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
    sys.path.append('/home/zhouw/WorkSpace/00sortware/spateo-release-main/')
    import spateo
    import spateo as st
    import torch
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    slice1 = adata1.copy()
    slice2 = adata2.copy()
    del adata1, adata2

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
    sys.path.append('/home/zhouw/WorkSpace/00sortware/spateo-release-main/')
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
    sys.path.append('/home/zhouw/JupyterCode/CellCloudX/')
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
    sys.path.append('/home/zhouw/JupyterCode/CellCloudX/')
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
    sys.path.append('/home/zhouw/JupyterCode/CellCloudX/')
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

def ground_truth_state(TX, TY, X_feat, Y_feat, xlabels=None, knn=1, ylabels=None, CI=0.95):
    from scipy import stats
    from sklearn.metrics.cluster import adjusted_rand_score
    from sklearn.metrics import adjusted_mutual_info_score
    from sklearn.metrics import normalized_mutual_info_score

    sys.path.append('/home/zhouw/JupyterCode/CellCloudX/')
    import cellcloudx as cc
    if not CI is None:
        K = int(TY.shape[0]*CI) * knn

    sys.path.append('/home/zhouw/JupyterCode/CellCloudX/')
    import cellcloudx as cc
    TX_n = cc.ag.utility.centerlize(TX)[0]
    TY_n = cc.ag.utility.centerlize(TY)[0]

    src2, dst2, dist2 = cc.tl.coord_edges(TX, TY, knn = knn,  method='sknn')

    if not CI is None:
        kidx = np.argpartition(dist2, K,)[:K]
        # kidx2 = np.argsort(dist2)[:K]
        src2, dst2, dist2 = src2[kidx], dst2[kidx], dist2[kidx]

    # normd = np.sum(np.square(TX_n[src2] - TY_n[dst2]), axis=1)
    # normd = np.exp(-0.5* normd)

    cos2 = cc.tl.mtx_similarity(X_feat, Y_feat, method='cosine', pairidx=[src2, dst2])
    # pea2 = cc.tl.mtx_similarity(X_feat, Y_feat, method='pearson', pairidx=[src2, dst2])
    pea2 = stats.pearsonr(X_feat[src2], Y_feat[dst2], axis=1).statistic

    df = pd.DataFrame(np.c_[dist2, cos2, pea2],
                         columns=['distance(mean)', 'cosine(mean)', 'pearson(mean)'])

    if not (xlabels is None or ylabels is None):
        types = xlabels.columns.tolist()
        methods = [['ARI','AMI', 'NMI'], [adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score]]
        scores = []
        for itl in types:
            for inm, imd in zip(*methods):
                iscore = imd(xlabels[itl].values[src2], ylabels[itl].values[dst2])
                scores.append([f'{inm}({itl})', iscore])
        scores = pd.DataFrame(scores, columns=['types', 'score'])
        return df, scores
    else:
        return df

def plt_hist0(dfs,  figsize=(4.5, 4.5), y_lim = (0,100), ys_lim = [-1,1], linewidth=1, save=None):
    import seaborn as sns
    palette = ['#e41a1c','#4daf4a','#009db2', 'gold']
    #['#009db2', '#765005']
    fig, axs = plt.subplots(1, 3, figsize=figsize,
                             constrained_layout=False, sharex=False, sharey=False)

    sns.kdeplot(hue='method', y='distance(mean)',  linewidth=linewidth, cut=2,
                palette =palette,
                data=dfs, ax=axs[0], legend=False)
    axs[0].grid(False)
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    axs[0].set_ylim(*y_lim)
    
    sns.kdeplot(hue='method', y='cosine(mean)',  linewidth=linewidth, cut=2,
                palette =palette,
                data=dfs, ax=axs[1], legend=False)
    axs[1].grid(False)
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    axs[1].set_ylim(*ys_lim)
    
    ax2 = sns.kdeplot(hue='method', y='pearson(mean)',  linewidth=linewidth, cut=2,
                palette =palette,
                data=dfs,ax=axs[2], legend=True)
    axs[2].grid(False)
    axs[2].get_xaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)
    axs[2].set_ylim(*ys_lim)
    
    fig.tight_layout()
    if save:
        fig.savefig(f'{save}.score.pdf')
        dfs.to_csv(f'{save}.score.csv')
    plt.show()

def plot_regist_2d(posreg, save=None, figsize=(10,5), size=0.5, sharex=True, 
                   slice_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3']):
    X = posreg[(posreg['group'] == 'f')]
    Y = posreg[(posreg['group'] == 'm')]

    fig, ax = plt.subplots(1,2, figsize=figsize, sharex=sharex,)
    ax[0].scatter(X['X'], X['Y'], s=size, edgecolors='none', c='#b39ddf')
    ax[0].scatter(Y['X'], Y['Y'], s=size, edgecolors='none', c='c')
    ax[0].grid(False)
    ax[0].set_aspect('equal', adjustable='box')

    ax[1].scatter(X['X0'], X['Y0'], s=size, edgecolors='none', c='#b39ddf')
    ax[1].scatter(Y['X0'], Y['Y0'], s=size, edgecolors='none', c='c')
    ax[1].grid(False)
    ax[1].set_aspect('equal', adjustable='box')
    fig.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()

def plt_dens(dfs, figsize=(4.5, 4.5), y_lim = (0,200), ys_lim = [-1,1], linewidth=1, save=None):
    import seaborn as sns
    palette = COLOR
    dfs = dfs.reset_index(drop=True)
    #['#009db2', '#765005']
    fig, axs = plt.subplots(1, 3, figsize=figsize,
                             constrained_layout=False, sharex=False, sharey=False)

    sns.kdeplot(hue='method', y='distance(mean)',  linewidth=linewidth, cut=2,
                palette =palette,
                data=dfs, ax=axs[0], legend=False)
    axs[0].grid(False)
    # axs[0].get_xaxis().set_visible(False)
    # axs[0].get_yaxis().set_visible(False)
    axs[0].set_ylim(*y_lim)

    sns.kdeplot(hue='method', y='cosine(mean)',  linewidth=linewidth, cut=2,
                palette =palette,
                data=dfs, ax=axs[1], legend=False)
    axs[1].grid(False)
    # axs[1].get_xaxis().set_visible(False)
    # axs[1].get_yaxis().set_visible(False)
    axs[1].set_ylim(*ys_lim)

    ax2 = sns.kdeplot(hue='method', y='pearson(mean)',  linewidth=linewidth, cut=2,
                palette =palette,
                data=dfs,ax=axs[2], legend=True)
    axs[2].grid(False)
    # axs[2].get_xaxis().set_visible(False)
    # axs[2].get_yaxis().set_visible(False)
    axs[2].set_ylim(*ys_lim)

    fig.tight_layout()
    if save:
        fig.savefig(f'{save}.score.pdf',  transparent=True )
        dfs.to_csv(f'{save}.score.csv')
    plt.show()

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

def plt_hist(dfs, figsize=(4.5, 6), y_lim = (0,100), ys_min = None, ys_max = None, error =0,
             linewidth=1, save=None):
    import seaborn as sns
    palette = COLOR
    dfs = dfs.copy().reset_index(drop=True)

    idf = dfs.groupby('method').mean(0).reset_index()
    fig, axs = plt.subplots( 3,1, figsize=figsize,
                             constrained_layout=False, sharex=False, sharey=False)

    g = sns.barplot( y='method', x='distance(mean)',  linewidth=linewidth, 
                palette =palette,  errorbar=('ci', 95),
                data=dfs.copy(), ax=axs[0], legend=False)

    axs[0].grid(False)
    # axs[0].get_xaxis().set_visible(False)
    # axs[0].get_yaxis().set_visible(False)
    axs[0].set_xlim(y_lim[0], idf['distance(mean)'].max() if y_lim[1] is None else y_lim[1])
    axs[0].axvline(idf['distance(mean)'].min(), c='gray', linestyle='-.')

    g1 = sns.barplot( y='method', x='cosine(mean)',  linewidth=linewidth, 
                palette =palette, errorbar=('ci', 95),
                data=dfs.copy(), ax=axs[1], legend=False)
    axs[1].grid(False)
    # axs[1].get_xaxis().set_visible(False)
    # axs[1].get_yaxis().set_visible(False)
    ys_lim = [idf['cosine(mean)'].min() if ys_min is None else ys_min, 
              idf['cosine(mean)'].max()+error if ys_max is None else ys_max]
    axs[1].set_xlim(*ys_lim)

    g2 = sns.barplot( y='method', x='pearson(mean)',  linewidth=linewidth, 
                palette =palette, errorbar=('ci', 95),
                data=dfs, ax=axs[2], legend=False)
    axs[2].grid(False)
    # axs[2].get_xaxis().set_visible(False)
    # axs[2].get_yaxis().set_visible(False)
    ys_lim = [idf['pearson(mean)'].min() if ys_min is None else ys_min, 
              idf['pearson(mean)'].max()+error if ys_max is None else ys_max]
    axs[2].set_xlim(*ys_lim)

    fig.tight_layout()
    if save:
        fig.savefig(f'{save}.score.hist.pdf',transparent=True ,)
        idf.to_csv(f'{save}.score.hist.csv')
    plt.show()

def plt_scatter(dfs, txs, tys, save=None, size=1.5, wspace=0, hspace=0, sharey=True,sharex=True,
                rasterized=False, alpha=None, dpi=1000, title='use',
                invert_xaxis=False, invert_yaxis=False, figsize=(17,8),):
    MTs = ['raw'] + MT
    palette = ['gray'] + COLOR 
    import matplotlib.ticker as ticker
    
    fig, ax = plt.subplots(1,len(MTs), figsize=figsize, sharey=sharey,sharex=sharex)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    for i, (ilc, iname) in enumerate(zip(palette, MTs)):
        ialpha = alpha if i>0 else 1
        if i ==0:
            if invert_xaxis:
                ax[i].invert_xaxis()
            if invert_yaxis:
                ax[i].invert_yaxis()
        iX = txs[i].astype(np.float64)
        iY = tys[i].astype(np.float64)
        shift = iX.mean(0)

        iX -= shift
        iY -= shift
        ax[i].scatter(iX[:,0], iX[:,1], s=size, edgecolor = 'none', rasterized=rasterized, c='#b39ddf')
        ax[i].scatter(iY[:,0], iY[:,1], s=size, edgecolor = 'none', rasterized=rasterized, alpha=ialpha, c=palette[i])
        ax[i].grid(False)
        ax[i].set_aspect('equal', adjustable='box', anchor='C')
        if title=='use':
            ax[i].set_title(iname)
        elif not title is None:
            ax[i].set_title(title)
        # ax[i].set_axis_off()
        ax[i].xaxis.set_major_locator(ticker.NullLocator())
        ax[i].yaxis.set_major_locator(ticker.NullLocator())
    fig.tight_layout()
    if save:
        plt.savefig(f'{save}.scatter.pdf', dpi=dpi, transparent=True ,)
    plt.show()

def scatter3d_mpl(adata, groupby, basis='spatial',
                  colors=None, transparent=True,
                  figsize=(5,5), size= 0.4,
                  rasterized=True, alpha =1,
                  edgecolors='none' ,frame_type=4, xyz=[0,1,2],
                  show_legend=True, loc="center left", lncol=2, lfs=7, 
                  sharex=True, sharey=True, save=None, show=True, dpi=None,
                  markerscale=10, scatterpoints=1, bbox_to_anchor=(0.93, 0, 0.5, 1),
                  elev=None, azim=None, roll=None, ax=None, saveargs={},):
    from matplotlib.lines import Line2D
    colors = adata.uns[f'{groupby}_colors'] if colors is None else colors
    labels = adata.obs[groupby].cat.remove_unused_categories().cat.categories.tolist()
    colors1 = adata.obs[groupby].copy().cat.rename_categories(colors)
    mapdata= adata.obsm[basis][:,xyz]

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize,
                                 sharex=sharex, sharey=sharey,
                                 subplot_kw=dict(projection="3d"))
        plt.subplots_adjust(wspace=0,hspace=0)

    ax.view_init(elev=elev, azim=azim, roll=roll)

    ax.scatter(mapdata[:, 0], mapdata[:, 1], mapdata[:, 2], 
            s=size, 
            c=colors1, 
            edgecolors=edgecolors,
            rasterized=rasterized, 
            alpha=alpha)
    # for i, c, label in zip(range(len(labels)), colors, labels):
    #     widx = (adata.obs[groupby] == label)
    #     imap = mapdata[widx,:]
    #     ax.scatter(imap[:, 0], imap[:, 1], imap[:, 2], 
    #             s=size, 
    #             c=c, 
    #             label=label, 
    #             edgecolors=edgecolors,
    #             rasterized=rasterized, 
    #             alpha=alpha)

    cc.pl.add_frame(ax, frame_linewidth=0.3,
                    #xlims=(-1,10), ylims=(0,10),zlims=(-1,15),
                    frame_type=frame_type)
    ax.set_aspect('equal', 'box')
    ax.set_axis_off()

    legend_elements = [
                   Line2D([0], [0], marker='o', color=icl, label=ila,
                          linewidth=0,
                          markerfacecolor=icl, markersize=size)
    for icl, ila in zip(colors, labels) ]

    if show_legend:
        ax.legend( #title=groupby, 
                    handles=legend_elements, 
                    loc=loc, ncol=lncol,
                    prop={'size':lfs},
                    #alignment='left',
                    #scatterpoints=scatterpoints,
                    bbox_to_anchor=bbox_to_anchor,
                    #frameon=frameon,
                    #mode=mode,
                    markerscale=markerscale)
    try:
        fig.tight_layout()
    except:
        pass

    if save:
        fig.savefig(save, dpi=dpi, transparent=True, **saveargs)
    if show is None:
        return fig, ax
    elif show is True:
        plt.show()
    else:
        plt.close()

def spider(df, id_column=None, columns=None, max_values=None, 
            title=None, alpha=0.15, 
            color_bg='#A0A0A0', alpha_bg=0.05, 
            colors=None, fs='xx-small', fs_format = '.3f',
            padding=1.05, figsize=(5,5), rotate_label=True,
             show_legend=True, bbox_to_anchor=(0.1, 0.1),
            saveargs={}, show=True, save=None, ax=None,
            **kargs):
    columns = df._get_numeric_data().columns.tolist() if columns is None else columns
    data = df[columns].to_dict(orient='list')
    ids = df.index.tolist() if id_column is None else df[id_column].tolist()
    if max_values is None:
        max_values = {key: padding*max(value) for key, value in data.items()}

    normalized_data = {key: np.array(value) / max_values[key] for key, value in data.items()}
    num_vars = len(data.keys())
    tiks = list(data.keys())
    tiks += tiks[:1]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    degrees = [ np.degrees(angle-np.pi if angle>np.pi else angle ) - 90 for angle in angles]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    for i, model_name in enumerate(ids):
        values = [normalized_data[key][i] for key in data.keys()]
        actual_values = [data[key][i] for key in data.keys()]
        values += values[:1]  # Close the plot for a better look
        icolor = None if colors is None else colors[i]

        ax.plot(angles, values, c=icolor, label=model_name, **kargs)
        ax.fill(angles, values, c=icolor, alpha=alpha)
        for _x, _y, t, r in zip(angles, values, actual_values, degrees):
            t = f'{t :{fs_format}}' if isinstance(t, float) else str(t)
            ax.text(_x, _y, t, size=fs, rotation=r,
                    rotation_mode='anchor')

    if not color_bg is None:
        ax.fill(angles, np.ones(num_vars + 1), alpha=alpha_bg)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(tiks)
    # ax.grid(linewidth=3)

    if rotate_label:
        for label, degree in zip(ax.get_xticklabels(), degrees):
            x,y = label.get_position()
            lab = ax.text(x,y, label.get_text(), transform=label.get_transform(),
                          ha=label.get_ha(), va=label.get_va())
            lab.set_rotation(degree)
        ax.set_xticklabels([])

    if show_legend: ax.legend(loc='upper right', bbox_to_anchor=bbox_to_anchor)
    if title is not None: plt.suptitle(title)

    try:
        fig.tight_layout()
    except:
        pass

    if save:
        fig.savefig(save, **saveargs)
    if show is None:
        return fig, ax
    elif show is True:
        plt.show()
    else:
        plt.close()
