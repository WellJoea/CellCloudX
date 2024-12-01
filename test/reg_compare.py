import os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import seaborn as sns
import sys
import matplotlib.pyplot as plt


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

    posreg = pd.DataFrame(np.r_[ TX, TY], 
                          index=np.r_[Xidx, Yidx],
                          columns=columns)
    posreg['group'] = np.repeat(['f', 'm'], (TX.shape[0], TY.shape[0]))
    posreg[['X0', 'Y0']] = (np.r_[ train_ad1.obsm['spatial'], train_ad2.obsm['spatial']]) 
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
                  normal=True,  transtype='rigid',
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

    slice1.layers["counts"] = slice1.X.copy()
    # sc.pp.filter_cells(slice1, min_genes=min_cells)
    sc.pp.filter_genes(slice1, min_cells=min_cells)
    if normal:
        sc.pp.normalize_total(slice1)
        sc.pp.log1p(slice1)
    sc.pp.highly_variable_genes(slice1, n_top_genes=filter_gene)
    slice1 = slice1[:, slice1.var.highly_variable] 

    slice2.layers["counts"] = slice2.X.copy()
    # sc.pp.filter_cells(slice2, min_genes=10)
    sc.pp.filter_genes(slice2, min_cells=min_cells)
    if normal:
        sc.pp.normalize_total(slice2)
        sc.pp.log1p(slice2)
    sc.pp.highly_variable_genes(slice2, n_top_genes=filter_gene)
    slice2 = slice2[:, slice2.var.highly_variable]

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

    if transtype == 'rigid':
        TX = aligned_slices[0].obsm[key_added]
        TY = aligned_slices[1].obsm[key_added]
    else:
        TX = aligned_slices[0].obsm[key_added+'_nonrigid']
        TY = aligned_slices[1].obsm[key_added+'_nonrigid']
    Xidx = adata1.obs_names
    Yidx = adata2.obs_names
    columns = list('XYZ')[:TX.shape[1]]

    posreg = pd.DataFrame(np.r_[ TX, TY], 
                          index=np.r_[Xidx, Yidx],
                          columns=columns)
    posreg['group'] = np.repeat(['f', 'm'], (TX.shape[0], TY.shape[0]))
    posreg[['X0', 'Y0']] = (np.r_[ adata1.obsm['spatial'], adata2.obsm['spatial']]) 
    return posreg

def cast_clust(adata, groupby, output_path, basis='spatial', k=10, filter_gene=15000,
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
    sc.tl.leiden(adatas,resolution=1)
    sc.tl.umap(adatas)

    # cc.pl.splitplot(adatas, groupby='SID', splitby="kmeans", basis='spatial', size=5)

    from sklearn.cluster import KMeans, MiniBatchKMeans
    embed_stack = adatas.obsm['cast_emb']
    kmeans = KMeans(n_clusters=k,random_state=0).fit(embed_stack) if minibatch == False else MiniBatchKMeans(n_clusters=k,random_state=0).fit(embed_stack)
    adatas.obs['kmeans'] = kmeans.labels_.astype(str)
    sc.pl.umap(adatas, color=["leiden", 'kmeans'])
    return adatas

def cast_regist(adata1, adata2, output_path, filter_gene=15000, gpu_t=None,
                iterations=1000, iterations_bs=1000,
                  basis='spatial',  normal=True, gpu=0, transform_type='affine'):
    import CAST
    from CAST import CAST_MARK
    from CAST import CAST_STACK
    from CAST.CAST_Stack import reg_params
    import torch
    from scipy.sparse import issparse, csr_matrix
    adata1 = adata1.copy()
    adata2 = adata2.copy()

    idx1 = sc.pp.filter_genes(adata1, min_cells=8, inplace=False)[0]
    idx2 = sc.pp.filter_genes(adata2, min_cells=8, inplace=False)[0]

    cgenes = list(set(adata1.var_names[idx1]) & set(adata1.var_names[idx2]))
    adataxy = ad.concat({'X': adata1[:, cgenes], 'Y':adata2[:, cgenes]}, label='Batch')
    adataxy.obs.index = adataxy.obs.index + adataxy.obs['Batch'].astype(str)
    if normal:
        adataxy.layers['norm_1e4'] = sc.pp.normalize_total(adataxy, target_sum=1e4, inplace=False)['X']
    else:
        adataxy.layers['norm_1e4'] = adataxy.X.copy()

    cgenes = hgvs(adataxy, n_top_genes=filter_gene, batch_key='Batch')
    print(f'common gene: {len(cgenes)}')
    adataxy = adataxy[:, cgenes]

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
        coords_raw[sample_t] = torch.DoubleTensor(adataxy[idx].obsm[basis])
        exp_dict[sample_t] =  adataxy[idx].layers['norm_1e4']
        cellidx[sample_t] = idx
    embed_dict = CAST_MARK(coords_raw,exp_dict,output_path,  gpu_t=gpu_t,)


    if  transform_type == 'affine':
        iterations_bs = [0]
    elif transform_type == 'deformable':
         iterations_bs = [iterations_bs]
    params_dist = CAST.reg_params(dataname = samples[0], # S2 is the query sample
                                gpu = gpu,
                                diff_step = 5,
                                #### Affine parameters
                                iterations=iterations,
                                dist_penalty1=0,
                                bleeding=500,
                                d_list = [3,2,1,1/2,1/3],
                                attention_params = [None,3,1,0],
                                #### FFD parameters
                                dist_penalty2 = [0],
                                alpha_basis_bs = [500],
                                meshsize = [8],
                                iterations_bs = iterations_bs,
                                attention_params_bs = [[None,3,1,0]],
                                mesh_weight = [None])
    params_dist.alpha_basis = torch.Tensor([1/1000,1/1000,1/50,5,5]).reshape(5,1).to(params_dist.device)
    # Run CAST Stack
    coords_reg = CAST.CAST_STACK(coords_raw,embed_dict,output_path,samples,params_dist)

    TX = coords_reg['X']
    TY = coords_reg['Y']
    Xidx = adata1.obs_names
    Yidx = adata2.obs_names

    posreg = pd.DataFrame(np.r_[ TX, TY], 
                          index=np.r_[Xidx, Yidx],
                          columns=['X', 'Y'])
    posreg['group'] = np.repeat(['f', 'm'], (TX.shape[0], TY.shape[0]))
    posreg[['X0', 'Y0']] = (np.r_[ adata1.obsm['spatial'], adata2.obsm['spatial']]) 
    return posreg

def ccf_embedding(adata1, adata2,  hidden_dims=[512, 40], n_epochs=1000, donormal=True,
               knn=8, n_top_genes=5000,  n_comps=50,
               use_gate = False, device='cuda:1',  gconvs='gatv3',**kargs):
    sys.path.append('/home/zhouw/JupyterCode/')
    import cellcloud3d as cc

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
        ccal = cc.nn.GATE( Lambda=0)
        ccal.train(adataxy, groupby='Batch',
                            basis='spatial', hidden_dims=hidden_dims, n_epochs=n_epochs, lr=1e-3, 
                        device=device, gconvs=gconvs,
                        step_size =500, step_gamma=0.5, use_scheduler=False)
    sc.tl.pca(adataxy, n_comps=n_comps)
    return adataxy
    # ccf_reg = cc.ag.ccf_wrap(adataxy.obsm['spatial'], adataxy.obsm['GATE'], adataxy.obs['Batch'], levels=['X', 'Y'])
    # ccf_reg.regists(root='X', method=method, transformer=transformer, tol=tol, **kargs)
    # return ccf_reg.TY

def ccf_regist(adata1, adata2, use_gate=False,
                hidden_dims=[512, 40], n_epochs=500, donormal=True,
               knn=8, n_top_genes=5000,  n_comps=40,
                device='cuda:1', 

                method=['ansac', 'ccd'], transformer=['rigid', 'rigid'],
                    feat_normal='l2',
                    K = 20,
                    KF = None,
                    kd_method='sknn',
                    w=None,
                    # w_clip=[1e-5, 1-1e-5],
                    maxiter=[100,300],**kargs
             ):
    sys.path.append('/home/zhouw/JupyterCode/')
    import cellcloud3d as cc
    adataS = ccf_embedding(adata1, adata2, use_gate=use_gate, n_top_genes=n_top_genes,
                            donormal=donormal, device=device,knn=knn,
                           hidden_dims=hidden_dims, n_epochs=n_epochs, n_comps=n_comps )
    if use_gate:
        ccf_reg = cc.ag.ccf_wrap(adataS.obsm['spatial'], adataS.obsm['GATE'], 
                                 adataS.obs['Batch'], levels=['X', 'Y'])
    else:
        ccf_reg = cc.ag.ccf_wrap(adataS.obsm['spatial'], adataS.obsm['X_pca'],
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
    posreg = pd.DataFrame(np.c_[adataS.obsm['ccf'], adataS.obsm['spatial']],
                          index=adataS.obs_names,
                          columns=columns)
    posreg['group'] = adataS.obs['Batch'].cat.rename_categories({'X':'f', 'Y':'m'})
    return adataS, ccf_reg, posreg

def ground_truth_state(TX, TY, X_feat, Y_feat, CI=0.95):
    from scipy import stats
    sys.path.append('/home/zhouw/JupyterCode/')
    import cellcloud3d as cc
    K = int(TY.shape[0]*CI)

    src2, dst2, dist2 = cc.tl.coord_edges(TX, TY, knn = 1,  method='sknn')
    kidx = np.argpartition(dist2, K,)[:K]
    kidx2 = np.argsort(dist2)[:K]

    src2, dst2, dist2 = src2[kidx], dst2[kidx], dist2[kidx]
    cos2 = cc.tl.mtx_similarity(X_feat, Y_feat, method='cosine', pairidx=[src2, dst2])
    # pea2 = cc.tl.mtx_similarity(X_feat, Y_feat, method='pearson', pairidx=[src2, dst2])
    pea2 = stats.pearsonr(X_feat[src2], Y_feat[dst2], axis=1).statistic
    df =pd.DataFrame(np.c_[dist2, cos2, pea2],
                         columns=['distance(mean)', 'cosine(mean)', 'pearson(mean)'])
    return df

def plt_hist(dfs,  figsize=(4.5, 4.5), y_lim = (0,100), ys_lim = [-1,1], linewidth=1, save=None):
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