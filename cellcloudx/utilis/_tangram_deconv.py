import pandas as pd
import numpy as np
import scanpy as sc
from tqdm import tqdm

class seg_deconv(object):
    def __init__(self, adata_st, adata_sc, adata_map, img_seg=None, na_to_cell=True,
                 threshold=0.5, full_match=False):
        self.adata_st = adata_st
        self.adata_sc = adata_sc
        self.adata_map = adata_map
        self.img_seg = img_seg
        self.na_to_cell = na_to_cell
        self.threshold = threshold
        self.full_match = full_match

    def get_segment(self, adata_st=None, img_seg=None):
        if adata_st is None: 
            adata_st = self.adata_st

        if img_seg is None:
            assert "image_features" in adata_st.obsm.keys()
            img_seg = adata_st.obsm["image_features"]

        img_seg = img_seg.copy()

        centroids = img_seg[["segmentation_centroid"]].copy()

        centroids["centroids_idx"] = [
            np.array([f"{k}_{j}" for j in np.arange(i)], dtype="object")
            for k, i in zip(
                adata_st.obs.index.values,
                adata_st.obsm["image_features"]["segmentation_label"],
            )
        ]
        centroids_idx = centroids.explode("centroids_idx")
        centroids_coords = centroids.explode("segmentation_centroid")
        segmentation_df = pd.DataFrame(
            centroids_coords["segmentation_centroid"].to_list(),
            columns=["y", "x"],
            index=centroids_coords.index,
        )
        segmentation_df["centroids"] = centroids_idx["centroids_idx"].values
        segmentation_df.index.set_names("spot_idx", inplace=True)
        segmentation_df.reset_index(
            drop=False, inplace=True,
        )
        zero_idx = (segmentation_df['centroids'].isna() | 
                    segmentation_df[['x', 'y']].isna().any(axis=1))
    
        if self.na_to_cell and (zero_idx.any()):
            spa_pos = pd.DataFrame(adata_st.obsm["spatial"],
                                    columns= ['y', 'x'],
                                    index=adata_st.obs_names)
            zero_obs = segmentation_df['spot_idx'].values[zero_idx]
            segmentation_df.loc[zero_idx, 'centroids'] = zero_obs
            segmentation_df.loc[zero_idx, ['y', 'x']] = spa_pos.loc[zero_obs, ['y', 'x']].values
        else:
            if zero_idx.any():
                print(f'Zero sigmantation index: {segmentation_df["spot_idx"].values[zero_idx]}')
            segmentation_df = segmentation_df[(~zero_idx)]

        vcounts = segmentation_df['spot_idx'].value_counts()
        segmentation_df['cell_counts'] = vcounts[segmentation_df['spot_idx']].values

        segmentation_df.loc[segmentation_df['centroids'].isna(), 'cell_counts']=0
        na_seg = segmentation_df['spot_idx'].values[(segmentation_df['cell_counts'] == 0)]
        if len(na_seg):
            print(f'Zero sigmantation index: {na_seg}')

        adata_st.uns["seg_infor"] = segmentation_df
        adata_st.obsm["tangram_spot_centroids"] = centroids["centroids_idx"]

    @staticmethod
    def one_hot_encoding(l, keep_aggregate=False):
        df_enriched = pd.DataFrame({"cl": l})
        for i in l.unique():
            df_enriched[i] = list(map(int, df_enriched["cl"] == i))
        if not keep_aggregate:
            del df_enriched["cl"]
        return df_enriched

    def project_annot( self, adata_map=None, annotation="cell_type", seg_infor = None, threshold=None):
        threshold = self.threshold if threshold is None else threshold

        if adata_map is None: 
            adata_map = self.adata_map

        F_map = adata_map.copy()
        if "F_out" in adata_map.obs.keys():
            F_map = adata_map[adata_map.obs["F_out"] > threshold]

        df = self.one_hot_encoding(F_map.obs[annotation])
        match_frac = F_map.X.T @ df
        match_frac.index = F_map.var.index

        idx = adata_map.obs["F_out"]> threshold
        mapX_df = adata_map[idx].to_df()

        match_pair = mapX_df.idxmax(1).to_frame('sp_idx')
        match_pair['sc_idx'] = match_pair.index
        match_pair['map_type'] = 'max'

        if self.full_match:
            match_val_dict = {  i: match_pair[(match_pair['sp_idx']==i)]['sc_idx'].values 
                                    for i in match_pair['sp_idx'] }
            if seg_infor is None:
                seg_infor = self.adata_st.uns["seg_infor"]
            cell_count_dict = dict(zip(seg_infor['spot_idx'], seg_infor['cell_counts']))
            match_add = []
            for cidx, cval in cell_count_dict.items():
                mat_cell = match_val_dict.get(cidx, [])
                mval = len(mat_cell)
                if mval < cval:
                    add_val = cval - mval
                    add_cells = mapX_df[cidx]
                    add_cells = add_cells.loc[list(set(add_cells.index) - set(mat_cell))]
                    ind = np.argpartition(add_cells.values, -add_val)[-add_val:]
                    add_cells = add_cells.index.values[ind]
                    match_add.append([ [cidx]*add_val, add_cells])
            match_add = np.hstack(match_add).T
            match_add = pd.DataFrame(match_add, columns=['sp_idx', 'sc_idx'])
            match_add['map_type'] = 'add'
            match_add.index = match_add['sc_idx']
            match_pair = pd.concat([match_pair, match_add], axis=0)

        match_pair['sc_sp_annt'] = adata_map.obs.loc[match_pair.index, annotation]
        match_counts = match_pair[['sp_idx', 'sc_sp_annt']].value_counts().to_frame('counts').reset_index()
        match_counts = match_counts.pivot(index='sp_idx', columns='sc_sp_annt', values='counts')

        self.match_pair = match_pair
        self.match_counts = match_counts
        self.match_frac = match_frac

    def de_annot(self, adata_st=None, match_counts=None, is_frec=False, seg_infor=None):
        from tqdm import tqdm
        if adata_st is None: 
            adata_st = self.adata_st
        if match_counts is None:
            if is_frec:
                match_counts = self.match_frac
            else:
                match_counts = self.match_counts
        if seg_infor is None:
            seg_infor = adata_st.uns["seg_infor"]
        
        seg_infor = seg_infor[(seg_infor['cell_counts']>0)].copy()
        match_counts = match_counts.copy().fillna(0)
        match_counts = match_counts[(match_counts.sum(1)>0)]

        com_cells = set(seg_infor['spot_idx']) & set(match_counts.index)
        print(f'{len(com_cells)} spots will be deconvoluted.')
        cell_count_dict = dict(zip(seg_infor['spot_idx'], seg_infor['cell_counts']))

        annotate = []
        if is_frec:
            for cidx in tqdm(com_cells):
                cval = cell_count_dict[cidx]
                ict_val = match_counts.loc[cidx]
                ict_val = (ict_val/ict_val.sum())*cval
                ict_val = ict_val[(ict_val>0)].sort_values(ascending=False)
                ict_val = ict_val.round(0).astype(np.int64).clip(1, None)
                ikeep_n = int(min(ict_val.sum(), cval))

                idf = seg_infor[(seg_infor['spot_idx']==cidx)].head(ikeep_n).copy()
                ict_annt = np.repeat(ict_val.index, ict_val.values)[:ikeep_n]
                idf['annot'] = ict_annt
                annotate.append(idf)

        else:
            for cidx in tqdm(com_cells):
                cval = cell_count_dict[cidx]
                ict_val = match_counts.loc[cidx]
                ict_val = ict_val[(ict_val>0)].sort_values(ascending=False)
                ikeep_n = int(min(ict_val.sum(), cval))

                idf = seg_infor[(seg_infor['spot_idx']==cidx)].head(ikeep_n).copy()
                ict_annt = np.repeat(ict_val.index, ict_val.values)[:ikeep_n]
                idf['annot'] = ict_annt
                annotate.append(idf)

        annotate = pd.concat(annotate, axis=0)
        return annotate

    def de_sp(self, adata_st=None, adata_sc=None, match_pair=None, use_raw=True, seg_infor=None):
        if adata_st is None:
            adata_st = self.adata_st
        if adata_sc is None:
            adata_sc = self.adata_sc
        if match_pair is None:
            match_pair = self.match_pair

        idata_st = (adata_st.raw.to_adata() if use_raw else adata_st).copy()
        idata_sc = (adata_sc.raw.to_adata() if use_raw else adata_sc).copy()

        if hasattr(idata_st.X, "toarray"):
            idata_st.X = idata_st.X.toarray()
        if hasattr(idata_sc.X, "toarray"):
            idata_sc.X = idata_sc.X.toarray()

        if seg_infor is None:
            seg_infor = adata_st.uns["seg_infor"]
        
        seg_infor = seg_infor[(seg_infor['cell_counts']>0)].copy()
        com_cells = set(seg_infor['spot_idx']) & set(match_pair['sp_idx'])
        com_gene = list(set(idata_sc.var_names) & set(idata_st.var_names))
        idata_st = idata_st[:, com_gene]
        idata_sc = idata_sc[:, com_gene]

        print(f'{len(com_cells)} spots will be deconvoluted.')
        print(f'{len(com_gene)} gene will be deconvoluted.')
        cell_count_dict = dict(zip(seg_infor['spot_idx'], seg_infor['cell_counts']))

        anno_info = []
        anno_gene = []
        for cidx in tqdm(com_cells):
            cval = cell_count_dict[cidx]
            i_pair = match_pair[(match_pair['sp_idx']==cidx)]
            ikeep_n = int(min(i_pair.shape[0], cval))

            idf = seg_infor[(seg_infor['spot_idx']==cidx)].head(ikeep_n).copy()
            i_pair = i_pair.head(ikeep_n)
            idf[['sc_idx', 'sc_sp_annt']] = i_pair[['sc_idx', 'sc_sp_annt']].values
            
            X_c = idata_sc[idf['sc_idx'], :].X.copy()
            X_w = X_c/X_c.sum(0)
            X_w[np.isnan(X_w)] = 1/idf.shape[0]
            X_t = idata_st[idf['spot_idx'], :].X * X_w

            anno_info.append(idf)
            anno_gene.append(X_t)

        anno_info = pd.concat(anno_info, axis=0)
        anno_gene = np.concatenate(anno_gene, axis=0)
        sc_gene = idata_sc[anno_info.sc_idx.values,:].X

        var_df = pd.DataFrame({"Gene": com_gene}, index=com_gene)
        obs_df = anno_info
        obs_df.index = anno_info['centroids']
        obsm = {'spatial': obs_df[['y', 'x']].values}
        uns = adata_st.uns

        adata_C = sc.AnnData( X=sc_gene, obs=obs_df, var=var_df, obsm=obsm, uns=uns)
        adata_T = sc.AnnData( X=anno_gene, obs=obs_df, var=var_df, obsm=obsm, uns=uns)
        return adata_C, adata_T

# devg = seg_deconv(adata_st, adata_sc, ad_map, full_match=False, threshold=0.5)
# devg.get_segment()
# devg.project_annot( annotation="cell_subclass")
# adata_C, adata_T = devg.de_sp(use_raw=True)