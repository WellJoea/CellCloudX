import numpy as np
import pandas as pd
import os
import sys
import json
import SimpleITK as sitk
import glob
import anndata as ad
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix

from ..plotting._colors import rgb_to_hex

#wget -r -np -R "index.html*" http://download.alleninstitute.org/publications/allen_mouse_brain_common_coordinate_framework/
# wget -r -np -R "index.html*" http://download.alleninstitute.org/informatics-archive/

_Times = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P28', 'P56']
_Test_data = 'Mouse Brain'

class load_gene_expression():
    def __init__(self, Times=None, Times_type=None, check_assert=True, ftype='csv'):
        '''
        Times_type: [1,3]
        1: Mouse Brain
        3: Developing Mouse Brain
        '''
        self.Times = Times if not Times is None else _Times
        self.Times_type = Times_type if not Times_type is None else [3]*len(self.Times)

        assert len(self.Times) == len(self.Times_type), 'Times and Times_type should have the same length'
    
        self.check_assert = check_assert        
        self.ftype = ftype
    
    def guess_ttype(self, Time, ttype):
        if (ttype is None) and Time in self.Times and Time != 'P56':
            ttype = 3
        elif (ttype is None) and Time == 'P56':
            print('Type will set to 1 for P56')
            ttype = 1
        if self.check_assert:
            assert ttype in [1, 3], 'Developing Mouse Brain 3, Mouse Brain 1'
        return ttype

    def _get_time_gene_infor(self, Time, ttype = None):
        ttype = self.guess_ttype(Time, ttype)
        self.gene_url = f"http://api.brain-map.org/api/v2/data/Gene/query.{self.ftype}?criteria=products[id$eq{ttype}]&num_rows=all&start_row=0"
        self.all_gene_slice = f"http://api.brain-map.org/api/v2/data/query.{self.ftype}?criteria=model::SectionDataSet,rma::criteria,[failed$_of_section],rma::options,[tabular$eq'plane_of_sections.name+as+plane','genes.acronym+as+gene','data_,data_sets.id']&num_rows=all&start_row=0"
        gene_df = pd.read_csv(self.gene_url, header=0, index_col=None)
        gene_df['Time'] = Time
        gene_df['Time_type'] = ttype

        return gene_df
    
    def get_times_genes_infor(self, ttype = None):
        genes_df = []
        for Time, ttype in zip(self.Times, self.Times_type):
            gene_df = self._get_time_gene_infor(Time, ttype)
            genes_df.append(gene_df)
        genes_df = pd.concat(genes_df, axis=0)
        return genes_df

    def _get_time_gene_id(self, Time, gene, ttype = None):
        ttype = self.guess_ttype(Time, ttype)
        time_url = f"http://api.brain-map.org/api/v2/data/SectionDataSet/query.csv?criteria=[failed$eq'false'],"\
                   f"products[id$eq{ttype}],genes[acronym$eq'{gene}'],specimen(donor(age[name$eq'{Time}']))&num_rows=all&start_row=0"
        data =pd.read_csv(time_url, header=0, index_col=None)
        data['Time'] = Time
        data['gene'] = gene
        data['Time_type'] = ttype
        return data

    def get_times_genes_id(self, n_jobs= 50, verbose=2):
        genes_df = self.get_times_genes_infor()
        # gene_slice = []
        # for i, irow in genes_df.iterrows():
        #     data = self._get_time_gene_id(irow['Time'], irow['acronym'], ttype=irow['Time_type'])
        #     gene_slice.append(data)

        genes_slice  = Parallel(n_jobs= n_jobs, backend='loky', verbose=verbose)\
                                (delayed(self._get_time_gene_id)(irow['Time'], irow['acronym'], ttype=irow['Time_type'])
                                for i, irow in genes_df.iterrows())
        genes_slice = pd.concat(genes_slice, axis=0)
        return genes_slice, genes_df

    def _get_gene_expression_from_url(self, slice_id, get_exp_type='energy', down_exp_types='energy,density,intensity',
                                    size=None,
                                    execute = 'url',
                                    timeout= 120,
                                    outdir=None):
        import requests, zipfile, io
        exp_url = f"http://api.brain-map.org/grid_data/download/{slice_id}?include={down_exp_types}"
        if execute == 'read':
            data = requests.get(exp_url) 
            z = zipfile.ZipFile(io.BytesIO(data.content))
            with z.open(f'{get_exp_type}.raw') as energy:
                content = energy.read()
            data = np.frombuffer(content, dtype="float32")
            if not size is None:
                data = data.resize(size)
            return data
        elif execute == 'url':
            return exp_url
        elif execute == 'write':
            try:
                r = requests.get(exp_url,  timeout=timeout)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                os.makedirs(outdir, exist_ok=True)
                z.extractall(outdir)
            except:
                print(f'{slice_id} failed')

    def get_genes_expression_data(self, genes_slice, writedir,
                                   get_exp_type='energy', down_exp_types='energy,density,intensity',
                                    check= True, download = True, deep_check = False,
                                    timeout=60, size = None, space=None,
                                    n_jobs= 15, verbose=2):
        import glob
        import SimpleITK as sitk
        import os
        self.fglist = []
        for i, irow in genes_slice.iterrows():
            id   = irow['id']
            gene = irow['gene']
            Time = irow['Time']
            Time_type = irow['Time_type']

            ipath = f'{writedir}/{Time_type}/{Time}/{gene}_{Time}_{id}/'
            if check:
                try:
                    if deep_check:
                        iexp = sitk.ReadImage( f'{ipath}/{get_exp_type}.mhd')
                        assert np.all(np.array(iexp.GetSize()) == np.array(size)) and \
                               np.all(np.array(iexp.GetSpacing()) == np.array(space))
                    else:
                        assert os.path.exists(f'{ipath}/{get_exp_type}.mhd') and os.path.exists(f'{ipath}/{get_exp_type}.raw')
                except:
                    self.fglist.append([id, ipath])
            else:
                self.fglist.append([id, ipath])
        
        if len(self.fglist) == 0:
            print('All data was found.')
        else:
            print(f'{len(self.fglist)} data not found.')
            print(f'Please check the detial gene list in `self.fglist`')

            if download and len(self.fglist) > 0:
                print('Downloading data ...')
                Parallel(n_jobs= n_jobs, backend='loky', verbose=verbose)\
                        (delayed(self._get_gene_expression_from_url)
                            (id, execute='write', outdir=ipath, get_exp_type=get_exp_type,  down_exp_types=down_exp_types, timeout=timeout)
                                for  id, ipath in self.fglist)
    
    def get_spacesize(self, vol_url=None, annt_url=None):
        vol_url = 'https://community.brain-map.org/uploads/short-url/vqGJrTYNDyoJZ6qCfAZon8GbnF8.csv' if vol_url is None else vol_url
        vol_df = pd.read_csv(vol_url)

        annt_url = 'https://community.brain-map.org/uploads/short-url/2nTPC7WTUwUTVeoiULI6A6zUDg2.csv' if annt_url is None else annt_url
        annt_df = pd.read_csv(annt_url)
        return vol_df, annt_df
        
    def _get_expression(self, ipath, exp_type ='energy', space=None, size=None):
        iexp = sitk.ReadImage( f'{ipath}/{exp_type}.mhd')
        iexp_np = sitk.GetArrayFromImage(iexp).transpose([2,1,0])

        if size is not None:
            assert np.all(np.array(iexp_np.shape) == np.array(size))
        if space is not None:
            assert np.all(np.array(iexp.GetSpacing()) == np.array(space))
        return iexp_np

    def get_Time_adata(self, path, genes_slice=None, n_jobs=50,  verbose=2, exp_type ='energy', space=None, size=None):
        if genes_slice is None:
            apaths = glob.glob(f'{path}/*', recursive=False)
        else:
            apaths = [ f'{path}/{irow["gene"]}_{irow["Time"]}_{irow["id"]}'
                       for i, irow in genes_slice.iterrows() ]
        paths = []
        dpaths =[]
        for ipath in apaths:
            if os.path.exists(f'{ipath}/{exp_type}.mhd') and os.path.exists(f'{ipath}/{exp_type}.raw'):
                paths.append(ipath)
            else:
                dpaths.append(ipath)

        self.paths = paths
        self.dpaths = dpaths
        if len(dpaths):
            print(f'Warning: {len(dpaths)} data not found!!')
            print(f'Please check the detial gene list in `self.dpaths`')

        genes = np.array([  '_'.join(os.path.basename(i).split('_')[:-2]) for i in paths ])
        slics = np.array([ os.path.basename(i).split('_')[-1] for i in paths ])
        assert len(paths) == len(genes)
        exps  = Parallel(n_jobs= n_jobs, backend='loky', verbose=verbose)\
                (delayed(self._get_expression)
                    (ipath, exp_type =exp_type, space=space, size=size )
                    for ipath in paths)
        exps = np.array(exps)

        ugenes = np.unique(genes)
        uexps = []
        uslics = []
        for igene in ugenes:
            idx = np.where(genes == igene)[0]
            uexps.append(exps[idx].mean(axis=0)) #
            uslics.append(';'.join(slics[idx].tolist()))
        uexps = np.array(uexps)
        assert uexps.shape == (len(ugenes), *size)
        uexps = uexps.reshape(len(ugenes),-1).T

        dim = size
        Gdid = np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]]
        locs = np.vstack([i.ravel() for i in Gdid]).T
        loct = locs * np.array(space)
        
        cells = [ '_'.join([ f'{i :02d}' for i in iline ]) for iline in locs ]

        adata = ad.AnnData(uexps.clip(0,None))
        adata.var_names = ugenes
        adata.var['Gene'] = ugenes
        adata.obs_names = cells
        adata.obs['gird_ident'] = cells
        adata.obsm['spatial_gird'] = locs
        adata.obsm['spatial'] = loct
        adata.uns['grid_infor'] = {'space' :space, 'size': size, 
                                    'exp_type': exp_type,'slice_id': uslics}
        adata.X = csr_matrix(adata.X)
        return adata
        kidx = np.any(uexps > 0, axis=1)
        return adata[kidx]

    def add_adata_infor(self, adata, Time = None, Time_type=None, gene_list=None, ganno_np=None, struct_df=None, tree_dff=None, inplace=True):
        if not inplace:
            adata = adata.copy()

        if not Time is None:
            adata.obs['Time'] = Time
            adata.obs_names = f'{Time}:' + adata.obs_names
        if not Time_type is None:
            adata.obs['Time_type'] = Time_type

        if not gene_list is None:
            adata.var.merge(gene_list, how='left', left_on='Gene', right_on='acronym', sort=False)
            adata.var.index = adata.var['Gene']
        
        if not ganno_np is None:
            #P56_1_gannt = 'http://download.alleninstitute.org/informatics-archive/current-release/mouse_annotation/P56_Mouse_gridAnnotation.zip'
            adata.obs['grid_id'] = ganno_np[adata.obsm['spatial_gird'][:,0], 
                                            adata.obsm['spatial_gird'][:,1],
                                            adata.obsm['spatial_gird'][:,2]]
            
            adata = adata[ (adata.obs['grid_id'] != 0) ].copy()
        
        if (not tree_dff is None) and (not struct_df is None):
            struct_df = struct_df.copy().infer_objects()
            drop_col = struct_df.columns[((struct_df.values == None).sum(0) == struct_df.shape[0])]
            struct_df.drop(columns=drop_col, inplace=True)

            tree_dff = tree_dff.copy()
            tree_dff.columns = tree_dff.columns.astype(str)
            adata.uns['structure_annot'] = {'struct_df': struct_df, 'tree_dff' : tree_dff}

            struct_df = struct_df.copy().set_index('id').sort_index()
            adata.obs[struct_df.columns] = struct_df.loc[adata.obs['grid_id']].values
            
            map_dict = struct_df['acronym'].to_dict()
            map_dict = (struct_df.index.astype(str) + ':' + struct_df['acronym'] + ':' +  struct_df['name']).to_dict()

            tree_dff = tree_dff.copy().replace(map_dict)
            tree_dff.columns = 'level_' + tree_dff.columns.astype(str)
            adata.obs[tree_dff.columns] = tree_dff.loc[adata.obs['grid_id']].values.astype(str)

            for col in ['acronym', 'name']:
                try:
                    adata.obs[col] = pd.Categorical(adata.obs[col], categories=struct_df[col])
                except:
                    adata.obs[col] = adata.obs[col].astype('category')
                adata.obs[col] = adata.obs[col].cat.remove_unused_categories()
                adata.uns[f'{col}_colors'] = [ '#' + dict(zip(struct_df[col], struct_df['color_hex_triplet']))[i]
                                            for i in adata.obs[col].cat.categories ]

        adata.obs = adata.obs.infer_objects()
        adata.var = adata.var.infer_objects()   
        if not inplace:
            return adata

    
class structure_json():
    def __init__(self, file=None, sgid=17):
        '''
        sgid:
            1:  "Mouse Brain Atlas"
            17: "Developing Mouse Brain Atlas"
            10: "Human Brain Atlas"
            16: "Developing Human Brain Atlas"
            8:  "Non-Human Primate Brain Atlas"
            15: "Glioblastoma"
        '''
        if file is None:
            file = f'https://api.brain-map.org/api/v2/structure_graph_download/{sgid}.json'
        self.file = file 
        self.sgid = sgid
        self.read_json()
        self.get_msg()

    def read_json(self):
        import json
        try:
            with open(self.file, 'r') as f:
                self.struct = json.load(f)
        except:
            import urllib.request, json 
            with urllib.request.urlopen(self.file) as url:
                self.struct = json.load(url)

    def get_msg(self):
        self.msg = self.struct['msg']
        assert isinstance(self.msg, list) and len(self.msg) == 1
        self.msg = self.msg[0]

    def get_attribute(self, curdict):
        return pd.Series({ k:v for k,v in curdict.items() if k != 'children'}) 

    def get_children(self, curdict, check=False):
        for icurdict in curdict['children']:
            if check: assert curdict['id'] == icurdict['parent_structure_id']
            self.infors.append(self.get_attribute(icurdict))
            self.get_children(icurdict)

    def get_trees(self):
        curdict = self.msg
        self.infors = []
        # self.infors = [self.get_attribute(curdict)]
        self.get_children(curdict)

    def to_structure_df(self):
        self.get_trees()
        struct_df =  pd.concat(self.infors, axis=1).T
        struct_df['sgid'] = self.sgid
        return struct_df.infer_objects()

    def to_tree_flatten(self, fill_by_pid=False, add_self=True):
        struct_df = self.to_structure_df()
        # struct_df['parent_structure_id'] = struct_df['parent_structure_id'].fillna(-2)
        struct_df[['st_level', 'id', 'parent_structure_id']] = struct_df[['st_level', 'id', 'parent_structure_id']].astype(np.int64)

        levels = np.unique(struct_df['st_level'].astype(np.int64))
        tree_mtx = -10 * np.ones( (struct_df.shape[0], len(levels)), dtype=np.int64)
        id_level = dict(zip(struct_df['id'], struct_df['st_level']))
        id_pid   = dict(zip(struct_df['id'], struct_df['parent_structure_id']))

        for irow, id in enumerate(struct_df['id'].values):
            while (id in id_pid.keys()):
                ilev = id_level[id]
                ipid = id_pid[id]
                icol = np.where(levels == ilev)[0]
                tree_mtx[irow, icol] = ipid
                id = ipid

        tree_mtx = pd.DataFrame(tree_mtx, columns=levels, index=struct_df['id'])
        if fill_by_pid:
            tree_mtx[tree_mtx == -10] = np.nan
            tree_mtx = tree_mtx.fillna(method='ffill', axis=1)
        if add_self:
            tree_mtx[tree_mtx.columns.shape[0]] = tree_mtx.index
        return tree_mtx.astype(np.int64)

def get_itk_label(path=None,):
    if path is None:
        itklab = '/home/zhouw/WorkSpace/11Project/06cerebellum_for_3d/CCF/DevCCF_ITK-SNAP_label_3.6.txt'
    else:
        itklab = path

    itklab = pd.read_csv(itklab, header=None, index_col=None, delimiter=r"\s+", quotechar = '"', comment="#")
    addlines = []
    if not 16001 in itklab.values:
        addlines.append([16001, 243,25, 60, 1, 1, 1, 'DPall'])
    if not 16114 in itklab.values:
        addlines.append([16114, 255, 72, 102, 1, 1, 1, 'MPall'])
    
    if len(addlines) > 0:
        add_annot = pd.DataFrame(addlines)
        itklab = pd.concat([itklab, add_annot], axis=0)

    itklab.columns=['IDX','R', 'G', 'B', 'A', 'VIS', 'MSH', 'LABEL']
    itklab['IDX'] = itklab['IDX'].astype(np.int64)
    itklab['RGB'] =  itklab[['R', 'G', 'B']].astype(np.int64).apply(tuple, axis=1)
    itklab['COLOR'] = itklab['RGB'].apply(rgb_to_hex)
    itklab = itklab.sort_values('IDX').reset_index(drop=True)

    colordict = dict(zip(itklab['IDX'].astype(str), itklab['COLOR']))
    labeldict = dict(zip(itklab['IDX'].astype(str), itklab['LABEL']))
    idxdict = dict(zip(itklab['LABEL'].astype(str), itklab['IDX']))

    itklab.index = itklab.IDX
    itklab['LABEL'] = pd.Categorical(itklab['LABEL'], categories=itklab['LABEL'])
    return itklab, colordict, labeldict, idxdict
