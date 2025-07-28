import pandas as pd
import numpy as np
import scanpy as sc

Xenium_sf = [0.2125, 0.425, 0.8500, 1.700, 3.400, 6.800, 13.6, 27.2]
def imread_pyramid(file, level=2, verbose=0):
    import tifffile

    imread_pyramid.sf = Xenium_sf
    with tifffile.TiffFile(file) as tif:
        for i in range(len(tif.series[0].levels)):
            if verbose: print(i, tif.series[0].levels[i].shape)
        image = tif.series[0].levels[level].asarray()
    return image


def imread_h5(file, key=None, get='h5_data', **kargs):
    import h5py
    with h5py.File(file, 'r',  **kargs) as h5f:
        if get == 'h5':
            return h5f
        key = list(h5f.keys())[0] if key is None else key
        img  = h5f.get(key)
        if get in ['h5_data']:
            return (img, h5f)
        elif get in ['data','image','img']:
            return img
        elif get=='array':
            img = np.array(img)
            h5f.close()
            return img

def read_image_info(adata, img_key="hires", basis = None,  order=None,
                    library_id=None, get_pix_loc=False, rescale=None):

    basis = basis or 'spatial'
    rescale = rescale or 1
    library_id = list(adata.uns[basis].keys())[0] if (library_id is None) else library_id
    
    img_dict = adata.uns[basis][library_id]
    iimg = img_dict['images'][img_key]
    scale_factor = img_dict['scalefactors'].get(f'tissue_{img_key}_scalef', 1)
    spot_diameter_fullres = img_dict['scalefactors'].get('spot_diameter_fullres',1)
    
    scales = scale_factor*rescale
    if rescale != 1:
        # import cv2
        # rsize = np.round(np.array(iimg.shape[:2])*rescale, 0)[::-1].astype(np.int32)
        # iimg = cv2.resize(iimg[:,:,::-1].copy(), rsize, interpolation= cv2.INTER_LINEAR)        
        import skimage as ski
        rsize = np.round(np.array(iimg.shape[:2])*rescale, 0).astype(np.int32)
        iimg = ski.transform.resize(iimg.copy(), rsize, order=order)
    locs = pd.DataFrame(adata.obsm[basis] * scales, 
                        index=adata.obs_names)

    st_loc = np.round(locs, decimals=0).astype(np.int32)
    #iimg = np.round(iimg*255)
    #iimg = np.clip(iimg, 0, 255).astype(np.uint32)
    if get_pix_loc:
        st_img = np.zeros(iimg.shape[:2], dtype=bool)
        st_img[st_loc[:,1], st_loc[:,0]] = True
        from scipy.ndimage.morphology import binary_dilation
        pix = np.round(spot_diameter_fullres*scale_factor/2).astype(np.int32)
        strel = np.ones((pix, pix))
        st_img = binary_dilation(st_img, structure=strel).astype(np.int32)
    else:
        st_img = st_loc
    return {"img":iimg,
            "locs":locs, 
            'loc_img':st_img,
            'scale_factor':scale_factor, 
            'rescale': scales,
            'spot_size':spot_diameter_fullres }

def read_h5_st(path, sid, use_diopy=False,
               assay_name='Spatial',
               slice_name=None):
    slice_name = sid if slice_name is None else slice_name 
    if use_diopy:
        import diopy 
        adata =  diopy.input.read_h5(f'{path}/{sid}.h5', assay_name=assay_name)
    else:
        adata = sc.read(f'{path}/{sid}.h5ad')

    with open(f'{path}/{sid}.scale.factors.json', 'r') as f:
        sf_info = json.load(f)

    coor = pd.read_csv( f'{path}/{sid}.coordinates.csv',index_col=0)
    #coor.index = coor.index + f':{sid}'
    image = np.transpose(np.load( f'{path}/{sid}.image.npy'), axes=(1,0,2))
    image = np.clip(np.round(image*255), 0, 255).astype(np.uint8)

    print(sf_info, coor.shape, adata.shape, image.shape)
    assert (coor.index != adata.obs_names).sum() == 0
    adata.obs[coor.columns] = coor
    adata.obsm['spatial'] = coor[['imagerow', 'imagecol']].values
    adata.uns['spatial'] = {}
    adata.uns['spatial'][slice_name] ={
        'images':{'hires':image, 'lowres':image},
        #unnormalized.radius <- scale.factors$fiducial_diameter_fullres * scale.factors$tissue_lowres_scalef
        #spot.radius <-  unnormalized.radius / max(dim(x = image))
        'scalefactors': {'spot_diameter_fullres': sf_info['fiducial'], ##??
                         'fiducial_diameter_fullres': sf_info['fiducial'],
                         'tissue_hires_scalef': sf_info['hires'], # ~0.17.
                         'tissue_lowres_scalef': sf_info['lowres'],
                         'spot.radius': sf_info['spot.radius'], 
                        },
        'metadata': {'chemistry_description': 'custom',
                       'spot.radius':  sf_info['spot.radius'], 
                       'assay': sf_info['assay'], 
                       'key': sf_info['key'], 
                      }
    }
    return(adata)

def svgtoimage(image, resize=None, save = None, return_np=True, bg=0x000000, dpi=72, format='PNG',
               MAX_IMAGE_PIXELS=None):
    import PIL
    PIL.Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

    from svglib.svglib import svg2rlg
    from io import BytesIO
    from reportlab.graphics import renderPM

    svgfile = svg2rlg(image)
    if save:
        renderPM.drawToFile(svgfile, save, bg=bg, dpi=dpi, fmt=format)
    if return_np:
        bytespng = BytesIO()
        renderPM.drawToFile(svgfile, bytespng, bg=bg, dpi=dpi, fmt="PNG")
        img = Image.open(bytespng)
        img = img.resize(resize, Image.Resampling.LANCZOS)
        img.image = img
        return np.array(img)

def points2ply(POINTs, COLORs, normal=True, save=None, 
               radius=0.1, max_nn=30, alpha = None,
               otype='o3d'):
    # import pyvista as pv
    # pvd = pv.PolyData(np.asarray(downpcd.points)[:,[0,1,2]].astype(np.float64))
    # pvd['point_color'] = np.asarray(downpcd.colors)
    # pvd = pv.PolyDataFilters.compute_normals(pvd)
    # pvd.save

    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    if isinstance(POINTs, np.ndarray):
        cloud.points = o3d.utility.Vector3dVector(POINTs)
    else:
        cloud.points = POINTs

    if not alpha is None:
        if type(alpha) in [float, int]:
            alpha = np.full(len(COLORs), alpha, dtype=COLORs.dtype)
        COLORs = np.c_[COLORs[:,:3], alpha]

    if isinstance(COLORs, np.ndarray):
        if  np.issubdtype(COLORs.dtype, np.floating):
            cloud.colors = o3d.utility.Vector3dVector(COLORs[:,:3]) #rgba
        elif np.issubdtype(COLORs.dtype, np.integer):
            cloud.colors = o3d.utility.Vector3iVector(COLORs[:,:3])
    else:
        cloud.colors = COLORs

    if len(cloud.normals) == 0:
        if normal is True:
            cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        elif not normal is None:
            if isinstance(normal, np.ndarray):
                cloud.normals = o3d.utility.Vector3dVector( normal )
            elif isinstance(normal, o3d.utility.Vector3dVector):
                cloud.normals = normal
            else:
                raise ValueError('normal must be a numpy array or o3d.utility.Vector3dVector')

    if otype == 'o3d':
        cloud = cloud
        if save:
            o3d.io.write_point_cloud(save, cloud)
    elif otype == 'pyntcloud':
        from pyntcloud import PyntCloud
        import pandas as pd

        if np.issubdtype(COLORs.dtype, np.floating) and (COLORs.max()<= 1):
            COLORs = COLORs*255

        data = np.concatenate([POINTs,
                                np.asarray(cloud.normals), 
                                COLORs], axis=1)
        columns = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue', 'alpha']
        data = pd.DataFrame(data, 
                            columns=columns[:data.shape[1]])
        cloud = PyntCloud(data)
        if save:
            cloud.to_file(save, as_text=True)
    return cloud

