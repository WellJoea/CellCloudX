import numpy as np
import pandas as pd
from ..plotting._colors import cmap3a


def points2ply(POINTs, RGBs, normal=True, save=None, 
               radius=0.5, max_nn=30, alpha = None,
               otype='pyntcloud'):
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

    if (RGBs.shape[1] ==3) and (not alpha is None):
        if type(alpha) in [float, int]:
            alpha = np.full(len(RGBs), alpha, dtype=RGBs.dtype)
        RGBs = np.c_[RGBs[:,:3], alpha]

    if isinstance(RGBs, np.ndarray):
        cloud.colors = o3d.utility.Vector3dVector(RGBs[:,:3])
        # if  np.issubdtype(RGBs.dtype, np.floating):
        #     cloud.colors = o3d.utility.Vector3dVector(RGBs[:,:3]) #rgba
        # elif np.issubdtype(RGBs.dtype, np.integer):
        #     cloud.colors = o3d.utility.Vector3iVector(RGBs[:,:3])
    else:
        cloud.colors = RGBs

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

        if np.issubdtype(RGBs.dtype, np.floating) and (RGBs.max()<= 1):
            RGBs = np.uint8(RGBs*255)
        print(RGBs)
        data = np.concatenate([POINTs,
                                np.asarray(cloud.normals), 
                                RGBs], axis=1)
        columns = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue', 'alpha']
        data = pd.DataFrame(data, 
                            columns=columns[:data.shape[1]])

        cloud = PyntCloud(data)
        if save:
            cloud.to_file(save, as_text=True)
    return cloud

def save_ply(POINTs, COLORs, save):
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(POINTs)
    cloud.colors = o3d.utility.Vector3dVector(COLORs)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    from pyntcloud import PyntCloud
    data = np.concatenate([np.asarray(cloud.points),
                            np.asarray(cloud.normals), 
                            np.asarray(cloud.colors)], axis=1)
    data = pd.DataFrame(data, 
                        columns=['x', 'y', 'z', 
                                'nx', 'ny', 'nz',
                                'red', 'green', 'blue'])
    plydt = PyntCloud(data)
    plydt.to_file(save, as_text=True)
    return plydt

def readply(path):
    """
    Reads a PLY file and returns a point cloud object.
    """
    from pyntcloud import PyntCloud
    return  PyntCloud.from_file(path)


# def get_colorrgb(pld_data, ity, cmap = cc.pl.cmap3a, vmax=8, cat_dict = cate_dict):
#     if ity in markers:
#         idata = pld_data[ity].clip(0,vmax).values
#         icolors = cmap(idata/idata.max())
#         # icolors[:,:3] = np.uint8(icolors[:,:3]*255)

#     else:
#         idata = pld_data[ity]
#         iorder = idata.cat.categories
#         icolor = [ i[:3] for i in cat_dict[ity]['colrgb']]
#         colmap = dict(zip(iorder, icolor))
#         icolors = idata.astype(str).map(colmap)
#         icolors = np.float32(icolors.map(list).tolist())/255
#     return icolors
